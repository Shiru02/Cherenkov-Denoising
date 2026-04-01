import torch
import torch.nn as nn
from network_module import *
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ----------------------------------------
#         MC Dropout Block
# ----------------------------------------
class MCDropoutBlock(nn.Module):
    """
    Spatial (channel-wise) dropout that stays ACTIVE at inference time.

    nn.Dropout2d zeros entire feature map channels rather than individual pixels.
    This is correct for CNNs: adjacent pixels in a channel are correlated, so
    element-wise dropout is trivially compensated — spatial dropout forces the
    network to be robust to missing channels, producing genuine uncertainty.

    Placed after the full DCR stacks in the res and e2e branches, just before
    the final projection convolutions (D4/D5). This probes epistemic uncertainty
    at the decision boundary — after all feature assembly is complete — which is
    the most physically meaningful location.

    self.dropout.train() is called explicitly in forward() so dropout remains
    stochastic even when the parent model is in eval() mode (needed for MC inference).
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        self.dropout.train()   # always stochastic, even during model.eval()
        return self.dropout(x)


# ----------------------------------------
#      Auxiliary blocks (unchanged)
# ----------------------------------------
class Block_of_DMT1(nn.Module):
    def __init__(self):
        super(Block_of_DMT1, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(160, affine=True)
        self.relu1_1 = nn.ReLU()

    def forward(self, x):
        return self.relu1_1(self.bn1_1(self.conv1_1(x)))


class Block_of_DMT2(nn.Module):
    def __init__(self):
        super(Block_of_DMT2, self).__init__()
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(256, affine=True)
        self.relu2_1 = nn.ReLU()

    def forward(self, x):
        return self.relu2_1(self.bn2_1(self.conv2_1(x)))


class Block_of_DMT3(nn.Module):
    def __init__(self):
        super(Block_of_DMT3, self).__init__()
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, affine=True)
        self.relu3_1 = nn.ReLU()

    def forward(self, x):
        return self.relu3_1(self.bn3_1(self.conv3_1(x)))


class Block_of_DMT4(nn.Module):
    def __init__(self):
        super(Block_of_DMT4, self).__init__()
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256, affine=True)
        self.relu4_1 = nn.ReLU()

    def forward(self, x):
        return self.relu4_1(self.bn4_1(self.conv4_1(x)))


class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        return out


# ----------------------------------------
#         Main network
# ----------------------------------------
class MyDNN(nn.Module):
    """
    Wavelet U-Net denoiser with:
      1. Sinusoidal PSNR embedding injected at every decoder stage (unchanged)
      2. Three output branches from the shared D3 feature map:
           - res branch:  subtracts estimated noise residual  (original)
           - e2e branch:  direct tanh image estimate          (original)
           - log_var head: per-pixel log-variance (aleatoric uncertainty) [NEW]
      3. MC Dropout in res and e2e branches, after full DCR stacks and
         before final projection convolutions (D4/D5)             [NEW]

    Forward returns: (mean, log_var)
      mean    = (e2e_part + res_part) / 2   — denoised image
      log_var = clamped per-pixel log σ²    — aleatoric uncertainty

    mc_uncertainty() runs N stochastic forward passes to decompose:
      epistemic_std  = std of mean predictions across passes  (model uncertainty)
      aleatoric_var  = mean of exp(log_var) across passes     (physics/shot noise)
      combined       = sqrt(epistemic² + aleatoric)           (total uncertainty)
      confidence     = 1 / (1 + combined/tau)                 (trustworthiness map)
    """

    def __init__(self, opt, time_emb_dim=256, dropout_p=0.1):
        super(MyDNN, self).__init__()

        # ---- Time/PSNR embedding ----
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        self.E11_time = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 320))
        self.E12_time = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 320))
        self.E13_time = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 320))
        self.E14_time = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 320))
        self.E2_time  = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 512))
        self.E3_time  = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 512))
        self.E4_time  = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(time_emb_dim, 256))

        # ---- Wavelet transforms ----
        self.DWT  = DWTForward(J=1, wave='haar').cuda()
        self.IDWT = DWTInverse(wave='haar').cuda()

        # ---- Encoder ----
        self.E1 = Conv2dLayer(in_channels=1,      out_channels=160, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)
        self.E2 = Conv2dLayer(in_channels=1*4,    out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)
        self.E3 = Conv2dLayer(in_channels=1*4*4,  out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)
        self.E4 = Conv2dLayer(in_channels=1*4*16, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)

        # ---- Bottleneck ----
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(256, 3, 1, 1, pad_type=opt.pad, norm=opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type=opt.pad, norm=opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type=opt.pad, norm=opt.norm),
            ResConv2dLayer(256, 3, 1, 1, pad_type=opt.pad, norm=opt.norm)
        )

        # ---- DCR blocks ----
        self.blockDMT11 = self.make_layer(_DCR_block, 320)
        self.blockDMT12 = self.make_layer(_DCR_block, 320)
        self.blockDMT13 = self.make_layer(_DCR_block, 320)
        self.blockDMT14 = self.make_layer(_DCR_block, 320)
        self.blockDMT21 = self.make_layer(_DCR_block, 512)
        self.blockDMT31 = self.make_layer(_DCR_block, 512)
        self.blockDMT41 = self.make_layer(_DCR_block, 256)

        # ---- Decoder projections ----
        self.D1 = Conv2dLayer(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)
        self.D2 = Conv2dLayer(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)
        self.D3 = Conv2dLayer(in_channels=512, out_channels=640,  kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, activation='prelu', norm=opt.norm)

        # res branch final conv (predicts noise residual)
        self.D4 = Conv2dLayer(in_channels=320, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, norm='none', activation='none')
        # e2e branch final conv (predicts clean image directly)
        self.D5 = Conv2dLayer(in_channels=320, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, pad_type=opt.pad, norm='none', activation='tanh')

        # ---- Channel shuffle / mixing ----
        self.S1 = Conv2dLayer(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, pad_type=opt.pad, activation='none', norm='none')
        self.S2 = Conv2dLayer(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, pad_type=opt.pad, activation='none', norm='none')
        self.S3 = Conv2dLayer(in_channels=320, out_channels=320, kernel_size=1, stride=1, padding=0, dilation=1, pad_type=opt.pad, activation='none', norm='none')
        self.S4 = Conv2dLayer(in_channels=320, out_channels=320, kernel_size=1, stride=1, padding=0, dilation=1, groups=3*320, pad_type=opt.pad, activation='none', norm='none')

        # ---- MC Dropout — placed after full DCR stacks, before D4/D5 ----
        # Probes epistemic uncertainty at the prediction decision boundary.
        # Spatial Dropout2d drops whole channels so uncertainty is not trivially
        # interpolated away by adjacent pixels sharing the same channel.
        self.drop_res = MCDropoutBlock(p=dropout_p)
        self.drop_e2e = MCDropoutBlock(p=dropout_p)

        # ---- log_var head — lightweight third branch off D3 ----
        # Estimates per-pixel aleatoric (Poisson shot noise) uncertainty.
        # Two conv layers only — no DCR blocks needed, the network just needs to
        # learn a smooth spatial variance map from the shared D3 features.
        # Bias initialised to -6 → initial σ² ≈ exp(-6) ≈ 0.0025 (small),
        # preventing the NLL term from dominating before the network is stable.
        self.log_var_head = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=160, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=160, out_channels=1, kernel_size=1),
        )
        nn.init.constant_(self.log_var_head[-1].bias, -6.0)

    def make_layer(self, block, channel_in):
        return nn.Sequential(block(channel_in))

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh

    def forward(self, x, t):
        """
        Single forward pass — used during training.

        Args:
            x: noisy input  (B, 1, H, W)
            t: PSNR index   (B,)  — used as the noise-level conditioning signal

        Returns:
            mean    (B, 1, H, W) — denoised image
            log_var (B, 1, H, W) — per-pixel log σ² (aleatoric uncertainty)
        """
        noisy = x
        t_emb = self.time_mlp(t)

        # ---- Encoder ----
        E1 = self.E1(x)

        DMT1_yl, DMT1_yh = self.DWT(x)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        E2 = self.E2(DMT1)

        DMT2_yl, DMT2_yh = self.DWT(DMT1)
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        E3 = self.E3(DMT2)

        DMT3_yl, DMT3_yh = self.DWT(DMT2)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        E4 = self.E4(DMT3)

        # ---- Decoder with skip connections and time conditioning ----
        E4 = self.blockDMT41(E4) + self.E4_time(t_emb)[..., None, None]

        D1 = self.D1(E4)
        D1 = self._Itransformer(D1)
        IDMT4 = self.IDWT(D1)
        D1 = torch.cat((IDMT4, E3), 1)
        D1 = self.S1(D1)

        D2 = self.blockDMT31(D1) + self.E3_time(t_emb)[..., None, None]
        D2 = self.D2(D2)
        D2 = self._Itransformer(D2)
        IDMT3 = self.IDWT(D2)
        D2 = torch.cat((IDMT3, E2), 1)
        D2 = self.S2(D2)

        D3 = self.blockDMT21(D2) + self.E2_time(t_emb)[..., None, None]
        D3 = self.D3(D3)
        D3 = self._Itransformer(D3)
        IDMT2 = self.IDWT(D3)
        D3 = torch.cat((IDMT2, E1), 1)
        # D3 is now (B, 320, H, W) — shared features for all three branches

        # ---- res branch ----
        res_part1 = self.S3(D3)
        res_part2 = self.blockDMT11(res_part1) + self.E11_time(t_emb)[..., None, None]
        res_part3 = res_part2 + res_part1
        res_part4 = self.blockDMT12(res_part3) + self.E12_time(t_emb)[..., None, None]
        res_part5 = res_part4 + res_part3
        # MC Dropout here: after full DCR stack, before final projection
        res_part5 = self.drop_res(res_part5)
        res_part6 = self.D4(res_part5)
        res_part  = noisy - res_part6

        # ---- e2e branch ----
        e2e_part1 = self.S4(D3)
        e2e_part2 = self.blockDMT13(e2e_part1) + self.E13_time(t_emb)[..., None, None]
        e2e_part3 = e2e_part2 + e2e_part1
        e2e_part4 = self.blockDMT14(e2e_part3) + self.E14_time(t_emb)[..., None, None]
        e2e_part5 = e2e_part4 + e2e_part3
        # MC Dropout here: after full DCR stack, before final projection
        e2e_part5 = self.drop_e2e(e2e_part5)
        e2e_part  = self.D5(e2e_part5)

        # ---- mean output ----
        mean = (e2e_part + res_part) / 2

        # ---- log_var branch: lightweight head off shared D3 features ----
        # No DCR blocks, no dropout — pure aleatoric signal from data statistics.
        # Clamped to [-10, 4] to keep σ² in a numerically stable range.
        log_var = torch.clamp(self.log_var_head(D3), min=-10.0, max=4.0)

        return mean, log_var

    @torch.no_grad()
    def mc_uncertainty(self, x, t, n_samples=20):
        """
        Monte Carlo inference: run N stochastic forward passes and decompose uncertainty.

        MCDropoutBlock forces dropout to stay active regardless of self.training,
        so this works correctly under model.eval() — BatchNorm/GroupNorm stats are
        frozen while dropout stays stochastic.

        Uncertainty decomposition:
            epistemic_std:    std of mean predictions across passes
                              → model doesn't know (reducible with more data / training)
            aleatoric_var:    mean of exp(log_var) across passes
                              → irreducible Poisson shot noise from the physics
            combined:         sqrt(epistemic² + aleatoric)  [quadrature sum]
                              → total uncertainty per pixel
            confidence:       1 / (1 + combined / tau)      [range 0–1]
                              → 1 = trustworthy denoising, 0 = high uncertainty

        Args:
            x:         (B, 1, H, W) noisy input
            t:         (B,)         PSNR index (noise-level conditioning)
            n_samples: int          number of MC forward passes (default 20)

        Returns dict:
            mean          (B, 1, H, W)
            epistemic_std (B, 1, H, W)
            aleatoric_var (B, 1, H, W)
            combined      (B, 1, H, W)
            confidence    (B, 1, H, W)
            raw_means     (n_samples, B, 1, H, W)  — all passes, for analysis
        """
        self.eval()  # freeze BN stats; MCDropoutBlock stays stochastic internally

        all_means    = []
        all_log_vars = []

        for _ in range(n_samples):
            mean_i, log_var_i = self.forward(x, t)
            all_means.append(mean_i)
            all_log_vars.append(log_var_i)

        # Stack to (n_samples, B, 1, H, W)
        all_means    = torch.stack(all_means,    dim=0)
        all_log_vars = torch.stack(all_log_vars, dim=0)

        # Epistemic: variance of mean predictions across passes
        mean_pred     = all_means.mean(dim=0)   # (B, 1, H, W)
        epistemic_std = all_means.std(dim=0)     # (B, 1, H, W)

        # Aleatoric: mean of predicted variances across passes
        aleatoric_var = torch.exp(all_log_vars).mean(dim=0)  # (B, 1, H, W)

        # Combined uncertainty (quadrature sum)
        combined = torch.sqrt(epistemic_std ** 2 + aleatoric_var)

        # Confidence map: adaptive tau = mean combined uncertainty across batch
        tau = max(combined.mean().item(), 1e-8)
        confidence = 1.0 / (1.0 + combined / tau)

        return {
            'mean':          mean_pred,
            'epistemic_std': epistemic_std,
            'aleatoric_var': aleatoric_var,
            'combined':      combined,
            'confidence':    confidence,
            'raw_means':     all_means,
        }
