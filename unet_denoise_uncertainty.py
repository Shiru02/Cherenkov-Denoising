"""
U-Net Denoising for Cherenkov Images - MC Dropout Uncertainty

Uncertainty approach: Monte Carlo Dropout (Gal & Ghahramani, 2016)
  - Dropout layers remain ACTIVE at inference time
  - N forward passes produce N different predictions (each pass drops different units)
  - Mean across passes  → best denoised estimate
  - Std  across passes  → epistemic uncertainty  (model doesn't know)
  - MAD  across passes  → aleatoric proxy        (data-level noise)
  - Combined (quadrature sum) → total uncertainty map

Why MC Dropout on top of the existing heteroscedastic model:
  - Heteroscedastic log_var head captures ALEATORIC uncertainty (photon shot noise)
  - MC Dropout captures EPISTEMIC uncertainty (model unsure about this input)
  - Together they cover: "region is noisy by physics" AND "model is out-of-distribution"

Key design decisions:
  - Spatial Dropout2d (drops entire feature channels) rather than element-wise dropout.
    This is more effective for CNNs: adjacent pixels share channels, so element-wise
    dropout is too easily interpolated away.
  - Dropout only in DECODER (not encoder/bottleneck).
    Encoder extracts features; adding dropout there hurts representation quality.
    Decoder is where predictions are assembled — this is where epistemic uncertainty
    should be expressed.
  - dropout_p=0.1 is conservative. For Cherenkov denoising where image quality
    matters, aggressive dropout (>0.2) degrades PSNR noticeably.
  - n_samples=20 is a good default. Returns diminish sharply after ~30 samples.

Loss: identical to hybrid A+B (NLL + L1 + SSIM + physics prior).
      MC Dropout adds NO extra training loss — it's purely an inference technique.

Author: Claude
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
import os
import sys
import glob
import math


# ==============================================================================
# ========================== LOSS FUNCTIONS ====================================
# ==============================================================================

class SSIMLoss(nn.Module):
    """SSIM loss: Loss = 1 - SSIM."""
    def __init__(self, window_size=11, sigma=1.5, channel=1):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, sigma, channel))
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _create_window(self, window_size, sigma, channel):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, pred, target):
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        channel = pred.size(1)
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel).to(pred.device)
            self.channel = channel

        mu_p = F.conv2d(pred,    self.window, padding=self.window_size//2, groups=channel)
        mu_t = F.conv2d(target,  self.window, padding=self.window_size//2, groups=channel)

        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t

        sig_p  = torch.clamp(F.conv2d(pred**2,        self.window, padding=self.window_size//2, groups=channel) - mu_p2, min=0)
        sig_t  = torch.clamp(F.conv2d(target**2,      self.window, padding=self.window_size//2, groups=channel) - mu_t2, min=0)
        sig_pt =              F.conv2d(pred * target,  self.window, padding=self.window_size//2, groups=channel) - mu_pt

        num = (2*mu_pt + self.C1) * (2*sig_pt + self.C2)
        den = (mu_p2 + mu_t2 + self.C1) * (sig_p + sig_t + self.C2)
        return 1 - (num / (den + 1e-8)).mean()


class HeteroscedasticLoss(nn.Module):
    """
    Hybrid A+B loss for uncertainty-aware denoising.

    L = w_nll × NLL + w_l1 × L1(μ,y) + w_ssim × SSIM(μ,y) + w_phys × |σ²-σ²_prior|

    NLL = 0.5 × [(y-μ)²/σ² + log(σ²)]
    σ²_prior = physics_scale × (μ × noise_level + variance_floor)   [Poisson prior]
    """
    def __init__(self, nll_weight=1.0, l1_weight=0.5, ssim_weight=0.5,
                 physics_weight=0.1, physics_scale=0.01, variance_floor=1e-4,
                 ssim_window_size=11):
        super().__init__()
        self.nll_weight     = nll_weight
        self.l1_weight      = l1_weight
        self.ssim_weight    = ssim_weight
        self.physics_weight = physics_weight
        self.physics_scale  = physics_scale
        self.variance_floor = variance_floor
        self.l1_loss   = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size)

    def forward(self, mean, log_var, target, noise_level, nll_scale=1.0):
        var = torch.exp(log_var)

        # Gaussian NLL
        nll = 0.5 * (((target - mean)**2) / (var + 1e-8) + log_var).mean()

        # Reconstruction losses on mean
        l1   = self.l1_loss(mean, target)
        ssim = self.ssim_loss(mean, target)

        # Poisson physics prior
        noise_level_map = noise_level[:, :, None, None]
        sigma2_prior = self.physics_scale * (mean.detach() * noise_level_map + self.variance_floor)
        physics_reg  = F.l1_loss(var, sigma2_prior)

        total = (self.nll_weight * nll_scale * nll
                 + self.l1_weight   * l1
                 + self.ssim_weight * ssim
                 + self.physics_weight * physics_reg)

        return total, {
            'total':    total.item(),
            'nll':      nll.item(),
            'l1':       l1.item(),
            'ssim':     ssim.item(),
            'physics':  physics_reg.item(),
            'mean_var': var.mean().item(),
            'nll_scale': nll_scale,
        }


# ==============================================================================
# ========================== DATASET ===========================================
# ==============================================================================

class PatchDataset(Dataset):
    """
    Loads .npy Cherenkov patch files of shape (H, W, num_cumulative_levels).
    Input: randomly selected early level (low SNR).
    Target: last level (high SNR ground truth).
    """
    def __init__(self, data_path, input_levels=None):
        self.data_path = data_path
        self.files = glob.glob(os.path.join(data_path, '*.npy'))
        if not self.files:
            raise ValueError(f"No .npy files found in {data_path}")

        sample = np.load(self.files[0])
        self.num_levels  = sample.shape[-1]
        self.patch_shape = sample.shape[:2]
        self.input_levels = input_levels if input_levels else list(range(self.num_levels - 1))

        print(f"Found {len(self.files)} patch files | shape {sample.shape} | "
              f"input levels {self.input_levels} | target level {self.num_levels-1}")
        print(f"Value range: [{sample.min():.2f}, {sample.max():.2f}]")

    def __len__(self):
        return len(self.files) * len(self.input_levels)

    def __getitem__(self, idx):
        file_idx     = idx // len(self.input_levels)
        level_idx    = idx %  len(self.input_levels)
        input_level  = self.input_levels[level_idx]

        patch = np.load(self.files[file_idx])           # (H, W, num_levels)
        if patch.max() > 1.0:
            patch = patch / patch.max()

        noisy = torch.tensor(patch[:, :, input_level], dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(patch[:, :, -1],          dtype=torch.float32).unsqueeze(0)
        noisy = torch.clamp(noisy, 0, 1)
        clean = torch.clamp(clean, 0, 1)

        noise_level = torch.tensor(
            [1.0 - input_level / (self.num_levels - 1)], dtype=torch.float32
        )
        return {'noisy': noisy, 'clean': clean,
                'noise_level': noise_level, 'level_idx': input_level}


# ==============================================================================
# ========================== MODEL =============================================
# ==============================================================================

class ConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU + residual."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        ng = min(8, out_channels)
        while out_channels % ng != 0:
            ng -= 1

        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(ng, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(ng, out_channels)
        self.act   = nn.SiLU(inplace=True)
        self.skip  = (nn.Conv2d(in_channels, out_channels, 1)
                      if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        ng = min(8, channels)
        while channels % ng != 0:
            ng -= 1
        self.norm  = nn.GroupNorm(ng, channels)
        self.qkv   = nn.Conv2d(channels, channels * 3, 1)
        self.proj  = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H*W)
        k = k.view(B, self.num_heads, self.head_dim, H*W)
        v = v.view(B, self.num_heads, self.head_dim, H*W)
        attn = F.softmax(torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale, dim=-1)
        out  = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class MCDropoutBlock(nn.Module):
    """
    Spatial (channel-wise) dropout that stays ACTIVE at inference.

    nn.Dropout2d zeros entire feature map channels rather than individual pixels.
    This is correct for CNNs: adjacent pixels in a channel are correlated, so
    element-wise dropout is trivially compensated — spatial dropout forces the
    network to be robust to missing channels, producing genuine uncertainty.

    The key trick: we call self.train() inside forward() so dropout is never
    disabled even when the parent model is in eval() mode.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p       = p
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        # Force training mode so F.dropout2d is always stochastic
        self.dropout.train()
        return self.dropout(x)


class UNetMCDropout(nn.Module):
    """
    U-Net with:
      1. Noise-level conditioning (MLP embedding injected at every decoder stage)
      2. Dual output heads: mean (denoised) + log_var (heteroscedastic uncertainty)
      3. MC Dropout in the decoder for epistemic uncertainty at inference

    Dropout placement rationale:
      - After every decoder ConvBlock output (d4, d3, d2, d1)
      - NOT in encoder or bottleneck (preserves feature extraction quality)
      - Spatial Dropout2d (drops whole channels, not individual activations)

    At inference:
      - Call mc_uncertainty(x, noise_level, n_samples=20)
      - Returns mean prediction + epistemic std + aleatoric variance + combined map
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 noise_embed_dim=128, dropout_p=0.1):
        super().__init__()
        ch = base_channels
        self.dropout_p = dropout_p

        # Noise level embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(1, noise_embed_dim), nn.SiLU(),
            nn.Linear(noise_embed_dim, noise_embed_dim), nn.SiLU(),
        )

        # Encoder (no dropout — preserve feature quality)
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch,    ch*2)
        self.enc3 = ConvBlock(ch*2,  ch*4)
        self.enc4 = ConvBlock(ch*4,  ch*8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (no dropout)
        self.bottleneck = nn.Sequential(
            ConvBlock(ch*8, ch*8),
            AttentionBlock(ch*8),
            ConvBlock(ch*8, ch*8),
        )

        # Noise conditioning projections (one per decoder stage + bottleneck)
        self.noise_proj_b = nn.Linear(noise_embed_dim, ch*8)
        self.noise_proj_4 = nn.Linear(noise_embed_dim, ch*4)
        self.noise_proj_3 = nn.Linear(noise_embed_dim, ch*2)
        self.noise_proj_2 = nn.Linear(noise_embed_dim, ch)
        self.noise_proj_1 = nn.Linear(noise_embed_dim, ch)

        # Decoder with MC Dropout after each stage
        self.up4   = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.dec4  = ConvBlock(ch*4 + ch*8, ch*4)
        self.drop4 = MCDropoutBlock(p=dropout_p)  # epistemic dropout

        self.up3   = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec3  = ConvBlock(ch*2 + ch*4, ch*2)
        self.drop3 = MCDropoutBlock(p=dropout_p)

        self.up2   = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.dec2  = ConvBlock(ch + ch*2, ch)
        self.drop2 = MCDropoutBlock(p=dropout_p)

        self.up1   = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1  = ConvBlock(ch + ch, ch)
        self.drop1 = MCDropoutBlock(p=dropout_p)

        # Dual output heads
        self.final_mean = nn.Conv2d(ch, out_channels, 1)

        # Log-var head: small extra conv to give it its own capacity
        # Bias initialised to -6 → initial variance ≈ exp(-6) ≈ 0.0025 (small)
        self.final_logvar = nn.Sequential(
            nn.Conv2d(ch, ch//2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch//2, out_channels, 1),
        )
        nn.init.constant_(self.final_logvar[-1].bias, -6.0)

    def forward(self, x, noise_level):
        """
        Single forward pass — used during training.

        Args:
            x:           Noisy input (B, 1, H, W)
            noise_level: Noise level (B, 1), range [0, 1]

        Returns:
            mean:    Denoised image (B, 1, H, W)
            log_var: Per-pixel log-variance (B, 1, H, W)
        """
        n_emb = self.noise_embed(noise_level)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck + noise conditioning
        b = self.bottleneck(self.pool(e4))
        b = b + self.noise_proj_b(n_emb)[:, :, None, None]

        # Decoder + noise conditioning + MC Dropout
        d4 = self.drop4(self.dec4(torch.cat([self.up4(b),  e4], dim=1)))
        d4 = d4 + self.noise_proj_4(n_emb)[:, :, None, None]

        d3 = self.drop3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)))
        d3 = d3 + self.noise_proj_3(n_emb)[:, :, None, None]

        d2 = self.drop2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d2 = d2 + self.noise_proj_2(n_emb)[:, :, None, None]

        d1 = self.drop1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))
        d1 = d1 + self.noise_proj_1(n_emb)[:, :, None, None]

        mean    = torch.clamp(x + self.final_mean(d1), 0, 1)
        log_var = torch.clamp(self.final_logvar(d1), min=-10.0, max=4.0)
        return mean, log_var

    @torch.no_grad()
    def mc_uncertainty(self, x, noise_level, n_samples=20):
        """
        Monte Carlo inference: run N stochastic forward passes, aggregate results.

        MCDropoutBlock.forward() forces dropout to stay active regardless of
        self.training, so this works correctly with model.eval().

        Uncertainty decomposition:
            epistemic_std:      std of mean predictions across passes
                                → model doesn't know (reducible with more data)
            aleatoric_variance: mean of exp(log_var) across passes
                                → irreducible physics noise (Poisson shot noise)
            combined:           sqrt(epistemic² + aleatoric)
                                → total uncertainty in quadrature

        Args:
            x:           (B, 1, H, W) noisy input
            noise_level: (B, 1) noise level
            n_samples:   number of MC forward passes (20 is a good default)

        Returns dict with keys:
            mean            (B, 1, H, W)  — best denoised estimate (mean of means)
            epistemic_std   (B, 1, H, W)  — epistemic uncertainty (model)
            aleatoric_var   (B, 1, H, W)  — aleatoric uncertainty (physics)
            combined        (B, 1, H, W)  — total uncertainty
            confidence      (B, 1, H, W)  — 1/(1 + combined/tau), range [0,1]
            raw_means       (N, B, 1, H, W) — all mean predictions (for analysis)
        """
        self.eval()  # freeze BN/GN stats — but MCDropoutBlock stays stochastic

        all_means   = []
        all_log_vars = []

        for _ in range(n_samples):
            mean_i, log_var_i = self.forward(x, noise_level)
            all_means.append(mean_i)
            all_log_vars.append(log_var_i)

        # Stack: (n_samples, B, 1, H, W)
        all_means    = torch.stack(all_means,    dim=0)
        all_log_vars = torch.stack(all_log_vars, dim=0)

        # --- Epistemic: variance of the mean predictions across passes ---
        mean_pred     = all_means.mean(dim=0)         # (B, 1, H, W)
        epistemic_std = all_means.std(dim=0)          # (B, 1, H, W)

        # --- Aleatoric: mean of the predicted variances across passes ---
        aleatoric_var = torch.exp(all_log_vars).mean(dim=0)  # (B, 1, H, W)

        # --- Combined uncertainty (quadrature sum) ---
        combined = torch.sqrt(epistemic_std**2 + aleatoric_var)

        # --- Confidence map: high = trustworthy denoising ---
        # tau = mean combined uncertainty (adaptive per-batch reference)
        tau = combined.mean().item()
        tau = max(tau, 1e-8)
        confidence = 1.0 / (1.0 + combined / tau)

        return {
            'mean':           mean_pred,
            'epistemic_std':  epistemic_std,
            'aleatoric_var':  aleatoric_var,
            'combined':       combined,
            'confidence':     confidence,
            'raw_means':      all_means,
        }


# ==============================================================================
# ========================== METRICS ===========================================
# ==============================================================================

def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return float('inf') if mse == 0 else 20*math.log10(max_val) - 10*math.log10(mse.item())

def compute_ssim(pred, target):
    with torch.no_grad():
        return 1 - SSIMLoss()(pred, target).item()


# ==============================================================================
# ========================== TRAINING CONFIG ===================================
# ==============================================================================

class TrainingConfig:
    def __init__(self):
        # === DATA — UPDATE THIS ===
        self.data_path = "output_noisy_patches/noisy_patches_20251015_143344"
        self.save_dir  = "experiments/unet_mc_dropout"

        # === MODEL ===
        self.model_config = {
            'in_channels':     1,
            'out_channels':    1,
            'base_channels':   64,
            'noise_embed_dim': 128,
            'dropout_p':       0.15,   # conservative; try 0.15 if uncertainty maps look flat
        }

        # === MC INFERENCE ===
        self.mc_n_samples = 20   # passes at inference; 10 = fast, 30 = more stable

        # === TRAINING ===
        self.num_epochs        = 150
        self.batch_size        = 16
        self.learning_rate     = 5e-5
        self.weight_decay      = 1e-4
        self.train_val_split   = 0.8

        # === LOSS (heteroscedastic NLL + L1 + SSIM + physics prior) ===
        self.loss_config = {
            'nll_weight':       1.0,
            'l1_weight':        0.5,
            'ssim_weight':      0.5,
            'physics_weight':   0.1,
            'physics_scale':    0.01,
            'variance_floor':   1e-4,
            'ssim_window_size': 11,
            'nll_warmup_epochs': 10,
        }

        # === SCHEDULER ===
        self.warmup_epochs  = 5
        self.min_lr_factor  = 0.01

        # === TRAINING OPTIONS ===
        self.mixed_precision    = True
        self.gradient_clip_norm = 1.0

        # === LOGGING ===
        self.save_freq        = 10
        self.eval_freq        = 5
        self.sample_freq      = 10
        self.num_save_samples = 4

        # === HARDWARE ===
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(4, os.cpu_count() or 2)
        self.pin_memory  = True

        # === AUTO PATHS ===
        self.checkpoint_dir  = Path(self.save_dir) / 'checkpoints'
        self.samples_dir     = Path(self.save_dir) / 'samples'
        self.logs_dir        = Path(self.save_dir) / 'logs'
        self.tensorboard_dir = Path(self.save_dir) / 'tensorboard'


# ==============================================================================
# ========================== TRAINER ===========================================
# ==============================================================================

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._setup_directories()
        self._setup_logging()
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        self.global_step   = 0
        self.train_losses  = []

    # ------------------------------------------------------------------
    def _setup_directories(self):
        for d in [self.config.save_dir, self.config.checkpoint_dir,
                  self.config.samples_dir, self.config.logs_dir,
                  self.config.tensorboard_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        log_file = self.config.logs_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

        self.loss_csv = self.config.logs_dir / 'training_history.csv'
        cols = ['epoch', 'train_loss', 'train_nll', 'train_l1', 'train_ssim',
                'train_physics', 'train_mean_var',
                'val_loss',   'val_nll',   'val_l1',   'val_ssim',
                'val_physics','val_mean_var', 'val_psnr', 'val_ssim_metric', 'lr']
        pd.DataFrame(columns=cols).to_csv(self.loss_csv, index=False)

        self.writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))
        self.logger.info(f"TensorBoard: tensorboard --logdir={self.config.tensorboard_dir}")

    def _setup_data(self):
        self.logger.info(f"Loading data from: {self.config.data_path}")
        full = PatchDataset(data_path=self.config.data_path)
        total = len(full)
        train_size = int(total * self.config.train_val_split)
        val_size   = total - train_size
        self.train_dataset, self.val_dataset = random_split(
            full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        kw = dict(num_workers=self.config.num_workers,
                  pin_memory=self.config.pin_memory)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                                       shuffle=True,  drop_last=True,  **kw)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=self.config.batch_size,
                                       shuffle=False, drop_last=False, **kw)
        self.logger.info(f"Train: {train_size} | Val: {val_size}")

    def _setup_model(self):
        self.model = UNetMCDropout(**self.config.model_config).to(self.config.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"UNetMCDropout | params: {n_params:,} | dropout_p: {self.config.model_config['dropout_p']}")

    def _setup_training(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        n_warmup = self.config.warmup_epochs * len(self.train_loader)
        n_total  = self.config.num_epochs    * len(self.train_loader)
        def lr_lambda(step):
            if step < n_warmup:
                return step / max(1, n_warmup)
            p = (step - n_warmup) / max(1, n_total - n_warmup)
            return max(self.config.min_lr_factor, 0.5*(1.0 + math.cos(math.pi*p)))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        lc = self.config.loss_config
        self.criterion = HeteroscedasticLoss(
            nll_weight=lc['nll_weight'], l1_weight=lc['l1_weight'],
            ssim_weight=lc['ssim_weight'], physics_weight=lc['physics_weight'],
            physics_scale=lc['physics_scale'], variance_floor=lc['variance_floor'],
            ssim_window_size=lc['ssim_window_size'],
        ).to(self.config.device)
        self.nll_warmup_epochs = lc['nll_warmup_epochs']

        self.scaler = torch.amp.GradScaler('cuda') if self.config.mixed_precision else None
        self.logger.info(f"Loss: HeteroscedasticLoss | NLL warmup: {self.nll_warmup_epochs} epochs")

    def _nll_scale(self, epoch):
        return min(1.0, epoch / max(1, self.nll_warmup_epochs))

    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        # During training, MCDropoutBlock.dropout is in train mode via the
        # parent model.train() call — dropout is active as normal.
        losses = {k: [] for k in ['total','nll','l1','ssim','physics','mean_var']}
        nll_s  = self._nll_scale(epoch)
        pbar   = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch in pbar:
            noisy       = batch['noisy'].to(self.config.device)
            clean       = batch['clean'].to(self.config.device)
            noise_level = batch['noise_level'].to(self.config.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                mean, log_var = self.model(noisy, noise_level)
                loss, ld      = self.criterion(mean, log_var, clean, noise_level, nll_scale=nll_s)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            for k in losses:
                if k in ld:
                    losses[k].append(ld[k])

            if self.global_step % 100 == 0:
                for k, v in ld.items():
                    if isinstance(v, float):
                        self.writer.add_scalar(f'Train/{k}_step', v, self.global_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.global_step)

            pbar.set_postfix({
                'loss': f"{ld['total']:.4f}",
                'var':  f"{ld['mean_var']:.5f}",
                'nll_s': f"{nll_s:.2f}",
            })

        avg = {k: float(np.mean(v)) for k, v in losses.items() if v}
        for k, v in avg.items():
            self.writer.add_scalar(f'Train/{k}_epoch', v, epoch)
        return avg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        """
        Validation uses single forward passes (fast).
        MC uncertainty is only computed in generate_samples for visualization.
        """
        self.model.eval()
        # For validation loss, we want deterministic outputs (no dropout variance).
        # Temporarily disable MCDropoutBlock stochasticity by patching:
        for m in self.model.modules():
            if isinstance(m, MCDropoutBlock):
                m.dropout.eval()

        losses = {k: [] for k in ['total','nll','l1','ssim','physics','mean_var']}
        psnrs, ssims = [], []
        nll_s = self._nll_scale(epoch)

        for batch in tqdm(self.val_loader, desc="Validating"):
            noisy       = batch['noisy'].to(self.config.device)
            clean       = batch['clean'].to(self.config.device)
            noise_level = batch['noise_level'].to(self.config.device)

            mean, log_var = self.model(noisy, noise_level)
            _, ld = self.criterion(mean, log_var, clean, noise_level, nll_scale=nll_s)

            for k in losses:
                if k in ld:
                    losses[k].append(ld[k])
            for i in range(mean.shape[0]):
                psnrs.append(compute_psnr(mean[i:i+1], clean[i:i+1]))
                ssims.append(compute_ssim(mean[i:i+1], clean[i:i+1]))

        # Re-enable MC dropout for inference
        for m in self.model.modules():
            if isinstance(m, MCDropoutBlock):
                m.dropout.train()

        avg      = {k: float(np.mean(v)) for k, v in losses.items() if v}
        avg_psnr = float(np.mean(psnrs))
        avg_ssim = float(np.mean(ssims))

        for k, v in avg.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)
        self.writer.add_scalar('Val/PSNR',        avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM_Metric',  avg_ssim, epoch)
        self.writer.add_scalars('Loss/Comparison', {
            'train': self.train_losses[-1] if self.train_losses else avg['total'],
            'val':   avg['total']
        }, epoch)

        self.logger.info(
            f"Val  loss={avg['total']:.4f}  PSNR={avg_psnr:.2f} dB  "
            f"SSIM={avg_ssim:.4f}  mean_var={avg.get('mean_var',0):.5f}"
        )
        return avg, avg_psnr, avg_ssim

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_samples(self, epoch):
        """
        Visualize denoising quality + full MC uncertainty decomposition.

        Columns: Noisy | MC Mean | Ground Truth | Epistemic Std | Aleatoric Var | Combined | Confidence
        """
        batch       = next(iter(self.val_loader))
        noisy       = batch['noisy'].to(self.config.device)
        clean       = batch['clean'].to(self.config.device)
        noise_level = batch['noise_level'].to(self.config.device)
        n = min(self.config.num_save_samples, noisy.shape[0])
        noisy, clean, noise_level = noisy[:n], clean[:n], noise_level[:n]

        # Full MC inference
        unc = self.model.mc_uncertainty(noisy, noise_level, n_samples=self.config.mc_n_samples)
        output      = unc['mean']
        epi_std     = unc['epistemic_std']
        ale_var     = unc['aleatoric_var']
        combined    = unc['combined']
        confidence  = unc['confidence']

        # Log scalars
        self.writer.add_scalar('MC/Epistemic_Mean', epi_std.mean().item(), epoch)
        self.writer.add_scalar('MC/Aleatoric_Mean', ale_var.mean().item(), epoch)
        self.writer.add_scalar('MC/Combined_Mean',  combined.mean().item(), epoch)
        self.writer.add_scalar('MC/Confidence_Mean',confidence.mean().item(), epoch)

        # Log images for first sample
        self.writer.add_images('MC/Epistemic',  epi_std[:1],  epoch)
        self.writer.add_images('MC/Aleatoric',  ale_var[:1],  epoch)
        self.writer.add_images('MC/Combined',   combined[:1], epoch)
        self.writer.add_images('MC/Confidence', confidence[:1], epoch)

        # ---- Figure ----
        cols   = ['Noisy Input', f'MC Mean\n({self.config.mc_n_samples} passes)',
                  'Ground Truth', 'Epistemic Std\n(model unc.)',
                  'Aleatoric Var\n(physics noise)', 'Combined Unc.', 'Confidence']
        cmaps  = ['gray', 'gray', 'gray', 'plasma', 'hot', 'hot', 'RdYlGn']
        tensors= [noisy, output, clean, epi_std, ale_var, combined, confidence]

        fig, axes = plt.subplots(n, 7, figsize=(28, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i in range(n):
            psnr_noisy  = compute_psnr(noisy[i:i+1],  clean[i:i+1])
            psnr_denois = compute_psnr(output[i:i+1], clean[i:i+1])
            for j, (t, title, cmap) in enumerate(zip(tensors, cols, cmaps)):
                ax  = axes[i, j]
                img = t[i, 0].cpu().numpy()
                im  = ax.imshow(img, cmap=cmap,
                                vmin=0 if cmap != 'RdYlGn' else 0,
                                vmax=1 if j == 6 else None)
                if i == 0:
                    ax.set_title(title, fontsize=9, pad=4)
                if j == 0:
                    ax.set_ylabel(f'Sample {i+1}\nPSNR noisy: {psnr_noisy:.1f}dB', fontsize=8)
                if j == 1:
                    ax.set_xlabel(f'PSNR: {psnr_denois:.1f} dB', fontsize=8)
                ax.axis('off')
                if j >= 3:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(
            f'Epoch {epoch+1} — MC Dropout Uncertainty (n_samples={self.config.mc_n_samples})',
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        path = self.config.samples_dir / f'samples_epoch_{epoch:03d}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved samples: {path}")

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch, val_loss, val_psnr, is_best=False):
        ckpt = {
            'epoch':               epoch,
            'global_step':         self.global_step,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict':self.scheduler.state_dict(),
            'val_loss':            val_loss,
            'val_psnr':            val_psnr,
            'model_config':        self.config.model_config,
            'loss_config':         self.config.loss_config,
        }
        torch.save(ckpt, self.config.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        torch.save(ckpt, self.config.checkpoint_dir / 'latest.pt')
        if is_best:
            torch.save(ckpt, self.config.checkpoint_dir / 'best_model.pt')
            self.logger.info(f"New best model! PSNR: {val_psnr:.2f} dB")

    def save_history(self, epoch, train_l, val_l=None, val_psnr=None, val_ssim=None):
        df  = pd.read_csv(self.loss_csv)
        row = {
            'epoch':          epoch,
            'train_loss':     train_l.get('total',    np.nan),
            'train_nll':      train_l.get('nll',      np.nan),
            'train_l1':       train_l.get('l1',       np.nan),
            'train_ssim':     train_l.get('ssim',     np.nan),
            'train_physics':  train_l.get('physics',  np.nan),
            'train_mean_var': train_l.get('mean_var', np.nan),
            'val_loss':       val_l.get('total',    np.nan) if val_l else np.nan,
            'val_nll':        val_l.get('nll',      np.nan) if val_l else np.nan,
            'val_l1':         val_l.get('l1',       np.nan) if val_l else np.nan,
            'val_ssim':       val_l.get('ssim',     np.nan) if val_l else np.nan,
            'val_physics':    val_l.get('physics',  np.nan) if val_l else np.nan,
            'val_mean_var':   val_l.get('mean_var', np.nan) if val_l else np.nan,
            'val_psnr':       val_psnr or np.nan,
            'val_ssim_metric':val_ssim or np.nan,
            'lr':             self.scheduler.get_last_lr()[0],
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.loss_csv, index=False)

    def plot_training_curves(self):
        df = pd.read_csv(self.loss_csv)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0,0].plot(df['epoch'], df['train_loss'], label='Train')
        axes[0,0].plot(df['epoch'], df['val_loss'],   label='Val')
        axes[0,0].set_title('Total Loss'); axes[0,0].legend(); axes[0,0].grid(True)

        axes[0,1].plot(df['epoch'], df['val_psnr'], 'g-o', markersize=3)
        axes[0,1].set_title('Validation PSNR (dB)'); axes[0,1].grid(True)

        axes[0,2].plot(df['epoch'], df['train_nll'], label='Train')
        axes[0,2].plot(df['epoch'], df['val_nll'],   label='Val')
        axes[0,2].set_title('NLL Loss'); axes[0,2].legend(); axes[0,2].grid(True)

        axes[1,0].plot(df['epoch'], df['train_physics'], label='Train')
        axes[1,0].plot(df['epoch'], df['val_physics'],   label='Val')
        axes[1,0].set_title('Physics Regularizer'); axes[1,0].legend(); axes[1,0].grid(True)

        axes[1,1].plot(df['epoch'], df['train_mean_var'], label='Train')
        axes[1,1].plot(df['epoch'], df['val_mean_var'],   label='Val')
        axes[1,1].set_title('Mean Predicted Variance σ²')
        axes[1,1].set_yscale('log'); axes[1,1].legend(); axes[1,1].grid(True)

        axes[1,2].plot(df['epoch'], df['lr'], 'r-')
        axes[1,2].set_title('Learning Rate')
        axes[1,2].set_yscale('log'); axes[1,2].grid(True)

        for ax in axes.flat:
            ax.set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(self.config.logs_dir / 'training_curves.png', dpi=150)
        plt.close()

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_psnr = ckpt.get('val_psnr', 0.0)
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        self.global_step   = ckpt.get('global_step', 0)
        self.logger.info(f"Resumed from epoch {ckpt['epoch']}, step {self.global_step}")
        return ckpt['epoch'] + 1

    # ------------------------------------------------------------------
    def train(self, resume_path=None):
        start_epoch = 0
        if resume_path and Path(resume_path).exists():
            start_epoch = self.load_checkpoint(resume_path)

        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"MC samples (inference): {self.config.mc_n_samples}")
        self.logger.info(f"Dropout p: {self.config.model_config['dropout_p']}")
        self.logger.info("=" * 60)

        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                train_l = self.train_epoch(epoch)
                self.train_losses.append(train_l['total'])

                val_l, val_psnr, val_ssim = None, None, None
                if (epoch + 1) % self.config.eval_freq == 0:
                    val_l, val_psnr, val_ssim = self.validate(epoch)
                    is_best = val_psnr > self.best_val_psnr
                    if is_best:
                        self.best_val_psnr = val_psnr
                        self.best_val_loss = val_l['total']
                    self.save_checkpoint(epoch, val_l['total'], val_psnr, is_best)

                self.save_history(epoch, train_l, val_l, val_psnr, val_ssim)

                if (epoch + 1) % self.config.sample_freq == 0:
                    self.generate_samples(epoch)

                if (epoch + 1) % 10 == 0:
                    self.plot_training_curves()

                msg = (f"Epoch {epoch+1}/{self.config.num_epochs} | "
                       f"Train {train_l['total']:.4f} | "
                       f"NLL warmup {self._nll_scale(epoch):.2f}")
                if val_l:
                    msg += f" | Val {val_l['total']:.4f} | PSNR {val_psnr:.2f} dB"
                self.logger.info(msg)

        except KeyboardInterrupt:
            self.logger.info("Interrupted — saving checkpoint")
            self.save_checkpoint(epoch, 0, 0)
            self.plot_training_curves()

        finally:
            self.writer.close()
            self.logger.info("Done.")

        self.logger.info(f"Best PSNR: {self.best_val_psnr:.2f} dB")


# ==============================================================================
# ===================== STANDALONE INFERENCE UTILITIES =========================
# ==============================================================================

def load_model(checkpoint_path, device='cuda'):
    """Load a saved UNetMCDropout checkpoint for inference."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNetMCDropout(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | PSNR {ckpt.get('val_psnr','?'):.2f} dB")
    return model


@torch.no_grad()
def denoise_with_uncertainty(model, noisy_image, noise_level=0.5,
                             n_samples=20, device='cuda'):
    """
    Convenience function: denoise a single image and return full uncertainty maps.

    Args:
        model:        Loaded UNetMCDropout (from load_model)
        noisy_image:  numpy array (H, W), values in [0, 1]
        noise_level:  float in [0, 1]; 0=clean, 1=very noisy
        n_samples:    MC passes (more = smoother uncertainty, slower)
        device:       'cuda' or 'cpu'

    Returns:
        dict of numpy arrays (H, W):
            denoised, epistemic_std, aleatoric_var, combined, confidence
    """
    x  = torch.tensor(noisy_image, dtype=torch.float32)[None, None].to(device)
    nl = torch.tensor([[noise_level]], dtype=torch.float32).to(device)

    unc = model.mc_uncertainty(x, nl, n_samples=n_samples)

    return {k: v[0, 0].cpu().numpy() for k, v in unc.items() if k != 'raw_means'}


# ==============================================================================
# ========================== MAIN ==============================================
# ==============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    config = TrainingConfig()

    # ============================================================
    # UPDATE THIS PATH TO YOUR DATA
    # ============================================================
    config.data_path = "noisy_patches"
    config.save_dir  = "experiments/unet_mc_dropout"
    # ============================================================

    if not os.path.exists(config.data_path):
        print(f"ERROR: data path not found: {config.data_path}")
        print("Update config.data_path to your .npy patch directory.")
        sys.exit(1)

    trainer = Trainer(config)
    resume  = config.checkpoint_dir / 'latest.pt'
    trainer.train(resume_path=resume if resume.exists() else None)


if __name__ == "__main__":
    main()