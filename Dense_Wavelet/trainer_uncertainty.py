import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import dataset
import utils
import os
# import noise_generator
from tensorboardX import SummaryWriter
import scipy.io
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from utils import load_dict
import cv2
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ----------------------------------------
#              Utility losses
# ----------------------------------------
def TVLoss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    loss = 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    return loss


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def Asymmetricloss(noise_est, noise_level_map, alpha=0.3):
    batch_size = noise_est.size()[0]
    h = noise_est.size()[2]
    w = noise_est.size()[3]
    x = abs(noise_est) - abs(noise_level_map)
    mse = torch.mul(noise_est - noise_level_map, noise_est - noise_level_map)
    mask = torch.lt(x, 0)
    res = 0.3 * mse
    res[mask] = (1 - alpha) * mse[mask]
    res = torch.mean(res)
    return res


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


# ----------------------------------------
#     Uncertainty-aware loss function
# ----------------------------------------
class UncertaintyLoss(nn.Module):
    """
    Hybrid loss for uncertainty-aware denoising.

    L = w_l1   * L1(mean, target)
      + w_lap   * Laplacian(mean, target)
      + w_nll   * NLL(mean, log_var, target)        [NEW]
      + w_phys  * PhysicsPrior(var, mean, I_index)  [NEW]

    NLL term (Gaussian negative log-likelihood):
        NLL = 0.5 * [ (target - mean)² / σ² + log σ² ]
        Teaches the log_var head to predict high uncertainty where the
        reconstruction error is large.

    Physics prior (Poisson shot noise):
        σ²_prior = physics_scale * (|mean| * noise_level + variance_floor)
        Regularises the predicted variance toward the expected Poisson variance.
        noise_level is derived from I_index (input PSNR): lower PSNR → higher noise.
        L1 distance between predicted σ² and σ²_prior is minimised.

    NLL warmup:
        nll_scale ramps from 0 → 1 over nll_warmup_epochs.
        Prevents the NLL term from destabilising training before L1/Lap
        have brought the mean prediction to a reasonable quality.
    """
    def __init__(self,
                 l1_weight=1.0,
                 lap_weight=0.5,
                 nll_weight=0.5,
                 physics_weight=0.1,
                 physics_scale=0.01,
                 variance_floor=1e-4):
        super().__init__()
        self.l1_weight       = l1_weight
        self.lap_weight      = lap_weight
        self.nll_weight      = nll_weight
        self.physics_weight  = physics_weight
        self.physics_scale   = physics_scale
        self.variance_floor  = variance_floor
        self.l1_loss         = nn.L1Loss()

    def forward(self, mean, log_var, target, I_index, laplacian_loss, nll_scale=1.0):
        """
        Args:
            mean:           (B, 1, H, W)  denoised image
            log_var:        (B, 1, H, W)  per-pixel log σ²
            target:         (B, 1, H, W)  clean ground truth
            I_index:        float         batch PSNR of noisy input vs target
                                          (proxy for noise level; lower = noisier)
            laplacian_loss: float         pre-computed Laplacian edge loss scalar
            nll_scale:      float         warmup factor in [0, 1]

        Returns:
            total loss scalar, dict of individual loss components
        """
        var = torch.exp(log_var)

        # L1 reconstruction loss on the mean prediction
        l1 = self.l1_loss(mean, target)

        # Gaussian NLL: teaches log_var to track where reconstruction fails
        nll = 0.5 * (((target - mean) ** 2) / (var + 1e-8) + log_var).mean()

        # Poisson physics prior:
        # Convert I_index (PSNR) to a noise level in [0, 1].
        # PSNR range ~20 (very noisy) to ~50 (near-clean); we invert and normalise.
        # Clamp so the noise_level stays in a physically meaningful range.
        noise_level = torch.clamp(
            torch.tensor(1.0 - (I_index - 20.0) / 30.0, dtype=torch.float32),
            min=0.0, max=1.0
        ).to(mean.device)

        # σ²_prior = physics_scale * (|μ| * noise_level + variance_floor)
        # |μ| makes variance scale with local image intensity (Poisson property).
        sigma2_prior = self.physics_scale * (
            mean.detach().abs() * noise_level + self.variance_floor
        )
        physics_reg = nn.functional.l1_loss(var, sigma2_prior)

        # Total loss
        total = (self.l1_weight      * l1
                 + self.lap_weight   * laplacian_loss
                 + self.nll_weight   * nll_scale * nll
                 + self.physics_weight * physics_reg)

        return total, {
            'total':       total.item(),
            'l1':          l1.item(),
            'laplacian':   laplacian_loss if isinstance(laplacian_loss, float) else laplacian_loss.item(),
            'nll':         nll.item(),
            'physics':     physics_reg.item(),
            'mean_var':    var.mean().item(),
            'nll_scale':   nll_scale,
            'noise_level': noise_level.item(),
        }


# ----------------------------------------
#               Training
# ----------------------------------------
def MyDNN(opt):
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Directories
    save_folder   = os.path.join(opt.save_path, opt.task)
    sample_folder = os.path.join(opt.sample_path, opt.task)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Loss
    criterion = UncertaintyLoss(
        l1_weight      = 1.0,
        lap_weight     = 0.5,
        nll_weight     = 0.5,   # ramped up via nll_scale warmup
        physics_weight = 0.1,
        physics_scale  = 0.01,
        variance_floor = 1e-4,
    ).cuda()

    # NLL warmup: ramp nll_scale from 0 → 1 over this many epochs.
    # Keeps training stable while L1+Laplacian bring mean to a good quality first.
    nll_warmup_epochs = getattr(opt, 'nll_warmup_epochs', 20)

    # Generator
    generator = utils.create_MyDNN(opt)
    use_checkpoint = False
    if use_checkpoint:
        checkpoint_path = r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\final_denoise_epoch139_best.pth"
        pretrained_net = torch.load(checkpoint_path + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')

    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
    generator = generator.cuda()

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=opt.lr_g,
        betas=(opt.b1, opt.b2),
        weight_decay=opt.weight_decay
    )

    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        initial_lr = 0.0001
        final_lr   = 0.00001
        total_epochs = max(getattr(opt, 'epochs', 1), 1)
        lr = initial_lr + (final_lr - initial_lr) * min(epoch, total_epochs) / total_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def nll_scale(epoch):
        """Linearly ramp NLL weight from 0 to 1 over nll_warmup_epochs."""
        return min(1.0, epoch / max(1, nll_warmup_epochs))

    def compute_psnr(pred, target, max_val=255.0):
        """Compute PSNR between prediction and target."""
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val) - 10 * np.log10(mse)

    @torch.no_grad()
    def save_uncertainty_samples(generator, val_loader, sample_folder, epoch, n_samples=20, num_save_samples=4):
        """
        Visualize denoising quality + full MC uncertainty decomposition.
        
        Columns: Noisy | MC Mean | Ground Truth | Epistemic Std | Aleatoric Var | Combined | Confidence
        
        Args:
            generator: Model with mc_uncertainty method
            val_loader: Validation dataloader
            sample_folder: Directory to save samples
            epoch: Current epoch number
            n_samples: Number of MC forward passes
            num_save_samples: Number of samples to visualize
        """
        from network_emb_uncertainty import MCDropoutBlock
        
        # Get a batch from validation loader
        batch = next(iter(val_loader))
        true_input  = batch[0].cuda()
        true_target = batch[1].cuda()
        
        # Limit number of samples
        n = min(num_save_samples, true_input.shape[0])
        true_input  = true_input[:n]
        true_target = true_target[:n]
        
        # Compute I_index (PSNR) for conditioning
        I_index = utils.psnr(true_input, true_target, 255)
        batch_size = true_input.shape[0]
        I_index_tensor = torch.full((batch_size,), I_index, dtype=torch.float32).cuda()
        
        # Handle DataParallel models
        model_to_use = generator.module if hasattr(generator, 'module') else generator
        
        # Enable MC Dropout for uncertainty computation
        model_to_use.eval()
        for m in model_to_use.modules():
            if isinstance(m, MCDropoutBlock):
                m.dropout.train()  # Enable stochastic dropout
        
        # Run MC uncertainty inference
        unc = model_to_use.mc_uncertainty(true_input, I_index_tensor, n_samples=n_samples)
        
        output      = unc['mean']
        epi_std     = unc['epistemic_std']
        ale_var     = unc['aleatoric_var']
        combined    = unc['combined']
        confidence  = unc['confidence']
        
        # Disable MC Dropout after inference
        for m in model_to_use.modules():
            if isinstance(m, MCDropoutBlock):
                m.dropout.eval()
        
        # Normalize images to [0, 1] for visualization
        def normalize_tensor(t):
            t_min = t.min()
            t_max = t.max()
            if t_max > t_min:
                return (t - t_min) / (t_max - t_min)
            return t
        
        # Prepare tensors for visualization
        noisy_norm = normalize_tensor(true_input)
        output_norm = normalize_tensor(output)
        target_norm = normalize_tensor(true_target)
        epi_norm = normalize_tensor(epi_std)
        ale_norm = normalize_tensor(ale_var)
        comb_norm = normalize_tensor(combined)
        conf_norm = confidence  # Already in [0, 1]
        
        # Create figure
        cols = ['Noisy Input', f'MC Mean\n({n_samples} passes)',
                'Ground Truth', 'Epistemic Std\n(model unc.)',
                'Aleatoric Var\n(physics noise)', 'Combined Unc.', 'Confidence']
        cmaps = ['gray', 'gray', 'gray', 'plasma', 'hot', 'hot', 'RdYlGn']
        tensors = [noisy_norm, output_norm, target_norm, epi_norm, ale_norm, comb_norm, conf_norm]
        
        fig, axes = plt.subplots(n, 7, figsize=(28, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            # Compute PSNR for this sample
            psnr_noisy  = compute_psnr(true_input[i:i+1], true_target[i:i+1], max_val=255.0)
            psnr_denois = compute_psnr(output[i:i+1], true_target[i:i+1], max_val=255.0)
            
            for j, (t, title, cmap) in enumerate(zip(tensors, cols, cmaps)):
                ax  = axes[i, j]
                img = t[i, 0].cpu().numpy()
                
                # Set vmin/vmax for visualization
                vmin = 0 if cmap != 'RdYlGn' else 0
                vmax = 1 if j == 6 else None  # Confidence is already normalized
                
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                
                if i == 0:
                    ax.set_title(title, fontsize=9, pad=4)
                if j == 0:
                    ax.set_ylabel(f'Sample {i+1}\nPSNR noisy: {psnr_noisy:.1f}dB', fontsize=8)
                if j == 1:
                    ax.set_xlabel(f'PSNR: {psnr_denois:.1f} dB', fontsize=8)
                ax.axis('off')
                
                # Add colorbar for uncertainty maps
                if j >= 3:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(
            f'Epoch {epoch+1} — MC Dropout Uncertainty (n_samples={n_samples})',
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        
        # Save figure
        sample_path = os.path.join(sample_folder, f'uncertainty_samples_epoch_{epoch+1:03d}.png')
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Saved uncertainty samples: {sample_path}')

    def save_model(opt, epoch, iteration, len_dataset, generator, val_PSNR, best_PSNR):
        if opt.save_best_model and best_PSNR == val_PSNR:
            torch.save(generator, 'final_%s_epoch%d_best.pth' % (opt.task, epoch))
            print('The best model is successfully saved at epoch %d' % epoch)
        if opt.multi_gpu:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module,
                                   os.path.join(opt.save_path, opt.task,
                                                'MyDNN1_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size)))
                        print('The trained model is successfully saved at epoch %d' % epoch)
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module,
                                   'MyDNN1_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % iteration)
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator,
                                   'final_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % epoch)
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator,
                                   'final_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % iteration)

    # ----------------------------------------
    #             Datasets
    # ----------------------------------------
    trainset = dataset.Noise2CleanDataset(opt)
    print('The overall number of training images:', len(trainset))
    valset = dataset.ValDataset(opt)
    print('The overall number of val images:', len(valset))

    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(valset,   batch_size=opt.batch_size, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)

    # ----------------------------------------
    #               Training loop
    # ----------------------------------------
    prev_time  = time.time()
    best_PSNR  = 0
    start_epoch = getattr(opt, 'start_epoch', 1)

    for epoch in range(start_epoch - 1, opt.epochs):

        total_loss    = 0
        total_nll     = 0
        total_lap     = 0
        total_sobel   = 0
        total_physics = 0
        total_var     = 0

        generator.train()

        for i, (true_input, true_target) in enumerate(dataloader):

            true_input  = true_input.cuda()
            true_target = true_target.cuda()

            # Noise-level index from batch PSNR (used as conditioning signal)
            I_index = utils.psnr(true_input, true_target, 255)
            print(I_index)

            optimizer_G.zero_grad()

            batch_size      = true_input.shape[0]
            I_index_tensor  = torch.full((batch_size,), I_index, dtype=torch.float32).cuda()

            # Forward pass — now returns (mean, log_var)
            print("true_input.shape: ", true_input.shape)
            print("I_index_tensor.shape: ", I_index_tensor.shape)
            pre_clean, log_var = generator(true_input, I_index_tensor)
            # out = generator(true_input, I_index_tensor)

            # Laplacian edge loss (unchanged from original)
            pre  = pre_clean[0, :, :, :].data.permute(1, 2, 0).cpu().numpy()
            true = true_target[0, :, :, :].data.permute(1, 2, 0).cpu().numpy()
            laplacian_pre = cv2.Laplacian(pre,  cv2.CV_32F)
            laplacian_gt  = cv2.Laplacian(true, cv2.CV_32F)
            sobel_pre = 0.5 * (cv2.Sobel(pre,  cv2.CV_32F, 1, 0, ksize=5)
                               + cv2.Sobel(pre,  cv2.CV_32F, 0, 1, ksize=5))
            sobel_gt  = 0.5 * (cv2.Sobel(true, cv2.CV_32F, 1, 0, ksize=5)
                               + cv2.Sobel(true, cv2.CV_32F, 0, 1, ksize=5))
            sobel_loss     = mean_squared_error(sobel_pre, sobel_gt)
            laplacian_loss = mean_squared_error(laplacian_pre, laplacian_gt)

            # Uncertainty-aware loss
            loss, loss_dict = criterion(
                mean           = pre_clean,
                log_var        = log_var,
                target         = true_target,
                I_index        = I_index,
                laplacian_loss = laplacian_loss,
                nll_scale      = nll_scale(epoch + 1),
            )

            loss.backward()
            optimizer_G.step()

            # Timing
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left  = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time  = time.time()

            # Accumulate for epoch summary
            total_loss    += loss_dict['l1']
            total_nll     += loss_dict['nll']
            total_lap     += loss_dict['laplacian']
            total_sobel   += sobel_loss
            total_physics += loss_dict['physics']
            total_var     += loss_dict['mean_var']

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] "
                "[L1: %.4f] [Lap: %.4f] [NLL: %.4f (scale=%.2f)] "
                "[Physics: %.4f] [σ²: %.5f] Time_left: %s"
                % (
                    (epoch + 1), opt.epochs, i, len(dataloader),
                    loss_dict['l1'], loss_dict['laplacian'],
                    loss_dict['nll'], loss_dict['nll_scale'],
                    loss_dict['physics'], loss_dict['mean_var'],
                    time_left
                )
            )

            img_list  = [pre_clean, true_target, true_input]
            name_list = ['pred', 'gt', 'noise']
            utils.save_sample_png(
                sample_folder=sample_folder,
                sample_name='MyDNN_MS_epoch%d' % (epoch + 1),
                img_list=img_list,
                name_list=name_list,
                pixel_max_cnt=255
            )

            lr = adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)

        n_batches = max(len(dataloader), 1)
        print(
            "\r[Epoch %d/%d] SUMMARY "
            "[L1: %.4f] [Lap: %.4f] [NLL: %.4f] "
            "[Physics: %.4f] [σ²: %.5f] [Sobel: %.4f]"
            % (
                (epoch + 1), opt.epochs,
                total_loss    / n_batches,
                total_lap     / n_batches,
                total_nll     / n_batches,
                total_physics / n_batches,
                total_var     / n_batches,
                total_sobel   / n_batches,
            )
        )

        # ----------------------------------------
        #             Validation
        # ----------------------------------------
        val_PSNR         = 0
        be_PSNR          = 0
        num_of_val_image = 0

        # For deterministic validation loss, disable MCDropoutBlock stochasticity.
        # We re-enable it after the loop.
        from network_emb_uncertainty import MCDropoutBlock
        generator.eval()
        for m in generator.modules():
            if isinstance(m, MCDropoutBlock):
                m.dropout.eval()

        for j, (true_input, true_target) in enumerate(val_loader):

            true_input  = true_input.cuda()
            true_target = true_target.cuda()

            I_index = utils.psnr(true_input, true_target, 255)

            batch_size     = true_input.shape[0]
            I_index_tensor = torch.full((batch_size,), I_index, dtype=torch.float32).cuda()

            with torch.no_grad():
                # mean only — log_var not needed for PSNR tracking
                pre_clean, _ = generator(true_input, I_index_tensor)

            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(pre_clean, true_target, 255) * true_input.shape[0]
            be_PSNR  += utils.psnr(true_input, true_target, 255) * true_input.shape[0]

        # Re-enable MC Dropout for next training epoch
        for m in generator.modules():
            if isinstance(m, MCDropoutBlock):
                m.dropout.train()
        generator.train()

        val_PSNR = val_PSNR / num_of_val_image
        be_PSNR  = be_PSNR  / num_of_val_image

        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
        print('PSNR before denoising %d: %.4f' % ((epoch + 1), be_PSNR))
        print('NLL warmup scale: %.3f' % nll_scale(epoch + 1))

        # Save uncertainty samples periodically (every 10 epochs or at best model)
        save_freq = getattr(opt, 'uncertainty_sample_freq', 10)
        if (epoch + 1) % save_freq == 0 or val_PSNR >= best_PSNR:
            n_mc_samples = getattr(opt, 'mc_n_samples', 20)
            num_samples = getattr(opt, 'num_uncertainty_samples', 4)
            save_uncertainty_samples(
                generator, val_loader, sample_folder, epoch,
                n_samples=n_mc_samples, num_save_samples=num_samples
            )

        best_PSNR = max(val_PSNR, best_PSNR)
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader),
                   generator, val_PSNR, best_PSNR)
