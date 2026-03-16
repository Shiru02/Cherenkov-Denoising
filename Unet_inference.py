"""
Inference script for UNetMCDropout Cherenkov Denoiser
=====================================================
Supports:
  - Single image denoising with full MC uncertainty maps
  - Batch inference over a directory of .npy patch files
  - Uncertainty calibration plot (epistemic vs aleatoric vs combined)
  - Export results to .npy or .png

Usage:
  # Single image
  python inference_mc_dropout.py --checkpoint best_model.pt --image patch.npy

  # Batch over directory
  python inference_mc_dropout.py --checkpoint best_model.pt --data_dir noisy_patches/ --output_dir results/

  # Tune MC samples
  python inference_mc_dropout.py --checkpoint best_model.pt --image patch.npy --n_samples 30

  # Specify noise level manually (0=clean, 1=very noisy)
  python inference_mc_dropout.py --checkpoint best_model.pt --image patch.npy --noise_level 0.7
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ========================== MODEL (copied from training) ======================
# ==============================================================================

class ConvBlock(nn.Module):
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
    """Spatial dropout — stays ACTIVE at inference (forces self.dropout.train())."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p       = p
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        self.dropout.train()
        return self.dropout(x)


class UNetMCDropout(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 noise_embed_dim=128, dropout_p=0.1):
        super().__init__()
        ch = base_channels
        self.dropout_p = dropout_p

        self.noise_embed = nn.Sequential(
            nn.Linear(1, noise_embed_dim), nn.SiLU(),
            nn.Linear(noise_embed_dim, noise_embed_dim), nn.SiLU(),
        )
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch,    ch*2)
        self.enc3 = ConvBlock(ch*2,  ch*4)
        self.enc4 = ConvBlock(ch*4,  ch*8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(ch*8, ch*8), AttentionBlock(ch*8), ConvBlock(ch*8, ch*8),
        )
        self.noise_proj_b = nn.Linear(noise_embed_dim, ch*8)
        self.noise_proj_4 = nn.Linear(noise_embed_dim, ch*4)
        self.noise_proj_3 = nn.Linear(noise_embed_dim, ch*2)
        self.noise_proj_2 = nn.Linear(noise_embed_dim, ch)
        self.noise_proj_1 = nn.Linear(noise_embed_dim, ch)

        self.up4   = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.dec4  = ConvBlock(ch*4 + ch*8, ch*4)
        self.drop4 = MCDropoutBlock(p=dropout_p)

        self.up3   = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec3  = ConvBlock(ch*2 + ch*4, ch*2)
        self.drop3 = MCDropoutBlock(p=dropout_p)

        self.up2   = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.dec2  = ConvBlock(ch + ch*2, ch)
        self.drop2 = MCDropoutBlock(p=dropout_p)

        self.up1   = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1  = ConvBlock(ch + ch, ch)
        self.drop1 = MCDropoutBlock(p=dropout_p)

        self.final_mean   = nn.Conv2d(ch, out_channels, 1)
        self.final_logvar = nn.Sequential(
            nn.Conv2d(ch, ch//2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch//2, out_channels, 1),
        )
        nn.init.constant_(self.final_logvar[-1].bias, -6.0)

    def forward(self, x, noise_level):
        n_emb = self.noise_embed(noise_level)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        b  = b + self.noise_proj_b(n_emb)[:, :, None, None]
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
        self.eval()
        all_means, all_log_vars = [], []
        for _ in range(n_samples):
            m, lv = self.forward(x, noise_level)
            all_means.append(m)
            all_log_vars.append(lv)
        all_means    = torch.stack(all_means,    dim=0)
        all_log_vars = torch.stack(all_log_vars, dim=0)
        mean_pred     = all_means.mean(dim=0)
        epistemic_std = all_means.std(dim=0)
        aleatoric_var = torch.exp(all_log_vars).mean(dim=0)
        combined      = torch.sqrt(epistemic_std**2 + aleatoric_var)
        tau        = max(combined.mean().item(), 1e-8)
        confidence = 1.0 / (1.0 + combined / tau)
        return {
            'mean':          mean_pred,
            'epistemic_std': epistemic_std,
            'aleatoric_var': aleatoric_var,
            'combined':      combined,
            'confidence':    confidence,
            'raw_means':     all_means,
        }


# ==============================================================================
# ========================== METRICS ===========================================
# ==============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    return float('inf') if mse == 0 else 20 * math.log10(max_val) - 10 * math.log10(mse)


def compute_ssim_np(pred: np.ndarray, target: np.ndarray,
                    window_size: int = 11, sigma: float = 1.5) -> float:
    """Numpy SSIM (single-channel 2-D images in [0,1])."""
    from scipy.ndimage import uniform_filter
    # Gaussian approximation via repeated uniform filters
    def blur(img):
        return uniform_filter(img.astype(np.float64), size=window_size)

    C1, C2 = 0.01**2, 0.03**2
    mu_p, mu_t = blur(pred), blur(target)
    sig_p  = np.maximum(blur(pred**2)    - mu_p**2, 0)
    sig_t  = np.maximum(blur(target**2)  - mu_t**2, 0)
    sig_pt = blur(pred * target) - mu_p * mu_t
    num = (2*mu_p*mu_t + C1) * (2*sig_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return float(np.mean(num / (den + 1e-8)))


# ==============================================================================
# ========================== LOADING ===========================================
# ==============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda') -> UNetMCDropout:
    """Load a saved UNetMCDropout checkpoint."""
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNetMCDropout(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch    = ckpt.get('epoch', '?')
    val_psnr = ckpt.get('val_psnr', float('nan'))
    print(f"✓ Loaded checkpoint  |  epoch {epoch}  |  val PSNR {val_psnr:.2f} dB")
    return model


# ==============================================================================
# ========================== SINGLE-IMAGE INFERENCE ============================
# ==============================================================================

@torch.no_grad()
def infer_single(model: UNetMCDropout,
                 noisy_image: np.ndarray,
                 noise_level: float = 0.5,
                 n_samples: int = 20,
                 device: str = 'cuda') -> dict:
    """
    Denoise a single (H, W) numpy image and return full uncertainty maps.

    Returns dict of (H, W) numpy arrays:
        denoised, epistemic_std, aleatoric_var, combined, confidence
    Plus scalar metrics if ground truth is not provided.
    """
    if noisy_image.max() > 1.0:
        noisy_image = noisy_image / noisy_image.max()

    x  = torch.tensor(noisy_image, dtype=torch.float32)[None, None].to(device)
    nl = torch.tensor([[noise_level]], dtype=torch.float32).to(device)

    unc = model.mc_uncertainty(x, nl, n_samples=n_samples)
    return {k: v[0, 0].cpu().numpy() for k, v in unc.items() if k != 'raw_means'}


# ==============================================================================
# ========================== VISUALISATION =====================================
# ==============================================================================

def visualise_single(noisy: np.ndarray,
                     results: dict,
                     ground_truth: np.ndarray = None,
                     save_path: str = None,
                     noise_level: float = None,
                     n_samples: int = None):
    """
    7-panel figure:
      [Noisy | Denoised | GT (optional)] [Epistemic | Aleatoric | Combined | Confidence]
    """
    denoised   = results['mean']
    epi_std    = results['epistemic_std']
    ale_var    = results['aleatoric_var']
    combined   = results['combined']
    confidence = results['confidence']

    has_gt = ground_truth is not None

    # --- compute metrics ---
    psnr_noisy   = compute_psnr(noisy,    ground_truth) if has_gt else None
    psnr_denoise = compute_psnr(denoised, ground_truth) if has_gt else None
    ssim_denoise = compute_ssim_np(denoised, ground_truth) if has_gt else None

    ncols = 7 if has_gt else 6
    fig   = plt.figure(figsize=(4 * ncols, 4.5))
    gs    = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.05)

    def show(ax, img, title, cmap='gray', vmin=None, vmax=None, colorbar=False):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, pad=5)
        ax.axis('off')
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.3f')
        return im

    col = 0
    noisy_title = f'Noisy Input\n(noise={noise_level:.2f})' if noise_level else 'Noisy Input'
    if has_gt:
        noisy_title += f'\nPSNR: {psnr_noisy:.1f} dB'
    show(fig.add_subplot(gs[0, col]), noisy,    noisy_title); col += 1

    den_title = f'MC Mean\n({n_samples} passes)' if n_samples else 'MC Mean'
    if has_gt:
        den_title += f'\nPSNR: {psnr_denoise:.1f} dB  SSIM: {ssim_denoise:.3f}'
    show(fig.add_subplot(gs[0, col]), denoised, den_title); col += 1

    if has_gt:
        show(fig.add_subplot(gs[0, col]), ground_truth, 'Ground Truth'); col += 1

    show(fig.add_subplot(gs[0, col]), epi_std,    'Epistemic Std\n(model unc.)',   'plasma',  colorbar=True); col += 1
    show(fig.add_subplot(gs[0, col]), ale_var,    'Aleatoric Var\n(physics noise)', 'hot',    colorbar=True); col += 1
    show(fig.add_subplot(gs[0, col]), combined,   'Combined Unc.',                  'hot',    colorbar=True); col += 1
    show(fig.add_subplot(gs[0, col]), confidence, 'Confidence',                     'RdYlGn',
         vmin=0, vmax=1, colorbar=True)

    suptitle = 'MC Dropout Uncertainty Decomposition'
    if n_samples:
        suptitle += f'  |  {n_samples} MC passes'
    fig.suptitle(suptitle, fontsize=13, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved figure → {save_path}")
    plt.show()
    plt.close()

    if has_gt:
        print(f"\n{'─'*45}")
        print(f"  PSNR  noisy   → {psnr_noisy:.2f} dB")
        print(f"  PSNR  denoise → {psnr_denoise:.2f} dB  (+{psnr_denoise - psnr_noisy:.2f} dB)")
        print(f"  SSIM  denoise → {ssim_denoise:.4f}")
        print(f"  Epistemic std  mean: {epi_std.mean():.5f}")
        print(f"  Aleatoric var  mean: {ale_var.mean():.5f}")
        print(f"  Combined       mean: {combined.mean():.5f}")
        print(f"{'─'*45}\n")


def visualise_uncertainty_histogram(results: dict, save_path: str = None):
    """
    Histogram showing epistemic vs aleatoric uncertainty distributions side by side.
    Useful for diagnosing whether the model is over- or under-confident.
    """
    epi  = results['epistemic_std'].ravel()
    ale  = results['aleatoric_var'].ravel()
    comb = results['combined'].ravel()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, label, color in zip(
        axes,
        [epi, ale, comb],
        ['Epistemic Std', 'Aleatoric Var', 'Combined'],
        ['steelblue', 'tomato', 'mediumorchid']
    ):
        ax.hist(data, bins=80, color=color, alpha=0.8, density=True)
        ax.axvline(data.mean(), color='black', ls='--', lw=1.5, label=f'mean={data.mean():.4f}')
        ax.axvline(np.median(data), color='gray',  ls=':',  lw=1.5, label=f'median={np.median(data):.4f}')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Uncertainty value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Pixel-wise Uncertainty Distributions', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved histogram → {save_path}")
    plt.show()
    plt.close()


def visualise_mc_variance_vs_samples(model: UNetMCDropout,
                                     noisy: np.ndarray,
                                     noise_level: float,
                                     sample_counts=(5, 10, 20, 30, 50),
                                     device: str = 'cuda',
                                     save_path: str = None):
    """
    Shows how epistemic uncertainty estimate stabilises as n_samples increases.
    Run this once to pick the right n_samples for your use case.
    """
    x  = torch.tensor(noisy, dtype=torch.float32)[None, None].to(device)
    nl = torch.tensor([[noise_level]], dtype=torch.float32).to(device)

    mean_epis = []
    for n in sample_counts:
        unc = model.mc_uncertainty(x, nl, n_samples=n)
        mean_epis.append(unc['epistemic_std'][0, 0].cpu().mean().item())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sample_counts, mean_epis, 'o-', color='steelblue', lw=2, ms=7)
    ax.set_xlabel('Number of MC samples')
    ax.set_ylabel('Mean epistemic std')
    ax.set_title('Epistemic Uncertainty Convergence vs. MC Samples')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved convergence plot → {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# ========================== BATCH INFERENCE ===================================
# ==============================================================================

def batch_infer(model: UNetMCDropout,
                data_dir: str,
                output_dir: str,
                n_samples: int = 20,
                noise_level_override: float = None,
                save_npy: bool = True,
                save_png: bool = True,
                device: str = 'cuda'):
    """
    Run inference on every .npy patch file in data_dir.

    Each .npy file is expected to be shape (H, W, num_levels).
    Input:  level 0    (lowest SNR)
    Target: last level (highest SNR)

    Saves per-file .npy bundles and optional .png summaries to output_dir.
    Returns a summary DataFrame.
    """
    import glob
    import pandas as pd

    files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    if not files:
        raise ValueError(f"No .npy files found in {data_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for i, fpath in enumerate(files):
        patch = np.load(fpath)                       # (H, W, num_levels)
        if patch.max() > 1.0:
            patch = patch / patch.max()

        num_levels  = patch.shape[-1]
        noisy       = patch[:, :, 0]                 # worst level
        clean       = patch[:, :, -1]                # best level

        nl = noise_level_override if noise_level_override is not None else 1.0

        results = infer_single(model, noisy, noise_level=nl,
                               n_samples=n_samples, device=device)

        denoised = results['mean']
        psnr_noisy   = compute_psnr(noisy,    clean)
        psnr_denoise = compute_psnr(denoised, clean)
        ssim_denoise = compute_ssim_np(denoised, clean)

        stem = Path(fpath).stem
        if save_npy:
            np.save(os.path.join(output_dir, f'{stem}_results.npy'), {
                'noisy':         noisy,
                'denoised':      denoised,
                'clean':         clean,
                'epistemic_std': results['epistemic_std'],
                'aleatoric_var': results['aleatoric_var'],
                'combined':      results['combined'],
                'confidence':    results['confidence'],
            })

        if save_png:
            visualise_single(
                noisy, results, ground_truth=clean,
                save_path=os.path.join(output_dir, f'{stem}_vis.png'),
                noise_level=nl, n_samples=n_samples
            )

        records.append({
            'file':           os.path.basename(fpath),
            'psnr_noisy':     psnr_noisy,
            'psnr_denoised':  psnr_denoise,
            'psnr_gain_dB':   psnr_denoise - psnr_noisy,
            'ssim_denoised':  ssim_denoise,
            'mean_epistemic': results['epistemic_std'].mean(),
            'mean_aleatoric': results['aleatoric_var'].mean(),
            'mean_combined':  results['combined'].mean(),
            'mean_confidence':results['confidence'].mean(),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(files)}]  {os.path.basename(fpath)}"
                  f"  PSNR {psnr_noisy:.1f} → {psnr_denoise:.1f} dB"
                  f"  SSIM {ssim_denoise:.3f}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'inference_summary.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Batch complete  |  {len(files)} files  |  results in {output_dir}")
    print(f"  PSNR gain :  {df['psnr_gain_dB'].mean():.2f} dB  (mean)")
    print(f"  SSIM      :  {df['ssim_denoised'].mean():.4f}     (mean)")
    print(f"  Confidence:  {df['mean_confidence'].mean():.4f}     (mean)")
    print(f"  Summary CSV → {csv_path}")
    print(f"{'='*55}\n")

    # Summary plot
    _plot_batch_summary(df, output_dir)
    return df


def _plot_batch_summary(df, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df['psnr_gain_dB'], bins=30, color='steelblue', alpha=0.8)
    axes[0].axvline(df['psnr_gain_dB'].mean(), color='black', ls='--', lw=1.5,
                    label=f"mean={df['psnr_gain_dB'].mean():.2f} dB")
    axes[0].set_title('PSNR Gain (dB)')
    axes[0].set_xlabel('PSNR denoised − PSNR noisy (dB)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].hist(df['ssim_denoised'], bins=30, color='seagreen', alpha=0.8)
    axes[1].axvline(df['ssim_denoised'].mean(), color='black', ls='--', lw=1.5,
                    label=f"mean={df['ssim_denoised'].mean():.4f}")
    axes[1].set_title('SSIM (denoised)')
    axes[1].set_xlabel('SSIM')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(df['mean_epistemic'], df['mean_aleatoric'],
                    alpha=0.4, c=df['psnr_gain_dB'], cmap='RdYlGn', s=20)
    axes[2].set_xlabel('Mean Epistemic Std')
    axes[2].set_ylabel('Mean Aleatoric Var')
    axes[2].set_title('Epistemic vs Aleatoric\n(colour = PSNR gain)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Batch Inference Summary', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved batch summary → {os.path.join(output_dir, 'batch_summary.png')}")


# ==============================================================================
# ========================== CLI ENTRYPOINT ====================================
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='MC Dropout inference for Cherenkov denoising')
    p.add_argument('--checkpoint',    type=str, required=True,
                   help='Path to .pt checkpoint (best_model.pt or latest.pt)')
    p.add_argument('--device',        type=str, default='cuda',
                   help='cuda or cpu  (default: cuda)')

    # Single image mode
    p.add_argument('--image',         type=str, default=None,
                   help='Path to .npy patch file  [single-image mode]')
    p.add_argument('--input_level',   type=int, default=0,
                   help='Which level index to use as input (default: 0 = noisiest)')
    p.add_argument('--noise_level',   type=float, default=None,
                   help='Override noise_level fed to the model (0=clean, 1=noisy). '
                        'Default: inferred from input_level / num_levels.')

    # Batch mode
    p.add_argument('--data_dir',      type=str, default=None,
                   help='Directory of .npy patches  [batch mode]')
    p.add_argument('--output_dir',    type=str, default='inference_results',
                   help='Where to save batch results (default: inference_results/)')
    p.add_argument('--no_npy',        action='store_true',
                   help='Skip saving .npy result bundles in batch mode')
    p.add_argument('--no_png',        action='store_true',
                   help='Skip saving .png visualisations in batch mode')

    # MC settings
    p.add_argument('--n_samples',     type=int, default=20,
                   help='Number of MC forward passes (default: 20)')
    p.add_argument('--convergence',   action='store_true',
                   help='Plot epistemic uncertainty vs. n_samples (requires --image)')

    return p.parse_args()


def main():
    args = parse_args()

    # Auto-fall back to CPU
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print(" CUDA unavailable — falling back to CPU")
        device = 'cpu'

    model = load_model(args.checkpoint, device=device)

    # ------------------------------------------------------------------ SINGLE
    if args.image:
        patch = np.load(args.image)                           # (H, W, levels) or (H, W)
        if patch.ndim == 2:
            noisy = patch
            clean = None
            num_levels = 1
        else:
            num_levels = patch.shape[-1]
            if patch.max() > 1.0:
                patch = patch / patch.max()
            noisy = patch[:, :, args.input_level]
            clean = patch[:, :, -1] if num_levels > 1 else None

        nl = args.noise_level
        if nl is None:
            nl = 1.0 - args.input_level / max(1, num_levels - 1)
        print(f"Input level {args.input_level}/{num_levels-1}  →  noise_level={nl:.3f}")

        results = infer_single(model, noisy, noise_level=nl,
                               n_samples=args.n_samples, device=device)

        stem    = Path(args.image).stem
        out_vis = f'{stem}_inference.png'
        out_his = f'{stem}_uncertainty_hist.png'

        visualise_single(noisy, results, ground_truth=clean,
                         save_path=out_vis, noise_level=nl,
                         n_samples=args.n_samples)

        visualise_uncertainty_histogram(results, save_path=out_his)

        if args.convergence:
            visualise_mc_variance_vs_samples(
                model, noisy, nl, device=device,
                save_path=f'{stem}_convergence.png'
            )

    # ------------------------------------------------------------------ BATCH
    elif args.data_dir:
        batch_infer(
            model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            noise_level_override=args.noise_level,
            save_npy=not args.no_npy,
            save_png=not args.no_png,
            device=device,
        )

    else:
        print("Specify --image for single-image mode or --data_dir for batch mode.")
        print("Run with --help for full usage.")


if __name__ == '__main__':
    main()