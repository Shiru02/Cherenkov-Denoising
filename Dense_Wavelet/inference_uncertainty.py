#!/usr/bin/env python
"""
Inference script for Wavelet MyDNN Cherenkov Denoiser with MC Dropout Uncertainty
=================================================================================
Supports:
  - Single image denoising with full MC uncertainty maps
  - Batch inference over a directory of .npy patch files or image files (PNG, TIFF, etc.)
  - Uncertainty calibration plot (epistemic vs aleatoric vs combined)
  - Export results to .npy or .png

Usage:
  # Single image (.npy patch)
  python inference_uncertainty.py --checkpoint model.pth --image patch.npy

  # Single image file (PNG, TIFF, etc.)
  python inference_uncertainty.py --checkpoint model.pth --image noisy.png --gt_image clean.png

  # Batch over directory (.npy files)
  python inference_uncertainty.py --checkpoint model.pth --data_dir noisy_patches/ --output_dir results/

  # Batch over directory (image files)
  python inference_uncertainty.py --checkpoint model.pth --data_dir noisy_images/ --gt_dir clean_images/ --output_dir results/

  # Tune MC samples
  python inference_uncertainty.py --checkpoint model.pth --image patch.npy --n_samples 30

  # Specify PSNR level manually (typical range: 20-50 dB)
  python inference_uncertainty.py --checkpoint model.pth --image patch.npy --psnr_level 35
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
from PIL import Image
import cv2
import utils


# ==============================================================================
# ========================== UTILITY FUNCTIONS =================================
# ==============================================================================

def pad_for_wavelet(tensor):
    """
    Pad tensor to make dimensions divisible by 8 (required for 3-level Haar wavelet).
    
    Args:
        tensor: Input tensor of shape [B, C, H, W]
        
    Returns:
        padded_tensor: Padded tensor
        original_shape: (H, W) tuple if padding was applied, None otherwise
    """
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return padded, (h, w)
    return tensor, None


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR between prediction and target."""
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

def load_image_file(image_path: str) -> np.ndarray:
    """
    Load an image file (PNG, TIFF, JPEG, etc.) and normalize to [0, 1].
    
    Args:
        image_path: Path to image file
    
    Returns:
        (H, W) numpy array in [0, 1] range
    """
    img = Image.open(image_path)
    img_np = np.array(img).astype(np.float32)
    
    # Handle grayscale vs color
    if img_np.ndim == 3:
        # Convert to grayscale if color
        if img_np.shape[2] == 3:
            # RGB to grayscale
            img_np = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
        elif img_np.shape[2] == 4:
            # RGBA to grayscale (ignore alpha)
            img_np = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
        elif img_np.shape[2] == 1:
            img_np = img_np[:, :, 0]
    
    # Normalize to [0, 1]
    min_val, max_val = img_np.min(), img_np.max()
    if max_val > min_val:
        normalized = (img_np - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(img_np)
    
    return normalized.astype(np.float32)


def load_model(checkpoint_path: str, opt, device: str = 'cuda'):
    """
    Load a saved MyDNN checkpoint.
    
    Args:
        checkpoint_path: Path to .pth model file
        opt: Options object with model configuration
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model structure
    model = utils.create_MyDNN(opt)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, nn.Module):
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model


# ==============================================================================
# ========================== SINGLE-IMAGE INFERENCE ============================
# ==============================================================================

@torch.no_grad()
def infer_single(model,
                 noisy_image: np.ndarray,
                 psnr_level: float = 35.0,
                 n_samples: int = 20,
                 device: str = 'cuda') -> dict:
    """
    Denoise a single (H, W) numpy image and return full uncertainty maps.

    Args:
        model: MyDNN model with mc_uncertainty method
        noisy_image: (H, W) numpy array, values in [0, 1]
        psnr_level: Expected PSNR level in dB (typical range: 20-50)
        n_samples: Number of MC forward passes
        device: 'cuda' or 'cpu'

    Returns dict of (H, W) numpy arrays:
        mean, epistemic_std, aleatoric_var, combined, confidence
    """
    if noisy_image.max() > 1.0:
        noisy_image = noisy_image / noisy_image.max()

    # Convert to tensor and pad for wavelet
    x = torch.tensor(noisy_image, dtype=torch.float32)[None, None].to(device)
    padded_x, orig_shape = pad_for_wavelet(x)
    
    # Create PSNR conditioning tensor (batch_size,)
    batch_size = padded_x.shape[0]
    I_index_tensor = torch.full((batch_size,), psnr_level, dtype=torch.float32, device=device)

    # Run MC uncertainty inference
    unc = model.mc_uncertainty(padded_x, I_index_tensor, n_samples=n_samples)
    
    # Crop back to original size if padded
    if orig_shape is not None:
        h, w = orig_shape
        for key in ['mean', 'epistemic_std', 'aleatoric_var', 'combined', 'confidence']:
            unc[key] = unc[key][:, :, :h, :w]
    
    # Convert to numpy and return
    return {k: v[0, 0].cpu().numpy() for k, v in unc.items() if k != 'raw_means'}


# ==============================================================================
# ========================== VISUALISATION =====================================
# ==============================================================================

def visualise_single(noisy: np.ndarray,
                     results: dict,
                     ground_truth: np.ndarray = None,
                     save_path: str = None,
                     psnr_level: float = None,
                     n_samples: int = None):
    """
    7-panel figure:
      [Noisy | Denoised | GT (optional)] [Epistemic | Aleatoric | Combined | Confidence]
    """
    denoised   = results['mean']
    epi_std    = results['epistemic_std']

    has_gt = ground_truth is not None

    # --- compute metrics ---
    psnr_noisy   = compute_psnr(noisy,    ground_truth) if has_gt else None
    psnr_denoise = compute_psnr(denoised, ground_truth) if has_gt else None
    ssim_denoise = compute_ssim_np(denoised, ground_truth) if has_gt else None

    # Layout: [Noisy | Denoised | (optional GT) | Epistemic]
    ncols = 4 if has_gt else 3
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
    noisy_title = f'Noisy Input\n(PSNR={psnr_level:.1f} dB)' if psnr_level else 'Noisy Input'
    if has_gt:
        noisy_title += f'\nPSNR: {psnr_noisy:.1f} dB'
    show(fig.add_subplot(gs[0, col]), noisy,    noisy_title); col += 1

    den_title = f'MC Mean\n({n_samples} passes)' if n_samples else 'MC Mean'
    if has_gt:
        den_title += f'\nPSNR: {psnr_denoise:.1f} dB  SSIM: {ssim_denoise:.3f}'
    show(fig.add_subplot(gs[0, col]), denoised, den_title); col += 1

    if has_gt:
        show(fig.add_subplot(gs[0, col]), ground_truth, 'Ground Truth'); col += 1

    # Only epistemic uncertainty map
    show(fig.add_subplot(gs[0, col]), epi_std, 'Epistemic Std\n(model unc.)',
         'plasma', colorbar=True)

    suptitle = 'MC Dropout Uncertainty Decomposition'
    if n_samples:
        suptitle += f'  |  {n_samples} MC passes'
    fig.suptitle(suptitle, fontsize=13, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved figure → {save_path}")
    plt.close()

    if has_gt:
        print(f"\n{'─'*45}")
        print(f"  PSNR  noisy   → {psnr_noisy:.2f} dB")
        print(f"  PSNR  denoise → {psnr_denoise:.2f} dB  (+{psnr_denoise - psnr_noisy:.2f} dB)")
        print(f"  SSIM  denoise → {ssim_denoise:.4f}")
        print(f"  Epistemic std  mean: {epi_std.mean():.5f}")
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
    plt.close()


def visualise_mc_variance_vs_samples(model,
                                     noisy: np.ndarray,
                                     psnr_level: float,
                                     sample_counts=(5, 10, 20, 30, 50),
                                     device: str = 'cuda',
                                     save_path: str = None):
    """
    Shows how epistemic uncertainty estimate stabilises as n_samples increases.
    Run this once to pick the right n_samples for your use case.
    """
    x = torch.tensor(noisy, dtype=torch.float32)[None, None].to(device)
    padded_x, orig_shape = pad_for_wavelet(x)
    
    batch_size = padded_x.shape[0]
    I_index_tensor = torch.full((batch_size,), psnr_level, dtype=torch.float32, device=device)

    mean_epis = []
    for n in sample_counts:
        unc = model.mc_uncertainty(padded_x, I_index_tensor, n_samples=n)
        if orig_shape is not None:
            h, w = orig_shape
            epi = unc['epistemic_std'][:, :, :h, :w]
        else:
            epi = unc['epistemic_std']
        mean_epis.append(epi[0, 0].cpu().mean().item())

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
    plt.close()


# ==============================================================================
# ========================== BATCH INFERENCE ===================================
# ==============================================================================

def batch_infer(model,
                data_dir: str,
                output_dir: str,
                n_samples: int = 20,
                psnr_level_override: float = None,
                save_npy: bool = True,
                save_png: bool = True,
                gt_dir: str = None,
                device: str = 'cuda'):
    """
    Run inference on every .npy patch file or image file in data_dir.

    For .npy files: Each file is expected to be shape (H, W, num_levels).
                    Input: level 0 (lowest SNR), Target: last level (highest SNR)
    
    For image files: Each image is treated as a noisy input.
                     If gt_dir is provided, matching GT images are loaded for comparison.

    Saves per-file .npy bundles and optional .png summaries to output_dir.
    Returns a summary DataFrame.
    """
    import glob
    import pandas as pd

    # Find all supported files
    npy_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    image_extensions = ['*.png', '*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
    image_files = sorted(image_files)
    
    files = npy_files + image_files
    if not files:
        raise ValueError(f"No .npy or image files found in {data_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for i, fpath in enumerate(files):
        fpath_obj = Path(fpath)
        
        # Determine if it's .npy or image file
        if fpath_obj.suffix.lower() == '.npy':
            # Handle .npy file (multi-level format)
            patch = np.load(fpath)                       # (H, W, num_levels)
            if patch.max() > 1.0:
                patch = patch / patch.max()

            num_levels  = patch.shape[-1]
            noisy       = patch[:, :, 0]                 # worst level
            clean       = patch[:, :, -1]                # best level

            # Estimate PSNR from input level if not overridden
            if psnr_level_override is not None:
                psnr_lvl = psnr_level_override
            else:
                # Infer PSNR from level index (similar to noise_level inference in Unet)
                # Lower level index = lower PSNR
                # Typical range: 20-50 dB
                psnr_lvl = 50.0 - (30.0 * (0 / max(1, num_levels - 1)))
        else:
            # Handle image file
            noisy = load_image_file(fpath)
            clean = None
            
            # Try to find matching GT image if gt_dir is provided
            if gt_dir:
                noisy_name = fpath_obj.name
                # Try exact match
                gt_path = os.path.join(gt_dir, noisy_name)
                if not os.path.exists(gt_path):
                    # Try with _gt suffix
                    base_name = fpath_obj.stem
                    gt_path = os.path.join(gt_dir, f'{base_name}_gt{fpath_obj.suffix}')
                    if not os.path.exists(gt_path):
                        # Try any file with similar base name
                        for gt_candidate in glob.glob(os.path.join(gt_dir, '*')):
                            gt_candidate_obj = Path(gt_candidate)
                            if base_name in gt_candidate_obj.stem or gt_candidate_obj.stem.replace('_gt', '') in base_name:
                                gt_path = gt_candidate
                                break
                
                if os.path.exists(gt_path):
                    clean = load_image_file(gt_path)
            
            # Use provided PSNR level or default
            psnr_lvl = psnr_level_override if psnr_level_override is not None else 35.0

        results = infer_single(model, noisy, psnr_level=psnr_lvl,
                               n_samples=n_samples, device=device)

        denoised = results['mean']
        
        # Compute metrics if GT is available
        if clean is not None:
            psnr_noisy   = compute_psnr(noisy,    clean)
            psnr_denoise = compute_psnr(denoised, clean)
            ssim_denoise = compute_ssim_np(denoised, clean)
        else:
            psnr_noisy = None
            psnr_denoise = None
            ssim_denoise = None

        stem = fpath_obj.stem
        if save_npy:
            result_dict = {
                'noisy':         noisy,
                'denoised':      denoised,
                'epistemic_std': results['epistemic_std'],
                'aleatoric_var': results['aleatoric_var'],
                'combined':      results['combined'],
                'confidence':    results['confidence'],
            }
            if clean is not None:
                result_dict['clean'] = clean
            np.save(os.path.join(output_dir, f'{stem}_results.npy'), result_dict)

        if save_png:
            visualise_single(
                noisy, results, ground_truth=clean,
                save_path=os.path.join(output_dir, f'{stem}_wavelet_vis.png'),
                psnr_level=psnr_lvl, n_samples=n_samples
            )

        records.append({
            'file':           os.path.basename(fpath),
            'psnr_noisy':     psnr_noisy if psnr_noisy is not None else np.nan,
            'psnr_denoised':  psnr_denoise if psnr_denoise is not None else np.nan,
            'psnr_gain_dB':   (psnr_denoise - psnr_noisy) if (psnr_noisy is not None and psnr_denoise is not None) else np.nan,
            '%PSNR_gain':     ((psnr_denoise - psnr_noisy) / psnr_noisy) * 100 if (psnr_noisy is not None and psnr_denoise is not None) else np.nan,
            'ssim_denoised':  ssim_denoise if ssim_denoise is not None else np.nan,
            'mean_epistemic': results['epistemic_std'].mean(),
            'mean_aleatoric': results['aleatoric_var'].mean(),
            'mean_combined':  results['combined'].mean(),
            'mean_confidence':results['confidence'].mean(),
        })

        if (i + 1) % 10 == 0 or i == 0:
            status_msg = f"  [{i+1:4d}/{len(files)}]  {os.path.basename(fpath)}"
            if psnr_noisy is not None and psnr_denoise is not None:
                status_msg += f"  PSNR {psnr_noisy:.1f} → {psnr_denoise:.1f} dB  SSIM {ssim_denoise:.3f}"
            else:
                status_msg += f"  (no GT for metrics)"
            print(status_msg)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'inference_summary.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Batch complete  |  {len(files)} files  |  results in {output_dir}")
    if not df['psnr_gain_dB'].isna().all():
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

    # Plot PSNR Gain (dB)
    axes[0].hist(df['psnr_gain_dB'], bins=30, color='steelblue', alpha=0.8)
    axes[0].axvline(df['psnr_gain_dB'].mean(), color='black', ls='--', lw=1.5,
                    label=f"mean={df['psnr_gain_dB'].mean():.2f} dB")
    axes[0].set_title('PSNR Gain (dB)')
    axes[0].set_xlabel('PSNR denoised − PSNR noisy (dB)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Plot Percent PSNR Gain (%) on its own plot, not as part of the main subplot
    fig2, ax2 = plt.subplots(figsize=(5, 4))  # separate figure for percent gain
    ax2.hist(df['%PSNR_gain'], bins=30, color='darkorange', alpha=0.8)
    ax2.axvline(df['%PSNR_gain'].mean(), color='black', ls='--', lw=1.5,
                label=f"mean={df['%PSNR_gain'].mean():.2f}%")
    ax2.set_title('Percent PSNR Gain (%)')
    ax2.set_xlabel('100 * (PSNR denoised − PSNR noisy) / PSNR noisy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percent_psnr_gain_hist.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f" Saved % PSNR gain histogram → {os.path.join(output_dir, 'percent_psnr_gain_hist.png')}")

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
    plt.savefig(os.path.join(output_dir, 'wavelet_batch_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Saved batch summary → {os.path.join(output_dir, 'wavelet_batch_summary.png')}")


# ==============================================================================
# ========================== CLI ENTRYPOINT ====================================
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='MC Dropout inference for Wavelet Cherenkov denoising')
    p.add_argument('--checkpoint',    type=str, default=r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Wavelet_Denoiser(Send2Shiru)\output\output_FT_GAN\final_denoise_epoch75_bs3.pth",
                   help='Path to .pth checkpoint')
    p.add_argument('--device',        type=str, default='cuda',
                   help='cuda or cpu  (default: cuda)')

    # Single image mode
    p.add_argument('--image',         type=str, default=None, #r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Data\Organized_Test_Data\Patches\2021-01-28 09-43-28-767_img100_2_patch1.png",
                   help='Path to image file (.npy, .png, .tiff, .tif, .jpg, etc.)  [single-image mode]')
    p.add_argument('--gt_image',      type=str, default=None, #r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Data\Organized_Test_Data\Patches_GT\2021-01-28 09-43-28-767_img100_2_patch1_gt.png",
                   help='Optional ground truth image for comparison (only used if --image is an image file, not .npy)')
    p.add_argument('--input_level',   type=int, default=0,
                   help='Which level index to use as input (for .npy files only, default: 0 = noisiest)')
    p.add_argument('--psnr_level',   type=float, default=35,
                   help='Override PSNR level fed to the model (typical range: 20-50 dB). '
                        'Default: inferred from input_level / num_levels for .npy, or 35.0 for image files.')

    # Batch mode
    p.add_argument('--data_dir',      type=str, default=r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Data\Organized_Test_Data\Patches",
                   help='Directory of .npy patches or image files  [batch mode]')
    p.add_argument('--gt_dir',        type=str, default=r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Data\Organized_Test_Data\Patches_GT",
                   help='Optional directory of ground truth images (for batch mode with image files)')
    p.add_argument('--output_dir',    type=str, default=r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Wavelet_Denoiser(Send2Shiru)\inference_results_GAN",
                   help='Where to save results (single-image outputs and batch results).')
    p.add_argument('--no_npy',        action='store_true',
                   help='Skip saving .npy result bundles in batch mode')
    p.add_argument('--no_png',        action='store_true',
                   help='Skip saving .png visualisations in batch mode')

    # MC settings
    p.add_argument('--n_samples',     type=int, default=20,
                   help='Number of MC forward passes (default: 20)')
    p.add_argument('--convergence',   action='store_true',
                   help='Plot epistemic uncertainty vs. n_samples (requires --image)')

    # Model configuration (needed to recreate model structure)
    p.add_argument('--pad', type=str, default='reflect', help='Padding type')
    p.add_argument('--norm', type=str, default='none', help='Normalization type')
    p.add_argument('--init_type', type=str, default='xavier', help='Initialization type')
    p.add_argument('--init_gain', type=float, default=0.02, help='Initialization gain')

    return p.parse_args()


def main():
    args = parse_args()

    # Auto-fall back to CPU
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print(" CUDA unavailable — falling back to CPU")
        device = 'cpu'

    # Create options object (minimal required for model creation)
    class Opt:
        pass
    opt = Opt()
    opt.pad = args.pad
    opt.norm = args.norm
    opt.init_type = args.init_type
    opt.init_gain = args.init_gain

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")

    model = load_model(args.checkpoint, opt, device=device)

    # Ensure output directory exists (used for both single-image and batch modes)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ SINGLE
    if args.image:
        image_path = Path(args.image)
        
        # Check if it's a .npy file or an image file
        if image_path.suffix.lower() == '.npy':
            # Load .npy file (multi-level patch format)
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

            # Infer PSNR from level index if not overridden
            psnr_lvl = args.psnr_level
            if psnr_lvl is None:
                # Lower level index = lower PSNR
                # Typical range: 20-50 dB
                psnr_lvl = 50.0 - (30.0 * (args.input_level / max(1, num_levels - 1)))
            print(f"Input level {args.input_level}/{num_levels-1}  →  PSNR level={psnr_lvl:.1f} dB")
        else:
            # Load image file (PNG, TIFF, etc.)
            noisy = load_image_file(args.image)
            clean = None
            
            # Load ground truth if provided
            if args.gt_image:
                clean = load_image_file(args.gt_image)
                print(f"Loaded noisy image: {args.image}")
                print(f"Loaded ground truth: {args.gt_image}")
            else:
                print(f"Loaded image: {args.image} (no ground truth provided)")
            
            # Use provided PSNR level or default
            psnr_lvl = args.psnr_level if args.psnr_level is not None else 35.0
            print(f"Using PSNR level: {psnr_lvl:.1f} dB")

        results = infer_single(model, noisy, psnr_level=psnr_lvl,
                               n_samples=args.n_samples, device=device)

        stem    = image_path.stem
        out_vis = output_dir / f'{stem}_wavelet_inference.png'
        out_his = output_dir / f'{stem}_wavelet_uncertainty_hist.png'

        # Save full result bundle as .npy (similar structure to batch mode)
        result_dict = {
            'noisy':         noisy,
            'denoised':      results['mean'],
            'epistemic_std': results['epistemic_std'],
            'aleatoric_var': results['aleatoric_var'],
            'combined':      results['combined'],
            'confidence':    results['confidence'],
        }
        if clean is not None:
            result_dict['clean'] = clean
        np.save(output_dir / f'{stem}_results.npy', result_dict)

        visualise_single(noisy, results, ground_truth=clean,
                         save_path=out_vis, psnr_level=psnr_lvl,
                         n_samples=args.n_samples)

        visualise_uncertainty_histogram(results, save_path=out_his)

        if args.convergence:
            visualise_mc_variance_vs_samples(
                model, noisy, psnr_lvl, device=device,
                save_path=output_dir / f'{stem}_wavelet_convergence.png'
            )

    # ------------------------------------------------------------------ BATCH
    elif args.data_dir:
        batch_infer(
            model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            psnr_level_override=args.psnr_level,
            save_npy=not args.no_npy,
            save_png=not args.no_png,
            gt_dir=args.gt_dir,
            device=device,
        )

    else:
        print("Specify --image for single-image mode or --data_dir for batch mode.")
        print("Run with --help for full usage.")


if __name__ == '__main__':
    main()
