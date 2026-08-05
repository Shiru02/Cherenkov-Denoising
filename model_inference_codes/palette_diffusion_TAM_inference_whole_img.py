"""
Evaluation script for Palette-Style TAM Diffusion (x0-Parameterization)
=======================================================================
Evaluates whole images of arbitrary sizes by automatically
cropping them into patches, denoising via batched inference,
and stitching them back together with Gaussian windowed blending.
Saves ALL denoised images and comparison figures with residual maps.

Features:
  - Proper 16-bit TIFF handling (no .convert('L'))
  - Gaussian/cosine windowed blending to eliminate patch seams
  - Inferno residual maps for visual error comparison

IMPROVED EVALUATION (v2):
  - Bootstrap 95% confidence intervals (replaces naive s.d.)
  - Regional sub-block analysis (16 regions per image → 224 data points)
  - Texture vs flat region separation
  - Frequency-domain error analysis (low vs high freq)
  - Edge preservation index (Sobel correlation)
  - Noise residual analysis (bias, skew, kurtosis)
  - Per-image profile plots
  - Improved bar charts with bootstrap CIs
"""

import argparse
import math
import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats, ndimage
from scipy.signal import convolve2d

# ==============================================================================
# ========================== IMPORT FROM TRAINING SCRIPT =======================
# ==============================================================================
try:
    from palette_diffusion_TAM import PaletteUNet, DiffusionSchedule, DiffusionSampler
except ImportError:
    print("Error: Could not import from palette_diffusion_TAM.py")
    print("Please make sure this evaluation script is saved in the exact same directory!")
    exit(1)

try:
    import lpips
    import piq
except ImportError:
    print("Error: Missing metric libraries.")
    print("Please install them by running: pip install lpips piq")
    exit(1)

# ==============================================================================
# ========================== IMAGE I/O HELPERS =================================
# ==============================================================================

def load_image_as_float(path: str) -> np.ndarray:
    """
    Load a grayscale image (8-bit or 16-bit) and normalise to [0, 1] float32.
    Does NOT call .convert('L'), which would silently collapse 16-bit to 8-bit.
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3:
        arr = (0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2])

    if arr.dtype == np.uint8:
        max_val = 255.0
    elif arr.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
        max_val = 65535.0
    elif arr.dtype == np.uint32:
        max_val = 4294967295.0
    elif np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    else:
        obs_max = arr.max()
        max_val = float(obs_max) if obs_max > 0 else 1.0
        print(f"    [WARN] Unknown dtype {arr.dtype} in {path}, normalising by observed max ({max_val})")

    return arr.astype(np.float32) / max_val


def save_image_matching_depth(arr_float: np.ndarray, save_path: str, reference_path: str):
    """
    Save a [0,1] float32 array back to disk with the same bit-depth as the
    reference image.
    """
    ref_img = Image.open(reference_path)
    ref_arr = np.array(ref_img)

    if ref_arr.dtype in (np.uint16, np.dtype('>u2'), np.dtype('<u2')):
        out = (np.clip(arr_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
    else:
        out = (np.clip(arr_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(out).save(save_path)


# ==============================================================================
# ========================== ORIGINAL METRICS ==================================
# ==============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    return float('inf') if mse == 0 else 20 * math.log10(max_val) - 10 * math.log10(mse)

def compute_ssim_np(pred: np.ndarray, target: np.ndarray, window_size: int = 11) -> float:
    from scipy.ndimage import uniform_filter
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

def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))

def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))

def compute_fsim(pred: np.ndarray, target: np.ndarray, device: str) -> float:
    p_t = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return piq.fsim(p_t, t_t, data_range=1.0, chromatic=False).item()

def compute_lpips(pred: np.ndarray, target: np.ndarray, loss_fn_vgg, device: str) -> float:
    p_t = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) * 2.0 - 1.0
    t_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) * 2.0 - 1.0
    p_t = p_t.repeat(1, 3, 1, 1)
    t_t = t_t.repeat(1, 3, 1, 1)
    with torch.no_grad():
        return loss_fn_vgg(p_t, t_t).item()

# ==============================================================================
# ========================== NEW IMPROVED METRICS ==============================
# ==============================================================================

def compute_edge_preservation(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Edge Preservation Index (EPI): Pearson correlation between Sobel edge
    magnitudes of prediction and target. Higher = better edge fidelity.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T

    def edge_mag(img):
        gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
        gy = convolve2d(img, sobel_y, mode='same', boundary='symm')
        return np.sqrt(gx**2 + gy**2)

    e_pred = edge_mag(pred)
    e_target = edge_mag(target)

    if np.std(e_pred) < 1e-10 or np.std(e_target) < 1e-10:
        return 0.0
    return float(np.corrcoef(e_pred.ravel(), e_target.ravel())[0, 1])


def compute_freq_error(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Frequency-domain error analysis.
    Splits error into low-frequency (smooth structures) and high-frequency (details).
    freq_ratio > 1 means the model struggles more with fine details.
    """
    F_pred = np.fft.fft2(pred)
    F_target = np.fft.fft2(target)
    F_err = np.abs(F_pred - F_target)

    h, w = pred.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

    F_err_shifted = np.fft.fftshift(F_err)

    radius = min(cy, cx)
    low_mask = dist <= radius * 0.25
    high_mask = dist >= radius * 0.5

    low_err = float(np.mean(F_err_shifted[low_mask])) if np.any(low_mask) else 0.0
    high_err = float(np.mean(F_err_shifted[high_mask])) if np.any(high_mask) else 0.0
    ratio = high_err / (low_err + 1e-10)

    return {'FreqErr_Low': low_err, 'FreqErr_High': high_err, 'FreqErr_Ratio': ratio}


def compute_regional_metrics(pred: np.ndarray, target: np.ndarray, n_blocks: int = 4) -> dict:
    """
    Divide image into n_blocks x n_blocks sub-regions and compute MAE + SSIM
    per region. With 14 images × 16 regions = 224 data points.
    Returns dict with lists of regional MAE and SSIM values.
    """
    h, w = pred.shape
    bh, bw = h // n_blocks, w // n_blocks
    regional_mae = []
    regional_ssim = []

    for i in range(n_blocks):
        for j in range(n_blocks):
            y0, y1 = i * bh, (i + 1) * bh
            x0, x1 = j * bw, (j + 1) * bw
            p_block = pred[y0:y1, x0:x1]
            t_block = target[y0:y1, x0:x1]
            regional_mae.append(compute_mae(p_block, t_block))
            regional_ssim.append(compute_ssim_np(p_block, t_block))

    return {'regional_MAE': regional_mae, 'regional_SSIM': regional_ssim}


def compute_texture_vs_flat(pred: np.ndarray, target: np.ndarray, threshold: float = 0.02) -> dict:
    """
    Separate analysis for texture-rich vs flat/smooth regions.
    Uses local standard deviation of the ground truth to classify regions.
    """
    local_var = ndimage.uniform_filter(target**2, size=11) - ndimage.uniform_filter(target, size=11)**2
    local_var = np.maximum(local_var, 0)
    local_std = np.sqrt(local_var)

    texture_mask = local_std > threshold
    flat_mask = ~texture_mask

    results = {}
    for name, mask in [('texture', texture_mask), ('flat', flat_mask)]:
        if np.sum(mask) > 100:
            results[f'MAE_{name}'] = float(np.mean(np.abs(pred[mask] - target[mask])))
        else:
            results[f'MAE_{name}'] = np.nan

    return results


def compute_noise_residual_stats(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Analyze the residual (pred - target) for systematic bias and structure.
    Ideal model: mean≈0, low std, skew≈0, kurtosis≈0 (Gaussian-like errors).
    """
    residual = pred - target
    return {
        'residual_mean': float(np.mean(residual)),
        'residual_std': float(np.std(residual)),
        'residual_skew': float(stats.skew(residual.ravel())),
        'residual_kurtosis': float(stats.kurtosis(residual.ravel())),
    }


# ==============================================================================
# ========================== STATISTICAL HELPERS ===============================
# ==============================================================================

def bootstrap_ci(values, n_boot=5000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) < 3:
        return float(np.mean(values)), float(np.mean(values)), float(np.mean(values))
    n = len(values)
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(values, n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(lo), float(np.mean(values)), float(hi)


# ==============================================================================
# ========================== VISUALISATION =====================================
# ==============================================================================

def save_evaluation_sample(noisy: np.ndarray, denoised: np.ndarray, gt: np.ndarray, save_path: str):
    """
    Save a 2-row comparison figure:
      Row 1: Noisy | Denoised | Ground Truth | Denoised − GT (Raw)
      Row 2: |Noisy − GT| | |Denoised − GT| | Abs Colorbar | Raw Colorbar
    
    Absolute residuals use the 'inferno' colormap (dark=low, bright=high).
    Raw residuals use the 'bwr' diverging colormap (blue=negative, red=positive).
    """
    abs_residual_noisy    = np.abs(noisy - gt)
    abs_residual_denoised = np.abs(denoised - gt)
    raw_residual_denoised = denoised - gt

    vmax_abs = max(np.max(abs_residual_noisy), np.max(abs_residual_denoised))
    if vmax_abs < 1e-8:
        vmax_abs = 1.0

    vmax_raw = np.max(np.abs(raw_residual_denoised))
    if vmax_raw < 1e-8:
        vmax_raw = 1.0

    fig, axes = plt.subplots(2, 4, figsize=(24, 10),
                             gridspec_kw={'height_ratios': [1, 1]})

    # ---- Row 1: images ----
    axes[0, 0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Original (Noisy Full Image)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Denoised (Stitched)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title("Ground Truth")
    axes[0, 2].axis("off")

    # ---- Row 1, Col 4: Raw residual (bwr) ----
    im_raw = axes[0, 3].imshow(raw_residual_denoised, cmap='bwr', vmin=-vmax_raw, vmax=vmax_raw)
    axes[0, 3].set_title("Denoised − GT (Raw Error)")
    axes[0, 3].axis("off")

    # ---- Row 2: absolute residuals (inferno) ----
    im0 = axes[1, 0].imshow(abs_residual_noisy, cmap='inferno', vmin=0, vmax=vmax_abs)
    noisy_mae = np.mean(abs_residual_noisy)
    axes[1, 0].set_title(f"|Noisy − GT|  (MAE={noisy_mae:.4f})")
    axes[1, 0].axis("off")

    im1 = axes[1, 1].imshow(abs_residual_denoised, cmap='inferno', vmin=0, vmax=vmax_abs)
    denoised_mae = np.mean(abs_residual_denoised)
    axes[1, 1].set_title(f"|Denoised − GT|  (MAE={denoised_mae:.4f})")
    axes[1, 1].axis("off")

    # Colorbar in the third column of row 2 (Absolute)
    axes[1, 2].axis("off")
    cbar_abs = fig.colorbar(im1, ax=axes[1, 2], fraction=0.6, pad=0.05, shrink=0.8)
    cbar_abs.set_label("Absolute pixel error (dark = low, bright = high)", fontsize=10)

    # Colorbar in the fourth column of row 2 (Raw)
    axes[1, 3].axis("off")
    cbar_raw = fig.colorbar(im_raw, ax=axes[1, 3], fraction=0.6, pad=0.05, shrink=0.8)
    cbar_raw.set_label("Raw pixel error (blue = negative, red = positive)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ==============================================================================
# =============== NEW: IMPROVED SUMMARY PLOTS ==================================
# ==============================================================================

def save_bootstrap_bar_chart(values: list, metric_name: str, save_path: str,
                              color: str = '#5B8DB8'):
    """
    Bar chart with a single model showing bootstrap 95% CI + individual points.
    Replaces the old mean ± s.d. bars with tighter, more honest intervals.
    """
    lo, mean, hi = bootstrap_ci(values)

    fig, ax = plt.subplots(figsize=(5, 6))

    ax.bar([0], [mean], color=color, alpha=0.7, width=0.5, edgecolor='gray', linewidth=0.8)
    ax.errorbar([0], [mean], yerr=[[mean - lo], [hi - mean]], fmt='none',
                ecolor='black', capsize=8, capthick=2, linewidth=2)

    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(values))
    ax.scatter(jitter, values, color=color, alpha=0.5, s=40, edgecolor='white', linewidth=0.5, zorder=5)

    ax.set_xticks([0])
    ax.set_xticklabels(['Palette-Diffusion\nTAM'], fontsize=11)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} — Bootstrap 95% CI\n'
                 f'Mean={mean:.5f}  CI=[{lo:.5f}, {hi:.5f}]', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_per_image_profile(records: list, save_path: str):
    """
    Line plot showing all key metrics per image — reveals which images are
    easy/hard and whether the model has consistent or erratic performance.
    """
    filenames = [r['filename'].replace('.tiff', '').replace('.tif', '') for r in records]
    x = np.arange(len(filenames))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    vals = [r['MAE'] for r in records]
    axes[0].bar(x, vals, color='#E8593C', alpha=0.7, edgecolor='gray', linewidth=0.5)
    axes[0].axhline(np.mean(vals), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(vals):.5f}')
    axes[0].set_ylabel('MAE', fontsize=11)
    axes[0].set_title('Per-Image Metric Profile', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    vals = [r['SSIM'] for r in records]
    axes[1].bar(x, vals, color='#5DCAA5', alpha=0.7, edgecolor='gray', linewidth=0.5)
    axes[1].axhline(np.mean(vals), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(vals):.4f}')
    axes[1].set_ylabel('SSIM', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    vals = [r['EdgePreservation'] for r in records]
    axes[2].bar(x, vals, color='#7F77DD', alpha=0.7, edgecolor='gray', linewidth=0.5)
    axes[2].axhline(np.mean(vals), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(vals):.4f}')
    axes[2].set_ylabel('Edge Preservation', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(filenames, rotation=55, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_regional_analysis_plot(all_regional_mae: list, all_regional_ssim: list, save_path: str):
    """
    Histogram + stats of regional (sub-block) MAE and SSIM values.
    Shows the distribution across 16 regions × N images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lo, mean, hi = bootstrap_ci(all_regional_mae)
    axes[0].hist(all_regional_mae, bins=30, color='#E8593C', alpha=0.7, edgecolor='white')
    axes[0].axvline(mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean={mean:.5f}')
    axes[0].axvline(lo, color='gray', linestyle=':', linewidth=1, label=f'95% CI=[{lo:.5f}, {hi:.5f}]')
    axes[0].axvline(hi, color='gray', linestyle=':', linewidth=1)
    axes[0].set_xlabel('Regional MAE', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'Regional MAE Distribution (n={len(all_regional_mae)} sub-blocks)', fontsize=11)
    axes[0].legend(fontsize=9)

    lo, mean, hi = bootstrap_ci(all_regional_ssim)
    axes[1].hist(all_regional_ssim, bins=30, color='#5DCAA5', alpha=0.7, edgecolor='white')
    axes[1].axvline(mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean={mean:.4f}')
    axes[1].axvline(lo, color='gray', linestyle=':', linewidth=1, label=f'95% CI=[{lo:.4f}, {hi:.4f}]')
    axes[1].axvline(hi, color='gray', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Regional SSIM', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(f'Regional SSIM Distribution (n={len(all_regional_ssim)} sub-blocks)', fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_texture_vs_flat_plot(records: list, save_path: str):
    """Bar chart comparing MAE in texture-rich vs flat regions per image."""
    filenames = [r['filename'].replace('.tiff', '').replace('.tif', '') for r in records]
    tex_vals = [r.get('MAE_texture', np.nan) for r in records]
    flat_vals = [r.get('MAE_flat', np.nan) for r in records]

    x = np.arange(len(filenames))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, tex_vals, width, label='Texture regions', color='#D85A30', alpha=0.7)
    ax.bar(x + width/2, flat_vals, width, label='Flat regions', color='#378ADD', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(filenames, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('MAE: Texture-Rich vs Flat Regions per Image', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_frequency_analysis_plot(records: list, save_path: str):
    """Bar chart of low vs high frequency error per image."""
    filenames = [r['filename'].replace('.tiff', '').replace('.tif', '') for r in records]
    low_vals = [r['FreqErr_Low'] for r in records]
    high_vals = [r['FreqErr_High'] for r in records]

    x = np.arange(len(filenames))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, low_vals, width, label='Low-freq error (structure)', color='#5DCAA5', alpha=0.7)
    ax.bar(x + width/2, high_vals, width, label='High-freq error (details)', color='#D85A30', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(filenames, rotation=55, ha='right', fontsize=8)
    ax.set_ylabel('Frequency-Domain Error', fontsize=11)
    ax.set_title('Frequency Error: Low (Structure) vs High (Details) per Image', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_residual_analysis_plot(records: list, save_path: str):
    """Multi-panel residual statistics per image."""
    filenames = [r['filename'].replace('.tiff', '').replace('.tif', '') for r in records]
    x = np.arange(len(filenames))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    stats_list = [
        ('residual_mean', 'Residual Mean (Bias)', '#378ADD'),
        ('residual_std', 'Residual Std Dev', '#D85A30'),
        ('residual_skew', 'Residual Skewness', '#7F77DD'),
        ('residual_kurtosis', 'Residual Kurtosis', '#1D9E75'),
    ]

    for ax, (key, title, color) in zip(axes.ravel(), stats_list):
        vals = [r[key] for r in records]
        ax.bar(x, vals, color=color, alpha=0.7, edgecolor='gray', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=55, ha='right', fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Noise Residual Analysis (pred − GT)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_summary_radar(record_means: dict, save_path: str):
    """
    Radar chart showing normalised metric profile for the model.
    Each metric is normalised to [0, 1] where 1 = best observed value.
    """
    metric_info = [
        ('MAE', False),
        ('SSIM', True),
        ('PSNR_Result', True),
        ('EdgePreservation', True),
        ('FSIM', True),
        ('LPIPS', False),
    ]

    labels = []
    values = []
    for name, higher_better in metric_info:
        if name in record_means:
            labels.append(name)
            values.append(record_means[name])

    if len(labels) < 3:
        return

    norm_vals = []
    for i, (name, higher_better) in enumerate(metric_info):
        if name not in record_means:
            continue
        norm_vals.append(values[len(norm_vals)])

    norm_arr = np.array(norm_vals)
    norm_arr = (norm_arr - norm_arr.min()) / (norm_arr.max() - norm_arr.min() + 1e-10)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    norm_list = norm_arr.tolist()
    angles += angles[:1]
    norm_list += norm_list[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, norm_list, 'o-', linewidth=2, color='#7F77DD')
    ax.fill(angles, norm_list, alpha=0.15, color='#7F77DD')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Model Metric Profile (normalised)', fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ==============================================================================
# ========================== WHOLE IMAGE PROCESSING ============================
# ==============================================================================

def _build_blending_window(size: int, mode: str = 'gaussian', sigma_fraction: float = 0.25) -> np.ndarray:
    if mode == 'gaussian':
        sigma = size * sigma_fraction
        ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
        kern_1d = np.exp(-0.5 * (ax / sigma) ** 2)
        window = np.outer(kern_1d, kern_1d)
    elif mode == 'cosine':
        hann_1d = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(size, dtype=np.float64) / (size - 1)))
        window = np.outer(hann_1d, hann_1d)
    else:
        raise ValueError(f"Unknown blending mode: {mode}")

    window = window / window.max()
    window = np.maximum(window, 1e-6)
    return window.astype(np.float32)


def process_whole_image(noisy_np: np.ndarray, sampler: DiffusionSampler, args, device: str) -> np.ndarray:
    """
    Crops a full image into overlapping patches, runs diffusion denoising on
    batches, and stitches them back using soft (Gaussian) windowed blending.
    """
    h, w = noisy_np.shape
    p_size = args.patch_size
    stride = args.stride
    blend_mode = getattr(args, 'blend_mode', 'gaussian')

    n_h = math.ceil(max(h - p_size, 0) / stride) + 1 if h > p_size else 1
    n_w = math.ceil(max(w - p_size, 0) / stride) + 1 if w > p_size else 1

    pad_h = max((n_h - 1) * stride + p_size - h, 0)
    pad_w = max((n_w - 1) * stride + p_size - w, 0)

    img_padded = np.pad(noisy_np, ((0, pad_h), (0, pad_w)), mode='reflect')

    canvas = np.zeros_like(img_padded, dtype=np.float32)
    weight_map = np.zeros_like(img_padded, dtype=np.float32)

    blend_window = _build_blending_window(p_size, mode=blend_mode)

    patches = []
    coords = []

    for i in range(n_h):
        for j in range(n_w):
            y = i * stride
            x = j * stride
            patch = img_padded[y:y+p_size, x:x+p_size]
            patches.append(patch)
            coords.append((y, x))

    denoised_patches = []
    nl_tensor = torch.tensor([[args.noise_level]], dtype=torch.float32).to(device)

    print(f"    -> Extracted {len(patches)} patches. Processing in batches of {args.batch_size}...")
    print(f"    -> Blending mode: {blend_mode}")

    for i in range(0, len(patches), args.batch_size):
        batch_patches = patches[i:i+args.batch_size]

        batch_tensor = torch.tensor(np.array(batch_patches), dtype=torch.float32).unsqueeze(1).to(device)
        batch_nl = nl_tensor.repeat(len(batch_patches), 1)

        with torch.no_grad():
            denoised_batch = sampler.ddim_sample(
                y=batch_tensor,
                noise_level=batch_nl,
                n_steps=args.ddim_steps,
                eta=args.eta
            )

        denoised_patches.extend(denoised_batch.squeeze(1).cpu().numpy())

    for patch, (y, x) in zip(denoised_patches, coords):
        canvas[y:y+p_size, x:x+p_size]     += patch * blend_window
        weight_map[y:y+p_size, x:x+p_size] += blend_window

    reconstructed_padded = canvas / np.clip(weight_map, 1e-8, None)

    reconstructed = reconstructed_padded[:h, :w]
    return reconstructed

# ==============================================================================
# ========================== DATASET EVALUATION ================================
# ==============================================================================

def evaluate_dataset(sampler: DiffusionSampler, args, device: str = 'cuda'):
    noisy_files = sorted(glob.glob(os.path.join(args.data_dir, '*.tiff')) +
                         glob.glob(os.path.join(args.data_dir, '*.tif')))

    if not noisy_files:
        raise ValueError(f"No .tiff or .tif files found in {args.data_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    denoised_dir = os.path.join(args.output_dir, "denoised_images")
    figures_dir  = os.path.join(args.output_dir, "comparison_figures")
    analysis_dir = os.path.join(args.output_dir, "improved_analysis")

    Path(denoised_dir).mkdir(exist_ok=True)
    Path(figures_dir).mkdir(exist_ok=True)
    Path(analysis_dir).mkdir(exist_ok=True)

    print("Initializing LPIPS metric model (AlexNet)...")
    loss_fn_lpips = lpips.LPIPS(net='alex', verbose=False).to(device)
    loss_fn_lpips.eval()

    records = []
    all_regional_mae = []
    all_regional_ssim = []

    print(f"Starting TAM Diffusion DDIM evaluation on {len(noisy_files)} FULL images...")
    print(f"  (With improved metrics: regional, frequency, edge, texture/flat, residual analysis)")

    for i, noisy_path in enumerate(noisy_files):
        filename = os.path.basename(noisy_path)
        gt_path = os.path.join(args.gt_dir, filename)

        if not os.path.exists(gt_path):
            print(f"Skipping {filename}, Ground Truth missing.")
            continue

        noisy_np = load_image_as_float(noisy_path)
        gt_np    = load_image_as_float(gt_path)

        print(f"  [{i+1:4d}/{len(noisy_files)}] Processing {filename} ({noisy_np.shape[1]}x{noisy_np.shape[0]})...")

        denoised_np = process_whole_image(noisy_np, sampler, args, device)

        denoised_save_path = os.path.join(denoised_dir, filename)
        save_image_matching_depth(denoised_np, denoised_save_path, noisy_path)

        fig_name = f"compare_{filename.replace('.tiff', '.png').replace('.tif', '.png')}"
        fig_save_path = os.path.join(figures_dir, fig_name)
        save_evaluation_sample(noisy_np, denoised_np, gt_np, fig_save_path)
        print(f"    -> Saved raw image and comparison figure.")

        # ===================== ORIGINAL METRICS =====================
        mae = compute_mae(denoised_np, gt_np)
        mse = compute_mse(denoised_np, gt_np)
        ssim_val = compute_ssim_np(denoised_np, gt_np)
        fsim_val = compute_fsim(denoised_np, gt_np, device)
        lpips_val = compute_lpips(denoised_np, gt_np, loss_fn_lpips, device)

        psnr_baseline = compute_psnr(noisy_np, gt_np)
        psnr_result   = compute_psnr(denoised_np, gt_np)
        psnr_diff     = compute_psnr(noisy_np, denoised_np)

        # ===================== NEW IMPROVED METRICS =================
        epi = compute_edge_preservation(denoised_np, gt_np)
        freq_results = compute_freq_error(denoised_np, gt_np)
        regional = compute_regional_metrics(denoised_np, gt_np, n_blocks=4)
        tex_flat = compute_texture_vs_flat(denoised_np, gt_np)
        res_stats = compute_noise_residual_stats(denoised_np, gt_np)

        all_regional_mae.extend(regional['regional_MAE'])
        all_regional_ssim.extend(regional['regional_SSIM'])

        record = {
            'filename': filename,
            # Original
            'MAE': mae, 'MSE': mse, 'SSIM': ssim_val,
            'FSIM': fsim_val, 'LPIPS': lpips_val,
            'PSNR_Baseline': psnr_baseline, 'PSNR_Result': psnr_result, 'PSNR_Diff': psnr_diff,
            # New
            'EdgePreservation': epi,
            **freq_results,
            **tex_flat,
            **res_stats,
            'regional_MAE_mean': np.mean(regional['regional_MAE']),
            'regional_MAE_std': np.std(regional['regional_MAE']),
            'regional_SSIM_mean': np.mean(regional['regional_SSIM']),
            'regional_SSIM_std': np.std(regional['regional_SSIM']),
        }
        records.append(record)

        print(f"    -> PSNR: {psnr_baseline:.2f} -> {psnr_result:.2f} dB | LPIPS: {lpips_val:.4f} | "
              f"SSIM: {ssim_val:.4f} | EdgePres: {epi:.4f}")

    # ===================== SAVE RAW DATA ============================
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, 'evaluation_metrics_full_images.csv')
    df.to_csv(csv_path, index=False)

    # ===================== PRINT SUMMARY ============================
    print(f"\n{'='*65}")
    print(f"Evaluation Complete! Results saved to {args.output_dir}")
    print(f"--- Standard Metrics ---")
    print(f"Average MAE:   {df['MAE'].mean():.6f}")
    print(f"Average MSE:   {df['MSE'].mean():.6f}")
    print(f"Average SSIM:  {df['SSIM'].mean():.4f}")
    print(f"--- Perceptual Metrics ---")
    print(f"Average FSIM:  {df['FSIM'].mean():.4f}")
    print(f"Average LPIPS: {df['LPIPS'].mean():.4f}")
    print(f"--- PSNR Breakdown ---")
    print(f"Avg PSNR Baseline (Noisy vs GT): {df['PSNR_Baseline'].mean():.2f} dB")
    print(f"Avg PSNR Result (Denoised vs GT): {df['PSNR_Result'].mean():.2f} dB")
    print(f"Avg PSNR Diff (Noisy vs Denoised): {df['PSNR_Diff'].mean():.2f} dB")

    print(f"\n--- NEW: Improved Metrics ---")
    print(f"Average Edge Preservation: {df['EdgePreservation'].mean():.4f}")
    print(f"Average Freq Error (Low):  {df['FreqErr_Low'].mean():.4f}")
    print(f"Average Freq Error (High): {df['FreqErr_High'].mean():.4f}")
    print(f"Average Freq Ratio (H/L):  {df['FreqErr_Ratio'].mean():.4f}")

    mae_tex = df['MAE_texture'].dropna()
    mae_flat = df['MAE_flat'].dropna()
    if len(mae_tex) > 0:
        print(f"Average MAE (Texture):     {mae_tex.mean():.6f}")
    if len(mae_flat) > 0:
        print(f"Average MAE (Flat):        {mae_flat.mean():.6f}")

    print(f"Average Residual Bias:     {df['residual_mean'].mean():.6f}")
    print(f"Average Residual Std:      {df['residual_std'].mean():.6f}")
    print(f"Average Residual Skew:     {df['residual_skew'].mean():.4f}")
    print(f"Average Residual Kurtosis: {df['residual_kurtosis'].mean():.4f}")

    lo, mean, hi = bootstrap_ci(all_regional_mae)
    print(f"\n--- Regional Analysis ({len(all_regional_mae)} sub-blocks) ---")
    print(f"Regional MAE:  mean={mean:.6f}  95% CI=[{lo:.6f}, {hi:.6f}]")
    lo, mean, hi = bootstrap_ci(all_regional_ssim)
    print(f"Regional SSIM: mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    print(f"\n--- Bootstrap 95% Confidence Intervals (whole-image) ---")
    for metric in ['MAE', 'SSIM', 'PSNR_Result', 'LPIPS', 'FSIM', 'EdgePreservation']:
        vals = df[metric].dropna().tolist()
        if len(vals) >= 3:
            lo, mean, hi = bootstrap_ci(vals)
            print(f"  {metric:20s}: mean={mean:.5f}  CI=[{lo:.5f}, {hi:.5f}]")

    print(f"{'='*65}\n")

    # ===================== GENERATE IMPROVED PLOTS ==================
    print("Generating improved analysis plots...")

    for metric, color in [('MAE', '#E8593C'), ('SSIM', '#5DCAA5'), ('LPIPS', '#D85A30'),
                           ('PSNR_Result', '#378ADD'), ('EdgePreservation', '#7F77DD'),
                           ('FSIM', '#1D9E75')]:
        vals = df[metric].dropna().tolist()
        if len(vals) >= 3:
            save_bootstrap_bar_chart(vals, metric,
                                     os.path.join(analysis_dir, f'bootstrap_{metric}.png'), color)

    save_per_image_profile(records, os.path.join(analysis_dir, 'per_image_profile.png'))
    save_regional_analysis_plot(all_regional_mae, all_regional_ssim,
                                os.path.join(analysis_dir, 'regional_analysis.png'))
    save_texture_vs_flat_plot(records, os.path.join(analysis_dir, 'texture_vs_flat.png'))
    save_frequency_analysis_plot(records, os.path.join(analysis_dir, 'frequency_analysis.png'))
    save_residual_analysis_plot(records, os.path.join(analysis_dir, 'residual_analysis.png'))

    means_dict = {col: df[col].mean() for col in df.columns if df[col].dtype in [np.float64, np.float32, float]}
    save_summary_radar(means_dict, os.path.join(analysis_dir, 'metric_radar.png'))

    print(f"  All improved analysis plots saved to: {analysis_dir}")
    print(f"  Files: bootstrap_*.png, per_image_profile.png, regional_analysis.png,")
    print(f"         texture_vs_flat.png, frequency_analysis.png, residual_analysis.png, metric_radar.png")


# ==============================================================================
# ========================== CLI ENTRYPOINT ====================================
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Evaluation script for Palette TAM Diffusion on Full Images')

    p.add_argument('--checkpoint', type=str,
                   default=r"E:\Chr_denoise\experiments\palette_cherenkov_tam\checkpoints\best_model.pth",
                   help='Path to the TAM model checkpoint (.pth or .pt)')
    p.add_argument('--device', type=str, default='cuda')

    p.add_argument('--data_dir', type=str,
                   default=r"E:\Chr_denoise\test_imgs\Noise_whole_img",
                   help='Directory of noisy full .tiff images')
    p.add_argument('--gt_dir', type=str,
                   default=r"E:\Chr_denoise\test_imgs\GT_whole_img",
                   help='Directory of ground truth full .tiff images')
    p.add_argument('--output_dir', type=str,
                   default=r"E:\Chr_denoise\tam_evaluation_results_full",
                   help='Where to save results (metrics CSV and image samples)')

    p.add_argument('--patch_size', type=int, default=128,
                   help='Size of the patches to extract (should match your training size, e.g., 256)')
    p.add_argument('--stride', type=int, default=64,
                   help='Stride for patch extraction. If less than patch_size, patches will overlap.')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Number of patches to denoise simultaneously.')

    p.add_argument('--ddim_steps', type=int, default=3,
                   help='Number of DDIM steps for inference (default: 100)')
    p.add_argument('--eta', type=float, default=0.0,
                   help='DDIM eta parameter. (default: 0.0)')
    p.add_argument('--noise_level', type=float, default=0.8,
                   help='Noise level condition fed to the model. Default: 1.0')
    p.add_argument('--blend_mode', type=str, default='gaussian', choices=['gaussian', 'cosine'],
                   help='Blending window for patch stitching. "gaussian" (default) or "cosine" (Hann window).')

    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    model = PaletteUNet(base_channels=64, embed_dim=128).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print(f" Loaded TAM model checkpoint from epoch {ckpt.get('epoch', '?')} (Best Val Loss: {ckpt.get('best_loss', 'N/A')})")
    else:
        model.load_state_dict(ckpt)

    model.eval()

    schedule = DiffusionSchedule(T=1000).to(device)
    sampler = DiffusionSampler(schedule, model, device)

    evaluate_dataset(sampler, args, device=device)

if __name__ == '__main__':
    main()