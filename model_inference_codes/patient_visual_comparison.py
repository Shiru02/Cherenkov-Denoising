"""
Patient Visual Comparison — End-to-End
========================================
Step 1: Interactive ROI selection (click to place zoom box for each patient)
Step 2: Model inference + figure generation

Run from E:\\Chr_denoise\\model_training\\:
    python patient_visual_comparison_e2e.py

Workflow:
  1. For each patient, a window opens showing the GT image in inferno colormap
  2. Left-click to place the zoom box, click again to adjust
  3. Close window to confirm → next patient
  4. After all patients selected, models load and inference runs automatically
  5. Final figure saved as PNG + PDF

To skip ROI selection and reuse saved coordinates, set SKIP_ROI_SELECTION = True
"""

import math
import os
import re
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

# ══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PATIENT_ROOT = r"E:\Chr_denoise\test_imgs\Patient_imgs"
OUTPUT_DIR   = r"E:\Chr_denoise\patient_evaluation_results\figures"

PATIENT_NAMES = ["img14", "img22", "img50"]
FRAME_IDX     = 0

# Set True to skip interactive selection and use saved coords from last run
SKIP_ROI_SELECTION = False
SAVED_COORDS_PATH  = os.path.join(OUTPUT_DIR, "edge_zoom_coords.json")

# Models to show (order = column order)
MODELS_TO_SHOW = [
    "unet_mc",
    "gan_baseline",
    "palette_noise",
    "palette_tam_ssim_freq",
    "palette_fullwavelet",
]

COLUMN_LABELS = [
    "Ground\nTruth",
    "Noisy\nInput",
    "UNet",
    "cGAN",
    "Diffusion",
    "Diffusion\n+ TAM + Freq",
    "Diffusion\n+ Wavelet",
]

CHECKPOINTS = {
    "palette_tam_ssim_freq": r"E:\Chr_denoise\experiments\palette_cherenkov_tam_freq\checkpoints\best_model.pth",
    "palette_noise":         r"E:\Chr_denoise\experiments\palette_cherenkov_refactored\checkpoints\best_model.pth",
    "palette_fullwavelet":   r"E:\Chr_denoise\experiments\palette_cherenkov_fullwavelet\checkpoints\best_model.pth",
    "unet_mc":               r"E:\Chr_denoise\experiments\unet_mc_dropout\checkpoints\best_model.pt",
    "gan_baseline":          r"E:\Chr_denoise\experiments\gan_baseline\checkpoints\best_model.pth",
}

# Inference parameters
PATCH_SIZE   = 128
STRIDE       = 64
BATCH_SIZE   = 8
DDIM_STEPS   = 3
ETA          = 0.0
NOISE_LEVEL  = 0.8
BLEND_MODE   = "gaussian"
TAM_HISTORY  = 5
N_MC_SAMPLES = 20
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ROI config
SIGNAL_THRESHOLD = 0.05
ROI_PADDING      = 20
EDGE_ZOOM_SIZE   = 160
INSET_FRAC       = 0.32

# Figure style
EDGE_BOX_COLOR = "#44BBFF"


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE I/O
# ══════════════════════════════════════════════════════════════════════════════
def load_image_as_float(path: str) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = 0.2989*arr[:,:,0] + 0.5870*arr[:,:,1] + 0.1140*arr[:,:,2]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif arr.dtype in (np.uint16, np.dtype(">u2"), np.dtype("<u2")):
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint32:
        return arr.astype(np.float32) / 4294967295.0
    elif np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    else:
        m = arr.max(); m = float(m) if m > 0 else 1.0
        return arr.astype(np.float32) / m


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT DATA DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
def discover_patient_data(root):
    patients = {}
    for subfolder in sorted(os.listdir(root)):
        fpath = os.path.join(root, subfolder)
        if not os.path.isdir(fpath):
            continue
        def ckv_idx(p):
            m = re.search(r"CKV_(\d+)", os.path.basename(p), re.IGNORECASE)
            return int(m.group(1)) if m else -1
        VALID_EXTS = {".tif", ".tiff", ".png"}
        index_to_path = {}
        for fname in os.listdir(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS: continue
            idx = ckv_idx(fname)
            if idx < 0: continue
            if idx not in index_to_path:
                index_to_path[idx] = os.path.join(fpath, fname)
        ckv_files = list(index_to_path.values())
        if len(ckv_files) < 2: continue
        ckv_sorted = sorted(ckv_files, key=ckv_idx)
        patients[subfolder] = {"gt_path": ckv_sorted[-1], "noisy_paths": ckv_sorted[:-1]}
    return patients


def detect_signal_roi(gt_img, threshold=SIGNAL_THRESHOLD, padding=ROI_PADDING):
    mask = gt_img > (gt_img.max() * threshold)
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, gt_img.shape[0], 0, gt_img.shape[1]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    h, w = gt_img.shape
    return max(0, y0-padding), min(h, y1+padding), max(0, x0-padding), min(w, x1+padding)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: INTERACTIVE ROI SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def select_zoom_interactive(pname, gt_img):
    """Show GT image, let user click to place zoom box. Returns (cy, cx)."""
    matplotlib.use("TkAgg")
    import importlib
    importlib.reload(plt)

    roi = detect_signal_roi(gt_img)
    y0, y1, x0, x1 = roi
    roi_crop = gt_img[y0:y1, x0:x1]
    half = EDGE_ZOOM_SIZE // 2

    selected = {"cy": None, "cx": None}
    rect_patch = [None]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(roi_crop, cmap="inferno", aspect="equal",
              vmin=0, vmax=np.percentile(roi_crop, 99.5))
    ax.set_title(f"{pname} — Click to place zoom box, then close window",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Close window when done (X button)")

    def on_click(event):
        if event.inaxes != ax or event.button != 1: return
        cx_roi, cy_roi = int(event.xdata), int(event.ydata)
        selected["cy"] = y0 + cy_roi
        selected["cx"] = x0 + cx_roi
        if rect_patch[0] is not None:
            rect_patch[0].remove()
        rect = mpatches.Rectangle(
            (cx_roi - half, cy_roi - half), EDGE_ZOOM_SIZE, EDGE_ZOOM_SIZE,
            linewidth=2, edgecolor=EDGE_BOX_COLOR, facecolor="none")
        ax.add_patch(rect)
        rect_patch[0] = rect
        ax.set_title(f"{pname} — Selected: ({selected['cy']}, {selected['cx']})  |  Click again to adjust",
                     fontsize=12, fontweight="bold")
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    return selected["cy"], selected["cx"]


def run_roi_selection(all_patients):
    """Interactive ROI selection for all patients. Returns edge_zoom dict."""
    edge_zoom = {}
    for pname in PATIENT_NAMES:
        if pname not in all_patients:
            print(f"[ERROR] Patient '{pname}' not found"); continue
        gt = load_image_as_float(all_patients[pname]["gt_path"])
        print(f"\n{'='*50}")
        print(f"  {pname}: Click to place zoom box, then close window")
        print(f"{'='*50}")
        cy, cx = select_zoom_interactive(pname, gt)
        if cy is not None and cx is not None:
            edge_zoom[pname] = (int(cy), int(cx))
            print(f"  → {pname}: ({cy}, {cx})")
        else:
            print(f"  → {pname}: No selection, will auto-detect")

    # Save for reuse
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(SAVED_COORDS_PATH, "w") as f:
        json.dump({k: list(v) for k, v in edge_zoom.items()}, f, indent=2)
    print(f"\nCoordinates saved to: {SAVED_COORDS_PATH}")
    return edge_zoom


def load_saved_coords():
    """Load previously saved coordinates."""
    if not os.path.exists(SAVED_COORDS_PATH):
        print(f"[ERROR] No saved coords at {SAVED_COORDS_PATH}. Run with SKIP_ROI_SELECTION = False first.")
        sys.exit(1)
    with open(SAVED_COORDS_PATH) as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: INFERENCE & FIGURE (switches to Agg backend)
# ══════════════════════════════════════════════════════════════════════════════

# ---- Patch-based inference ----
def _build_blend_window(size, mode="gaussian"):
    if mode == "gaussian":
        sigma = size * 0.25
        ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
        k1d = np.exp(-0.5 * (ax / sigma) ** 2)
        w = np.outer(k1d, k1d)
    else:
        h1d = 0.5*(1.0-np.cos(2*np.pi*np.arange(size, dtype=np.float64)/(size-1)))
        w = np.outer(h1d, h1d)
    w = w / w.max()
    return np.maximum(w, 1e-6).astype(np.float32)

def denoise_image(noisy_np, infer_fn, batch_size):
    h, w = noisy_np.shape
    n_h = math.ceil(max(h-PATCH_SIZE,0)/STRIDE)+1 if h > PATCH_SIZE else 1
    n_w = math.ceil(max(w-PATCH_SIZE,0)/STRIDE)+1 if w > PATCH_SIZE else 1
    pad_h = max((n_h-1)*STRIDE+PATCH_SIZE-h, 0)
    pad_w = max((n_w-1)*STRIDE+PATCH_SIZE-w, 0)
    img_pad = np.pad(noisy_np, ((0,pad_h),(0,pad_w)), mode="reflect")
    canvas = np.zeros_like(img_pad, dtype=np.float32)
    wmap   = np.zeros_like(img_pad, dtype=np.float32)
    bw     = _build_blend_window(PATCH_SIZE, BLEND_MODE)
    patchlist, coords = [], []
    for i in range(n_h):
        for j in range(n_w):
            y, x = i*STRIDE, j*STRIDE
            patchlist.append(img_pad[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            coords.append((y, x))
    denoised = []
    for i in range(0, len(patchlist), batch_size):
        bt = torch.tensor(np.array(patchlist[i:i+batch_size]),
                          dtype=torch.float32).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            out = infer_fn(bt)
        denoised.extend(out.squeeze(1).cpu().numpy())
    for patch, (y, x) in zip(denoised, coords):
        canvas[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += patch * bw
        wmap[y:y+PATCH_SIZE,   x:x+PATCH_SIZE] += bw
    return (canvas / np.clip(wmap, 1e-8, None))[:h, :w]

# ---- SSIM ----
def compute_ssim(p, t, ws=11):
    def bl(x): return uniform_filter(x.astype(np.float64), size=ws)
    C1, C2 = 0.01**2, 0.03**2
    mp, mt = bl(p), bl(t)
    sp = np.maximum(bl(p**2)-mp**2, 0); st = np.maximum(bl(t**2)-mt**2, 0)
    spt = bl(p*t) - mp*mt
    return float(np.mean(((2*mp*mt+C1)*(2*spt+C2))/((mp**2+mt**2+C1)*(sp+st+C2)+1e-8)))

# ---- Model loaders (verbatim from evaluation scripts) ----
def load_palette_tam_ssim_freq(ckpt_path):
    from palette_TAM_SSIM_freq import PaletteUNet, DiffusionSchedule, DiffusionSampler
    model = PaletteUNet(base_channels=64, embed_dim=128).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt); model.eval()
    schedule = DiffusionSchedule(T=1000).to(DEVICE)
    sampler  = DiffusionSampler(schedule, model, DEVICE, tam_history_size=TAM_HISTORY)
    nl = torch.tensor([[NOISE_LEVEL]], dtype=torch.float32).to(DEVICE)
    def infer(bt):
        bs = bt.shape[0]
        return sampler.ddim_sample(y=bt, noise_level=nl.repeat(bs,1), n_steps=DDIM_STEPS, eta=ETA)
    return infer

def load_palette_noise(ckpt_path):
    from palette_diffusion_denoise_noise import PaletteUNet, DiffusionSchedule, DiffusionSampler
    model = PaletteUNet(base_channels=64, embed_dim=128).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt); model.eval()
    schedule = DiffusionSchedule(T=1000).to(DEVICE)
    sampler  = DiffusionSampler(schedule, model, DEVICE)
    nl = torch.tensor([[NOISE_LEVEL]], dtype=torch.float32).to(DEVICE)
    def infer(bt):
        bs = bt.shape[0]
        return sampler.ddim_sample(y=bt, noise_level=nl.repeat(bs,1), n_steps=DDIM_STEPS, eta=ETA)
    return infer

def load_palette_fullwavelet(ckpt_path):
    from palette_diffusion_denoise_fullwavelet import PaletteUNet, DiffusionSchedule, DiffusionSampler
    model = PaletteUNet(base_channels=64, embed_dim=128).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt); model.eval()
    schedule = DiffusionSchedule(T=1000).to(DEVICE)
    sampler  = DiffusionSampler(schedule, model, DEVICE)
    nl = torch.tensor([[NOISE_LEVEL]], dtype=torch.float32).to(DEVICE)
    def infer(bt):
        bs = bt.shape[0]
        return sampler.ddim_sample(y=bt, noise_level=nl.repeat(bs,1), n_steps=DDIM_STEPS, eta=ETA)
    return infer

def load_unet_mc(ckpt_path):
    import torch.nn as nn
    import torch.nn.functional as F
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            ng = min(8, out_channels)
            while out_channels % ng != 0: ng -= 1
            self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1)
            self.norm1=nn.GroupNorm(ng,out_channels)
            self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1)
            self.norm2=nn.GroupNorm(ng,out_channels)
            self.act=nn.SiLU(inplace=True)
            self.skip=(nn.Conv2d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity())
        def forward(self,x):
            r=self.skip(x); x=self.act(self.norm1(self.conv1(x)))
            x=self.norm2(self.conv2(x)); return self.act(x+r)
    class AttentionBlock(nn.Module):
        def __init__(self,channels,num_heads=4):
            super().__init__()
            self.num_heads=num_heads; self.head_dim=channels//num_heads
            ng=min(8,channels)
            while channels%ng!=0: ng-=1
            self.norm=nn.GroupNorm(ng,channels)
            self.qkv=nn.Conv2d(channels,channels*3,1)
            self.proj=nn.Conv2d(channels,channels,1)
            self.scale=self.head_dim**-0.5
        def forward(self,x):
            B,C,H,W=x.shape; h=self.norm(x)
            q,k,v=self.qkv(h).chunk(3,dim=1)
            q=q.view(B,self.num_heads,self.head_dim,H*W)
            k=k.view(B,self.num_heads,self.head_dim,H*W)
            v=v.view(B,self.num_heads,self.head_dim,H*W)
            attn=F.softmax(torch.einsum('bhdn,bhdm->bhnm',q,k)*self.scale,dim=-1)
            out=torch.einsum('bhnm,bhdm->bhdn',attn,v).reshape(B,C,H,W)
            return x+self.proj(out)
    class MCDropoutBlock(nn.Module):
        def __init__(self,p=0.1):
            super().__init__(); self.p=p; self.dropout=nn.Dropout2d(p=p)
        def forward(self,x): self.dropout.train(); return self.dropout(x)
    class UNetMCDropout(nn.Module):
        def __init__(self,in_channels=1,out_channels=1,base_channels=64,noise_embed_dim=128,dropout_p=0.1):
            super().__init__()
            ch=base_channels; self.dropout_p=dropout_p
            self.noise_embed=nn.Sequential(nn.Linear(1,noise_embed_dim),nn.SiLU(),nn.Linear(noise_embed_dim,noise_embed_dim),nn.SiLU())
            self.enc1=ConvBlock(in_channels,ch); self.enc2=ConvBlock(ch,ch*2)
            self.enc3=ConvBlock(ch*2,ch*4); self.enc4=ConvBlock(ch*4,ch*8)
            self.pool=nn.MaxPool2d(2)
            self.bottleneck=nn.Sequential(ConvBlock(ch*8,ch*8),AttentionBlock(ch*8),ConvBlock(ch*8,ch*8))
            self.noise_proj_b=nn.Linear(noise_embed_dim,ch*8)
            self.noise_proj_4=nn.Linear(noise_embed_dim,ch*4)
            self.noise_proj_3=nn.Linear(noise_embed_dim,ch*2)
            self.noise_proj_2=nn.Linear(noise_embed_dim,ch)
            self.noise_proj_1=nn.Linear(noise_embed_dim,ch)
            self.up4=nn.ConvTranspose2d(ch*8,ch*4,2,stride=2)
            self.dec4=ConvBlock(ch*4+ch*8,ch*4); self.drop4=MCDropoutBlock(p=dropout_p)
            self.up3=nn.ConvTranspose2d(ch*4,ch*2,2,stride=2)
            self.dec3=ConvBlock(ch*2+ch*4,ch*2); self.drop3=MCDropoutBlock(p=dropout_p)
            self.up2=nn.ConvTranspose2d(ch*2,ch,2,stride=2)
            self.dec2=ConvBlock(ch+ch*2,ch); self.drop2=MCDropoutBlock(p=dropout_p)
            self.up1=nn.ConvTranspose2d(ch,ch,2,stride=2)
            self.dec1=ConvBlock(ch+ch,ch); self.drop1=MCDropoutBlock(p=dropout_p)
            self.final_mean=nn.Conv2d(ch,out_channels,1)
            self.final_logvar=nn.Sequential(nn.Conv2d(ch,ch//2,3,padding=1),nn.SiLU(),nn.Conv2d(ch//2,out_channels,1))
            nn.init.constant_(self.final_logvar[-1].bias,-6.0)
        def forward(self,x,noise_level):
            n_emb=self.noise_embed(noise_level)
            e1=self.enc1(x); e2=self.enc2(self.pool(e1))
            e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
            b=self.bottleneck(self.pool(e4)); b=b+self.noise_proj_b(n_emb)[:,:,None,None]
            d4=self.drop4(self.dec4(torch.cat([self.up4(b),e4],1))); d4=d4+self.noise_proj_4(n_emb)[:,:,None,None]
            d3=self.drop3(self.dec3(torch.cat([self.up3(d4),e3],1))); d3=d3+self.noise_proj_3(n_emb)[:,:,None,None]
            d2=self.drop2(self.dec2(torch.cat([self.up2(d3),e2],1))); d2=d2+self.noise_proj_2(n_emb)[:,:,None,None]
            d1=self.drop1(self.dec1(torch.cat([self.up1(d2),e1],1))); d1=d1+self.noise_proj_1(n_emb)[:,:,None,None]
            mean=torch.clamp(x+self.final_mean(d1),0,1)
            log_var=torch.clamp(self.final_logvar(d1),min=-10.0,max=4.0)
            return mean,log_var
    ckpt=torch.load(ckpt_path,map_location=DEVICE,weights_only=False)
    model=UNetMCDropout(**ckpt["model_config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    nl=torch.tensor([[NOISE_LEVEL]],dtype=torch.float32).to(DEVICE)
    def infer(bt):
        bs=bt.shape[0]; acc=torch.zeros_like(bt)
        for _ in range(N_MC_SAMPLES):
            mean_pred,_=model(bt,nl.repeat(bs,1)); acc+=mean_pred
        return acc/N_MC_SAMPLES
    return infer

def load_gan_baseline(ckpt_path):
    import torch.nn as nn
    class GANConvBlock(nn.Module):
        def __init__(self,in_ch,out_ch):
            super().__init__()
            ng=min(8,out_ch)
            while out_ch%ng!=0: ng-=1
            self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1); self.norm1=nn.GroupNorm(ng,out_ch)
            self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1); self.norm2=nn.GroupNorm(ng,out_ch)
            self.act=nn.SiLU(inplace=True)
            self.skip=nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        def forward(self,x):
            r=self.skip(x); x=self.act(self.norm1(self.conv1(x)))
            x=self.norm2(self.conv2(x)); return self.act(x+r)
    class GANGenerator(nn.Module):
        def __init__(self,base_channels=64):
            super().__init__()
            ch=base_channels
            self.enc1=GANConvBlock(1,ch); self.enc2=GANConvBlock(ch,ch*2)
            self.enc3=GANConvBlock(ch*2,ch*4); self.enc4=GANConvBlock(ch*4,ch*8)
            self.pool=nn.MaxPool2d(2)
            self.bottleneck=nn.Sequential(GANConvBlock(ch*8,ch*8),GANConvBlock(ch*8,ch*8))
            self.up4=nn.ConvTranspose2d(ch*8,ch*4,2,stride=2); self.dec4=GANConvBlock(ch*4+ch*8,ch*4)
            self.up3=nn.ConvTranspose2d(ch*4,ch*2,2,stride=2); self.dec3=GANConvBlock(ch*2+ch*4,ch*2)
            self.up2=nn.ConvTranspose2d(ch*2,ch,2,stride=2); self.dec2=GANConvBlock(ch+ch*2,ch)
            self.up1=nn.ConvTranspose2d(ch,ch,2,stride=2); self.dec1=GANConvBlock(ch+ch,ch)
            self.final=nn.Sequential(nn.Conv2d(ch,1,1),nn.Sigmoid())
        def forward(self,x):
            e1=self.enc1(x); e2=self.enc2(self.pool(e1))
            e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
            b=self.bottleneck(self.pool(e4))
            d4=self.dec4(torch.cat([self.up4(b),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1))
            d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1))
            return self.final(d1)
    ckpt=torch.load(ckpt_path,map_location=DEVICE,weights_only=False)
    model=GANGenerator(base_channels=64).to(DEVICE)
    model.load_state_dict(ckpt["G"]); model.eval()
    def infer(bt): return model(bt)
    return infer

MODEL_LOADERS = {
    "palette_tam_ssim_freq": load_palette_tam_ssim_freq,
    "palette_noise":         load_palette_noise,
    "palette_fullwavelet":   load_palette_fullwavelet,
    "unet_mc":               load_unet_mc,
    "gan_baseline":          load_gan_baseline,
}

# ---- Figure generation ----
def generate_figure(all_patients, edge_zoom):
    """Run inference and generate the comparison figure."""
    # Switch to non-interactive backend for figure saving
    matplotlib.use("Agg")
    import importlib
    importlib.reload(plt)

    matplotlib.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    n_patients = len(PATIENT_NAMES)
    n_cols = 2 + len(MODELS_TO_SHOW)
    col_w = 1.3

    # Load models
    print("\nLoading models...")
    infer_fns = {}
    for tag in MODELS_TO_SHOW:
        ckpt = CHECKPOINTS[tag]
        if not os.path.exists(ckpt):
            print(f"  [SKIP] {tag}: not found"); continue
        print(f"  Loading {tag}...")
        infer_fns[tag] = MODEL_LOADERS[tag](ckpt)
    print(f"  Loaded {len(infer_fns)} models.\n")

    # Collect data
    all_data = []
    for pname in PATIENT_NAMES:
        pat = all_patients[pname]
        print(f"Processing: {pname} ({len(pat['noisy_paths'])} frames)")
        gt    = load_image_as_float(pat["gt_path"])
        noisy = load_image_as_float(pat["noisy_paths"][FRAME_IDX])

        roi = detect_signal_roi(gt)
        y0, y1, x0, x1 = roi

        # Edge zoom from selection
        if pname in edge_zoom:
            edge_cy, edge_cx = edge_zoom[pname]
        else:
            # Auto-detect fallback
            region = gt[y0:y1, x0:x1]
            sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
            gx = convolve2d(region, sx, mode="same", boundary="symm")
            gy = convolve2d(region, sx.T, mode="same", boundary="symm")
            grad_smooth = uniform_filter(np.sqrt(gx**2+gy**2), size=EDGE_ZOOM_SIZE//2)
            half = EDGE_ZOOM_SIZE//2
            grad_smooth[:half,:]=0; grad_smooth[-half:,:]=0
            grad_smooth[:,:half]=0; grad_smooth[:,-half:]=0
            if grad_smooth.max()<1e-6: cy,cx=region.shape[0]//2,region.shape[1]//2
            else: cy,cx=np.unravel_index(np.argmax(grad_smooth),grad_smooth.shape)
            edge_cy, edge_cx = y0+cy, x0+cx

        half = EDGE_ZOOM_SIZE // 2
        ey0, ey1 = max(0, edge_cy-half), min(gt.shape[0], edge_cy+half)
        ex0, ex1 = max(0, edge_cx-half), min(gt.shape[1], edge_cx+half)
        # Force square
        side = min(ey1-ey0, ex1-ex0)
        ey1 = ey0 + side; ex1 = ex0 + side
        print(f"  ROI: y=[{y0}:{y1}], x=[{x0}:{x1}]  Edge: ({edge_cy},{edge_cx}) {side}x{side}")

        # Denoise
        model_outputs = {}
        for tag in MODELS_TO_SHOW:
            if tag in infer_fns:
                print(f"    Denoising with {tag}...")
                model_outputs[tag] = denoise_image(noisy, infer_fns[tag], BATCH_SIZE)

        ssim_noisy  = compute_ssim(noisy, gt)
        ssim_models = {tag: compute_ssim(d, gt) for tag, d in model_outputs.items()}

        all_data.append({
            "name": pname, "gt": gt, "noisy": noisy,
            "model_outputs": model_outputs,
            "roi": roi, "edge_box": (ey0, ey1, ex0, ex1),
            "ssim_noisy": ssim_noisy, "ssim_models": ssim_models,
        })

    # Build figure
    print("\nGenerating figure...")
    total_w = n_cols * col_w
    # Use uniform row height (tallest ROI) so all rows are the same size
    max_rh = 0
    for data in all_data:
        y0, y1, x0, x1 = data["roi"]
        rh = col_w * 0.97 * ((y1-y0) / max(x1-x0, 1))
        max_rh = max(max_rh, rh)
    row_heights = [max_rh] * n_patients
    gap = 0.05
    total_h = sum(row_heights) + (n_patients-1) * gap
    fig = plt.figure(figsize=(total_w, total_h))

    y_cursor = 0.0
    for pi, data in enumerate(all_data):
        gt = data["gt"]; noisy = data["noisy"]
        y0,y1,x0,x1 = data["roi"]
        ey0,ey1,ex0,ex1 = data["edge_box"]
        vmin = 0.0; vmax = np.percentile(gt[y0:y1, x0:x1], 99.5)

        images = [gt, noisy] + [data["model_outputs"].get(t, np.zeros_like(gt)) for t in MODELS_TO_SHOW]
        ssim_vals = [None, data["ssim_noisy"]] + [data["ssim_models"].get(t) for t in MODELS_TO_SHOW]
        rh = row_heights[pi]

        for ci, (img, ssim_val) in enumerate(zip(images, ssim_vals)):
            ax = fig.add_axes([
                ci*col_w/total_w,
                1.0-(y_cursor+rh)/total_h,
                col_w/total_w*0.97,
                rh/total_h,
            ])
            ax.imshow(img[y0:y1,x0:x1], cmap="inferno", vmin=vmin, vmax=vmax,
                      aspect="auto", interpolation="bilinear")

            if ey0>=y0 and ex0>=x0:
                rect = mpatches.Rectangle((ex0-x0,ey0-y0), ex1-ex0, ey1-ey0,
                    lw=1.0, edgecolor=EDGE_BOX_COLOR, facecolor="none")
                ax.add_patch(rect)

            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

            if pi == 0:
                ax.set_title(COLUMN_LABELS[ci], fontsize=8, fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(f"Patient {pi+1}", fontsize=8, fontweight="bold",
                              rotation=90, labelpad=8)
            if ssim_val is not None:
                ax.text(0.97, 0.05, f"SSIM: {ssim_val:.3f}",
                       transform=ax.transAxes, ha="right", va="bottom",
                       fontsize=7, color="white", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                                 alpha=0.4, edgecolor="none"))

            ax_inset = ax.inset_axes([0.02, 0.02, INSET_FRAC, INSET_FRAC])
            ax_inset.imshow(img[ey0:ey1,ex0:ex1], cmap="inferno", vmin=vmin, vmax=vmax,
                            aspect="auto", interpolation="bilinear")
            ax_inset.set_xticks([]); ax_inset.set_yticks([])
            for sp in ax_inset.spines.values():
                sp.set_edgecolor(EDGE_BOX_COLOR); sp.set_linewidth(1.5)

        y_cursor += rh + gap

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = os.path.join(OUTPUT_DIR, f"patient_visual_comparison.{ext}")
        fig.savefig(p, dpi=300, facecolor="white")
        print(f"  Saved: {p}")
    plt.close(fig)
    print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    all_patients = discover_patient_data(PATIENT_ROOT)
    print(f"Found {len(all_patients)} patient folders.")

    # Validate patient names
    for pname in PATIENT_NAMES:
        if pname not in all_patients:
            print(f"[ERROR] Patient '{pname}' not found in {PATIENT_ROOT}")
            return

    # Step 1: ROI selection
    if SKIP_ROI_SELECTION:
        print("\nSkipping ROI selection, loading saved coordinates...")
        edge_zoom = load_saved_coords()
        print(f"  Loaded: {edge_zoom}")
    else:
        print("\n" + "="*60)
        print("  STEP 1: Interactive ROI Selection")
        print("  Click to place zoom box, close window to confirm")
        print("="*60)
        edge_zoom = run_roi_selection(all_patients)

    # Step 2: Inference + figure
    print("\n" + "="*60)
    print("  STEP 2: Model Inference & Figure Generation")
    print("="*60)
    generate_figure(all_patients, edge_zoom)


if __name__ == "__main__":
    main()