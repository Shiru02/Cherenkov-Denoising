"""
Palette-Style Conditional Diffusion for Cherenkov Image Denoising
=================================================================
Refactored Baseline Version:
- Clean L2 ε-MSE loss (removed conflicting spatial/freq auxiliary losses).
- Fixed DDIM clamping bounds [0, 1].
- Deterministic DDIM evaluation (η=0.0) with stochastic uncertainty mapping (η=0.5).
- Integrated logging, sampling, training curve generation, and resume capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pytorch_wavelets import DWTForward, DWTInverse

# --- MATPLOTLIB THREADING FIX ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import logging
import os
import sys
import glob
import math
import random
import cv2
from PIL import Image



# ==============================================================================
# ========================== NOISE SCHEDULE ====================================
# ==============================================================================

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps      = torch.arange(T + 1, dtype=torch.float64)
    alphas_bar = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas      = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, min=1e-6, max=0.999).float()

class DiffusionSchedule(nn.Module):
    """Precomputes all schedule-derived quantities."""
    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = T

        betas           = cosine_beta_schedule(T)
        alphas          = 1.0 - betas
        alphas_bar      = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.register_buffer('betas',         betas)
        self.register_buffer('alphas',        alphas)
        self.register_buffer('alphas_bar',    alphas_bar)
        self.register_buffer('sqrt_ab',       torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_m_ab', torch.sqrt(1.0 - alphas_bar))
        self.register_buffer('posterior_var',
            betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar).clamp(min=1e-8))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> tuple:
        """Closed-form forward process."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab    = self.sqrt_ab[t][:, None, None, None]
        sqrt_1m_ab = self.sqrt_one_m_ab[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                             eps: torch.Tensor) -> torch.Tensor:
        """Estimate clean image from current noisy state and predicted noise."""
        sqrt_ab    = self.sqrt_ab[t][:, None, None, None]
        sqrt_1m_ab = self.sqrt_one_m_ab[t][:, None, None, None]
        return (x_t - sqrt_1m_ab * eps) / sqrt_ab.clamp(min=1e-8)


# ==============================================================================
# ========================== LOSS FUNCTION =====================================
# ==============================================================================

class PaletteLoss(nn.Module):
    """
    x0-Parameterization Loss.
    Directly predicts the clean image, enabling stable spatial regularizers.
    """
    def __init__(self, l1_weight=1.0, l2_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, x0_pred, x0_target) -> tuple:
        l1 = F.l1_loss(x0_pred, x0_target)
        l2 = F.mse_loss(x0_pred, x0_target)
        
        # SSIM could easily be added here later if desired!
        total_loss = (self.l1_weight * l1) + (self.l2_weight * l2)

        return total_loss, {
            'total': total_loss.item(),
            'l1': l1.item(),
            'l2': l2.item(),
        }


# ==============================================================================
# ========================== MODEL =============================================
# ==============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim), nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None, :]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)





class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
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

class transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

class Itransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        ng = min(8, out_ch)
        while out_ch % ng != 0:
            ng -= 1
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(ng, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(ng, out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)

class Wavelet_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = transformer()
        # Use wavelet forward transform; device will be handled by the parent .to(device).
        self.dwt = DWTForward(J=1, wave='haar')

    def forward(self, x):
        # pytorch_wavelets backward is not fully AMP-safe on some setups.
        # Run DWT path in FP32 with autocast disabled to avoid Half/Float mismatch.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            DMT_yl, DMT_yh = self.dwt(x32)
            DMT = self.transformer(DMT_yl, DMT_yh)
        return DMT

class Wavelet_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Itransformer()
        # Inverse wavelet transform to reconstruct an image-like tensor.
        self.idwt = DWTInverse(wave='haar')

    def forward(self, x):
        # Keep inverse DWT in FP32 for the same AMP-compatibility reason as encoder.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            yl, yh = self.transformer(x32)
            # DWTInverse expects a (yl, yh_list) tuple.
            out = self.idwt((yl, yh))
        return out

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

class PaletteUNet(nn.Module):
    def __init__(self, base_channels: int = 64, embed_dim: int = 128):
        super().__init__()
        ch = base_channels
        ch_in_d1 = ch * 2  # d1 is formed by concatenating d2 (ch) and p1 (ch)

        self.t_embed  = SinusoidalTimestepEmbedding(embed_dim)
        self.nl_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(),
        )

        # ---- Wavelet transforms ----
        self.Wavelet_Encoder1 = Wavelet_Encoder()
        self.Wavelet_Encoder2 = Wavelet_Encoder()
        self.Wavelet_Encoder3 = Wavelet_Encoder()
        self.Wavelet_Decoder3 = Wavelet_Decoder()
        self.Wavelet_Decoder2 = Wavelet_Decoder()
        self.Wavelet_Decoder1 = Wavelet_Decoder()
        self.Wavelet_Decoder4 = Wavelet_Decoder()

        self.enc1 = ConvBlock(2,    ch)
        self.enc2 = ConvBlock(2*4,   ch*2)
        self.enc3 = ConvBlock(2*4*4, ch*2)
        # Encoder4 is the deepest feature stage; it must produce enough channels so the
        # subsequent wavelet inverse (decoder4) reduces to `ch*2`.
        self.enc4 = ConvBlock(2*4*16, ch*8)

        # Historically this model had pooling commented out; keep it as an identity so
        # spatial sizes stay aligned across wavelet encode/decode skip connections.
        self.pool = nn.Identity()

        # Lightweight merge blocks (no self-attention) to keep channel geometry valid.
        self.merge1 = ConvBlock(ch*2, ch)
        # concat(d3, p2): (ch) + (ch*2) = 3*ch -> project to 4*ch before wavelet decode.
        self.merge2 = ConvBlock(ch*3, ch*4)
        self.Attention_Block4 = nn.Sequential(
            ConvBlock(ch*8, ch*8),
            AttentionBlock(ch*8),
            ConvBlock(ch*8, ch*8),
        )

        self.cond_proj_b = nn.Linear(embed_dim, ch*8)
        # Must match channels of d4 (decoder4 output), d3 (decoder3 output), etc.
        self.cond_proj_4 = nn.Linear(embed_dim, ch*2)
        self.cond_proj_3 = nn.Linear(embed_dim, ch)
        self.cond_proj_2 = nn.Linear(embed_dim, ch)
        # mean is 1-channel, so project noise condition to 1 channel at the end.
        self.cond_proj_1 = nn.Linear(embed_dim, 1)

        self.up4  = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.dec4 = ConvBlock(ch*4 + ch*8, ch*4)
        self.up3  = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec3 = ConvBlock(ch*2 + ch*4, ch*2)
        self.up2  = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.dec2 = ConvBlock(ch + ch*2, ch)
        self.up1  = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch + ch, ch)
        # Late branches expect the same channel geometry as `d1` (which is ch*2).
        self.final = nn.Conv2d(1, 1, 1)
        self.S2 = nn.Conv2d(ch_in_d1, ch_in_d1, kernel_size=1, stride=1, padding=0)
        self.S1 = nn.Conv2d(ch_in_d1, ch_in_d1, kernel_size=1, stride=1, padding=0)

        # res branch final conv (predicts noise residual)
        self.D1 = nn.Conv2d(ch_in_d1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        # e2e branch final conv (predicts clean image directly)
        self.D2 = nn.Conv2d(ch_in_d1, 1, kernel_size=3, stride=1, padding=1, dilation=1)

        self.blockDMT11 = self.make_layer(_DCR_block, ch_in_d1)
        self.blockDMT12 = self.make_layer(_DCR_block, ch_in_d1)
        self.blockDMT13 = self.make_layer(_DCR_block, ch_in_d1)
        self.blockDMT14 = self.make_layer(_DCR_block, ch_in_d1)

    def make_layer(self, block, channel_in):
        return nn.Sequential(block(channel_in))

    def forward(self, x_t, y, t, noise_level):
        # Accept scalar / [B] / [B,1,1,...] and normalize to [B, 1] for Linear(1, embed_dim).
        noise_level = noise_level.float()
        if noise_level.ndim == 0:
            noise_level = noise_level.unsqueeze(0)
        if noise_level.ndim == 1:
            noise_level = noise_level.unsqueeze(-1)
        elif noise_level.ndim > 2:
            noise_level = noise_level.reshape(noise_level.shape[0], -1)
        if noise_level.shape[-1] != 1:
            raise ValueError(f"Expected noise_level with last dim 1, got shape {tuple(noise_level.shape)}")

        cond = self.t_embed(t) + self.nl_embed(noise_level) 
        inp  = torch.cat([x_t, y], dim=1)                    

        e1 = self.Wavelet_Encoder1(inp)
        e2 = self.Wavelet_Encoder2(self.pool(e1))
        e3 = self.Wavelet_Encoder3(self.pool(e2))

        p1 = self.enc1(inp)
        p2 = self.enc2(e1)
        p3 = self.enc3(e2)
        p4 = self.enc4(e3)

        d4 = self.Attention_Block4(p4)
        d4 = self.Wavelet_Decoder4(p4)
        d4 = d4 + self.cond_proj_4(cond)[:, :, None, None]
        d3 = torch.concatenate([d4, p3], dim=1)
        d3 = self.Wavelet_Decoder3(d3)
        d3 = d3 + self.cond_proj_3(cond)[:, :, None, None]
        d2 = torch.concatenate([d3, p2], dim=1)
        d2 = self.merge2(d2)
        d2 = self.Wavelet_Decoder2(d2)
        d2 = d2 + self.cond_proj_2(cond)[:, :, None, None]
        d1 = torch.concatenate([d2, p1], dim=1)

        # ---- res branch ----
        res_part1 = self.S1(d1)
        res_part2 = self.blockDMT11(res_part1)
        res_part3 = res_part2 + res_part1
        res_part4 = self.blockDMT12(res_part3)
        res_part5 = res_part4 + res_part3
        res_part6 = self.D1(res_part5)
        res_part  = x_t - res_part6

        # ---- e2e branch ----
        e2e_part1 = self.S2(d1)
        e2e_part2 = self.blockDMT13(e2e_part1)
        e2e_part3 = e2e_part2 + e2e_part1
        e2e_part4 = self.blockDMT14(e2e_part3)
        e2e_part5 = e2e_part4 + e2e_part3
        e2e_part  = torch.tanh(self.D2(e2e_part5))

        # ---- mean output ----
        mean = (e2e_part + res_part) / 2

        d1 = mean + self.cond_proj_1(cond)[:, :, None, None]

        return self.final(d1)


# ==============================================================================
# ========================== DATASET ===========================================
# ==============================================================================

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


class PatchDataset(Dataset):
    def __init__(self, in_root, gt_root):
        self.in_root = in_root
        self.gt_root = gt_root
        # randomly ignore 4 images from each directory to speed up training
        ignore = 4

        imgs = []
        target = []

        # Get sorted lists to ensure correspondence (assuming paired data)
        in_dirs = sorted([os.path.join(self.in_root, fname) for fname in os.listdir(self.in_root) if os.path.isdir(os.path.join(self.in_root, fname))])
        in_files = []
        for i, in_dir in enumerate(in_dirs):
            files = [os.path.join(in_dir, fname) for fname in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, fname))]
            files = sorted(files)
            if len(files) > ignore:
                files_to_keep = files.copy()
                files_to_ignore = random.sample(files_to_keep, ignore)
                files = [f for f in files_to_keep if f not in files_to_ignore]
            in_files.extend(files)
        gt_files = sorted([os.path.join(self.gt_root, fname) for fname in os.listdir(self.gt_root) if os.path.isfile(os.path.join(self.gt_root, fname))])
        print(len(in_files), len(gt_files))

        # Ensure same number of input and gt images
        # assert len(in_files) == len(gt_files), "Mismatch between in_root and gt_root file counts"

        imgs.extend(in_files)
        target.extend(gt_files)

        self.imgs = imgs
        self.target = target
        self.ignore = ignore

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean, angle_aug=False):
        # random rotate
        if angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  
        Ymin = np.min(arr)  
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last
    def img_sharpen(self, img):

        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0

        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        clean_path = self.target[math.floor(index/(12-self.ignore))]
        # Extract the number between the two underscores in the filename
        import re
        base_name = os.path.basename(noise_path)
        match = re.search(r'_(\d+)_', base_name)
        if match:
            noise_index = int(match.group(1))
        else:
            noise_index = 0  # or handle error if not found
        
        index_transfer = {
            1: 1, 2: 2, 3: 3, 5: 4, 8: 5, 10: 6, 20: 7, 40: 8, 50: 9, 80: 10, 120: 11, 200: 12
        }
        noise_index = index_transfer.get(noise_index, 0)

        noise_level = (noise_index / 12)
        noise_level = torch.tensor(noise_level, dtype=torch.float32)
        
        noise_r = Image.open(noise_path)
        noise_r_np = np.array(noise_r).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = noise_r_np.min(), noise_r_np.max()
        noise_r = ((noise_r_np - min_val) / (max_val - min_val))
        noise_r = np.array(noise_r).astype(np.float32)

        h, w = noise_r.shape[:2]

        rand_h, rand_w = self.random_crop_start(h, w, 136, 4)
        noise_r = noise_r[rand_h:rand_h+136, rand_w:rand_w+136]
         
        clean = Image.open(clean_path)
        clean_r_np = np.array(clean).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = clean_r_np.min(), clean_r_np.max()
        clean_r = ((clean_r_np - min_val) / (max_val - min_val))
        clean = np.array(clean_r).astype(np.float32)
        clean = clean[rand_h:rand_h+136, rand_w:rand_w+136]

        noise_r = torch.from_numpy(noise_r.astype(np.float32)).contiguous().unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).contiguous().unsqueeze(0)

        return {'noisy': noise_r, 'clean': clean,
                'noise_level': noise_level, 'level_idx': noise_index}
    
    def __len__(self):
        return len(self.imgs)

# class PatchDataset(Dataset):
#     def __init__(self, data_path: str, input_levels: list = None):
#         self.files = glob.glob(os.path.join(data_path, '*.npy'))
#         if not self.files:
#             raise ValueError(f"No .npy files found in {data_path}")
#         sample = np.load(self.files[0])
#         self.num_levels   = sample.shape[-1]
#         self.input_levels = input_levels or list(range(self.num_levels - 1))
#         print(f"Found {len(self.files)} patches | shape {sample.shape} | "
#               f"input levels {self.input_levels} | target {self.num_levels-1}")

#     def __len__(self):
#         return len(self.files) * len(self.input_levels)

#     def __getitem__(self, idx):
#         file_idx    = idx // len(self.input_levels)
#         level_idx   = idx %  len(self.input_levels)
#         input_level = self.input_levels[level_idx]

#         patch = np.load(self.files[file_idx])
#         if patch.max() > 1.0:
#             patch = patch / patch.max()

#         noisy = torch.tensor(patch[:, :, input_level], dtype=torch.float32).unsqueeze(0)
#         clean = torch.tensor(patch[:, :, -1],          dtype=torch.float32).unsqueeze(0)
#         noisy = torch.clamp(noisy, 0, 1)
#         clean = torch.clamp(clean, 0, 1)

#         noise_level = torch.tensor(
#             [1.0 - input_level / (self.num_levels - 1)], dtype=torch.float32
#         )
#         return {'noisy': noisy, 'clean': clean,
#                 'noise_level': noise_level, 'level_idx': input_level}


# ==============================================================================
# ========================== SAMPLER ===========================================
# ==============================================================================

class DiffusionSampler:
    def __init__(self, schedule: DiffusionSchedule, model: PaletteUNet, device):
        self.schedule = schedule
        self.model    = model
        self.device   = device

    @torch.no_grad()
    def ddpm_sample(self, y, noise_level):
        B   = y.shape[0]
        x_t = torch.randn_like(y)
        T   = self.schedule.T

        for t_val in tqdm(reversed(range(T)), desc="DDPM", total=T, leave=False):
            t_tensor = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            eps_pred = self.model(x_t, y, t_tensor, noise_level)

            alpha     = self.schedule.alphas[t_val]
            alpha_bar = self.schedule.alphas_bar[t_val]
            beta      = self.schedule.betas[t_val]
            post_var  = self.schedule.posterior_var[t_val]

            coef = beta / (1.0 - alpha_bar).sqrt()
            mean = (1.0 / alpha.sqrt()) * (x_t - coef * eps_pred)

            if t_val > 0:
                x_t = mean + post_var.sqrt() * torch.randn_like(x_t)
            else:
                x_t = mean

        return x_t.clamp(0, 1)

    @torch.no_grad()
    def ddim_sample(self, y, noise_level, n_steps=100, eta=0.0):
        # FIXED: eta=0.0 default for deterministic baseline testing
        B  = y.shape[0]
        T  = self.schedule.T
        step_size = max(T // n_steps, 1)
        timesteps = list(reversed(range(0, T, step_size)))

        x_t = torch.randn_like(y)

        for i, t_val in enumerate(tqdm(timesteps, desc=f"DDIM η={eta}", leave=False)):
            t_tensor = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            eps_pred = self.model(x_t, y, t_tensor, noise_level)

            ab_t   = self.schedule.alphas_bar[t_val]
            t_prev = timesteps[i+1] if i + 1 < len(timesteps) else -1
            ab_prev = self.schedule.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            x0_pred = (x_t - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(0, 1) # FIXED: clamping bounds corrected to [0, 1]

            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0).sqrt()
            dir_x = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            x_t   = ab_prev.sqrt() * x0_pred + dir_x
            
            if eta > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return x_t.clamp(0, 1)

    @torch.no_grad()
    def sample_n(self, y, noise_level, n=8, method='ddim', n_ddim_steps=100, eta=0.0):
        """Samples n times and returns the mean and standard deviation (uncertainty)."""
        self.model.eval()
        all_samples = []
        for _ in range(n):
            if method == 'ddim':
                s = self.ddim_sample(y, noise_level, n_steps=n_ddim_steps, eta=eta)
            else:
                s = self.ddpm_sample(y, noise_level)
            all_samples.append(s)
            
        samples = torch.stack(all_samples, dim=0) # (n, B, 1, H, W)
        return {
            'mean': samples.mean(dim=0),
            'std': samples.std(dim=0) if n > 1 else torch.zeros_like(samples[0]),
            'samples': samples,
        }

# ==============================================================================
# ========================== TRAINING CONFIG ===================================
# ==============================================================================

class TrainingConfig:
    def __init__(self):
        self.data_path = "noisy_patches"
        self.save_dir  = "experiments/palette_cherenkov_fullwavelet"
        self.clean_path = "clean_patches"
        self.model_config = {
            'base_channels': 64,
            'embed_dim': 128,
        }
        
        self.T = 1000
        self.sample_method = 'ddim'
        self.n_ddim_steps = 100
        self.ddim_eta = 0.0 # Strict determinism for evaluation
        self.n_uncertainty = 8 

        self.num_epochs = 600
        self.batch_size = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.train_val_split = 0.8
        self.preload = True

        self.warmup_epochs = 10
        self.min_lr_factor = 0.1 
        self.mixed_precision = True
        self.gradient_clip_norm = 1.0

        self.eval_freq = 5
        self.sample_freq = 20 
        self.num_save_samples = 4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(4, os.cpu_count() or 2)
        self.pin_memory = True

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
        self.global_step = 0

    def _setup_directories(self):
        for d in [self.config.checkpoint_dir, self.config.samples_dir,
                  self.config.logs_dir, self.config.tensorboard_dir]:
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
        if not self.loss_csv.exists():
            cols = ['epoch', 'train_loss', 'val_loss', 'val_psnr', 'lr']
            pd.DataFrame(columns=cols).to_csv(self.loss_csv, index=False)

        self.writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))

    def _setup_data(self):
        self.logger.info(f"Loading data from: {self.config.data_path}")
        full = PatchDataset(in_root=self.config.data_path, gt_root=self.config.clean_path)
        total = len(full)
        train_size = int(total * self.config.train_val_split)
        val_size = total - train_size

        self.train_dataset, self.val_dataset = random_split(
            full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        kw = dict(num_workers=self.config.num_workers, pin_memory=self.config.pin_memory)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True, **kw)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False, **kw)

    def _setup_model(self):
        self.schedule = DiffusionSchedule(T=self.config.T).to(self.config.device)
        self.model    = PaletteUNet(**self.config.model_config).to(self.config.device)
        self.sampler  = DiffusionSampler(self.schedule, self.model, self.config.device)
        self.criterion = PaletteLoss() 

    def _setup_training(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * self.config.min_lr_factor
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.mixed_precision)

    def load_checkpoint(self, checkpoint_path):
        """Loads a checkpoint and returns the epoch to start from."""
        if not os.path.exists(checkpoint_path) or self.config.preload == False:
            self.logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return 1

        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_val_loss = ckpt.get('best_loss', float('inf'))
        
        start_epoch = ckpt['epoch'] + 1 
        self.logger.info(f"Successfully loaded checkpoint. Resuming at epoch {start_epoch}.")
        return start_epoch

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")

        for batch in pbar:
            x0 = batch['clean'].to(self.config.device)
            y  = batch['noisy'].to(self.config.device)
            nl = batch['noise_level'].to(self.config.device)
            print(x0.shape, y.shape, nl.shape)
            B  = x0.shape[0]

            t = torch.randint(0, self.config.T, (B,), device=self.config.device).long()
            x_t, eps = self.schedule.q_sample(x0, t)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                eps_pred = self.model(x_t, y, t, nl)
                x0_pred = self.schedule.predict_x0_from_eps(x_t, t, eps_pred).clamp(0, 1)
                loss, metrics = self.criterion(x0_pred, x0)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_losses.append(metrics['total'])
            self.writer.add_scalar('Train/Loss', metrics['total'], self.global_step)
            self.global_step += 1
            
            pbar.set_postfix({'loss': f"{metrics['total']:.4f}"})

        return np.mean(epoch_losses)

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        val_losses = []
        psnrs = []

        for batch in self.val_loader:
            x0 = batch['clean'].to(self.config.device)
            y  = batch['noisy'].to(self.config.device)
            nl = batch['noise_level'].to(self.config.device)
            B  = x0.shape[0]

            t = torch.randint(0, self.config.T, (B,), device=self.config.device).long()
            x_t, eps = self.schedule.q_sample(x0, t)

            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                eps_pred = self.model(x_t, y, t, nl)
                x0_pred = self.schedule.predict_x0_from_eps(x_t, t, eps_pred).clamp(0, 1)
                loss, metrics = self.criterion(x0_pred, x0)

            val_losses.append(metrics['total'])
            
            x0_pred = self.schedule.predict_x0_from_eps(x_t, t, eps_pred).clamp(0, 1)
            mse_val = F.mse_loss(x0_pred, x0).item()
            psnrs.append(10 * math.log10(1.0 / (mse_val + 1e-8)))

        mean_loss = np.mean(val_losses)
        mean_psnr = np.mean(psnrs)
        
        self.writer.add_scalar('Val/Loss', mean_loss, epoch)
        self.writer.add_scalar('Val/PSNR', mean_psnr, epoch)
        
        return mean_loss, mean_psnr

    def log_epoch(self, epoch, tr_loss, val_loss, val_psnr):
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            f"Epoch {epoch} | Tr Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} | LR: {lr:.2e}"
        )
        df = pd.DataFrame([{
            'epoch': epoch, 'train_loss': tr_loss, 
            'val_loss': val_loss, 'val_psnr': val_psnr, 'lr': lr
        }])
        df.to_csv(self.loss_csv, mode='a', header=False, index=False)

    def plot_curves(self):
        """Reads the training history CSV and plots the loss and PSNR curves."""
        if not self.loss_csv.exists():
            return
        
        df = pd.read_csv(self.loss_csv)
        if len(df) == 0:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color=color, linestyle='-')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:orange', linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('PSNR (dB)', color=color)  
        ax2.plot(df['epoch'], df['val_psnr'], label='Val PSNR', color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title('Training Convergence')
        fig.tight_layout()  
        
        plot_path = self.config.logs_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def save_samples(self, epoch):
        self.model.eval()
        self.logger.info(f"Generating samples and uncertainty maps for epoch {epoch}...")
        
        batch = next(iter(self.val_loader))
        x0 = batch['clean'][:self.config.num_save_samples].to(self.config.device)
        y  = batch['noisy'][:self.config.num_save_samples].to(self.config.device)
        nl = batch['noise_level'][:self.config.num_save_samples].to(self.config.device)

        # Force stochasticity (eta > 0) to generate meaningful uncertainty maps
        sampling_eta = 0.5 
        
        results = self.sampler.sample_n(
            y, nl, 
            n=self.config.n_uncertainty, 
            method=self.config.sample_method, 
            n_ddim_steps=self.config.n_ddim_steps, 
            eta=sampling_eta
        )

        mean_pred = results['mean']
        std_pred  = results['std']
        
        fig, axes = plt.subplots(self.config.num_save_samples, 4, figsize=(16, 4 * self.config.num_save_samples))
        if self.config.num_save_samples == 1:
            axes = [axes]
            
        for i in range(self.config.num_save_samples):
            # Convert tensors to numpy arrays
            cond_img   = y[i, 0].cpu().numpy()
            target_img = x0[i, 0].cpu().numpy()
            mean_img   = mean_pred[i, 0].cpu().numpy()
            uncert_img = std_pred[i, 0].cpu().numpy()
            
            axes[i][0].imshow(cond_img, cmap='gray', vmin=0, vmax=1)
            axes[i][0].set_title("Condition (Noisy)")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(target_img, cmap='gray', vmin=0, vmax=1)
            axes[i][1].set_title("Ground Truth (Clean)")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(mean_img, cmap='gray', vmin=0, vmax=1)
            axes[i][2].set_title(f"Mean Prediction (n={self.config.n_uncertainty})")
            axes[i][2].axis('off')
            
            # Use 'hot' colormap for uncertainty map visibility
            im = axes[i][3].imshow(uncert_img, cmap='hot')
            axes[i][3].set_title(f"Uncertainty (std map, η={sampling_eta})")
            axes[i][3].axis('off')
            fig.colorbar(im, ax=axes[i][3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = self.config.samples_dir / f"samples_epoch_{epoch:04d}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        self.logger.info(f"Saved samples to {save_path}")

    def run(self, start_epoch=1):
        for epoch in range(start_epoch, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)

            # --- ALWAYS SAVE LATEST AT THE END OF EVERY EPOCH ---
            ckpt = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_val_loss,
            }
            torch.save(ckpt, self.config.checkpoint_dir / 'latest_model.pth')

            # --- EVALUATION & LOGGING ---
            if epoch % self.config.eval_freq == 0:
                val_loss, val_psnr = self.evaluate(epoch)
                self.log_epoch(epoch, train_loss, val_loss, val_psnr)
                self.plot_curves() # Update the training curves graph

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_psnr = val_psnr
                    ckpt['best_loss'] = self.best_val_loss
                    torch.save(ckpt, self.config.checkpoint_dir / 'best_model.pth')
                    self.logger.info(f"Saved best model at epoch {epoch}")

            # --- SAVING SAMPLES ---
            if epoch % self.config.sample_freq == 0:
                self.save_samples(epoch)

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
    # UPDATE THESE PATHS TO YOUR DATA
    # ============================================================
    config.data_path = r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Data\Simulated_singleframe2"
    config.clean_path = r"C:\Users\DoseOptics\Desktop\Denoising_Project\Denoising_Models\Dense_Wavelet\Data\Simulated_cumulative"
    config.save_dir  = "experiments/palette_cherenkov_fullwavelet"
    # ============================================================

    # Rebuild auto paths
    config.checkpoint_dir  = Path(config.save_dir) / 'checkpoints'
    config.samples_dir     = Path(config.save_dir) / 'samples'
    config.logs_dir        = Path(config.save_dir) / 'logs'
    config.tensorboard_dir = Path(config.save_dir) / 'tensorboard'

    if not os.path.exists(config.data_path):
        print(f"ERROR: data path not found: {config.data_path}")
        print("Update config.data_path to your .npy patch directory.")
        sys.exit(1)

    trainer = Trainer(config)

    # --- NEW RESUME LOGIC ---
    # Try to load latest_model.pth first, fallback to best_model.pth
    latest_path = config.checkpoint_dir / 'latest_model.pth'
    best_path = config.checkpoint_dir / 'best_model.pth'
    
    if latest_path.exists():
        start_epoch = trainer.load_checkpoint(latest_path)
    elif best_path.exists():
        start_epoch = trainer.load_checkpoint(best_path)
    else:
        start_epoch = 1
        
    trainer.run(start_epoch=start_epoch)

if __name__ == '__main__':
    main()