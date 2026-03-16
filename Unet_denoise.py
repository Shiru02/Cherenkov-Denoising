"""
U-Net Denoising for Cherenkov Images - Integrated Training Script

Features:
- Direct supervised denoising: low SNR → high SNR (last level as ground truth)
- Loss: L1 + L2 (MSE) + SSIM
- Data loading follows original PatchDataset format exactly

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
from datetime import datetime
from tqdm import tqdm
import logging
import os
import sys
import glob
import random
import math
from copy import deepcopy


# ==============================================================================
# ========================== LOSS FUNCTIONS ====================================
# ==============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    Loss = 1 - SSIM (minimizing loss maximizes SSIM)
    """
    def __init__(self, window_size=11, sigma=1.5, channel=1):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, sigma, channel))
        
        # Constants for numerical stability
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def _create_window(self, window_size, sigma, channel):
        """Create a Gaussian window for SSIM computation."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(self, pred, target):
        """Compute SSIM loss."""
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        channel = pred.size(1)
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel).to(pred.device)
            self.channel = channel
        
        mu_pred = F.conv2d(pred, self.window, padding=self.window_size//2, groups=channel)
        mu_target = F.conv2d(target, self.window, padding=self.window_size//2, groups=channel)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred ** 2, self.window, padding=self.window_size//2, groups=channel) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, self.window, padding=self.window_size//2, groups=channel) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=channel) - mu_pred_target
        
        sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0)
        sigma_target_sq = torch.clamp(sigma_target_sq, min=0)
        
        numerator = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)
        denominator = (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)
        
        ssim_map = numerator / (denominator + 1e-8)
        
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: L1 + L2 (MSE) + SSIM
    """
    def __init__(self, l1_weight=0.3, l2_weight=0.3, ssim_weight=0.4, ssim_window_size=11):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size)
    
    def forward(self, pred, target):
        """
        Compute combined loss.
        
        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Dictionary of individual loss values
        """
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.l2_weight * l2 + self.ssim_weight * ssim
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1.item(),
            'l2': l2.item(),
            'ssim': ssim.item()
        }
        
        return total_loss, loss_dict


# ==============================================================================
# ========================== DATASET ===========================================
# ==============================================================================

class PatchDataset(Dataset):
    """
    Dataset for loading Cherenkov patches - follows original format exactly.
    
    Loads .npy files with shape (H, W, num_cumulative_levels).
    For training: randomly selects input level, uses last level as ground truth.
    
    Args:
        data_path: Path to directory containing .npy patch files
        input_levels: List of level indices to use as inputs (default: all except last)
    """
    def __init__(self, data_path, input_levels=None):
        self.data_path = data_path
        self.files = glob.glob(os.path.join(data_path, '*.npy'))
        
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {data_path}")
        
        # Load one sample to check shape
        sample = np.load(self.files[0])
        self.num_levels = sample.shape[-1]
        self.patch_shape = sample.shape[:2]
        
        # Set input levels (all except last by default)
        if input_levels is None:
            self.input_levels = list(range(self.num_levels - 1))
        else:
            self.input_levels = input_levels
        
        print(f"Found {len(self.files)} patch files")
        print(f"Sample patch shape: {sample.shape}")
        print(f"Number of cumulative levels: {self.num_levels}")
        print(f"Input levels: {self.input_levels}")
        print(f"Target level: {self.num_levels - 1} (last)")
        print(f"Sample value range: [{sample.min():.2f}, {sample.max():.2f}]")
    
    def __len__(self):
        return len(self.files) * len(self.input_levels)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - 'noisy': Input image [1, H, W]
                - 'clean': Target image [1, H, W] (last cumulative level)
                - 'noise_level': Normalized noise level (for conditioning)
                - 'level_idx': Which input level was used
        """
        # Map idx to file and level
        file_idx = idx // len(self.input_levels)
        level_idx = idx % len(self.input_levels)
        input_level = self.input_levels[level_idx]
        
        # Load patch
        file_path = self.files[file_idx]
        patch = np.load(file_path)  # Shape: (H, W, num_levels)
        
        # Normalize to [0, 1] - exactly as original
        if patch.max() > 1.0:
            patch = patch / patch.max()
        
        # Extract noisy (input level) and clean (last level)
        noisy = patch[:, :, input_level]
        clean = patch[:, :, -1]  # Last level is ground truth
        
        # Convert to tensor: [H, W] -> [1, H, W]
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        
        # Clamp to valid range
        noisy = torch.clamp(noisy, 0, 1)
        clean = torch.clamp(clean, 0, 1)
        
        # Compute noise level (0 = cleanest, 1 = noisiest)
        noise_level = 1.0 - (input_level / (self.num_levels - 1))
        noise_level = torch.tensor([noise_level], dtype=torch.float32)
        
        return {
            'noisy': noisy,
            'clean': clean,
            'noise_level': noise_level,
            'level_idx': input_level
        }


# ==============================================================================
# ========================== MODEL =============================================
# ==============================================================================

class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU with residual."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU(inplace=True)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)
        
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        # Use reshape instead of view to handle non-contiguous tensors
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)


class UNetDenoiser(nn.Module):
    """
    U-Net for image denoising.
    
    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (1 for grayscale)
        base_channels: Base feature channels (64 recommended)
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        
        ch = base_channels
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)
        self.enc4 = ConvBlock(ch * 4, ch * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ConvBlock(ch * 8, ch * 8),
            AttentionBlock(ch * 8),
            ConvBlock(ch * 8, ch * 8)
        )
        
        # Decoder
        # up4(b) has ch*4 channels, e4 has ch*8 channels -> concat = ch*4 + ch*8 = ch*12
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch * 4 + ch * 8, ch * 4)  # 768 -> 256
        
        # up3(d4) has ch*2 channels, e3 has ch*4 channels -> concat = ch*2 + ch*4 = ch*6
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 2 + ch * 4, ch * 2)  # 384 -> 128
        
        # up2(d3) has ch channels, e2 has ch*2 channels -> concat = ch + ch*2 = ch*3
        self.up2 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch + ch * 2, ch)  # 192 -> 64
        
        # up1(d2) has ch channels, e1 has ch channels -> concat = ch + ch = ch*2
        self.up1 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch + ch, ch)  # 128 -> 64
        
        # Output
        self.final = nn.Conv2d(ch, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Output with residual learning
        out = x + self.final(d1)
        return torch.clamp(out, 0, 1)


# ==============================================================================
# ========================== METRICS ===========================================
# ==============================================================================

def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR in dB."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


def compute_ssim(pred, target):
    """Compute SSIM value (not loss)."""
    ssim_loss = SSIMLoss()
    with torch.no_grad():
        loss = ssim_loss(pred, target)
    return 1 - loss.item()


# ==============================================================================
# ========================== TRAINING CONFIG ===================================
# ==============================================================================

class TrainingConfig:
    """Training configuration."""
    
    def __init__(self):
        # === DATA PATHS - UPDATE THESE ===
        self.data_path = "output_noisy_patches/noisy_patches_20251015_143344"  # Your patch directory
        self.save_dir = "experiments/unet_denoising"
        
        # === MODEL ===
        self.model_config = {
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 64,
        }
        
        # === TRAINING ===
        self.num_epochs = 150
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.train_val_split = 0.8
        
        # === LOSS WEIGHTS ===
        self.loss_config = {
            'l1_weight': 1,
            'l2_weight': 1,
            'ssim_weight': 1,
            'ssim_window_size': 11
        }
        
        # === SCHEDULER ===
        self.warmup_epochs = 5
        self.min_lr_factor = 0.01
        
        # === TRAINING OPTIONS ===
        self.mixed_precision = True
        self.gradient_clip_norm = 1.0
        
        # === LOGGING & SAVING ===
        self.save_freq = 10
        self.eval_freq = 5
        self.sample_freq = 10
        self.num_save_samples = 4
        
        # === HARDWARE ===
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(4, os.cpu_count() or 2)
        self.pin_memory = True
        
        # === AUTO-GENERATED PATHS ===
        self.checkpoint_dir = Path(self.save_dir) / 'checkpoints'
        self.samples_dir = Path(self.save_dir) / 'samples'
        self.logs_dir = Path(self.save_dir) / 'logs'
        self.tensorboard_dir = Path(self.save_dir) / 'tensorboard'


# ==============================================================================
# ========================== TRAINER ===========================================
# ==============================================================================

class Trainer:
    """Trainer for U-Net denoising model."""
    
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
        """Create output directories."""
        for d in [self.config.save_dir, self.config.checkpoint_dir, 
                  self.config.samples_dir, self.config.logs_dir,
                  self.config.tensorboard_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging and TensorBoard."""
        log_file = self.config.logs_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # CSV for loss tracking
        self.loss_csv_path = self.config.logs_dir / 'training_history.csv'
        columns = ['epoch', 'train_loss', 'train_l1', 'train_l2', 'train_ssim',
                   'val_loss', 'val_l1', 'val_l2', 'val_ssim', 'val_psnr', 'val_ssim_metric', 'lr']
        pd.DataFrame(columns=columns).to_csv(self.loss_csv_path, index=False)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))
        self.logger.info(f"TensorBoard logging to: {self.config.tensorboard_dir}")
        self.logger.info(f"Run 'tensorboard --logdir={self.config.tensorboard_dir}' to view")
    
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        self.logger.info(f"Loading data from: {self.config.data_path}")
        
        # Create dataset using original PatchDataset format (no augmentation)
        full_dataset = PatchDataset(data_path=self.config.data_path)
        
        # Split
        total_size = len(full_dataset)
        train_size = int(total_size * self.config.train_val_split)
        val_size = total_size - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
        
        self.logger.info(f"Training samples: {train_size}")
        self.logger.info(f"Validation samples: {val_size}")
    
    def _setup_model(self):
        """Setup model."""
        self.model = UNetDenoiser(**self.config.model_config).to(self.config.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {num_params:,}")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, loss."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine scheduler with warmup
        num_warmup_steps = self.config.warmup_epochs * len(self.train_loader)
        num_training_steps = self.config.num_epochs * len(self.train_loader)
        
        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(self.config.min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss
        self.criterion = CombinedLoss(**self.config.loss_config).to(self.config.device)
        
        # Mixed precision - updated API
        self.scaler = torch.amp.GradScaler('cuda') if self.config.mixed_precision else None
        
        self.logger.info(f"Loss weights: L1={self.config.loss_config['l1_weight']}, "
                        f"L2={self.config.loss_config['l2_weight']}, "
                        f"SSIM={self.config.loss_config['ssim_weight']}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': [], 'l1': [], 'l2': [], 'ssim': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch in pbar:
            noisy = batch['noisy'].to(self.config.device)
            clean = batch['clean'].to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Updated autocast API
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                output = self.model(noisy)
                loss, loss_dict = self.criterion(output, clean)
            
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
            
            # Record losses
            for k in epoch_losses:
                epoch_losses[k].append(loss_dict[k] if k in loss_dict else loss_dict['total'])
            
            # TensorBoard logging (every 100 steps)
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss_Step', loss_dict['total'], self.global_step)
                self.writer.add_scalar('Train/L1_Step', loss_dict['l1'], self.global_step)
                self.writer.add_scalar('Train/L2_Step', loss_dict['l2'], self.global_step)
                self.writer.add_scalar('Train/SSIM_Step', loss_dict['ssim'], self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], self.global_step)
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}", 
                            'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"})
        
        # Log epoch averages to TensorBoard
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.writer.add_scalar('Train/Loss_Epoch', avg_losses['total'], epoch)
        self.writer.add_scalar('Train/L1_Epoch', avg_losses['l1'], epoch)
        self.writer.add_scalar('Train/L2_Epoch', avg_losses['l2'], epoch)
        self.writer.add_scalar('Train/SSIM_Epoch', avg_losses['ssim'], epoch)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        val_losses = {'total': [], 'l1': [], 'l2': [], 'ssim': []}
        psnrs, ssims = [], []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            noisy = batch['noisy'].to(self.config.device)
            clean = batch['clean'].to(self.config.device)
            
            output = self.model(noisy)
            _, loss_dict = self.criterion(output, clean)
            
            for k in val_losses:
                val_losses[k].append(loss_dict[k] if k in loss_dict else loss_dict['total'])
            
            # Metrics
            for i in range(output.shape[0]):
                psnrs.append(compute_psnr(output[i:i+1], clean[i:i+1]))
                ssims.append(compute_ssim(output[i:i+1], clean[i:i+1]))
        
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        
        # TensorBoard logging
        self.writer.add_scalar('Val/Loss', avg_losses['total'], epoch)
        self.writer.add_scalar('Val/L1', avg_losses['l1'], epoch)
        self.writer.add_scalar('Val/L2', avg_losses['l2'], epoch)
        self.writer.add_scalar('Val/SSIM_Loss', avg_losses['ssim'], epoch)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM_Metric', avg_ssim, epoch)
        
        # Log comparison: train vs val loss
        self.writer.add_scalars('Loss/Comparison', {
            'train': self.train_losses[-1] if hasattr(self, 'train_losses') and self.train_losses else avg_losses['total'],
            'val': avg_losses['total']
        }, epoch)
        
        self.logger.info(f"Val Loss: {avg_losses['total']:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
        
        return avg_losses, avg_psnr, avg_ssim
    
    @torch.no_grad()
    def generate_samples(self, epoch):
        """Generate and save sample images."""
        self.model.eval()
        
        batch = next(iter(self.val_loader))
        noisy = batch['noisy'].to(self.config.device)
        clean = batch['clean'].to(self.config.device)
        
        n = min(self.config.num_save_samples, noisy.shape[0])
        noisy, clean = noisy[:n], clean[:n]
        
        output = self.model(noisy)
        
        # Log images to TensorBoard
        # Create a grid: noisy | denoised | clean for each sample
        for i in range(min(n, 4)):  # Log up to 4 samples
            self.writer.add_images(f'Samples/Sample_{i}', 
                                   torch.stack([noisy[i], output[i], clean[i]]), 
                                   epoch, dataformats='NCHW')
        
        # Also log as a comparison grid
        comparison = torch.cat([noisy, output, clean], dim=0)  # Stack all
        self.writer.add_images('Samples/Comparison_Grid', comparison, epoch)
        
        # Create comparison figure for file saving
        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            axes[i, 0].imshow(noisy[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title('Noisy Input')
            axes[i, 0].axis('off')
            
            psnr = compute_psnr(output[i:i+1], clean[i:i+1])
            axes[i, 1].imshow(output[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title(f'Denoised (PSNR: {psnr:.1f} dB)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(clean[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.config.samples_dir / f'samples_epoch_{epoch:03d}.png', dpi=150)
        plt.close()
        self.logger.info(f"Saved samples to samples_epoch_{epoch:03d}.png")
    
    def save_checkpoint(self, epoch, val_loss, val_psnr, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
        }
        
        torch.save(checkpoint, self.config.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        torch.save(checkpoint, self.config.checkpoint_dir / 'latest.pt')
        
        if is_best:
            torch.save(checkpoint, self.config.checkpoint_dir / 'best_model.pt')
            self.logger.info(f"New best model! PSNR: {val_psnr:.2f} dB")
    
    def save_history(self, epoch, train_losses, val_losses=None, val_psnr=None, val_ssim=None):
        """Save training history to CSV."""
        df = pd.read_csv(self.loss_csv_path)
        
        row = {
            'epoch': epoch,
            'train_loss': train_losses['total'],
            'train_l1': train_losses['l1'],
            'train_l2': train_losses['l2'],
            'train_ssim': train_losses['ssim'],
            'val_loss': val_losses['total'] if val_losses else np.nan,
            'val_l1': val_losses['l1'] if val_losses else np.nan,
            'val_l2': val_losses['l2'] if val_losses else np.nan,
            'val_ssim': val_losses['ssim'] if val_losses else np.nan,
            'val_psnr': val_psnr if val_psnr else np.nan,
            'val_ssim_metric': val_ssim if val_ssim else np.nan,
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.loss_csv_path, index=False)
    
    def plot_training_curves(self):
        """Plot training curves."""
        df = pd.read_csv(self.loss_csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(df['epoch'], df['val_psnr'], 'g-o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Validation PSNR')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(df['epoch'], df['val_ssim_metric'], 'b-o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Validation SSIM')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(df['epoch'], df['lr'], 'r-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.config.logs_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_psnr = checkpoint.get('val_psnr', 0.0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}, global_step {self.global_step}")
        return checkpoint['epoch'] + 1
    
    def train(self, resume_path=None):
        """Main training loop."""
        start_epoch = 0
        if resume_path and Path(resume_path).exists():
            start_epoch = self.load_checkpoint(resume_path)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Starting training on {self.config.device}")
        self.logger.info("=" * 60)
        
        # Track losses for TensorBoard comparison
        self.train_losses = []
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                train_losses = self.train_epoch(epoch)
                self.train_losses.append(train_losses['total'])
                
                val_losses, val_psnr, val_ssim = None, None, None
                if (epoch + 1) % self.config.eval_freq == 0:
                    val_losses, val_psnr, val_ssim = self.validate(epoch)
                    
                    is_best = val_psnr > self.best_val_psnr
                    if is_best:
                        self.best_val_psnr = val_psnr
                        self.best_val_loss = val_losses['total']
                    
                    self.save_checkpoint(epoch, val_losses['total'], val_psnr, is_best)
                
                self.save_history(epoch, train_losses, val_losses, val_psnr, val_ssim)
                
                if (epoch + 1) % self.config.sample_freq == 0:
                    self.generate_samples(epoch)
                
                if (epoch + 1) % 10 == 0:
                    self.plot_training_curves()
                
                log_msg = f"Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {train_losses['total']:.4f}"
                if val_losses:
                    log_msg += f" | Val Loss: {val_losses['total']:.4f} | PSNR: {val_psnr:.2f} dB"
                self.logger.info(log_msg)
            
            self.plot_training_curves()
            self.logger.info(f"Training completed! Best PSNR: {self.best_val_psnr:.2f} dB")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
            self.save_checkpoint(epoch, 0, 0, is_best=False)
            self.plot_training_curves()
        
        finally:
            # Close TensorBoard writer
            self.writer.close()
            self.logger.info("TensorBoard writer closed")


# ==============================================================================
# ========================== MAIN ==============================================
# ==============================================================================

def main():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Config
    config = TrainingConfig()
    
    # ============================================
    # UPDATE THIS PATH TO YOUR DATA
    # ============================================
    config.data_path = "output_noisy_patches/noisy_patches_20251015_143344"
    config.save_dir = "experiments/unet_denoising"
    # ============================================
    
    if not os.path.exists(config.data_path):
        print(f"ERROR: Data path does not exist: {config.data_path}")
        print("Please update 'config.data_path' to your patch directory")
        sys.exit(1)
    
    # Train
    trainer = Trainer(config)
    
    resume_path = config.checkpoint_dir / 'latest.pt'
    trainer.train(resume_path=resume_path if resume_path.exists() else None)


if __name__ == "__main__":
    main()