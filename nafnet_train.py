"""
NAFNet Training Script for Cherenkov Image Denoising - FIXED VERSION

FIXES APPLIED:
1. Symmetric encoder-decoder architecture [2,2,4,8] on both sides
2. Consistent normalization - ALWAYS normalize by max value
3. Rebalanced loss weights - SSIM reduced to 0.2
4. Learning rate increased back to 5e-4
5. Validation loss properly tracked and plotted
6. Import from fixed model file

Loss: L1 + L2 (MSE) + SSIM (reweighted)
Data: .npy patches with shape (H, W, num_cumulative_levels)
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

from nafnet_model import NAFNet, count_parameters


# ==============================================================================
# ========================== LOSS FUNCTIONS ====================================
# ==============================================================================

class SSIMLoss(nn.Module):
    """SSIM Loss. Loss = 1 - SSIM."""
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
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, pred, target):
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        channel = pred.size(1)
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel).to(pred.device)
            self.channel = channel

        pad = self.window_size // 2
        mu_p = F.conv2d(pred, self.window, padding=pad, groups=channel)
        mu_t = F.conv2d(target, self.window, padding=pad, groups=channel)
        mu_pp, mu_tt, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

        sig_pp = F.conv2d(pred ** 2, self.window, padding=pad, groups=channel) - mu_pp
        sig_tt = F.conv2d(target ** 2, self.window, padding=pad, groups=channel) - mu_tt
        sig_pt = F.conv2d(pred * target, self.window, padding=pad, groups=channel) - mu_pt
        sig_pp = torch.clamp(sig_pp, min=0)
        sig_tt = torch.clamp(sig_tt, min=0)

        num = (2 * mu_pt + self.C1) * (2 * sig_pt + self.C2)
        den = (mu_pp + mu_tt + self.C1) * (sig_pp + sig_tt + self.C2)
        return 1 - (num / (den + 1e-8)).mean()


class CombinedLoss(nn.Module):
    """Combined loss: L1 + L2 (MSE) + SSIM"""
    def __init__(self, l1_weight=1.0, l2_weight=1.0, ssim_weight=0.2, ssim_window_size=11):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        total = self.l1_weight * l1 + self.l2_weight * l2 + self.ssim_weight * ssim
        return total, {'total': total.item(), 'l1': l1.item(), 'l2': l2.item(), 'ssim': ssim.item()}


# ==============================================================================
# ========================== DATASET ===========================================
# ==============================================================================

class PatchDataset(Dataset):
    """
    Loads .npy patches with shape (H, W, num_cumulative_levels).
    Input: any level except last. Target: last level (ground truth).
    
    FIXED: ALWAYS normalize by max value for consistency
    """
    def __init__(self, data_path, input_levels=None):
        self.data_path = data_path
        self.files = glob.glob(os.path.join(data_path, '*.npy'))
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {data_path}")

        # Sort for reproducibility across runs / platforms
        self.files.sort()

        sample = np.load(self.files[0])
        self.num_levels = sample.shape[-1]
        self.input_levels = input_levels or list(range(self.num_levels - 1))

        print(f"Found {len(self.files)} patch files")
        print(f"Sample patch shape: {sample.shape}")
        print(f"Number of cumulative levels: {self.num_levels}")
        print(f"Input levels: {self.input_levels}")
        print(f"Target level: {self.num_levels - 1} (last)")
        print(f"Sample value range: [{sample.min():.2f}, {sample.max():.2f}]")

    def __len__(self):
        return len(self.files) * len(self.input_levels)

    def __getitem__(self, idx):
        file_idx = idx // len(self.input_levels)
        level_idx = idx % len(self.input_levels)
        input_level = self.input_levels[level_idx]

        patch = np.load(self.files[file_idx])
        
        # FIXED: ALWAYS normalize consistently
        max_val = patch.max()
        if max_val > 0:
            patch = patch / max_val

        noisy = torch.tensor(patch[:, :, input_level], dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(patch[:, :, -1], dtype=torch.float32).unsqueeze(0)
        noisy = torch.clamp(noisy, 0, 1)
        clean = torch.clamp(clean, 0, 1)

        noise_level = torch.tensor([1.0 - (input_level / (self.num_levels - 1))], dtype=torch.float32)

        return {'noisy': noisy, 'clean': clean, 'noise_level': noise_level, 'level_idx': input_level}


# ==============================================================================
# ========================== METRICS ===========================================
# ==============================================================================

def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


# Shared SSIMLoss instance for metric computation (avoids re-creating every call)
_ssim_metric_fn = None

def compute_ssim(pred, target):
    global _ssim_metric_fn
    if _ssim_metric_fn is None:
        _ssim_metric_fn = SSIMLoss()
    with torch.no_grad():
        loss = _ssim_metric_fn(pred.cpu(), target.cpu())
    return 1 - loss.item()


# ==============================================================================
# ========================== CONFIG ============================================
# ==============================================================================

class TrainingConfig:
    def __init__(self):
        # === REPRODUCIBILITY ===
        self.seed = 42

        # === DATA ===
        self.data_path = "noisy_patches"
        self.save_dir = "experiments/nafnet_denoising_fixed"

        # === MODEL (FIXED: Symmetric architecture) ===
        self.model_config = {
            'in_channels': 1,
            'out_channels': 1,
            'width': 32,
            'enc_blk_nums': [2, 2, 4, 8],
            'dec_blk_nums': [2, 2, 4, 8],  # FIXED: Match encoder!
            'middle_blk_num': 12,
            'dw_expand': 2,
            'ffn_expand': 2,
        }

        # === TRAINING ===
        self.num_epochs = 150
        self.batch_size = 16
        # FIXED: Increased LR back to 5e-4 (was too conservative at 1e-4)
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5
        # Train / Val / Test split ratios (must sum to 1.0)
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        # === LOSS (FIXED: Rebalanced SSIM weight) ===
        self.loss_config = {
            'l1_weight': 1.0,
            'l2_weight': 1.0,
            'ssim_weight': 0.2,  # FIXED: Reduced from 1.0 to 0.2
            'ssim_window_size': 11,
        }

        # === SCHEDULER ===
        self.warmup_epochs = 5
        self.min_lr_factor = 0.01

        # === OPTIONS ===
        self.mixed_precision = True
        self.use_bf16 = True
        self.gradient_clip_norm = 1.0

        # === NaN SAFETY ===
        self.nan_patience = 3

        # === LOGGING ===
        self.save_freq = 10
        self.eval_freq = 5
        self.sample_freq = 10
        self.num_save_samples = 4

        # === HARDWARE ===
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(4, os.cpu_count() or 2)
        self.pin_memory = True

        # === PATHS ===
        self.checkpoint_dir = Path(self.save_dir) / 'checkpoints'
        self.samples_dir = Path(self.save_dir) / 'samples'
        self.logs_dir = Path(self.save_dir) / 'logs'
        self.tensorboard_dir = Path(self.save_dir) / 'tensorboard'


# ==============================================================================
# ========================== TRAINER ===========================================
# ==============================================================================

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._nan_count = 0
        self._setup_directories()
        self._setup_logging()
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        self.global_step = 0
        self.train_losses = []

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

        self.loss_csv_path = self.config.logs_dir / 'training_history.csv'
        columns = ['epoch', 'train_loss', 'train_l1', 'train_l2', 'train_ssim',
                   'val_loss', 'val_l1', 'val_l2', 'val_ssim', 'val_psnr', 'val_ssim_metric', 'lr']
        pd.DataFrame(columns=columns).to_csv(self.loss_csv_path, index=False)

        self.writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))
        self.logger.info(f"TensorBoard: {self.config.tensorboard_dir}")

    def _setup_data(self):
        self.logger.info(f"Loading data from: {self.config.data_path}")
        full_dataset = PatchDataset(data_path=self.config.data_path)
        total = len(full_dataset)

        train_size = int(total * self.config.train_ratio)
        val_size = int(total * self.config.val_ratio)
        test_size = total - train_size - val_size

        assert train_size > 0 and val_size > 0 and test_size > 0, (
            f"All splits must be > 0, got train={train_size}, val={val_size}, test={test_size} "
            f"from total={total}"
        )

        split_generator = torch.Generator().manual_seed(self.config.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=split_generator
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
            drop_last=True, generator=torch.Generator().manual_seed(self.config.seed)
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory, drop_last=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory, drop_last=False
        )

        self.logger.info(f"Training: {train_size}, Validation: {val_size}, Test: {test_size}")

    def _setup_model(self):
        self.model = NAFNet(**self.config.model_config).to(self.config.device)
        n_params = count_parameters(self.model)
        self.logger.info(f"NAFNet parameters: {n_params:,}")
        self.logger.info(f"Architecture: enc={self.config.model_config['enc_blk_nums']}, "
                        f"dec={self.config.model_config['dec_blk_nums']}")

    def _setup_training(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay, betas=(0.9, 0.999)
        )

        num_warmup = self.config.warmup_epochs * len(self.train_loader)
        num_total = self.config.num_epochs * len(self.train_loader)

        def lr_lambda(step):
            if step < num_warmup:
                return float(step) / float(max(1, num_warmup))
            progress = float(step - num_warmup) / float(max(1, num_total - num_warmup))
            return max(self.config.min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.criterion = CombinedLoss(**self.config.loss_config).to(self.config.device)

        self.use_amp = self.config.mixed_precision and torch.cuda.is_available()
        if self.use_amp and self.config.use_bf16:
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                self.logger.info("Mixed precision: BF16 (no GradScaler needed)")
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda')
                self.logger.warning("BF16 requested but not supported — falling back to FP16 + GradScaler")
        elif self.use_amp:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler('cuda')
            self.logger.info("Mixed precision: FP16 + GradScaler")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            self.logger.info("Mixed precision: disabled (FP32)")

        self.logger.info(f"Optimizer: AdamW  lr={self.config.learning_rate}  betas=(0.9, 0.999)")
        self.logger.info(f"Loss weights: L1={self.config.loss_config['l1_weight']}, "
                        f"L2={self.config.loss_config['l2_weight']}, "
                        f"SSIM={self.config.loss_config['ssim_weight']}")

    def _check_nan_and_recover(self, loss_val, epoch):
        """
        Detect NaN loss and attempt recovery by rolling back to the last checkpoint.
        Returns True if NaN was detected (caller should skip this batch).
        """
        if not math.isnan(loss_val):
            self._nan_count = 0
            return False

        self._nan_count += 1
        self.logger.warning(f"NaN loss detected! (count: {self._nan_count}/{self.config.nan_patience})")

        if self._nan_count >= self.config.nan_patience:
            latest_ckpt = self.config.checkpoint_dir / 'best_model.pt'
            if latest_ckpt.exists():
                self.logger.warning("Rolling back to best checkpoint...")
                checkpoint = torch.load(latest_ckpt, map_location=self.config.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= 0.5
                self.logger.warning(f"LR halved to {self.optimizer.param_groups[0]['lr']:.2e} after NaN recovery")
                self._nan_count = 0
            else:
                self.logger.error("No checkpoint to recover from — training may not recover")
        return True

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {'total': [], 'l1': [], 'l2': [], 'ssim': []}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch in pbar:
            noisy = batch['noisy'].to(self.config.device)
            clean = batch['clean'].to(self.config.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                output = self.model(noisy)
                loss, loss_dict = self.criterion(output, clean)

            if self._check_nan_and_recover(loss_dict['total'], epoch):
                pbar.set_postfix({'loss': 'NaN!', 'status': 'skip'})
                continue

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

            for k in epoch_losses:
                epoch_losses[k].append(loss_dict[k] if k in loss_dict else loss_dict['total'])

            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss_Step', loss_dict['total'], self.global_step)
                self.writer.add_scalar('Train/L1_Step', loss_dict['l1'], self.global_step)
                self.writer.add_scalar('Train/L2_Step', loss_dict['l2'], self.global_step)
                self.writer.add_scalar('Train/SSIM_Step', loss_dict['ssim'], self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.scheduler.get_last_lr()[0], self.global_step)

            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}",
                            'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"})

        if not epoch_losses['total']:
            return {k: float('nan') for k in epoch_losses}

        avg = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.writer.add_scalar('Train/Loss_Epoch', avg['total'], epoch)
        self.writer.add_scalar('Train/L1_Epoch', avg['l1'], epoch)
        self.writer.add_scalar('Train/L2_Epoch', avg['l2'], epoch)
        self.writer.add_scalar('Train/SSIM_Epoch', avg['ssim'], epoch)
        return avg

    @torch.no_grad()
    def validate(self, epoch):
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
            for i in range(output.shape[0]):
                psnrs.append(compute_psnr(output[i:i+1], clean[i:i+1]))
                ssims.append(compute_ssim(output[i:i+1], clean[i:i+1]))

        avg = {k: np.mean(v) for k, v in val_losses.items()}
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)

        self.writer.add_scalar('Val/Loss', avg['total'], epoch)
        self.writer.add_scalar('Val/L1', avg['l1'], epoch)
        self.writer.add_scalar('Val/L2', avg['l2'], epoch)
        self.writer.add_scalar('Val/SSIM_Loss', avg['ssim'], epoch)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM_Metric', avg_ssim, epoch)
        self.writer.add_scalars('Loss/Comparison', {
            'train': self.train_losses[-1] if self.train_losses else avg['total'],
            'val': avg['total']
        }, epoch)

        self.logger.info(f"Val Loss: {avg['total']:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
        return avg, avg_psnr, avg_ssim

    @torch.no_grad()
    def test(self):
        """Run evaluation on the held-out test set using the best checkpoint."""
        best_ckpt = self.config.checkpoint_dir / 'best_model.pt'
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from epoch {checkpoint['epoch']} for testing")
        else:
            self.logger.warning("No best_model.pt found — testing with current model weights")

        self.model.eval()
        test_losses = {'total': [], 'l1': [], 'l2': [], 'ssim': []}
        psnrs, ssims = [], []

        for batch in tqdm(self.test_loader, desc="Testing"):
            noisy = batch['noisy'].to(self.config.device)
            clean = batch['clean'].to(self.config.device)
            output = self.model(noisy)
            _, loss_dict = self.criterion(output, clean)

            for k in test_losses:
                test_losses[k].append(loss_dict[k] if k in loss_dict else loss_dict['total'])
            for i in range(output.shape[0]):
                psnrs.append(compute_psnr(output[i:i+1], clean[i:i+1]))
                ssims.append(compute_ssim(output[i:i+1], clean[i:i+1]))

        avg = {k: np.mean(v) for k, v in test_losses.items()}
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)

        self.logger.info("=" * 60)
        self.logger.info("TEST SET RESULTS")
        self.logger.info(f"  Loss:  {avg['total']:.4f}  (L1={avg['l1']:.4f}, L2={avg['l2']:.4f}, SSIM_loss={avg['ssim']:.4f})")
        self.logger.info(f"  PSNR:  {avg_psnr:.2f} dB")
        self.logger.info(f"  SSIM:  {avg_ssim:.4f}")
        self.logger.info("=" * 60)

        test_results = {
            'test_loss': avg['total'], 'test_l1': avg['l1'],
            'test_l2': avg['l2'], 'test_ssim_loss': avg['ssim'],
            'test_psnr': avg_psnr, 'test_ssim': avg_ssim,
        }
        results_path = self.config.logs_dir / 'test_results.csv'
        pd.DataFrame([test_results]).to_csv(results_path, index=False)
        self.logger.info(f"Test results saved to {results_path}")

        self.writer.add_scalar('Test/Loss', avg['total'], 0)
        self.writer.add_scalar('Test/PSNR', avg_psnr, 0)
        self.writer.add_scalar('Test/SSIM', avg_ssim, 0)

        return avg, avg_psnr, avg_ssim

    @torch.no_grad()
    def generate_samples(self, epoch):
        self.model.eval()
        batch = next(iter(self.val_loader))
        noisy = batch['noisy'].to(self.config.device)
        clean = batch['clean'].to(self.config.device)
        n = min(self.config.num_save_samples, noisy.shape[0])
        noisy, clean = noisy[:n], clean[:n]
        output = self.model(noisy)

        for i in range(min(n, 4)):
            self.writer.add_images(f'Samples/Sample_{i}',
                                   torch.stack([noisy[i], output[i], clean[i]]),
                                   epoch, dataformats='NCHW')
        comparison = torch.cat([noisy, output, clean], dim=0)
        self.writer.add_images('Samples/Comparison_Grid', comparison, epoch)

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
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'config_seed': self.config.seed,
        }
        torch.save(checkpoint, self.config.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        torch.save(checkpoint, self.config.checkpoint_dir / 'latest.pt')
        if is_best:
            torch.save(checkpoint, self.config.checkpoint_dir / 'best_model.pt')
            self.logger.info(f"New best model! PSNR: {val_psnr:.2f} dB")

    def save_history(self, epoch, train_losses, val_losses=None, val_psnr=None, val_ssim=None):
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
        pd.DataFrame([row]).to_csv(self.loss_csv_path, mode='a', header=False, index=False)

    def plot_training_curves(self):
        df = pd.read_csv(self.loss_csv_path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # FIXED: Plot validation loss with markers
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', marker='')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', marker='o', markersize=4)
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[0, 1].plot(df['epoch'], df['val_psnr'], 'g-o', markersize=4)
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Validation PSNR'); axes[0, 1].grid(True)

        axes[1, 0].plot(df['epoch'], df['val_ssim_metric'], 'b-o', markersize=4)
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('Validation SSIM'); axes[1, 0].grid(True)

        axes[1, 1].plot(df['epoch'], df['lr'], 'r-')
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule'); axes[1, 1].set_yscale('log'); axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.config.logs_dir / 'training_curves.png', dpi=150)
        plt.close()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_psnr = checkpoint.get('val_psnr', 0.0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
        return checkpoint['epoch'] + 1

    def train(self, resume_path=None):
        start_epoch = 0
        if resume_path and Path(resume_path).exists():
            start_epoch = self.load_checkpoint(resume_path)

        self.logger.info("=" * 60)
        self.logger.info(f"Starting NAFNet training (FIXED VERSION) on {self.config.device}")
        self.logger.info(f"Random seed: {self.config.seed}")
        self.logger.info(f"AMP dtype: {self.amp_dtype}")
        self.logger.info(f"Architecture: enc={self.config.model_config['enc_blk_nums']}, "
                        f"dec={self.config.model_config['dec_blk_nums']}")
        self.logger.info(f"Loss weights: L1={self.config.loss_config['l1_weight']}, "
                        f"L2={self.config.loss_config['l2_weight']}, "
                        f"SSIM={self.config.loss_config['ssim_weight']}")
        self.logger.info("=" * 60)

        epoch = start_epoch
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

            self.logger.info("Running test set evaluation...")
            self.test()

        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
            self.save_checkpoint(epoch, 0, 0, is_best=False)
            self.plot_training_curves()
        finally:
            self.writer.close()
            self.logger.info("TensorBoard writer closed")


# ==============================================================================
# ========================== MAIN ==============================================
# ==============================================================================

def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    config = TrainingConfig()

    # ============================================
    # UPDATE THIS PATH TO YOUR DATA
    # ============================================
    config.data_path = "noisy_patches"
    config.save_dir = "experiments/nafnet_denoising_fixed"
    config.seed = 42
    # ============================================

    seed_everything(config.seed)

    if not os.path.exists(config.data_path):
        print(f"ERROR: Data path does not exist: {config.data_path}")
        print(f"Please update the config.data_path in main() to point to your data")
        sys.exit(1)

    trainer = Trainer(config)
    resume_path = config.checkpoint_dir / 'latest.pt'
    trainer.train(resume_path=resume_path if resume_path.exists() else None)


if __name__ == "__main__":
    main()