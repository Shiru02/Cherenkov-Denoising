"""
Palette-Style Conditional Diffusion for Cherenkov Image Denoising
=================================================================
x0-Parameterization + Temporal Attention Modulation (TAM) Version:
- Model directly predicts the clean image (x0) instead of noise (eps).
- TAM module Cross-Attends current bottleneck with cached past semantic states.
- Enables stable spatial regularizers (L1 + L2) at all timesteps.
- DDIM/DDPM samplers enforce physical bounds [0, 1] at every prediction step.
- Efficient training generates 1 history step on-the-fly to train TAM without unrolling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

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


class PaletteUNet(nn.Module):
    def __init__(self, base_channels: int = 64, embed_dim: int = 128):
        super().__init__()
        ch = base_channels

        self.t_embed  = SinusoidalTimestepEmbedding(embed_dim)
        self.nl_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(),
        )

        self.enc1 = ConvBlock(2,    ch)
        self.enc2 = ConvBlock(ch,   ch*2)
        self.enc3 = ConvBlock(ch*2, ch*4)
        self.enc4 = ConvBlock(ch*4, ch*8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(ch*8, ch*8),
            AttentionBlock(ch*8),
            ConvBlock(ch*8, ch*8),
        )

        # --- NEW: Temporal Attention Modulation (TAM) Components ---
        self.ch_b = ch * 8
        self.tam_q = nn.Linear(embed_dim, self.ch_b)
        self.tam_k = nn.Linear(self.ch_b, self.ch_b)
        self.tam_v = nn.Linear(self.ch_b, self.ch_b)
        self.tam_out = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(self.ch_b, self.ch_b, 1)
        )
        # -----------------------------------------------------------

        self.cond_proj_b = nn.Linear(embed_dim, ch*8)
        self.cond_proj_4 = nn.Linear(embed_dim, ch*4)
        self.cond_proj_3 = nn.Linear(embed_dim, ch*2)
        self.cond_proj_2 = nn.Linear(embed_dim, ch)
        self.cond_proj_1 = nn.Linear(embed_dim, ch)

        self.up4  = nn.ConvTranspose2d(ch*8, ch*4, 2, stride=2)
        self.dec4 = ConvBlock(ch*4 + ch*8, ch*4)
        self.up3  = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec3 = ConvBlock(ch*2 + ch*4, ch*2)
        self.up2  = nn.ConvTranspose2d(ch*2, ch, 2, stride=2)
        self.dec2 = ConvBlock(ch + ch*2, ch)
        self.up1  = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1 = ConvBlock(ch + ch, ch)
        self.final = nn.Conv2d(ch, 1, 1)

    def forward(self, x_t, y, t, noise_level, history=None):
        cond = self.t_embed(t) + self.nl_embed(noise_level) 
        inp  = torch.cat([x_t, y], dim=1)                    

        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))
        b  = b  + self.cond_proj_b(cond)[:, :, None, None]

        # --- Extract Semantic State for Memory ---
        b_pooled = b.mean(dim=[2, 3]) # Global average pool -> (B, ch*8)

        # --- Apply Temporal Attention Modulation (TAM) ---
        if history is not None and len(history) > 0:
            # history is a list of tensors of shape (B, ch*8)
            H_t = torch.stack(history, dim=1) # (B, N_hist, ch*8)
            
            Q = self.tam_q(cond).unsqueeze(1) # (B, 1, ch*8)
            K = self.tam_k(H_t)               # (B, N_hist, ch*8)
            V = self.tam_v(H_t)               # (B, N_hist, ch*8)

            # Cross-Attention
            attn = torch.bmm(Q, K.transpose(1, 2)) * (self.ch_b ** -0.5) # (B, 1, N_hist)
            attn = F.softmax(attn, dim=-1)

            # Modulate current bottleneck
            mod = torch.bmm(attn, V) # (B, 1, ch*8)
            mod = mod.view(-1, self.ch_b, 1, 1) # Expand spatially to (B, ch*8, 1, 1)
            b = b + self.tam_out(mod)
        # -------------------------------------------------

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d4 = d4 + self.cond_proj_4(cond)[:, :, None, None]
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d3 = d3 + self.cond_proj_3(cond)[:, :, None, None]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d2 = d2 + self.cond_proj_2(cond)[:, :, None, None]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = d1 + self.cond_proj_1(cond)[:, :, None, None]

        # Notice we now return BOTH the prediction and the pooled bottleneck state
        return self.final(d1), b_pooled


# ==============================================================================
# ========================== DATASET ===========================================
# ==============================================================================

class PatchDataset(Dataset):
    def __init__(self, data_path: str, input_levels: list = None):
        self.files = glob.glob(os.path.join(data_path, '*.npy'))
        if not self.files:
            raise ValueError(f"No .npy files found in {data_path}")
        sample = np.load(self.files[0])
        self.num_levels   = sample.shape[-1]
        self.input_levels = input_levels or list(range(self.num_levels - 1))
        print(f"Found {len(self.files)} patches | shape {sample.shape} | "
              f"input levels {self.input_levels} | target {self.num_levels-1}")

    def __len__(self):
        return len(self.files) * len(self.input_levels)

    def __getitem__(self, idx):
        file_idx    = idx // len(self.input_levels)
        level_idx   = idx %  len(self.input_levels)
        input_level = self.input_levels[level_idx]

        patch = np.load(self.files[file_idx])
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
# ========================== SAMPLER ===========================================
# ==============================================================================

class DiffusionSampler:
    def __init__(self, schedule: DiffusionSchedule, model: PaletteUNet, device, tam_history_size=5):
        self.schedule = schedule
        self.model    = model
        self.device   = device
        self.tam_history_size = tam_history_size

    @torch.no_grad()
    def ddpm_sample(self, y, noise_level):
        B   = y.shape[0]
        x_t = torch.randn_like(y)
        T   = self.schedule.T
        history_buffer = []

        for t_val in tqdm(reversed(range(T)), desc="DDPM (TAM x0)", total=T, leave=False):
            t_tensor = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            
            # Predict x0 with TAM Memory
            x0_pred, b_pooled = self.model(x_t, y, t_tensor, noise_level, history=history_buffer)
            x0_pred = x0_pred.clamp(0, 1)

            # Update rolling history buffer
            history_buffer.append(b_pooled)
            if len(history_buffer) > self.tam_history_size:
                history_buffer.pop(0)

            alpha     = self.schedule.alphas[t_val]
            alpha_bar = self.schedule.alphas_bar[t_val]
            beta      = self.schedule.betas[t_val]
            post_var  = self.schedule.posterior_var[t_val]

            eps_pred = (x_t - alpha_bar.sqrt() * x0_pred) / (1.0 - alpha_bar).sqrt().clamp(min=1e-8)
            coef = beta / (1.0 - alpha_bar).sqrt()
            mean = (1.0 / alpha.sqrt()) * (x_t - coef * eps_pred)

            if t_val > 0:
                x_t = mean + post_var.sqrt() * torch.randn_like(x_t)
            else:
                x_t = mean

        return x_t.clamp(0, 1)

    @torch.no_grad()
    def ddim_sample(self, y, noise_level, n_steps=100, eta=0.0):
        B  = y.shape[0]
        T  = self.schedule.T
        step_size = max(T // n_steps, 1)
        timesteps = list(reversed(range(0, T, step_size)))

        x_t = torch.randn_like(y)
        history_buffer = []

        for i, t_val in enumerate(tqdm(timesteps, desc=f"DDIM η={eta} (TAM x0)", leave=False)):
            t_tensor = torch.full((B,), t_val, device=self.device, dtype=torch.long)
            
            # Predict x0 with TAM Memory
            x0_pred, b_pooled = self.model(x_t, y, t_tensor, noise_level, history=history_buffer)
            x0_pred = x0_pred.clamp(0, 1)

            # Update rolling history buffer
            history_buffer.append(b_pooled)
            if len(history_buffer) > self.tam_history_size:
                history_buffer.pop(0)

            ab_t   = self.schedule.alphas_bar[t_val]
            t_prev = timesteps[i+1] if i + 1 < len(timesteps) else -1
            ab_prev = self.schedule.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            eps_pred = (x_t - ab_t.sqrt() * x0_pred) / (1.0 - ab_t).sqrt().clamp(min=1e-8)
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0).sqrt()
            dir_x = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            x_t   = ab_prev.sqrt() * x0_pred + dir_x
            
            if eta > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return x_t.clamp(0, 1)

    @torch.no_grad()
    def sample_n(self, y, noise_level, n=8, method='ddim', n_ddim_steps=100, eta=0.0):
        self.model.eval()
        all_samples = []
        for _ in range(n):
            if method == 'ddim':
                s = self.ddim_sample(y, noise_level, n_steps=n_ddim_steps, eta=eta)
            else:
                s = self.ddpm_sample(y, noise_level)
            all_samples.append(s)
            
        samples = torch.stack(all_samples, dim=0) 
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
        self.save_dir  = "experiments/palette_cherenkov_tam"
        
        self.model_config = {
            'base_channels': 64,
            'embed_dim': 128,
        }
        
        self.T = 1000
        self.sample_method = 'ddim'
        self.n_ddim_steps = 100
        self.ddim_eta = 0.0 
        self.n_uncertainty = 8 
        
        self.tam_history_size = 5 # Size of the temporal memory buffer

        self.num_epochs = 1000
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.train_val_split = 0.8

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
        full = PatchDataset(data_path=self.config.data_path)
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
        self.sampler  = DiffusionSampler(self.schedule, self.model, self.config.device, self.config.tam_history_size)
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
        if not os.path.exists(checkpoint_path):
            self.logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return 1

        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        
        # Load weights with strict=False to gracefully initialize new TAM layers from scratch
        self.model.load_state_dict(ckpt['model'], strict=False) 
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
            B  = x0.shape[0]

            t = torch.randint(0, self.config.T, (B,), device=self.config.device).long()
            x_t, _ = self.schedule.q_sample(x0, t)

            # --- EFFICIENT TAM TRAINING: Simulate 1 previous step for history ---
            # Forward process t_prev is noisier, which translates to the past step in reverse process
            t_prev = torch.clamp(t + torch.randint(1, 5, (B,), device=self.config.device), max=self.config.T-1)
            x_t_prev, _ = self.schedule.q_sample(x0, t_prev)
            with torch.no_grad():
                _, b_prev = self.model(x_t_prev, y, t_prev, nl)
            # ----------------------------------------------------------------------

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                # Run the actual step with the simulated history
                x0_pred, _ = self.model(x_t, y, t, nl, history=[b_prev])
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
            x_t, _ = self.schedule.q_sample(x0, t)

            t_prev = torch.clamp(t + torch.randint(1, 5, (B,), device=self.config.device), max=self.config.T-1)
            x_t_prev, _ = self.schedule.q_sample(x0, t_prev)
            _, b_prev = self.model(x_t_prev, y, t_prev, nl)

            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                x0_pred, _ = self.model(x_t, y, t, nl, history=[b_prev])
                loss, metrics = self.criterion(x0_pred, x0)

            val_losses.append(metrics['total'])
            mse_val = F.mse_loss(x0_pred.clamp(0, 1), x0).item()
            psnrs.append(10 * math.log10(1.0 / (mse_val + 1e-8)))

        mean_loss = float(np.mean(val_losses))
        mean_psnr = float(np.mean(psnrs))
        
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

            ckpt = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_val_loss,
            }
            torch.save(ckpt, self.config.checkpoint_dir / 'latest_model.pth')

            if epoch % self.config.eval_freq == 0:
                val_loss, val_psnr = self.evaluate(epoch)
                self.log_epoch(epoch, train_loss, val_loss, val_psnr)
                self.plot_curves()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_psnr = val_psnr
                    ckpt['best_loss'] = self.best_val_loss
                    torch.save(ckpt, self.config.checkpoint_dir / 'best_model.pth')
                    self.logger.info(f"Saved best model at epoch {epoch}")

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
    config.data_path = "noisy_patches"
    config.save_dir  = "experiments/palette_cherenkov_tam"
    # ============================================================

    config.checkpoint_dir  = Path(config.save_dir) / 'checkpoints'
    config.samples_dir     = Path(config.save_dir) / 'samples'
    config.logs_dir        = Path(config.save_dir) / 'logs'
    config.tensorboard_dir = Path(config.save_dir) / 'tensorboard'

    if not os.path.exists(config.data_path):
        print(f"ERROR: data path not found: {config.data_path}")
        print("Update config.data_path to your .npy patch directory.")
        sys.exit(1)

    trainer = Trainer(config)

    # Note: If resuming from the previous checkpoint, strict=False inside load_checkpoint
    # ensures that the brand-new TAM layers are safely initialized while preserving your previous weights!
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