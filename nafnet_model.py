"""
NAFNet (Nonlinear Activation Free Network) for Image Denoising

Based on "Simple Baselines for Image Restoration" (Chen et al., ECCV 2022)
Key idea: Replace nonlinear activations with SimpleGate (channel split + multiply)
         and use Simplified Channel Attention instead of full self-attention.

This achieves SOTA denoising with lower compute than transformer-based methods.

FIXES APPLIED:
- Removed global residual connection (was causing artifacts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


class SimpleGate(nn.Module):
    """
    SimpleGate: Split channels in half, multiply element-wise.
    Replaces nonlinear activations (ReLU, GELU, etc.)
    f(x1, x2) = x1 * x2 where [x1, x2] = split(x)
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA).
    Uses global average pooling + 1x1 conv instead of full self-attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x * self.sca(x)


class NAFBlock(nn.Module):
    """
    NAFNet Block: The core building block.
    
    Structure:
    - LayerNorm -> 1x1 Conv (expand) -> 3x3 DWConv -> SimpleGate -> SCA -> 1x1 Conv
    - LayerNorm -> 1x1 Conv (expand) -> SimpleGate -> 1x1 Conv
    - Both branches have skip connections and learnable scaling (beta, gamma)
    """
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        # Spatial mixing branch
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1,
                               groups=dw_channels)  # Depthwise conv
        self.sg1 = SimpleGate()  # dw_channels -> dw_channels // 2
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        # Channel mixing branch (FFN)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()  # ffn_channels -> ffn_channels // 2
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        # Learnable scaling parameters
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x):
        # Spatial mixing
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.sg1(h)
        h = self.sca(h)
        h = self.conv3(h)
        h = self.dropout1(h)
        x = x + h * self.beta

        # Channel mixing (FFN)
        h = self.norm2(x)
        h = self.conv4(h)
        h = self.sg2(h)
        h = self.conv5(h)
        h = self.dropout2(h)
        x = x + h * self.gamma

        return x


class NAFNet(nn.Module):
    """
    NAFNet: Nonlinear Activation Free Network for Image Restoration.
    
    U-Net style encoder-decoder with NAFBlocks.
    
    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (1 for grayscale)
        width: Base channel width (default: 32)
        enc_blk_nums: Number of NAFBlocks at each encoder level
        dec_blk_nums: Number of NAFBlocks at each decoder level
        middle_blk_num: Number of NAFBlocks in bottleneck
        dw_expand: Depthwise expansion factor
        ffn_expand: FFN expansion factor
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=32,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        middle_blk_num=12,
        dw_expand=2,
        ffn_expand=2
    ):
        super().__init__()

        # Initial feature extraction
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for num_blocks in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels, dw_expand, ffn_expand) for _ in range(num_blocks)])
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        # Bottleneck
        self.middle = nn.Sequential(
            *[NAFBlock(channels, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
        )

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1),
                    nn.PixelShuffle(2)  # channels*2 -> channels//2 with 2x spatial
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels, dw_expand, ffn_expand) for _ in range(num_blocks)])
            )

        # Output projection
        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # REMOVED: residual = x  (global residual connection removed)

        x = self.intro(x)

        # Encoder
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle(x)

        # Decoder
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            # Handle size mismatch from odd spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)

        # FIXED: Direct output without global residual
        return torch.clamp(x, 0, 1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test NAFNet
    model = NAFNet(
        in_channels=1, out_channels=1, width=32,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 4, 8],  # FIXED: Symmetric with encoder
        middle_blk_num=12
    ).to(device)
    
    x = torch.randn(2, 1, 128, 128).to(device)
    out = model(x)
    
    print(f"NAFNet:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {count_parameters(model):,}")