# models/cbam.py
"""
CBAM — Convolutional Block Attention Module.

Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)

CBAM applies TWO sequential attention gates:
  1. Channel Attention Gate  — "which channels matter?" (R vs G vs B)
  2. Spatial Attention Gate  — "which pixels/regions matter?" (murky patches)

This is the upgraded replacement for the plain ChannelAttention used before.
For underwater images, the spatial gate is critical: it lets the model
identify and prioritise the turbid (blurry/hazy) patches over clear patches
before the image is passed to the YOLO backbone.

Architecture:
    Input feature map  [B, C, H, W]
        ↓
    Channel Attention  [B, C, 1, 1]  — squeeze global spatial info, excite channels
        ↓  (element-wise multiply)
    Spatially-refined  [B, C, H, W]
        ↓
    Spatial Attention  [B, 1, H, W]  — squeeze channels, highlight important pixels
        ↓  (element-wise multiply)
    CBAM output        [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionGate(nn.Module):
    """
    Channel attention gate (the 'what' — which feature channels to emphasise).

    Uses BOTH average-pool AND max-pool paths, then adds them before sigmoid.
    The dual-pooling is the key difference from a plain Squeeze-and-Excitation block:
    avg-pool captures overall channel statistics, max-pool captures the most
    discriminative activations.

    Args:
        channels:  Number of input channels (3 for RGB input, or C for feature maps)
        reduction: Bottleneck reduction ratio inside the MLP (default 8 → 1/8 hidden)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # squeeze spatial → [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)   # squeeze spatial → [B, C, 1, 1]

        # Shared MLP (applied to both pooled vectors)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape

        avg_out = self.mlp(self.avg_pool(x))   # [B, C]
        max_out = self.mlp(self.max_pool(x))   # [B, C]

        # Sum both paths, apply sigmoid, reshape to [B, C, 1, 1] for broadcasting
        gate = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * gate                        # [B, C, H, W]


class SpatialAttentionGate(nn.Module):
    """
    Spatial attention gate (the 'where' — which spatial locations to emphasise).

    Compresses the channel dimension into 2 descriptors (avg + max across channels),
    concatenates them → 2-channel map → 7×7 conv → sigmoid gate.
    The large 7×7 kernel is intentional: it captures a wide spatial context,
    which is important for detecting partially occluded mines in murky water.

    Args:
        kernel_size: Conv kernel for the spatial gate (7 is the CBAM default)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same-padding"
        padding = kernel_size // 2

        # 2 input channels (avg-pool + max-pool across C), 1 output channel
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool across channels → [B, 1, H, W] each
        avg_map = torch.mean(x, dim=1, keepdim=True)       # [B, 1, H, W]
        max_map, _ = torch.max(x, dim=1, keepdim=True)     # [B, 1, H, W]

        # Concatenate → [B, 2, H, W] → conv → sigmoid gate
        pooled = torch.cat([avg_map, max_map], dim=1)       # [B, 2, H, W]
        gate   = self.sigmoid(self.conv(pooled))            # [B, 1, H, W]
        return x * gate                                     # [B, C, H, W]


class CBAM(nn.Module):
    """
    Full CBAM block: Channel Attention → Spatial Attention (sequential).

    Drop-in replacement for the old ChannelAttention class.
    Can be used inside LearnableColorCorrection (channels=3) or
    inside any deeper feature map in the backbone (channels=64, 128, etc.)

    Args:
        channels:        Number of input channels
        reduction:       Channel attention bottleneck ratio (default 8)
        spatial_kernel:  Spatial attention conv kernel size (default 7)
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channel_gate  = ChannelAttentionGate(channels, reduction)
        self.spatial_gate  = SpatialAttentionGate(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_gate(x)    # refine WHICH channels
        x = self.spatial_gate(x)    # refine WHERE in those channels
        return x


# ── Quick sanity test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test on RGB input (3-channel, as used in LearnableColorCorrection)
    cbam_rgb = CBAM(channels=3, reduction=1)   # reduction=1 since only 3 channels
    dummy    = torch.randn(2, 3, 640, 640)
    out      = cbam_rgb(dummy)
    assert out.shape == dummy.shape
    params   = sum(p.numel() for p in cbam_rgb.parameters())
    print(f"[CBAM-RGB]   Output: {out.shape} | Params: {params:,}")

    # Test on a typical backbone feature map (256-channel)
    cbam_feat = CBAM(channels=256, reduction=8)
    dummy2    = torch.randn(2, 256, 80, 80)
    out2      = cbam_feat(dummy2)
    assert out2.shape == dummy2.shape
    params2   = sum(p.numel() for p in cbam_feat.parameters())
    print(f"[CBAM-256]   Output: {out2.shape} | Params: {params2:,}")
    print("[CBAM] All tests passed.")
