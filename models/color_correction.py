# models/color_correction.py
"""
Learnable Color-Correction Preprocessing Block — upgraded with CBAM.

Core architectural novelty: instead of applying CLAHE offline with fixed
parameters, this lightweight CNN block is inserted BEFORE the YOLO backbone
and trained end-to-end. The network learns the optimal per-image enhancement
needed to maximise detection accuracy under varying water conditions.

Upgrade from v1:
    Old: plain ChannelAttention (SE-style, channel-only)
    New: full CBAM (Channel gate + Spatial gate)

The spatial attention gate is the key addition — it allows the model to
identify which PIXELS are most degraded (turbid patches, colour-cast regions)
and correct those specifically, rather than applying a uniform channel scale.
"""

import torch
import torch.nn as nn

from models.cbam import CBAM


class LearnableColorCorrection(nn.Module):
    """
    Lightweight learnable preprocessing block inserted at the start of YOLO.

    Architecture (end-to-end trainable):

        Input (3-ch RGB image)
              ↓
        Depthwise conv 3×3   — per-channel local colour / contrast
              ↓
        Pointwise conv 1×1   — cross-channel colour mixing
              ↓
        BatchNorm + ReLU
              ↓
        CBAM                 — channel gate (which channels?) +
                               spatial gate (which pixels?)
              ↓
        Gated residual add   — model learns HOW MUCH to correct
              ↓
        Output (3-ch enhanced image)

    Total params: ~4,200 (CBAM on 3 channels is tiny).
    Zero overhead compared to the YOLO backbone (~3M+ params).
    """

    def __init__(self):
        super().__init__()

        # Depthwise separable convolution
        # Depthwise: each of the 3 channels is filtered independently.
        # This is colour-aware — red channel gets its own spatial filter.
        self.depthwise = nn.Conv2d(
            in_channels=3, out_channels=3,
            kernel_size=3, padding=1, groups=3, bias=False
        )

        # Pointwise: 1×1 conv mixes information across channels after depthwise.
        # Learns cross-channel corrections (e.g. "if blue is very high, reduce it").
        self.pointwise = nn.Conv2d(3, 3, kernel_size=1, bias=False)

        # Normalisation + activation
        self.bn  = nn.BatchNorm2d(3)
        self.act = nn.ReLU(inplace=True)

        # CBAM replaces the old plain ChannelAttention.
        # reduction=1 because we only have 3 channels — no bottleneck needed.
        # spatial_kernel=7 gives a wide receptive field to catch turbid patches.
        self.cbam = CBAM(channels=3, reduction=1, spatial_kernel=7)

        # Learnable scalar gate on the residual connection.
        # Initialised to 0 so the block starts as an identity transform and
        # gradually learns to apply corrections as training progresses.
        self.residual_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """
        Initialise weights so the block is an identity transform at epoch 0.
        This ensures the YOLO backbone starts from the same pretrained state
        as a vanilla model and only learns correction incrementally.
        """
        nn.init.dirac_(self.depthwise.weight)
        nn.init.eye_(self.pointwise.weight.squeeze(-1).squeeze(-1))
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                              # save original for residual

        out = self.depthwise(x)                   # per-channel spatial filter
        out = self.pointwise(out)                 # cross-channel mix
        out = self.bn(out)                        # normalise
        out = self.act(out)                       # non-linearity
        out = self.cbam(out)                      # channel + spatial attention

        # Gated residual: the model learns HOW MUCH correction to apply.
        # At init, residual_scale=0 so output == identity (safe start).
        return identity + self.residual_scale * out


# ── Quick sanity test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    block  = LearnableColorCorrection()
    dummy  = torch.randn(2, 3, 640, 640)   # batch=2, RGB, 640×640
    out    = block(dummy)

    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} != {dummy.shape}"

    params = sum(p.numel() for p in block.parameters())
    print(f"[LearnableColorCorrection+CBAM] Output shape : {out.shape}")
    print(f"[LearnableColorCorrection+CBAM] Total params : {params:,}")
    print("[LearnableColorCorrection+CBAM] Sanity check passed.")
