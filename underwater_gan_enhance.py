# underwater_gan_enhance.py
"""
Simple Underwater Image Enhancement using a GAN-style Generator.

This is a STANDALONE demo — no dataset, no training loop, no discriminator needed
to see results. It shows the full pipeline:

  Read image → Apply underwater degradation → Run through Generator → Compare output

Two modes:
  1. DEMO mode    — works on any image you give it (degrades it to simulate
                    underwater conditions, then enhances it back).
  2. REAL mode    — if you already HAVE an underwater image, just pass it directly.

The Generator architecture is inspired by FUnIE-GAN:
  - Encoder: downsample with conv blocks (extract features)
  - Bottleneck: deepest feature representation
  - Decoder: upsample with skip connections (reconstruct enhanced image)
  - This U-Net style architecture with skip connections is key for preserving
    structural details (edges of mines) while correcting colour/contrast.

Run:
    pip install torch torchvision opencv-python pillow matplotlib

    python underwater_gan_enhance.py --image your_image.jpg
    python underwater_gan_enhance.py --image your_image.jpg --real-underwater
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATOR ARCHITECTURE  (FUnIE-GAN / CycleGAN style U-Net)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Encoder block: Conv → InstanceNorm → LeakyReLU."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """Decoder block: ConvTranspose → InstanceNorm → ReLU (+ optional dropout)."""
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net style Generator — the core of FUnIE-GAN / CycleGAN.

    Encoder path: progressively downsamples and extracts features.
    Decoder path: progressively upsamples and reconstructs the image.
    Skip connections: concatenate encoder feature maps into decoder at each level.
                      This is what preserves fine structural details (mine edges,
                      cables, seabed texture) while the global colour is corrected.

    Input:  [B, 3, H, W]  — degraded underwater image (normalised -1 to 1)
    Output: [B, 3, H, W]  — enhanced image (normalised -1 to 1)
    """

    def __init__(self, in_channels=3, out_channels=3, base_features=64):
        super().__init__()

        f = base_features  # shorthand

        # ── Encoder (downsampling) ──────────────────────────────────────────
        # No norm on first layer (standard in GAN generators)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )                                   # → [B, 64,  H/2,  W/2]
        self.enc2 = ConvBlock(f,    f*2)    # → [B, 128, H/4,  W/4]
        self.enc3 = ConvBlock(f*2,  f*4)    # → [B, 256, H/8,  W/8]
        self.enc4 = ConvBlock(f*4,  f*8)    # → [B, 512, H/16, W/16]

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f*8, f*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )                                   # → [B, 512, H/32, W/32]

        # ── Decoder (upsampling with skip connections) ──────────────────────
        # in_ch is doubled at each level because of concatenated skip connection
        self.dec1 = DeconvBlock(f*8,   f*8, dropout=True)   # → [B, 512, H/16, W/16]
        self.dec2 = DeconvBlock(f*8*2, f*4, dropout=True)   # → [B, 256, H/8,  W/8]
        self.dec3 = DeconvBlock(f*4*2, f*2)                 # → [B, 128, H/4,  W/4]
        self.dec4 = DeconvBlock(f*2*2, f)                   # → [B, 64,  H/2,  W/2]

        # ── Output layer ────────────────────────────────────────────────────
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(f*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections (cat along channel dim)
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))

        out = self.output_conv(torch.cat([d4, e1], dim=1))
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  UNDERWATER DEGRADATION  (to simulate murky water on any input image)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_underwater(image_rgb: np.ndarray, depth: float = 0.6) -> np.ndarray:
    """
    Apply realistic underwater colour degradation to an RGB image.

    Physics modelled:
      - Red channel attenuates fastest with depth
      - Blue/green channels persist longer
      - Haze (scattering) increases with depth
      - Contrast drops, slight blur from particles

    Args:
        image_rgb: uint8 RGB numpy array [H, W, 3]
        depth:     0.0 (surface) to 1.0 (very deep / murky)

    Returns:
        Degraded uint8 RGB numpy array
    """
    img = image_rgb.astype(np.float32) / 255.0

    # Channel attenuation (Beer-Lambert law approximation)
    red_atten   = np.exp(-depth * 3.5)   # red absorbed fastest
    green_atten = np.exp(-depth * 1.2)
    blue_atten  = np.exp(-depth * 0.5)   # blue survives deepest

    img[:, :, 0] *= red_atten
    img[:, :, 1] *= green_atten
    img[:, :, 2] *= blue_atten

    # Add blue/green haze (backscatter from particles)
    haze_intensity = depth * 0.25
    img[:, :, 1] = np.clip(img[:, :, 1] + haze_intensity * 0.4, 0, 1)  # green haze
    img[:, :, 2] = np.clip(img[:, :, 2] + haze_intensity * 0.7, 0, 1)  # blue haze

    # Reduce contrast (contrast compression at depth)
    img = 0.5 + (img - 0.5) * (1 - depth * 0.4)

    # Add turbidity blur (suspended particles scatter light)
    blur_radius = max(1, int(depth * 5)) | 1   # must be odd
    img_bgr = cv2.GaussianBlur(
        (img * 255).astype(np.uint8),
        (blur_radius, blur_radius), 0
    )
    img = img_bgr.astype(np.float32) / 255.0

    # Add noise (sensor + scatter noise)
    noise = np.random.normal(0, depth * 0.03, img.shape).astype(np.float32)
    img   = np.clip(img + noise, 0, 1)

    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_rgb: np.ndarray, size: int = 256) -> torch.Tensor:
    """Resize → normalise to [-1, 1] → add batch dim."""
    img = cv2.resize(image_rgb, (size, size))
    tensor = torch.from_numpy(img).float() / 127.5 - 1.0   # [0,255] → [-1,1]
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)           # [H,W,C] → [1,C,H,W]
    return tensor


def postprocess(tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
    """Remove batch dim → denormalise to [0,255] → resize to original."""
    img = tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.resize(img, (original_size[1], original_size[0]))


def run_generator(generator: UNetGenerator,
                  image_rgb: np.ndarray,
                  size: int = 256) -> np.ndarray:
    """Run the generator on a single RGB image. Returns enhanced RGB image."""
    generator.eval()
    with torch.no_grad():
        inp  = preprocess(image_rgb, size)
        out  = generator(inp)
    return postprocess(out, image_rgb.shape[:2])


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS  (simple, no-reference quality measures)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray,
                    degraded: np.ndarray,
                    enhanced: np.ndarray) -> dict:
    """
    Compute simple image quality metrics to quantify the improvement.

    UICM-style colour cast: lower red/blue imbalance = better colour.
    Contrast (std of luminance): higher = more detail visible.
    Sharpness (Laplacian variance): higher = sharper edges.
    """

    def luminance(img):
        return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

    def sharpness(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())

    def contrast(img):
        return float(luminance(img.astype(np.float32)).std())

    def colour_balance(img):
        """Ratio of red to blue — closer to 1.0 = more balanced."""
        r = img[:,:,0].mean() + 1e-6
        b = img[:,:,2].mean() + 1e-6
        return float(r / b)

    return {
        "contrast":       {"degraded": round(contrast(degraded), 2),
                           "enhanced": round(contrast(enhanced), 2)},
        "sharpness":      {"degraded": round(sharpness(degraded), 1),
                           "enhanced": round(sharpness(enhanced), 1)},
        "colour_balance": {"degraded": round(colour_balance(degraded), 3),
                           "enhanced": round(colour_balance(enhanced), 3),
                           "original": round(colour_balance(original), 3)},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison(original, degraded, enhanced, metrics, out_path):
    """
    Save a side-by-side comparison figure:
      [Original] [Degraded / Underwater] [GAN Enhanced]
    with a metrics bar underneath.
    """
    fig = plt.figure(figsize=(15, 8), facecolor="#0d1117")
    fig.suptitle("Underwater GAN Enhancement — AquaThreat Demo",
                 fontsize=16, color="white", fontweight="bold", y=0.97)

    gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1], hspace=0.35, wspace=0.05)

    titles  = ["Original Image", "Simulated Underwater\n(Degraded Input)", "GAN Enhanced\n(Generator Output)"]
    images  = [original, degraded, enhanced]
    borders = ["#4CAF50", "#f44336", "#2196F3"]

    for i, (title, img, color) in enumerate(zip(titles, images, borders)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    # ── Metrics bar ─────────────────────────────────────────────────────────
    ax_m = fig.add_subplot(gs[1, :])
    ax_m.set_facecolor("#161b22")
    ax_m.axis("off")

    m = metrics
    lines = [
        f"Contrast     │  Degraded: {m['contrast']['degraded']:6.2f}   →   "
        f"Enhanced: {m['contrast']['enhanced']:6.2f}   "
        f"({'▲ +' if m['contrast']['enhanced'] > m['contrast']['degraded'] else '▼ '}"
        f"{abs(m['contrast']['enhanced'] - m['contrast']['degraded']):.2f})",

        f"Sharpness    │  Degraded: {m['sharpness']['degraded']:8.1f}   →   "
        f"Enhanced: {m['sharpness']['enhanced']:8.1f}   "
        f"({'▲ +' if m['sharpness']['enhanced'] > m['sharpness']['degraded'] else '▼ '}"
        f"{abs(m['sharpness']['enhanced'] - m['sharpness']['degraded']):.1f})",

        f"R/B Balance  │  Original: {m['colour_balance']['original']:.3f}   │   "
        f"Degraded: {m['colour_balance']['degraded']:.3f}   →   "
        f"Enhanced: {m['colour_balance']['enhanced']:.3f}   "
        f"(target ≈ {m['colour_balance']['original']:.3f})",
    ]

    for j, line in enumerate(lines):
        ax_m.text(0.02, 0.75 - j * 0.32, line,
                  transform=ax_m.transAxes, fontsize=10,
                  color="#e6edf3", fontfamily="monospace",
                  verticalalignment="top")

    note = ("NOTE: This demo uses an UNTRAINED generator (random weights).\n"
            "After training on paired underwater data, the output will be "
            "a properly colour-corrected, high-contrast enhanced image.")
    ax_m.text(0.98, 0.05, note,
              transform=ax_m.transAxes, fontsize=8,
              color="#8b949e", ha="right", va="bottom", style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Output] Comparison saved → {out_path}")


def save_side_by_side(degraded, enhanced, out_path):
    """Save a simple 2-panel PNG: degraded | enhanced."""
    h = max(degraded.shape[0], enhanced.shape[0])
    d_res = cv2.resize(degraded, (degraded.shape[1], h))
    e_res = cv2.resize(enhanced, (enhanced.shape[1], h))
    combined = np.hstack([d_res, e_res])
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[Output] Side-by-side PNG saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GAN-based underwater image enhancement demo"
    )
    p.add_argument("--image",           required=True,
                   help="Path to input image (any format OpenCV supports)")
    p.add_argument("--real-underwater", action="store_true",
                   help="Input is already an underwater image — skip degradation")
    p.add_argument("--depth",           type=float, default=0.65,
                   help="Simulated water depth 0.0–1.0 (default 0.65)")
    p.add_argument("--size",            type=int, default=256,
                   help="Internal generator resolution (default 256; use 128 if slow)")
    p.add_argument("--weights",         default=None,
                   help="Optional path to trained generator weights (.pt)")
    p.add_argument("--out-dir",         default=".",
                   help="Directory to save output images (default: current dir)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load image ─────────────────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[Error] Image not found: {img_path}")
        sys.exit(1)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[Error] Could not read image: {img_path}")
        sys.exit(1)

    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[Input]  {img_path.name}  |  size: {original_rgb.shape[1]}×{original_rgb.shape[0]}")

    # ── Determine the degraded / input image ───────────────────────────────
    if args.real_underwater:
        print(f"[Mode]   Real underwater image — no degradation applied.")
        degraded_rgb = original_rgb.copy()
    else:
        print(f"[Mode]   Demo — simulating underwater degradation (depth={args.depth})")
        degraded_rgb = simulate_underwater(original_rgb, depth=args.depth)

    # ── Build generator ────────────────────────────────────────────────────
    print(f"[Model]  Building UNetGenerator (base_features=64, size={args.size})")
    generator = UNetGenerator(in_channels=3, out_channels=3, base_features=64)

    if args.weights and Path(args.weights).exists():
        print(f"[Model]  Loading trained weights from: {args.weights}")
        generator.load_state_dict(torch.load(args.weights, map_location="cpu"))
    else:
        print("[Model]  No weights provided — using random initialisation.")
        print("         (The output will look noisy; train the model for real results.)")

    total_params = sum(p.numel() for p in generator.parameters())
    print(f"[Model]  Total parameters: {total_params:,}")

    # ── Run enhancement ────────────────────────────────────────────────────
    print(f"[Enhance] Running generator...")
    enhanced_rgb = run_generator(generator, degraded_rgb, size=args.size)

    # ── Compute metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(original_rgb, degraded_rgb, enhanced_rgb)

    print("\n[Metrics]")
    print(f"  Contrast      degraded={metrics['contrast']['degraded']:.2f}   "
          f"→  enhanced={metrics['contrast']['enhanced']:.2f}")
    print(f"  Sharpness     degraded={metrics['sharpness']['degraded']:.1f}   "
          f"→  enhanced={metrics['sharpness']['enhanced']:.1f}")
    print(f"  R/B Balance   original={metrics['colour_balance']['original']:.3f}   "
          f"degraded={metrics['colour_balance']['degraded']:.3f}   "
          f"→  enhanced={metrics['colour_balance']['enhanced']:.3f}")

    # ── Save outputs ────────────────────────────────────────────────────────
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = img_path.stem

    # 1. Full comparison figure (3 panels + metrics)
    save_comparison(
        original_rgb, degraded_rgb, enhanced_rgb, metrics,
        out_path=str(out_dir / f"{stem}_comparison.png"),
    )

    # 2. Simple side-by-side PNG (degraded | enhanced)
    save_side_by_side(
        degraded_rgb, enhanced_rgb,
        out_path=str(out_dir / f"{stem}_side_by_side.png"),
    )

    # 3. Enhanced image alone
    cv2.imwrite(
        str(out_dir / f"{stem}_enhanced.png"),
        cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    )
    print(f"[Output] Enhanced image saved → {out_dir / f'{stem}_enhanced.png'}")

    print("\n[Done] Output files:")
    print(f"   {stem}_comparison.png    ← 3-panel comparison + metrics")
    print(f"   {stem}_side_by_side.png  ← quick degraded | enhanced view")
    print(f"   {stem}_enhanced.png      ← enhanced image only")
    print("\nTo use trained weights (after training on real underwater data):")
    print("   python underwater_gan_enhance.py --image img.jpg --weights generator.pt")


if __name__ == "__main__":
    main()
