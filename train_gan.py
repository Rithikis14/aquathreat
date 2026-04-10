# train_gan.py
"""
AquaThreat — GAN Training Script (FUnIE-GAN style)

Trains a Generator to transform degraded underwater images into enhanced ones.
Uses a PatchGAN Discriminator that judges 70×70 patches — better at texture
quality than a single "real/fake" score for the whole image.

Supports two training modes:
  PAIRED   — you have (degraded_image, clean_image) pairs
             Loss: Adversarial + L1 pixel reconstruction
  UNPAIRED — you only have separate bags of clear and murky images (CycleGAN style)
             Loss: Adversarial + Cycle consistency (we implement paired here,
                   unpaired is left as an extension note)

Folder layout expected for PAIRED training:
    data/gan/
    ├── train/
    │   ├── degraded/   ← murky/degraded images
    │   └── clean/      ← corresponding enhanced/clear images (same filename)
    └── val/
        ├── degraded/
        └── clean/

Usage:
    python train_gan.py --epochs 100 --batch 8
    python train_gan.py --resume runs/gan/generator_epoch_50.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

from underwater_gan_enhance import UNetGenerator, simulate_underwater


# ─────────────────────────────────────────────────────────────────────────────
#  PATCHGAN DISCRIMINATOR
# ─────────────────────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator.

    Instead of outputting a single real/fake score for the whole image,
    it outputs a grid of scores — each score covers a 70×70 pixel patch.
    This makes it much better at judging local texture quality (important
    for checking if murky regions are properly enhanced).

    Input:  concatenated [degraded, enhanced] → 6 channels
    Output: [B, 1, H/16, W/16] grid of real/fake patch scores
    """

    def __init__(self, in_channels=6):
        super().__init__()

        def disc_block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_channels, 64,  stride=2, norm=False),  # no norm on first
            disc_block(64,  128, stride=2),
            disc_block(128, 256, stride=2),
            disc_block(256, 512, stride=1),   # stride 1 in last conv block
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # patch scores
        )

    def forward(self, degraded, enhanced):
        # Concatenate condition (degraded) with target (enhanced) along channels
        x = torch.cat([degraded, enhanced], dim=1)
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class UnderwaterDataset(Dataset):
    """
    Paired underwater image dataset.

    If clean/ folder doesn't exist, it auto-generates synthetic pairs by
    applying simulate_underwater() to the clean images — useful for demo/testing.
    """

    def __init__(self, root: str, split: str = "train", size: int = 256,
                 auto_degrade: bool = False):
        self.size        = size
        self.auto_degrade = auto_degrade

        root = Path(root) / split
        degraded_dir = root / "degraded"
        clean_dir    = root / "clean"

        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean image directory not found: {clean_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.clean_paths = sorted([
            p for p in clean_dir.iterdir() if p.suffix.lower() in exts
        ])

        if auto_degrade or not degraded_dir.exists():
            print(f"[Dataset] Auto-generating degraded pairs from {len(self.clean_paths)} clean images.")
            self.degraded_paths = None   # will degrade on-the-fly
        else:
            self.degraded_paths = sorted([
                p for p in degraded_dir.iterdir() if p.suffix.lower() in exts
            ])
            assert len(self.degraded_paths) == len(self.clean_paths), (
                f"Mismatch: {len(self.degraded_paths)} degraded vs "
                f"{len(self.clean_paths)} clean images"
            )

        print(f"[Dataset] {split}: {len(self.clean_paths)} pairs")

    def __len__(self):
        return len(self.clean_paths)

    def _load_rgb(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise IOError(f"Cannot read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img_rgb, (self.size, self.size))
        t   = torch.from_numpy(img).float() / 127.5 - 1.0   # [-1, 1]
        return t.permute(2, 0, 1)                             # [C, H, W]

    def __getitem__(self, idx):
        clean_rgb = self._load_rgb(self.clean_paths[idx])

        if self.degraded_paths is None:
            depth = np.random.uniform(0.4, 0.85)
            degraded_rgb = simulate_underwater(clean_rgb, depth=depth)
        else:
            degraded_rgb = self._load_rgb(self.degraded_paths[idx])

        return {
            "degraded": self._to_tensor(degraded_rgb),
            "clean":    self._to_tensor(clean_rgb),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  LOSSES
# ─────────────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Least-Squares GAN loss (LSGAN).
    More stable than BCE — doesn't saturate and has better gradients
    when discriminator is confident (which it often is at start of training).
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(generator, discriminator, dataloader,
                    opt_G, opt_D, gan_loss, l1_loss,
                    device, lambda_l1=100.0):
    """
    One full epoch of GAN training.

    Generator loss  = adversarial (fool D) + L1 pixel reconstruction
    Discriminator loss = real pair score + fake pair score

    lambda_l1=100: heavily weights pixel reconstruction so the generator
    doesn't just produce any plausible-looking image — it must match the
    target colour/detail closely.
    """
    generator.train()
    discriminator.train()

    metrics = {"G_adv": 0, "G_l1": 0, "G_total": 0, "D": 0}
    n = 0

    for batch in dataloader:
        degraded = batch["degraded"].to(device)
        clean    = batch["clean"].to(device)
        bs       = degraded.size(0)

        # ── Train Discriminator ──────────────────────────────────────────────
        opt_D.zero_grad()

        enhanced     = generator(degraded).detach()  # detach so grad doesn't flow to G
        real_score   = discriminator(degraded, clean)
        fake_score   = discriminator(degraded, enhanced)
        d_loss       = 0.5 * (gan_loss(real_score, True) + gan_loss(fake_score, False))
        d_loss.backward()
        opt_D.step()

        # ── Train Generator ──────────────────────────────────────────────────
        opt_G.zero_grad()

        enhanced     = generator(degraded)           # re-run (with grad this time)
        fake_score   = discriminator(degraded, enhanced)
        g_adv        = gan_loss(fake_score, True)    # fool discriminator
        g_l1         = l1_loss(enhanced, clean) * lambda_l1
        g_loss       = g_adv + g_l1
        g_loss.backward()
        opt_G.step()

        metrics["G_adv"]   += g_adv.item()
        metrics["G_l1"]    += (g_l1 / lambda_l1).item()
        metrics["G_total"] += g_loss.item()
        metrics["D"]       += d_loss.item()
        n += 1

    return {k: v / n for k, v in metrics.items()}


def save_sample_grid(generator, val_loader, device, epoch, out_dir):
    """Save a 4-column grid: [degraded | generated | clean] for 4 val samples."""
    generator.eval()
    rows = []
    with torch.no_grad():
        for batch in val_loader:
            degraded = batch["degraded"][:4].to(device)
            clean    = batch["clean"][:4]
            enhanced = generator(degraded).cpu()

            for d, e, c in zip(degraded.cpu(), enhanced, clean):
                def to_img(t):
                    return ((t.permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
                rows.append(np.hstack([to_img(d), to_img(e), to_img(c)]))
            break

    grid = np.vstack(rows[:4])
    out_path = Path(out_dir) / f"samples_epoch_{epoch:03d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[Samples] → {out_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = UnderwaterDataset(args.data, "train", args.size, auto_degrade=args.auto_degrade)
    val_ds   = UnderwaterDataset(args.data, "val",   args.size, auto_degrade=args.auto_degrade)
    if len(train_ds) == 0:
        print("\n[Error] No images found in data/gan/train/clean/")
        print("        Add at least 10 clear images to that folder and re-run.")
        import sys; sys.exit(1)
    if len(val_ds) == 0:
        print("\n[Error] No images found in data/gan/val/clean/")
        print("        Add at least 5 clear images to that folder and re-run.")
        import sys; sys.exit(1)

    # num_workers=0 required on Windows (multiprocessing causes errors otherwise)
    import platform
    nw = 0 if platform.system() == "Windows" else 2
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=nw)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=nw)

    # ── Models ───────────────────────────────────────────────────────────────
    G = UNetGenerator(3, 3, base_features=64).to(device)
    D = PatchGANDiscriminator(in_channels=6).to(device)

    if args.resume and Path(args.resume).exists():
        G.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[Train] Resumed generator from: {args.resume}")

    # ── Optimisers ────────────────────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Linear LR decay in second half of training (standard GAN practice)
    def lr_lambda(epoch):
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / (args.epochs - decay_start)

    sched_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # ── Losses ────────────────────────────────────────────────────────────────
    gan_loss = GANLoss().to(device)
    l1_loss  = nn.L1Loss()

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"G_adv": [], "G_l1": [], "D": []}

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        m = train_one_epoch(G, D, train_dl, opt_G, opt_D,
                            gan_loss, l1_loss, device, lambda_l1=100.0)
        sched_G.step()
        sched_D.step()

        history["G_adv"].append(m["G_adv"])
        history["G_l1"].append(m["G_l1"])
        history["D"].append(m["D"])

        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"G_adv={m['G_adv']:.4f}  G_l1={m['G_l1']:.4f}  "
              f"G_total={m['G_total']:.4f}  D={m['D']:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            pt_path = out_dir / f"generator_epoch_{epoch:03d}.pt"
            torch.save(G.state_dict(), str(pt_path))
            print(f"[Save] Generator → {pt_path}")
            save_sample_grid(G, val_dl, device, epoch, out_dir)

    # ── Plot loss curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["G_adv"], label="G adversarial")
    axes[0].plot(history["G_l1"],  label="G L1 reconstruction")
    axes[0].set_title("Generator Losses"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["D"], label="Discriminator", color="orange")
    axes[1].set_title("Discriminator Loss"); axes[1].legend(); axes[1].grid(True)
    plt.suptitle("GAN Training Loss Curves — AquaThreat")
    plt.savefig(str(out_dir / "loss_curves.png"), dpi=150, bbox_inches="tight")
    print(f"[Plot] Loss curves → {out_dir / 'loss_curves.png'}")

    print(f"\n[Done] Final generator weights → {out_dir}/generator_epoch_{args.epochs:03d}.pt")
    print("       Use with: python underwater_gan_enhance.py --image X.jpg "
          f"--weights {out_dir}/generator_epoch_{args.epochs:03d}.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Train FUnIE-GAN style underwater enhancer")
    p.add_argument("--data",         default="data/gan",
                   help="Root of dataset (expects train/ and val/ subdirs)")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch",        type=int, default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--size",         type=int, default=256,
                   help="Training image resolution (default 256)")
    p.add_argument("--save-every",   type=int, default=10)
    p.add_argument("--out-dir",      default="runs/gan")
    p.add_argument("--resume",       default=None)
    p.add_argument("--auto-degrade", action="store_true",
                   help="Auto-generate degraded images from clean ones (no real pairs needed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)