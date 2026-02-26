"""
Generate synthetic image pool from a trained cWGAN-GP generator.
Usage:
    python scripts/generate_synth.py --ckpt runs/gan/checkpoints/ckpt_epoch0100.pt \
                                      --n_per_class 500 \
                                      --out_dir data_synth
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

# allow running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from config import resolve_device
from models.gan import Generator

CLASS_NAMES = ["apple", "banana", "orange"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="data_synth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(resolve_device(cfg.device))
    num_classes = len(CLASS_NAMES)

    G = Generator(z_dim=cfg.z_dim, num_classes=num_classes).to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)

    state_dict = ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt
    G.load_state_dict(state_dict)
    G.eval()

    torch.manual_seed(args.seed)
    out_root = Path(args.out_dir)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = out_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        while generated < args.n_per_class:
            bs = min(args.batch_size, args.n_per_class - generated)
            z = torch.randn(bs, cfg.z_dim, device=device)
            y = torch.full((bs,), cls_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                imgs = G(z, y)

            for i in range(bs):
                save_image(
                    imgs[i],
                    cls_dir / f"{cls_name}_synth_{generated + i:05d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
            generated += bs

        print(f"{cls_name}: {generated} images saved to {cls_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
