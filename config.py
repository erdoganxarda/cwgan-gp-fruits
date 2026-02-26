from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Config:
    # Paths
    data_root: Path = Path("data_final")  # expects: data_final/train, data_final/val, data_final/test
    out_root: Path = Path("runs")

    # Image
    img_size: int = 64
    channels: int = 3

    # Repro
    seed: int = 42

    # GAN (cWGAN-GP)
    z_dim: int = 128
    gan_batch: int = 64
    gan_epochs: int = 100
    gan_lr_g: float = 1e-4       # generator lr
    gan_lr_d: float = 2e-4       # critic lr (reduced for more stable critic)
    n_critic: int = 3            # critic updates per G update
    gp_lambda: float = 10.0      # gradient penalty coefficient
    sample_every: int = 5        # save sample grid every N epochs
    ckpt_every: int = 20         # save checkpoint every N epochs
    fid_every: int = 5           # compute FID every N epochs (set <= 0 to disable)
    fid_n_samples: int = 512     # real/fake samples used for FID
    fid_eval_split: str = "val"  # use val split for stable FID

    # Classifier
    clf_batch: int = 64
    clf_epochs: int = 30
    clf_lr: float = 1e-3
    # M4 has 10 cores; cap raised to 8 so data loading saturates the performance cores
    num_workers: int = min(8, max(1, (os.cpu_count() or 4) // 2))
    persistent_workers: bool = True   # keep workers alive across epochs (no respawn cost)
    prefetch_factor: int = 2          # batches queued ahead per worker

    # Compute device
    # Use "auto" to pick the best available backend: CUDA -> MPS -> CPU.
    device: str = "auto"
    pin_memory: bool = False  # mainly useful for CUDA, not MPS


def resolve_device(requested: str) -> str:
    """
    Resolve a requested device string into an available torch device name.

    Supported values:
      - "auto" (default): cuda -> mps -> cpu
      - "cuda", "cuda:0", ...
      - "mps"
      - "cpu"
    """
    req = (requested or "auto").lower()
    try:
        import torch
    except Exception:
        return "cpu" if req == "auto" else requested

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if req.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"

    if req in {"mps", "mps:0"}:
        return "mps" if has_mps else "cpu"

    if req == "cpu":
        return "cpu"

    # allow passing through any other torch device string (best-effort)
    return requested
