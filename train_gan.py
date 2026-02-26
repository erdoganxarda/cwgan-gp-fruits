"""
cWGAN-GP Training Loop
----------------------
- WGAN loss + gradient penalty (λ=10)
- n_critic from Config
- Adam(β1=0.0, β2=0.9) for both G and D (TTUR: different lr okay)
- Saves checkpoints + sample grids every N epochs
- FID evaluation every fid_every epochs (default split: val, no augmentation)
"""

import json
import random
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from torchvision.models import inception_v3

from config import Config, resolve_device
from models.gan import Generator, ProjectionCritic

# ------------------------------------------------------------------ #
#  Gradient penalty                                                    #
# ------------------------------------------------------------------ #

def gradient_penalty(critic, real, fake, labels, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp, labels)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ------------------------------------------------------------------ #
#  FID                                                                 #
# ------------------------------------------------------------------ #

@torch.no_grad()
def get_inception_features(images, model, device, batch_size=64):
    """Extract pool3 features from InceptionV3 for a tensor of images."""
    # inception expects 299x299, RGB, normalized to ImageNet stats
    up = torch.nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        # images come in [-1,1], rescale to [0,1] then ImageNet normalize (vectorized)
        batch = (batch + 1) / 2
        batch = (batch - mean) / std
        batch = up(batch)
        f = model(batch)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


def calc_fid(mu1, sigma1, mu2, sigma2):
    """Numpy FID from precomputed statistics."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(G, real_loader, num_classes, cfg, inception_model, device):
    """Compute FID between real images and generated images."""
    # collect real images
    real_imgs = []
    count = 0
    target = cfg.fid_n_samples
    for imgs, _ in real_loader:
        real_imgs.append(imgs)
        count += imgs.size(0)
        if count >= target:
            break
    real_imgs = torch.cat(real_imgs)[:target]
    n_fid = real_imgs.size(0)

    # generate fake images
    G.eval()
    fake_imgs = []
    remaining = n_fid
    while remaining > 0:
        bs = min(cfg.gan_batch, remaining)
        z = torch.randn(bs, cfg.z_dim, device=device)
        y = torch.randint(0, num_classes, (bs,), device=device)
        fake_imgs.append(G(z, y).cpu())
        remaining -= bs
    fake_imgs = torch.cat(fake_imgs)[:n_fid]
    G.train()

    real_feats = get_inception_features(real_imgs, inception_model, device)
    fake_feats = get_inception_features(fake_imgs, inception_model, device)

    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sig_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    return calc_fid(mu_r, sig_r, mu_f, sig_f)


def load_inception(device):
    """Load InceptionV3 with pool3 output (2048-d)."""
    # Torchvision API changed across versions; support both.
    try:
        from torchvision.models import Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    except Exception:
        model = inception_v3(pretrained=True, transform_input=False)
    # remove final fc to get pool3 features
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def make_loader(cfg: Config, split: str, train: bool):
    tf_list = [
        transforms.Resize((cfg.img_size, cfg.img_size)),
    ]
    if train:
        tf_list.append(transforms.RandomHorizontalFlip())
    tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    tf = transforms.Compose(tf_list)
    ds = datasets.ImageFolder(str(cfg.data_root / split), transform=tf)
    loader = DataLoader(
        ds,
        batch_size=cfg.gan_batch,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=train,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    return loader, ds.classes


def save_samples(G, fixed_z, fixed_y, epoch, out_dir, nrow=8):
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z, fixed_y)
    vutils.save_image(imgs, out_dir / f"samples_epoch{epoch:04d}.png",
                      nrow=nrow, normalize=True, value_range=(-1, 1))
    G.train()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def train(cfg: Config):
    device = torch.device(resolve_device(cfg.device))
    print(f"Device: {device}")

    # reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, class_names = make_loader(cfg, split="train", train=True)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    fid_split = cfg.fid_eval_split
    fid_split_path = cfg.data_root / fid_split
    if not fid_split_path.exists():
        print(f"FID split '{fid_split}' not found, falling back to 'train'.")
        fid_split = "train"
    fid_loader, fid_classes = make_loader(cfg, split=fid_split, train=False)
    if fid_classes != class_names:
        raise RuntimeError(
            f"Class mismatch between train ({class_names}) and {fid_split} ({fid_classes})."
        )
    print(f"FID eval split: {fid_split}  samples={len(fid_loader.dataset)}")

    G = Generator(z_dim=cfg.z_dim, num_classes=num_classes).to(device)
    D = ProjectionCritic(num_classes=num_classes).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.gan_lr_g, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.gan_lr_d, betas=(0.0, 0.9))

    # fixed noise for visual tracking
    n_samples = 24  # 8 per class
    fixed_z = torch.randn(n_samples, cfg.z_dim, device=device)
    fixed_y = torch.arange(num_classes, device=device).repeat_interleave(n_samples // num_classes)

    out_dir = Path(cfg.out_root) / "gan"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # FID setup
    fid_log = []
    best_fid = float("inf")
    fid_enabled = cfg.fid_every is not None and cfg.fid_every > 0 and cfg.fid_n_samples > 0
    inception_model = load_inception(device) if fid_enabled else None
    if not fid_enabled:
        print("FID disabled (set Config.fid_every > 0 to enable).")

    global_step = 0

    for epoch in range(1, cfg.gan_epochs + 1):
        d_loss_acc, g_loss_acc, n_batches = 0.0, 0.0, 0
        d_real_acc, d_fake_acc, gp_acc = 0.0, 0.0, 0.0

        for real_imgs, real_labels in train_loader:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            B = real_imgs.size(0)

            # ---- Train Critic ---- #
            z = torch.randn(B, cfg.z_dim, device=device)
            # Keep fake conditioning aligned with the current real batch labels.
            # This also keeps GP conditioning consistent for interpolated samples.
            fake_labels = real_labels
            with torch.no_grad():
                fake_imgs = G(z, fake_labels)

            d_real = D(real_imgs, real_labels).mean()
            # Critic must score fake images with their own conditioning labels.
            d_fake = D(fake_imgs, fake_labels).mean()
            gp = gradient_penalty(D, real_imgs, fake_imgs, real_labels, device)
            d_loss = d_fake - d_real + cfg.gp_lambda * gp

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            d_loss_acc += d_loss.item()
            d_real_acc += d_real.item()
            d_fake_acc += d_fake.item()
            gp_acc += gp.item()

            # ---- Train Generator every n_critic steps ---- #
            global_step += 1
            if global_step % cfg.n_critic == 0:
                z = torch.randn(B, cfg.z_dim, device=device)
                gen_labels = torch.randint(0, num_classes, (B,), device=device)
                fake_imgs = G(z, gen_labels)
                g_loss = -D(fake_imgs, gen_labels).mean()

                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

                g_loss_acc += g_loss.item()

            n_batches += 1

        n_g_updates = max(1, n_batches // cfg.n_critic)
        d_avg = d_loss_acc / n_batches
        g_avg = g_loss_acc / n_g_updates
        d_real_avg = d_real_acc / n_batches
        d_fake_avg = d_fake_acc / n_batches
        gp_avg = gp_acc / n_batches
        w_dist = d_real_avg - d_fake_avg

        # FID evaluation
        fid_str = ""
        do_fid = fid_enabled and (epoch % cfg.fid_every == 0 or epoch == cfg.gan_epochs)
        if do_fid:
            fid = compute_fid(G, fid_loader, num_classes, cfg, inception_model, device)
            fid_log.append({
                "epoch": epoch,
                "fid": fid,
                "d_loss": d_avg,
                "g_loss": g_avg,
                "d_real": d_real_avg,
                "d_fake": d_fake_avg,
                "gp": gp_avg,
                "w_dist": w_dist,
            })
            fid_str = f"  FID: {fid:.2f}"

            if fid < best_fid:
                best_fid = fid
                torch.save({
                    "epoch": epoch,
                    "fid": fid,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                }, ckpt_dir / "best_fid.pt")
        else:
            fid_log.append({
                "epoch": epoch,
                "d_loss": d_avg,
                "g_loss": g_avg,
                "d_real": d_real_avg,
                "d_fake": d_fake_avg,
                "gp": gp_avg,
                "w_dist": w_dist,
            })

        print(
            f"[Epoch {epoch:03d}/{cfg.gan_epochs}]  "
            f"D_loss: {d_avg:.4f}  "
            f"G_loss: {g_avg:.4f}  "
            f"W_dist: {w_dist:.4f}  "
            f"GP: {gp_avg:.4f}{fid_str}"
        )

        # save samples
        if epoch % cfg.sample_every == 0 or epoch == 1:
            save_samples(G, fixed_z, fixed_y, epoch, out_dir)

        # save checkpoint
        if epoch % cfg.ckpt_every == 0 or epoch == cfg.gan_epochs:
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, ckpt_dir / f"ckpt_epoch{epoch:04d}.pt")

    # save training log
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(fid_log, f, indent=2)
    print(f"Training finished. Log saved to {out_dir / 'train_log.json'}")
    if best_fid < float("inf"):
        print(f"Best FID: {best_fid:.2f} (checkpoint: {ckpt_dir / 'best_fid.pt'})")
    return G, D


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
