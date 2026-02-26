"""
Classification training script — supports 3 scenarios:
  --scenario real    : train on real data only
  --scenario synth   : train on synthetic data only
  --scenario both    : train on real + synthetic combined

  --n_per_class N    : limit to N images per class (for data-size experiments)

Returns JSON with accuracy, per-class metrics, and training time.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

from config import Config, resolve_device
from models.classifier import FruitCNN


# ------------------------------------------------------------------ #
#  Dataset helpers                                                     #
# ------------------------------------------------------------------ #

def get_transform(img_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def subsample_dataset(dataset, n_per_class, seed=42):
    """Return a Subset with at most n_per_class images per class."""
    from collections import defaultdict
    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    selected = []
    for label, indices in sorted(class_indices.items()):
        rng.shuffle(indices)
        selected.extend(indices[:n_per_class])
    return Subset(dataset, selected)


def build_dataset(cfg, scenario, n_per_class, synth_dir="data_synth"):
    tf_train = get_transform(cfg.img_size, train=True)
    tf_test = get_transform(cfg.img_size, train=False)

    real_train = datasets.ImageFolder(str(cfg.data_root / "train"), transform=tf_train)
    test_ds = datasets.ImageFolder(str(cfg.data_root / "test"), transform=tf_test)

    if scenario == "real":
        train_ds = subsample_dataset(real_train, n_per_class, cfg.seed) if n_per_class else real_train
    elif scenario == "synth":
        synth_train = datasets.ImageFolder(synth_dir, transform=tf_train)
        train_ds = subsample_dataset(synth_train, n_per_class, cfg.seed) if n_per_class else synth_train
    elif scenario == "both":
        synth_train = datasets.ImageFolder(synth_dir, transform=tf_train)
        if n_per_class:
            real_sub = subsample_dataset(real_train, n_per_class, cfg.seed)
            synth_sub = subsample_dataset(synth_train, n_per_class, cfg.seed)
            train_ds = ConcatDataset([real_sub, synth_sub])
        else:
            train_ds = ConcatDataset([real_train, synth_train])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return train_ds, test_ds, real_train.classes


# ------------------------------------------------------------------ #
#  Train / Evaluate                                                    #
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def run(cfg, scenario, n_per_class, synth_dir, out_dir):
    device = torch.device(resolve_device(cfg.device))
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    train_ds, test_ds, class_names = build_dataset(cfg, scenario, n_per_class, synth_dir)

    _pw = cfg.persistent_workers and cfg.num_workers > 0
    _pf = cfg.prefetch_factor if cfg.num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=cfg.clf_batch, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False,
                              persistent_workers=_pw, prefetch_factor=_pf)
    test_loader = DataLoader(test_ds, batch_size=cfg.clf_batch, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                             persistent_workers=_pw, prefetch_factor=_pf)

    model = FruitCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.clf_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.clf_epochs)

    train_size = len(train_ds)
    print(f"[{scenario}] n_per_class={n_per_class or 'all'}  train_size={train_size}  device={device}")

    t0 = time.time()
    for epoch in range(1, cfg.clf_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if epoch % 10 == 0 or epoch == cfg.clf_epochs:
            print(f"  Epoch {epoch:03d}  loss={loss:.4f}  train_acc={acc:.4f}")
    train_time = time.time() - t0

    preds, labels = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    test_acc = report["accuracy"]

    print(f"  Test accuracy: {test_acc:.4f}  Train time: {train_time:.1f}s")

    result = {
        "scenario": scenario,
        "n_per_class": n_per_class or "all",
        "train_size": train_size,
        "test_accuracy": round(test_acc, 4),
        "train_time_sec": round(train_time, 1),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
            }
            for name in class_names
        },
    }

    # save result
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_n{n_per_class}" if n_per_class else f"{scenario}_all"
    with open(out_path / f"result_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["real", "synth", "both"], required=True)
    parser.add_argument("--n_per_class", type=int, default=None,
                        help="Limit training images per class (None = use all)")
    parser.add_argument("--synth_dir", type=str, default="data_synth")
    parser.add_argument("--out_dir", type=str, default="runs/clf")
    args = parser.parse_args()

    cfg = Config()
    run(cfg, args.scenario, args.n_per_class, args.synth_dir, args.out_dir)
