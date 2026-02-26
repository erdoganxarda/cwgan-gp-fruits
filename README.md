<div align="center">

# gan-image-gen

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP) for synthetic fruit image generation,
plus a classification pipeline that systematically compares classifiers trained on
**real data**, **synthetic data**, and a **combination of both**.

</div>

---

## Table of Contents

- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Tests](#tests)
- [Architecture](#architecture)
- [Key Config](#key-config)
- [References](#references)

---

## Results

<div align="center">
  <img src="docs/assets/samples_epoch100.png" width="720" alt="GAN-generated fruit images at epoch 100">
  <p><em>Generated images at epoch 100 — apple · banana · orange (3 × 8 grid)</em></p>
</div>

<br>

<div align="center">
  <img src="docs/assets/accuracy_vs_size.png" width="720" alt="Test accuracy vs training set size">
  <p><em>Test accuracy vs. training set size</em></p>
  <img src="docs/assets/per_class_f1.png" width="720" alt="Per-class F1 scores">
  <p><em>Per-class F1 scores</em></p>
</div>

<br>

Key findings from the 15-experiment grid (5 sizes × 3 scenarios, evaluated on 492 images/class held-out test set):

| Images / class | Real-only | Synth-only | Real + Synth |
|:--------------:|:---------:|:----------:|:------------:|
| 100            | 98.98%    | 97.29%     | 98.85%       |
| 200            | 99.12%    | 97.02%     | 98.78%       |
| 400            | 97.83%    | **98.71%** | 96.82%       |
| 800            | **99.73%**| 98.85%     | 98.85%       |
| 1300           | 99.12%    | 98.92%     | **99.32%**   |

> **Takeaway:** Synth-only consistently stays within 1–2% of real-only accuracy across all dataset sizes, validating the GAN's output quality. At 400 images/class, synth-only marginally *outperforms* real-only — a likely regularisation effect from the GAN's implicit augmentation. Combining both sources yields peak accuracy (99.32%) at full scale.

---

## Project Structure

<details>
<summary>Click to expand</summary>

```
gan-image-gen/
├── config.py                    # All hyperparameters (GAN + Classifier)
├── models/
│   ├── gan.py                   # Generator (cBN) + ProjectionCritic
│   └── classifier.py            # FruitCNN (trained from scratch)
├── train_gan.py                 # WGAN-GP training loop + FID logging
├── train_classifier.py          # Classification: real / synth / both scenarios
├── scripts/
│   ├── generate_synth.py        # Generate synthetic dataset from trained G
│   ├── run_experiments.py       # Full experiment grid (5 sizes × 3 scenarios)
│   └── plot_results.py          # Accuracy / time plots + per-class F1 charts
├── notebooks/
│   └── gan_image_gen_quickstart.ipynb
├── tests/
│   └── test_models.py           # Model shape smoke tests (no training data needed)
├── docs/assets/                 # Sample outputs committed for README
├── data_final/                  # Real dataset — see Getting the Data
└── data_synth/                  # Generated after Step 2
```

</details>

---

## Dataset

3 classes: **apple**, **banana**, **orange** — 64 × 64 RGB images.

Real images are sourced from the [Fruits 360](https://www.kaggle.com/datasets/moltean/fruits) dataset (subset prepared locally). Before redistributing any dataset files, comply with Fruits 360's license and attribution requirements.

| Split | Per class | Total |
|-------|:---------:|:-----:|
| Train | 1 300     | 3 900 |
| Val   | 159       | 477   |
| Test  | 492       | 1 476 |

### Getting the Data

1. Download [Fruits 360 from Kaggle](https://www.kaggle.com/datasets/moltean/fruits) and unzip it.
2. Copy the `apple`, `banana`, and `orange` folders from `fruits-360_dataset/fruits-360/Training/` and `Test/` into:

```
data_final/
  train/  apple/  banana/  orange/
  val/    apple/  banana/  orange/   ← carve out ~10 % of training images
  test/   apple/  banana/  orange/
```

The exact split sizes (1 300 / 159 / 492 per class) are not required — the code works with any balanced split. `data_final/` is excluded from git by `.gitignore`.

---

## Setup

```bash
conda create -n gan python=3.11 -y
conda activate gan
pip install -r requirements.txt
```

> **GPU note:** If `pip install -r requirements.txt` doesn't produce a working GPU build, install PyTorch/torchvision for your CUDA version first from [pytorch.org/get-started](https://pytorch.org/get-started/locally/), then run the rest of the install.

Open `notebooks/gan_image_gen_quickstart.ipynb` for an end-to-end interactive walkthrough (data checks → small training run → visualisations) before committing to a full run.

---

## Usage

A `Makefile` wraps all four steps. Run `make help` to see every target.

### Step 1 — Train the GAN

```bash
python train_gan.py      # or: make train
```

| Output | Location |
|--------|----------|
| Checkpoints | `runs/gan/checkpoints/` |
| Best-FID checkpoint | `runs/gan/checkpoints/best_fid.pt` |
| Sample grids (every 5 epochs) | `runs/gan/samples_epoch*.png` |
| Training log | `runs/gan/train_log.json` |

Key settings: cWGAN-GP · TTUR (G lr=1e-4, D lr=2e-4) · Adam(β₁=0, β₂=0.9) · n\_critic=3 · λ\_gp=10 · FID every 5 epochs on `val`.

> **Offline mode:** Set `fid_every=0` in `config.py` to skip FID (avoids downloading Inception V3 weights).

### Step 2 — Generate Synthetic Dataset

```bash
python scripts/generate_synth.py \
    --ckpt runs/gan/checkpoints/best_fid.pt \
    --n_per_class 1300 \
    --seed 42
# or: make generate
```

The fixed seed makes the synthetic dataset reproducible. Run once, then treat it as frozen.

### Step 3 — Run Experiment Grid

```bash
python scripts/run_experiments.py   # or: make experiment
```

Runs all 15 experiments (5 sizes × 3 scenarios). Results are written to `runs/clf/all_results.json`.

### Step 4 — Plot Results

```bash
python scripts/plot_results.py   # or: make plot
```

| Plot | File |
|------|------|
| Test accuracy vs. data size | `runs/clf/plots/accuracy_vs_size.png` |
| Training time vs. data size | `runs/clf/plots/time_vs_size.png` |
| Per-class F1 scores | `runs/clf/plots/per_class_f1.png` |

---

## Tests

Shape and sanity smoke tests for all three models — no dataset required:

```bash
pytest tests/ -v   # or: make test
```

---

## Architecture

### Generator — ~1.5 M parameters

```
z (128-d) + class label
        │
        ▼
   Linear → 256 × 4 × 4
        │
  ┌─────┴──────────────────────────────┐
  │  4 × GenBlock                       │
  │  Upsample(×2) → Conv → cBN → ReLU  │
  │  4×4 → 8×8 → 16×16 → 32×32 → 64×64│
  └─────────────────────────────────────┘
        │
   BN → ReLU → Conv(3) → Tanh
        │
  3 × 64 × 64 image
```

**ConditionalBatchNorm:** scale (γ) and shift (β) are predicted per class via learned embeddings, making the generator truly class-conditional.

### Projection Critic — ~4.9 M parameters

```
3 × 64 × 64 image
        │
  4 × CriticBlock (Conv→LReLU→Conv→LReLU→AvgPool, with skip)
  64×64 → 32×32 → 16×16 → 8×8 → 4×4
        │
  Global sum pooling → h (512-d)
        │
  score = Linear(h) + embed(y)·h      ← projection (Miyato & Koyama 2018)
```

### Classifier — ~813 K parameters

```
3 × 64 × 64 image
        │
  3 × (Conv-BN-ReLU → Conv-BN-ReLU → MaxPool → Dropout2d)
  64×64 → 32×32 → 16×16
        │
  AdaptiveAvgPool → 128 × 4 × 4
        │
  Flatten → Linear(2048→256) → ReLU → Dropout → Linear(256→3)
```

Trained from scratch — no pretrained weights.

---

## Key Config

All hyperparameters live in [`config.py`](config.py) as a frozen dataclass.

| Parameter        | Value  | Description                      |
|------------------|:------:|----------------------------------|
| `img_size`       | 64     | Image resolution (64 × 64 px)    |
| `z_dim`          | 128    | Latent vector dimension          |
| `gan_batch`      | 64     | GAN batch size                   |
| `gan_epochs`     | 100    | GAN training epochs              |
| `gan_lr_g`       | 1e-4   | Generator learning rate          |
| `gan_lr_d`       | 2e-4   | Critic learning rate (TTUR)      |
| `n_critic`       | 3      | Critic steps per generator step  |
| `gp_lambda`      | 10.0   | Gradient penalty coefficient     |
| `fid_every`      | 5      | FID evaluation frequency (epochs)|
| `clf_batch`      | 64     | Classifier batch size            |
| `clf_epochs`     | 30     | Classifier training epochs       |
| `clf_lr`         | 1e-3   | Classifier learning rate         |
| `device`         | auto   | CUDA → MPS → CPU fallback        |

---

## References

- Miyato, T. & Koyama, M. (2018). [cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637). *ICLR 2018.*
- Gulrajani, I. et al. (2017). [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028). *NeurIPS 2017.*
- Muresan, H. & Oltean, M. [Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits). Kaggle.

---

<div align="center">
  <sub>MIT License © 2026 Arda Erdogan</sub>
</div>
