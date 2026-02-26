"""
Smoke tests: verify model forward passes and output shapes.
No training data required — runs entirely with random tensors.
"""

import torch
import pytest

from models.gan import Generator, ProjectionCritic
from models.classifier import FruitCNN


BATCH = 4
Z_DIM = 128
NUM_CLASSES = 3
IMG = (BATCH, 3, 64, 64)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def labels(device):
    return torch.randint(0, NUM_CLASSES, (BATCH,), device=device)


# ---------- Generator ---------- #

def test_generator_output_shape(device, labels):
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES).to(device)
    z = torch.randn(BATCH, Z_DIM, device=device)
    out = G(z, labels)
    assert out.shape == torch.Size(IMG), f"Expected {IMG}, got {out.shape}"


def test_generator_output_range(device, labels):
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES).to(device)
    z = torch.randn(BATCH, Z_DIM, device=device)
    out = G(z, labels)
    assert out.min() >= -1.0 - 1e-5, "Generator output below -1 (Tanh expected)"
    assert out.max() <= 1.0 + 1e-5, "Generator output above +1 (Tanh expected)"


def test_generator_param_count(device):
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in G.parameters())
    # Sanity bound: Generator should be roughly 1–3 M params
    assert 500_000 < n_params < 5_000_000, f"Unexpected param count: {n_params}"


# ---------- Critic ---------- #

def test_critic_output_shape(device, labels):
    D = ProjectionCritic(num_classes=NUM_CLASSES).to(device)
    x = torch.randn(*IMG, device=device)
    out = D(x, labels)
    assert out.shape == torch.Size([BATCH, 1]), f"Expected ({BATCH}, 1), got {out.shape}"


def test_critic_param_count(device):
    D = ProjectionCritic(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in D.parameters())
    # Sanity bound: Critic should be roughly 3–8 M params
    assert 1_000_000 < n_params < 10_000_000, f"Unexpected param count: {n_params}"


# ---------- Classifier ---------- #

def test_classifier_output_shape(device):
    clf = FruitCNN(num_classes=NUM_CLASSES).to(device)
    x = torch.randn(*IMG, device=device)
    out = clf(x)
    assert out.shape == torch.Size([BATCH, NUM_CLASSES]), \
        f"Expected ({BATCH}, {NUM_CLASSES}), got {out.shape}"


def test_classifier_param_count(device):
    clf = FruitCNN(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in clf.parameters())
    # Sanity bound: FruitCNN should be roughly 500K–2M params
    assert 100_000 < n_params < 3_000_000, f"Unexpected param count: {n_params}"


# ---------- Generator → Critic round-trip ---------- #

def test_generator_critic_roundtrip(device, labels):
    G = Generator(z_dim=Z_DIM, num_classes=NUM_CLASSES).to(device)
    D = ProjectionCritic(num_classes=NUM_CLASSES).to(device)
    z = torch.randn(BATCH, Z_DIM, device=device)
    fake = G(z, labels)
    score = D(fake, labels)
    assert score.shape == torch.Size([BATCH, 1])
    assert torch.isfinite(score).all(), "Critic scores contain NaN or Inf"
