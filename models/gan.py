"""
cWGAN-GP with Projection Discriminator
---------------------------------------
Generator  : conditional batch-norm, transposed-conv stack  (z + label → 64×64 RGB)
Critic     : conv stack + projection (Miyato & Koyama, 2018)
"""

import torch
import torch.nn as nn


# ---------- helper ---------- #
class ConditionalBatchNorm2d(nn.Module):
    """BN whose γ,β are predicted from the class embedding."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gain = nn.Embedding(num_classes, num_features)
        self.bias = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.bn(x)
        gain = self.gain(y).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias(y).unsqueeze(-1).unsqueeze(-1)
        return h * gain + bias


class GenBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_classes: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.cbn = ConditionalBatchNorm2d(out_ch, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self.conv(h)
        h = self.cbn(h, y)
        return self.act(h)


# ---------- Generator ---------- #
class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, num_classes: int = 3, ch: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, ch * 4 * 4)  # → ch×4×4
        self.block1 = GenBlock(ch, ch, num_classes)        # 4→8
        self.block2 = GenBlock(ch, ch // 2, num_classes)   # 8→16
        self.block3 = GenBlock(ch // 2, ch // 4, num_classes)  # 16→32
        self.block4 = GenBlock(ch // 4, ch // 8, num_classes)  # 32→64
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(ch // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        h = self.block1(h, y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        return self.to_rgb(h)


# ---------- Critic (Projection Discriminator) ---------- #
class CriticBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if downsample:
            layers.append(nn.AvgPool2d(2))
        self.main = nn.Sequential(*layers)

        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            *(nn.AvgPool2d(2),) if downsample else (),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class ProjectionCritic(nn.Module):
    """
    Wasserstein critic with projection conditioning.
    Output = φ(x)·w  +  (y_embed)·φ(x)  (scalar per sample).
    """

    def __init__(self, num_classes: int = 3, ch: int = 64):
        super().__init__()
        self.block1 = CriticBlock(3, ch)           # 64→32
        self.block2 = CriticBlock(ch, ch * 2)      # 32→16
        self.block3 = CriticBlock(ch * 2, ch * 4)  # 16→8
        self.block4 = CriticBlock(ch * 4, ch * 8)  # 8→4
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.linear = nn.Linear(ch * 8, 1)  # unconditional path
        self.embed = nn.Embedding(num_classes, ch * 8)  # projection path

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.act(h)
        # global sum pooling
        h = h.sum(dim=[2, 3])  # (B, ch*8)

        out = self.linear(h)  # unconditional score
        proj = (self.embed(y) * h).sum(dim=1, keepdim=True)  # projection
        return out + proj
