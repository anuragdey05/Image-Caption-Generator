"""Custom CNN encoder for 128x128 WikiArt images."""
from __future__ import annotations
import torch
from torch import nn

class CNNEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(256, feature_dim)

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        pooled = self.pool(features).flatten(1)
        return self.projection(pooled)
