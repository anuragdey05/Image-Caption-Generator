from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 128
    patch_size: int = 16
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0


class PatchEmbedding(nn.Module):
    """Split image into patches and map to d_model."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> Tensor:
        # B,C,H,W -> B,d_model,num_patches -> B,num_patches,d_model
        patched = self.proj(images)
        return patched.flatten(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    """MHSA + FFN with residual connections."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class VisionEncoderViT(nn.Module):
    """
    Converts image tensors from ArtemisDataset into patch tokens.
    Returns image_tokens shaped (B, num_patches, d_model) for cross-attention.
    """

    def __init__(self, config: Optional[ViTConfig] = None, in_channels: int = 3) -> None:
        super().__init__()
        self.config = config or ViTConfig()
        self.patch_embed = PatchEmbedding(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=in_channels,
            d_model=self.config.d_model,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, self.config.d_model))
        self.dropout = nn.Dropout(self.config.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=self.config.d_model,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                    attn_dropout=self.config.attn_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: float tensor shaped (B, 3, 128, 128) from ArtemisDataset.

        Returns:
            image_tokens: (B, num_patches, d_model)
        """
        tokens = self.patch_embed(images) + self.pos_embedding
        tokens = self.dropout(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens