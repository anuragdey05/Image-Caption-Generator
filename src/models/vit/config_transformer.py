from __future__ import annotations

from dataclasses import dataclass

from src.models.vit.text_decoder import TextDecoderConfig
from src.models.vit.vision_encoder_vit import ViTConfig


@dataclass
class TransformerHyperParams:
    """Central place for ViT + text-decoder hyperparameters."""
    vocab_size: int
    pad_idx: int
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    patch_size: int = 16
    max_seq_len: int = 32
    image_size: int = 128
    dropout: float = 0.1
    attn_dropout: float = 0.0
    mlp_ratio: float = 4.0

    def vision_config(self) -> ViTConfig:
        return ViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
        )

    def decoder_config(self) -> TextDecoderConfig:
        return TextDecoderConfig(
            vocab_size=self.vocab_size,
            max_length=self.max_seq_len,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            pad_idx=self.pad_idx,
        )