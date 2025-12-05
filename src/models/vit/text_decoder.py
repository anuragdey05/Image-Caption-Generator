from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TextDecoderConfig:
    vocab_size: int
    max_length: int = 64
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    pad_idx: int = 0


class DecoderBlock(nn.Module):
    """Transformer decoder block with masked self-attn + cross-attn."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)

        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Tensor,
        self_padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Masked self-attention
        y, _ = self.self_attn(
            self.norm_self(x),
            self.norm_self(x),
            self.norm_self(x),
            attn_mask=self_mask,
            key_padding_mask=self_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(y)

        # Cross attention over image tokens
        y, _ = self.cross_attn(
            self.norm_cross(x),
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(y)

        # Feed-forward
        y = self.ffn(self.norm_ffn(x))
        x = x + self.dropout(y)
        return x


class TextTransformerDecoder(nn.Module):
    """Autoregressive text decoder conditioned on image tokens."""

    def __init__(self, config: TextDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.pos_emb = nn.Embedding(config.max_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, captions_in: Tensor, memory: Tensor) -> Tensor:
        """Return logits for next-token prediction."""
        B, T = captions_in.shape
        if T > self.config.max_length:
            raise ValueError("captions_in exceeds max_length configured for decoder")

        positions = torch.arange(T, device=captions_in.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(captions_in) + self.pos_emb(positions)
        x = self.dropout(x)

        self_mask = self._causal_mask(T, captions_in.device)
        padding_mask = captions_in.eq(self.config.pad_idx)

        for layer in self.layers:
            x = layer(
                x,
                memory,
                self_mask=self_mask,
                self_padding_mask=padding_mask,
                memory_padding_mask=None,
            )

        x = self.norm(x)
        logits = self.proj(x)
        return logits

    def generate(
        self,
        memory: Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: Optional[int] = None,
    ) -> Tensor:
        """Greedy generation conditioned on encoder memory."""
        max_len = max_length or self.config.max_length
        device = memory.device
        B = memory.size(0)
        seq = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(seq, memory)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return seq

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(torch.tril(torch.ones_like(mask), diagonal=0).bool(), 0.0)
        return mask
