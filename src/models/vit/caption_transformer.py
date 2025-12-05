from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from src.models.vit.text_decoder import TextDecoderConfig, TextTransformerDecoder
from src.models.vit.vision_encoder_vit import ViTConfig, VisionEncoderViT


class CaptionTransformer(nn.Module):
    """Wrapper that couples the ViT encoder with the text decoder."""

    def __init__(
        self,
        vision_config: Optional[ViTConfig] = None,
        decoder_config: Optional[TextDecoderConfig] = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoderViT(vision_config or ViTConfig())
        if decoder_config is None:
            raise ValueError("decoder_config must be provided with the target vocab size")
        self.text_decoder = TextTransformerDecoder(decoder_config)

    def forward(self, images: Tensor, captions_in: Tensor) -> Tensor:
        """Encode images, decode captions, return logits (B, T, vocab)."""
        memory = self.vision_encoder(images)
        logits = self.text_decoder(captions_in, memory)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: Optional[int] = None,
    ) -> Tensor:
        """Greedy caption generation given raw images."""
        memory = self.vision_encoder(images)
        return self.text_decoder.generate(memory, bos_token_id, eos_token_id, max_length=max_length)
