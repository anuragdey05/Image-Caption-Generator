"""Wrapper module for the CNN encoder + LSTM decoder."""
from __future__ import annotations
from typing import Tuple, Union
import torch
from torch import nn
from models.cnn_lstm.cnn_encoder import CNNEncoder
from models.cnn_lstm.lstm_decoder import LSTMDecoder


class ImageCaptioningCNNLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_feature_dim: int = 256,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(feature_dim=image_feature_dim)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            image_feat_dim=image_feature_dim,
        )

    def forward(self, images: torch.Tensor, captions_in: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder(images)
        return self.decoder(captions_in, image_features)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        greedy: bool = True,
        beam_size: int = 1,
        top_k: int = 1,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        image_features = self.encoder(images)
        sequences, lengths, scores = self.decoder.generate(
            image_features,
            max_length=max_length,
            greedy=greedy,
            beam_size=beam_size,
            top_k=top_k,
        )
        if top_k == 1:
            sequences = sequences[:, 0, :]
            lengths = lengths[:, 0]
            scores = scores[:, 0]
        if return_metadata:
            return sequences, lengths, scores
        return sequences
