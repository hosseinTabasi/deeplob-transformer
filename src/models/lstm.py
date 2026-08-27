"""LSTM classifier on raw LOB windows."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Single-layer LSTM then a linear head on the last hidden state."""

    def __init__(
        self,
        n_features: int = 40,
        hidden: int = 32,
        n_layers: int = 1,
        n_classes: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
