"""Flattened MLP baseline for LOB windows."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Two-layer MLP on flattened (T, F) windows.

    This is a weak baseline: it ignores temporal order except insofar
    as position is encoded by concatenation.
    """

    def __init__(
        self,
        seq_len: int = 100,
        n_features: int = 40,
        hidden: int = 64,
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        in_dim = seq_len * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        return self.net(x)
