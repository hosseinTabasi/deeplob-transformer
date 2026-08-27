"""Small Transformer encoder (TransLOB-inspired, not a dump) and fusion."""

from __future__ import annotations

import math

import torch
from torch import nn

from .deeplob import DeepLOB
from .features_ofi import OFIEncoder


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ControlledTransformer(nn.Module):
    """Tiny encoder: linear embed, sinusoidal PE, 2 layers, mean pool.

    TransLOB (Wallbridge, arXiv:2003.00130) uses a transformer on LOB
    windows. This module is a *controlled* small encoder for an
    ablation against MLP / LSTM / DeepLOB, not a reproduction of the
    full TransLOB width or training recipe.
    """

    def __init__(
        self,
        n_features: int = 40,
        d_model: int = 32,
        nhead: int = 4,
        n_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        z = self.enc(z)
        return self.head(z.mean(dim=1))


class FusionDeepLOB(nn.Module):
    """Ablation hook: DeepLOB encoding, OFI encoding, or concatenation.

    mode
        ``deeplob`` : CNN-LSTM only
        ``ofi``     : handcrafted OFI + imbalance MLP
        ``both``    : concat encodings then linear head
    """

    def __init__(
        self,
        mode: str = "deeplob",
        n_classes: int = 3,
        ofi_hidden: int = 16,
    ) -> None:
        super().__init__()
        if mode not in {"deeplob", "ofi", "both"}:
            raise ValueError(mode)
        self.mode = mode
        self.deeplob = DeepLOB(n_classes=n_classes)
        self.ofi = OFIEncoder(hidden=ofi_hidden)
        if mode == "deeplob":
            in_dim = self.deeplob.lstm_hidden
        elif mode == "ofi":
            in_dim = ofi_hidden
        else:
            in_dim = self.deeplob.lstm_hidden + ofi_hidden
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "deeplob":
            h = self.deeplob.encode(x)
        elif self.mode == "ofi":
            h = self.ofi(x)
        else:
            h = torch.cat([self.deeplob.encode(x), self.ofi(x)], dim=-1)
        return self.head(h)
