"""Handcrafted order-flow imbalance (OFI) and book imbalance features.

OFI follows the discrete Cont–Kukanov–Stoikov construction in spirit:
signed changes in size at a price level when that price is still the
best, plus full size when the best price moves. We compute this at
level 1 and as a sum over ten levels. These are *not* learned.

The classifier is a small MLP on the per-window summary of OFI and
imbalance. Used as an ablation against DeepLOB (raw-book CNN-LSTM).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _level_slice(book: np.ndarray | torch.Tensor, level: int) -> tuple:
    """Return (pa, va, pb, vb) at 0-based level from (..., 40)."""
    base = level * 4
    return book[..., base], book[..., base + 1], book[..., base + 2], book[..., base + 3]


def ofi_features_from_book(book: np.ndarray) -> np.ndarray:
    """Compute per-time OFI and imbalance.

    Parameters
    ----------
    book : (T, 40) or (B, T, 40)

    Returns
    -------
    feats : (T, 4) or (B, T, 4)
        columns = [ofi_l1, ofi_sum10, imb_l1, imb_depth]
    """
    squeeze = False
    if book.ndim == 2:
        book = book[None, ...]
        squeeze = True
    b, t, _ = book.shape
    ofi_l1 = np.zeros((b, t), dtype=np.float32)
    ofi_sum = np.zeros((b, t), dtype=np.float32)
    imb_l1 = np.zeros((b, t), dtype=np.float32)
    imb_d = np.zeros((b, t), dtype=np.float32)
    for i in range(10):
        pa, va, pb, vb = _level_slice(book, i)
        d_va = np.zeros_like(va)
        d_vb = np.zeros_like(vb)
        d_va[:, 1:] = va[:, 1:] - va[:, :-1]
        d_vb[:, 1:] = vb[:, 1:] - vb[:, :-1]
        # Price improved (ask down / bid up) counts as positive size at new best.
        d_pa = np.zeros_like(pa)
        d_pb = np.zeros_like(pb)
        d_pa[:, 1:] = pa[:, 1:] - pa[:, :-1]
        d_pb[:, 1:] = pb[:, 1:] - pb[:, :-1]
        signed = np.where(d_pb > 0, vb, 0.0) - np.where(d_pb < 0, vb, 0.0)
        signed += np.where(d_pb == 0, d_vb, 0.0)
        signed -= np.where(d_pa < 0, va, 0.0)
        signed += np.where(d_pa > 0, va, 0.0)
        signed -= np.where(d_pa == 0, d_va, 0.0)
        ofi_sum += signed.astype(np.float32)
        if i == 0:
            ofi_l1 = signed.astype(np.float32)
            imb_l1 = ((vb - va) / (vb + va + 1e-8)).astype(np.float32)
    va_all = sum(_level_slice(book, i)[1] for i in range(10))
    vb_all = sum(_level_slice(book, i)[3] for i in range(10))
    imb_d = ((vb_all - va_all) / (vb_all + va_all + 1e-8)).astype(np.float32)
    feats = np.stack([ofi_l1, ofi_sum, imb_l1, imb_d], axis=-1)
    if squeeze:
        return feats[0]
    return feats


class OFIEncoder(nn.Module):
    """Mean/std pooling of 4 OFI channels over time, then MLP to hidden."""

    def __init__(self, n_channels: int = 4, hidden: int = 16) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Linear(n_channels * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def _channels(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable OFI-like channels from a raw book batch."""
        # x: (B, T, 40)
        pa = x[..., 0]
        va = x[..., 1]
        pb = x[..., 2]
        vb = x[..., 3]
        d_va = torch.zeros_like(va)
        d_vb = torch.zeros_like(vb)
        d_va[:, 1:] = va[:, 1:] - va[:, :-1]
        d_vb[:, 1:] = vb[:, 1:] - vb[:, :-1]
        ofi_l1 = d_vb - d_va
        va_all = x[..., 1::4].sum(dim=-1)
        vb_all = x[..., 3::4].sum(dim=-1)
        d_va_all = torch.zeros_like(va_all)
        d_vb_all = torch.zeros_like(vb_all)
        d_va_all[:, 1:] = va_all[:, 1:] - va_all[:, :-1]
        d_vb_all[:, 1:] = vb_all[:, 1:] - vb_all[:, :-1]
        ofi_sum = d_vb_all - d_va_all
        imb_l1 = (vb - va) / (vb + va + 1e-8)
        imb_d = (vb_all - va_all) / (vb_all + va_all + 1e-8)
        return torch.stack([ofi_l1, ofi_sum, imb_l1, imb_d], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch = self._channels(x)  # (B, T, 4)
        mu = ch.mean(dim=1)
        sd = ch.std(dim=1, unbiased=False)
        return self.net(torch.cat([mu, sd], dim=-1))


class OFIClassifier(nn.Module):
    def __init__(self, hidden: int = 16, n_classes: int = 3) -> None:
        super().__init__()
        self.enc = OFIEncoder(hidden=hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))
