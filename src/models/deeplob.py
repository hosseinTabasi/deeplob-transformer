"""Small CNN-LSTM in the spirit of Zhang, Zohren, Roberts (2019).

This is an original, reduced-width implementation of the DeepLOB
pipeline described in arXiv:1808.03668 / IEEE TSP 2019:

1. Spatial 1x2 convolutions with stride 2 over (price, volume) pairs,
   then over adjacent levels, then a 1x10 mix across the ten levels.
2. An Inception block (1 / 3 / 5 / pool) along time.
3. An LSTM on the resulting time series and a 3-way softmax head.

It is **not** a copy of any GPL dump. Filter counts are smaller than
the paper so a CPU smoke train is feasible. Shapes match the paper's
input convention: (B, T, 40) with T typically 100.
"""

from __future__ import annotations

import torch
from torch import nn


class Inception1d(nn.Module):
    """Four-branch 1-D inception along the time axis."""

    def __init__(self, in_ch: int, out_ch: int = 8) -> None:
        super().__init__()
        self.b1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.b5 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
        )
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.pool(x)], dim=1)
        return self.act(y)


class DeepLOB(nn.Module):
    """Reduced-width DeepLOB: spatial CNN + inception + LSTM."""

    def __init__(
        self,
        n_features: int = 40,
        conv_ch: int = 8,
        inception_ch: int = 8,
        lstm_hidden: int = 16,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        if n_features != 40:
            raise ValueError("DeepLOB spatial stack assumes 40 LOB features")
        self.conv1 = nn.Conv2d(1, conv_ch, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(conv_ch, conv_ch, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(conv_ch, conv_ch, kernel_size=(1, 10))
        self.act = nn.LeakyReLU(0.01)
        self.inception = Inception1d(conv_ch, inception_ch)
        inception_out = 4 * inception_ch
        self.lstm = nn.LSTM(inception_out, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)
        self.lstm_hidden = lstm_hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return last LSTM hidden state, shape (B, lstm_hidden)."""
        # x: (B, T, 40) -> (B, 1, T, 40)
        z = x.unsqueeze(1)
        z = self.act(self.conv1(z))  # (B, C, T, 20)
        z = self.act(self.conv2(z))  # (B, C, T, 10)
        z = self.act(self.conv3(z))  # (B, C, T, 1)
        z = z.squeeze(-1)  # (B, C, T)
        z = self.inception(z)  # (B, 4*inc, T)
        z = z.transpose(1, 2)  # (B, T, 4*inc)
        out, _ = self.lstm(z)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))
