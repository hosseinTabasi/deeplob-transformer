"""FI-2010 loader with a synthetic LOB fallback.

Scientific dataset
------------------
FI-2010 (Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis, 2018) is the
intended empirical source: five Nasdaq Nordic names, ten trading days,
ten LOB levels, 40 raw features per event. Place the extracted archive
under ``data/raw/FI-2010/`` (see ``scripts/download_fi2010.md``).

This module does **not** download FI-2010. If the directory is absent,
``synthetic_lob`` builds a stationary-ish 10-level book used only for
shape tests and a tiny overfit-able smoke train. Synthetic output is
labelled TOY in every artefact that consumes it.

Feature layout (Zhang, Zohren, Roberts 2019, arXiv:1808.03668)
--------------------------------------------------------------
At each event t the 40-vector is
``[p_a^(i), v_a^(i), p_b^(i), v_b^(i)]`` for levels i = 1..10.
Windows of length T (default 100) are the network input.

Labels follow the FI-2010 / Ntakaris smoothing of future mids:
``l_t = (mean_{i=1..k} p_{t+i} - p_t) / p_t``, then
up / stationary / down by threshold ``alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

N_LEVELS = 10
FEATURE_DIM = 40  # 10 levels * (ask px, ask vol, bid px, bid vol)
N_CLASSES = 3  # 0=down, 1=stationary, 2=up
DEFAULT_SEQ_LEN = 100

SplitName = Literal["train", "val", "test"]


def _find_fi2010(root: Path) -> Path | None:
    candidates = [
        root / "data" / "raw" / "FI-2010",
        root / "data" / "raw" / "fi-2010",
        Path("data/raw/FI-2010"),
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def synthetic_lob(
    n_events: int = 800,
    n_days: int = 10,
    n_levels: int = N_LEVELS,
    seed: int = 0,
    tick: float = 0.01,
    inject_imbalance_signal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 10-level bid/ask book with mild stationarity.

    Mid-price is a slow random walk. Depth is exponential in level.
    If ``inject_imbalance_signal`` is True, a positive (negative) L1
    imbalance slightly drifts the next mid up (down) so a one-epoch
    smoke train can overfit. This is **not** a market simulator.

    Returns
    -------
    features : (n_days, n_events, 40)
    mids : (n_days, n_events)
    day_index : (n_days,)
    """
    rng = np.random.default_rng(seed)
    books = np.zeros((n_days, n_events, n_levels * 4), dtype=np.float32)
    mids = np.zeros((n_days, n_events), dtype=np.float32)
    for d in range(n_days):
        mid = 100.0 + rng.normal(0.0, 0.05)
        for t in range(n_events):
            shock = rng.normal(0.0, 0.02)
            if inject_imbalance_signal and t > 0:
                prev = books[d, t - 1]
                va1, vb1 = float(prev[1]), float(prev[3])
                imb = (vb1 - va1) / (vb1 + va1 + 1e-8)
                shock += 0.04 * np.tanh(3.0 * imb)
            mid = max(mid + shock, 50.0)
            spread = tick * (1 + rng.integers(1, 4))
            pa1 = mid + 0.5 * spread
            pb1 = mid - 0.5 * spread
            row = []
            for i in range(n_levels):
                pa = pa1 + i * tick
                pb = pb1 - i * tick
                # Exponential depth; mildly noisy, positive.
                va = float(rng.exponential(80.0 / (i + 1))) + 1.0
                vb = float(rng.exponential(80.0 / (i + 1))) + 1.0
                row.extend([pa, va, pb, vb])
            books[d, t] = np.asarray(row, dtype=np.float32)
            mids[d, t] = 0.5 * (books[d, t, 0] + books[d, t, 2])
    return books, mids, np.arange(n_days, dtype=np.int32)


def label_from_mids(
    mids: np.ndarray,
    k: int,
    alpha: float = 2e-4,
) -> np.ndarray:
    """Ntakaris-style smoothed future-mid labels.

    Parameters
    ----------
    mids : (n_days, n_events) or (n_events,)
    k : horizon in events
    alpha : relative-change threshold
    """
    if mids.ndim == 1:
        mids = mids[None, :]
    n_days, n_events = mids.shape
    y = np.ones((n_days, n_events), dtype=np.int64)  # stationary
    for d in range(n_days):
        p = mids[d]
        for t in range(n_events - k):
            m_plus = float(p[t + 1 : t + 1 + k].mean())
            rel = (m_plus - float(p[t])) / (float(p[t]) + 1e-12)
            if rel > alpha:
                y[d, t] = 2
            elif rel < -alpha:
                y[d, t] = 0
            else:
                y[d, t] = 1
        y[d, n_events - k :] = -1  # invalid tail
    return y if n_days > 1 or mids.ndim == 2 else y[0]


def window_samples(
    books: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows that do not cross day boundaries.

    Returns X (N, seq_len, 40), y (N,), day_id (N,).
    """
    n_days, n_events, f = books.shape
    xs, ys, days = [], [], []
    for d in range(n_days):
        for t in range(seq_len - 1, n_events):
            lab = int(labels[d, t])
            if lab < 0:
                continue
            xs.append(books[d, t - seq_len + 1 : t + 1])
            ys.append(lab)
            days.append(d)
    if not xs:
        return (
            np.zeros((0, seq_len, f), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(days, dtype=np.int64),
    )


def paper_setup2_split(
    day_id: np.ndarray,
    n_days: int = 10,
    n_train_days: int = 7,
) -> dict[str, np.ndarray]:
    """Zhang et al. Setup 2: first 7 days train, last 3 days test.

    No validation day is reserved in the original protocol. We still
    return an empty val mask so callers have a uniform interface.
    """
    train_days = np.arange(n_train_days)
    test_days = np.arange(n_train_days, n_days)
    return {
        "train": np.isin(day_id, train_days),
        "val": np.zeros_like(day_id, dtype=bool),
        "test": np.isin(day_id, test_days),
    }


def chronological_split(
    day_id: np.ndarray,
    n_days: int = 10,
    train_days: int = 6,
    val_days: int = 1,
) -> dict[str, np.ndarray]:
    """Stricter chronological split: train 1-6, val 7, test 8-10.

    Windows never mix days. No shuffling across the cut. This is
    *not* the original DeepLOB protocol; it is the robustness check.
    """
    tr = np.arange(train_days)
    va = np.arange(train_days, train_days + val_days)
    te = np.arange(train_days + val_days, n_days)
    return {
        "train": np.isin(day_id, tr),
        "val": np.isin(day_id, va),
        "test": np.isin(day_id, te),
    }


@dataclass
class FI2010Dataset:
    """In-memory windows. Source is ``fi2010`` or ``synthetic_toy``."""

    X: np.ndarray
    y: np.ndarray
    day_id: np.ndarray
    source: str
    k: int
    seq_len: int
    n_days: int

    @classmethod
    def load(
        cls,
        root: Path | str = ".",
        k: int = 10,
        seq_len: int = DEFAULT_SEQ_LEN,
        alpha: float = 2e-4,
        seed: int = 0,
        synthetic_events: int = 200,
        synthetic_days: int = 10,
        force_synthetic: bool = False,
    ) -> FI2010Dataset:
        root = Path(root)
        path = None if force_synthetic else _find_fi2010(root)
        if path is None:
            books, mids, _ = synthetic_lob(
                n_events=synthetic_events,
                n_days=synthetic_days,
                seed=seed,
            )
            source = "synthetic_toy"
            n_days = synthetic_days
        else:
            loaded = _load_fi2010_arrays(path)
            if loaded is None:
                books, mids, _ = synthetic_lob(
                    n_events=synthetic_events,
                    n_days=synthetic_days,
                    seed=seed,
                )
                source = "synthetic_toy"
                n_days = synthetic_days
            else:
                books, mids = loaded
                source = "fi2010"
                n_days = books.shape[0]
        labels = label_from_mids(mids, k=k, alpha=alpha)
        X, y, day_id = window_samples(books, labels, seq_len=seq_len)
        return cls(
            X=X, y=y, day_id=day_id, source=source, k=k, seq_len=seq_len, n_days=n_days
        )

    def split(self, protocol: str = "paper_setup2") -> dict[str, np.ndarray]:
        if protocol == "paper_setup2":
            return paper_setup2_split(self.day_id, n_days=self.n_days)
        if protocol == "chronological":
            return chronological_split(self.day_id, n_days=self.n_days)
        raise ValueError(f"unknown protocol {protocol!r}")

    def subset(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.X[mask], self.y[mask]


def _load_fi2010_arrays(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort load of common FI-2010 dumps (npy / csv).

    Returns None if the on-disk layout is not recognised so the caller
    can fall back to synthetic data rather than crash.
    """
    npy = list(path.glob("*.npy"))
    if npy:
        arr = np.load(npy[0])
        if arr.ndim == 3 and arr.shape[-1] >= FEATURE_DIM:
            books = arr[..., :FEATURE_DIM].astype(np.float32)
            mids = 0.5 * (books[..., 0] + books[..., 2])
            return books, mids
    return None
