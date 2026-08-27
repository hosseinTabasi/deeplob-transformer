"""Tensor-shape tests on a synthetic LOB batch. Must pass without FI-2010."""
from __future__ import annotations
import numpy as np
import torch
from data.fi2010 import FEATURE_DIM, FI2010Dataset, chronological_split, paper_setup2_split, synthetic_lob, window_samples, label_from_mids
from models import ControlledTransformer, DeepLOB, FusionDeepLOB, LSTMClassifier, MLPClassifier, OFIClassifier, ofi_features_from_book

BATCH, SEQ, FEAT = 4, 32, FEATURE_DIM

def _batch():
    return torch.randn(BATCH, SEQ, FEAT)

def test_synthetic_lob_shape():
    books, mids, days = synthetic_lob(n_events=80, n_days=10, seed=0)
    assert books.shape == (10, 80, 40)
    assert mids.shape == (10, 80)
    assert days.shape == (10,)
    assert np.isfinite(books).all()
    assert np.all(books[..., 2] < books[..., 0])

def test_windows_and_splits():
    books, mids, _ = synthetic_lob(n_events=80, n_days=10, seed=1)
    y = label_from_mids(mids, k=1, alpha=1e-6)
    X, labels, day_id = window_samples(books, y, seq_len=16)
    assert X.ndim == 3 and X.shape[-1] == 40
    assert len(labels) == len(day_id) == len(X)
    paper = paper_setup2_split(day_id, n_days=10)
    chrono = chronological_split(day_id, n_days=10)
    assert paper["train"].any() and paper["test"].any()
    assert chrono["train"].any() and chrono["val"].any() and chrono["test"].any()
    assert set(day_id[paper["train"]]) & set(day_id[paper["test"]]) == set()
    assert set(day_id[chrono["train"]]) & set(day_id[chrono["test"]]) == set()

def test_mlp_shape():
    m = MLPClassifier(seq_len=SEQ, n_features=FEAT)
    assert m(_batch()).shape == (BATCH, 3)

def test_lstm_shape():
    m = LSTMClassifier(n_features=FEAT)
    assert m(_batch()).shape == (BATCH, 3)

def test_deeplob_shape():
    m = DeepLOB()
    assert m(_batch()).shape == (BATCH, 3)

def test_transformer_shape():
    m = ControlledTransformer(n_features=FEAT, d_model=32, nhead=4, n_layers=1)
    assert m(_batch()).shape == (BATCH, 3)

def test_ofi_shape_and_features():
    x = _batch()
    m = OFIClassifier()
    assert m(x).shape == (BATCH, 3)
    feats = ofi_features_from_book(x.numpy())
    assert feats.shape == (BATCH, SEQ, 4)

def test_ablation_modes():
    x = _batch()
    for mode in ("deeplob", "ofi", "both"):
        m = FusionDeepLOB(mode=mode)
        assert m(x).shape == (BATCH, 3), mode

def test_dataset_falls_back_to_toy(tmp_path):
    ds = FI2010Dataset.load(root=tmp_path, k=1, seq_len=16, synthetic_events=60, synthetic_days=10, seed=0)
    assert ds.source == "synthetic_toy"
    assert ds.X.shape[1] == 16
    assert ds.X.shape[2] == 40
