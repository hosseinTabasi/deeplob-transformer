# Limit Order Book Forecasting: DeepLOB, TransLOB, and a Controlled Transformer

**Author:** Hossein Tabasi (2026). MIT licence. Junior research scaffold, not a trading system.

## Question

On FI-2010 (and optionally one extra market), how do MLP, LSTM, CNN-DeepLOB,
TransLOB-style attention, and a small controlled Transformer compare for
*mid-price move classification* at horizons k in {1, 2, 3, 5, 10}, under
(i) the original Zhang et al. Setup 2 split (first 7 days train, last 3 days
test) and (ii) a stricter chronological split (days 1-6 train, 7 validation,
8-10 test), with class-imbalance reporting via **macro-F1** rather than
accuracy? We do **not** claim tradable PnL, Sharpe, or execution-aware profit.

## Why it matters

Limit-order-book (LOB) mid-price classification is a standard probe of whether
deep models extract spatial structure (price-volume across levels) beyond
handcrafted imbalance. DeepLOB (Zhang, Zohren, Roberts, IEEE TSP 2019;
arXiv:1808.03668) and TransLOB (Wallbridge, arXiv:2003.00130) are widely
cited, but later work (e.g. arXiv:2403.09267) notes that reported accuracy
is sensitive to label thresholds, day-wise leakage, and class balance.
A controlled comparison that (a) keeps the original 7/3 protocol, (b) adds a
stricter chronological cut, and (c) reports macro-F1 is a prerequisite for
any claim that a Transformer improves on CNN-LSTM. This repository is the
*experimental skeleton* for that comparison. It is not a result paper.

## Data

**Scientific dataset:** FI-2010 (Ntakaris et al., Journal of Forecasting 2018):
five Nasdaq Nordic stocks, ten trading days, ten book levels, 40 raw features
per event. Place the archive under `data/raw/FI-2010/` (see
`scripts/download_fi2010.md`). This repo does not ship or scrape FI-2010.

**Toy fallback:** if FI-2010 is absent, `src/data/fi2010.py` builds a
stationary-ish 10-level bid/ask panel used only for shape tests and a
one-epoch smoke train. Every artefact from that path is labelled **TOY**.

Zhang et al. report horizons k=10, 20, 50 (event steps on downsampled
FI-2010), not k=1,2,3,5,10. We keep k=1,2,3,5,10 as the *research* grid
and will, when FI-2010 is available, also report k=10,20,50 to sit next
to the public DeepLOB tables.

## Method

- Input window: T events by 40 features, layout
  `[p_a^(i), v_a^(i), p_b^(i), v_b^(i)]_{i=1}^{10}` as in Zhang et al.
- Labels: Ntakaris smoothed future-mid percentage change, 3 classes
  (down / stationary / up), threshold `alpha`.
- Models (`src/models/`): MLP (flattened), LSTM, reduced-width DeepLOB
  (spatial 1x2 convolutions, inception along time, LSTM; original code,
  not a GPL dump), small Transformer encoder, OFI+imbalance MLP.
- Ablation: OFI-only vs DeepLOB vs concatenated encodings (`FusionDeepLOB`).
- Training: `src/train/train.py`, YAML configs, fixed seed, CPU-tiny
  defaults in `configs/toy.yaml`.

## Baselines

| Model | Role |
|---|---|
| MLP | unordered / flattened control |
| LSTM | temporal control without spatial CNN |
| DeepLOB (small) | Zhang et al. 2019 architecture in spirit |
| Controlled Transformer | TransLOB-inspired encoder, reduced width |
| OFI + imbalance | handcrafted microstructure features |

## Results

**NO FULL RESULTS YET.** FI-2010 has not been run. Any numbers in
`results/tables/toy_smoke.csv` are **TOY-only** (synthetic book, 1 epoch,
not comparable to Zhang et al. Table I/II). Do not cite them as empirical
findings. Macro-F1 on toy data is an integration check, not a forecast of
FI-2010 performance.

Paper split vs chronological split will be filled after a real FI-2010 pass.
We will not fill those cells with invented percentages.

## Limitations

Toy dynamics are near-stationary and inject a weak L1-imbalance signal so
that a smoke train can overfit. Real LOBs are non-stationary, irregularly
sampled, and dominated by cancellations. Classification of mid-price ticks
is not a trading strategy (spread, queue, fees, and impact are absent).

## Reproduce (toy)

```
PYTHONPATH=src python -m train.train --config configs/toy.yaml
PYTHONPATH=src python -m pytest -q
```

## References

Zhang, Zohren, Roberts (2019), arXiv:1808.03668. Wallbridge (2020),
arXiv:2003.00130. Ntakaris et al. (2018), FI-2010. LOB-ML survey
arXiv:2403.09267.
