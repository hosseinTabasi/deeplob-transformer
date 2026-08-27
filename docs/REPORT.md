# Workshop outline: Limit Order Book Forecasting with DeepLOB, TransLOB, and a Controlled Transformer

Hossein Tabasi, 2026. This document is a methods outline for a workshop-length
write-up. It describes the intended empirical study. **No full FI-2010
experiment has been run.** Sections that would normally contain tables of
accuracy and macro-F1 are explicit placeholders.

## 1. Motivation and research question

Electronic limit order books are the primary mechanism of price formation
in cash equities. At each event the book is a 2-by-L array of prices and
sizes. Predicting the sign of the next mid-price change is a well-studied
but still poorly identified problem: labels are threshold-dependent, the
unconditional class distribution is unbalanced, and standard train/test
cuts on ten days of FI-2010 can leak calendar structure. The question this
project asks is deliberately narrow.

On FI-2010, how do five model families — a flattened MLP, an LSTM on raw
features, a reduced-width DeepLOB CNN-LSTM, a TransLOB-inspired Transformer,
and a handcrafted OFI-plus-imbalance MLP — compare for three-class
mid-price classification at horizons k in {1,2,3,5,10}, under the original
Zhang et al. Setup 2 split (seven training days, three test days) and under
a stricter chronological split (train 1-6, validation 7, test 8-10)? The
primary metric is macro-F1, not accuracy. We do not estimate PnL.

The restriction to classification, and the refusal to convert class labels
into a backtest, is methodological rather than aesthetic. Mid-price
labels ignore the spread, queue priority, and fees. A model that is
accurate at k=10 on downsampled Nordic data from 2010 need not be a trading
signal on a 2026 lit venue. Keeping the task as classification lets us
compare architectures on the same public protocol that the literature
already uses, without inflating the claim.

## 2. Related work (what we actually cite)

Ntakaris, Magris, Kanniainen, Gabbouj and Iosifidis (Journal of Forecasting,
2018) released FI-2010: five Nasdaq Nordic names, ten days, ten levels,
with pre-computed z-score features and smoothed mid-price labels. Zhang,
Zohren and Roberts (IEEE Transactions on Signal Processing, 2019;
arXiv:1808.03668) proposed DeepLOB: spatial convolutions over
price-volume pairs and adjacent levels, an inception module along time,
and an LSTM. Their public tables use k=10, 20, 50 (and 100 in Setup 1),
not k=1,2,3,5,10. Setup 2 (7/3 days) is the deep-learning convention;
Setup 1 is an anchored day-forward fold. We will report both Setup 2 and
a stricter chronological cut; we will not pretend Setup 1 has been run
until it has.

Wallbridge (arXiv:2003.00130) replaced the recurrent stack with a
Transformer (TransLOB). Subsequent surveys, including arXiv:2403.09267,
catalogue CNN, RNN, and attention models on LOBs and document that
headline accuracy is fragile to label definition and to whether the
stationary class dominates. Cont, Kukanov and Stoikov style order-flow
imbalance remains the natural handcrafted baseline: if a deep model cannot
beat a two-feature imbalance classifier on macro-F1, the spatial CNN is
not doing identifiable work.

We do not treat GitHub dumps of DeepLOB as ground truth. The
implementation in `src/models/deeplob.py` follows the paper description
(1x2 stride-2 spatial stack, inception, LSTM) at reduced width so that a
CPU smoke train is feasible. It is not a copy of any GPL repository.

## 3. Data, labels, and splits

FI-2010 is the scientific sample. The loader looks for `data/raw/FI-2010/`
and, if the directory is empty, generates a synthetic ten-level book.
Synthetic paths are labelled `synthetic_toy` and every CSV row they produce
is tagged TOY. The synthetic generator is a slow random walk mid with
exponential depth and an optional injected L1-imbalance drift. It exists
so that tensor-shape tests and a one-epoch overfit check can run without
the proprietary-adjacent public archive. It is not a market simulator and
must not appear in a results table next to Zhang et al. without a TOY
flag.

Labels follow the Ntakaris construction: the relative difference between
the current mid and the mean of the next k mids, thresholded at `alpha`.
Classes are down, stationary, and up. We will report class counts per
split because FI-2010 is known to be unbalanced, which is why accuracy is
an insufficient headline metric.

Two split protocols are implemented.

- `paper_setup2`: days 0-6 train, days 7-9 test, no validation day, matching
  Zhang et al. Table II / Setup 2. Windows never cross day boundaries.
- `chronological`: days 0-5 train, day 6 validation, days 7-9 test. This is
  the robustness check. It is not the original protocol; if DeepLOB
  degrades under it, that is a finding about the 7/3 convention, not a
  failure to replicate.

We have not yet obtained FI-2010, so neither split has been evaluated on
real events. Optional extra-market data (e.g. a second public LOB) is out
of scope until FI-2010 is done.

## 4. Models and ablation

Five architectures share the same input tensor shape (B, T, 40) and the
same three-way head.

The MLP flattens the window. It is a control for "any nonlinear function
of the concatenated book" without temporal inductive bias. The LSTM reads
the 40-vector sequence. DeepLOB applies (1,2) convolutions with stride 2
to pair price with volume, then adjacent levels, then a (1,10) mix across
the ten levels, an inception block (1/3/5/pool) along time, and an LSTM.
The controlled Transformer linearly embeds the 40-vector, adds sinusoidal
positions, and uses two encoder layers with mean pooling. The OFI encoder
summarises signed size changes and L1 / depth imbalance, then an MLP.

The ablation hook `FusionDeepLOB` concatenates the DeepLOB LSTM state with
the OFI summary (`mode=both`), or uses either branch alone. If `both` does
not improve macro-F1 over DeepLOB on FI-2010, we will report that as a
negative result rather than dropping the cell.

Training uses Adam, cross-entropy, a YAML config, and a single seed in the
toy path. The full study will use several seeds and will not tune on the
test days. Early stopping, if used, will be on the chronological
validation day only, never on Setup 2 test days.

## 5. Metrics, what is not being claimed

Primary: macro-F1. Secondary: per-class precision and recall, accuracy
(for comparability with Zhang et al. Table I/II only), and class counts.
We will not report Sharpe, hit rate of a mid-mid simulator, or any profit
figure. Zhang et al. include a trading simulation in their paper; we
deliberately omit that module because it is not identified without
fees, queue, and impact, and because the present goal is architectural
comparison under a public classification protocol.

A further honesty constraint: Zhang et al. public numbers are at k=10,20,50
on FI-2010. Our k-grid includes shorter horizons that they did not
headline. When we fill tables, k=10 will be the overlap cell; k=1,2,3,5
are additional and cannot be compared to Table II.

## 6. Status of computation (honest)

As of this writing:

- Unit tests on synthetic batches check tensor shapes for every model and
  check that train and test day indices are disjoint under both splits.
- A one-epoch smoke train on the toy generator may exist in
  `results/tables/toy_smoke.csv`. Those numbers are TOY.
- FI-2010 is not on disk. No DeepLOB, TransLOB, or MLP number in this
  repository is an FI-2010 result.
- We have not reproduced Zhang et al. Table I or Table II. Replication is
  the companion repository `paper-replication-plus`.

This outline is therefore a pre-analysis plan plus a software skeleton.
Filling the results section before running FI-2010 would be fabrication.

## 7. Planned tables (empty by design)

Table A: Setup 2, macro-F1 and accuracy, k in {1,2,3,5,10}, five models.
Table B: chronological split, same grid. Table C: OFI vs DeepLOB vs both
at k=10. Table D: class counts. All cells: "not run".

When Table A is filled, we will also copy Zhang et al. Setup 2 DeepLOB
accuracy/F1 at k=10,20,50 from the public PDF (already extracted into
`paper-replication-plus/ORIGINAL_NUMBERS.csv`) as a *literature* column,
clearly marked as their numbers, not ours.

## 8. Risks and limitations

Ten days is a short panel. Deep models can fit day-specific artefacts.
The 7/3 split does not leave a clean validation day; any hyperparameter
choice made while looking at Setup 2 test numbers is leakage. Reduced-width
DeepLOB is not the 60k-parameter paper network; a gap versus Table II may
reflect width rather than a failed idea. FI-2010 is downsampled and
pre-normalised; LSE-style raw event data (the second dataset in Zhang et
al.) is not public here. Synthetic TOY data has an injected imbalance
signal and will overstate learnability.

## 9. Next actions

1. Obtain FI-2010 under the authors' terms; do not scrape.
2. Freeze configs and seeds before the first full train.
3. Run Setup 2 and chronological grids; write tables with CIs across seeds.
4. Only then compare to public DeepLOB cells at overlapping k.
5. Keep PnL out of the paper.

## 10. Workshop narrative (what to say in 12 minutes)

One slide on the question and the two splits. One slide on why macro-F1.
One slide on the five models and the OFI ablation. One slide that is
visibly empty: "FI-2010 not run; toy smoke only." One slide on what
Zhang et al. actually reported (k=10,20,50; Setup 1 vs 2). Closing: the
contribution of this stage is a reproducible protocol, not a leaderboard
number. If the audience remembers one sentence, it should be that
accuracy on an unbalanced mid-price task is not a trading result, and
that we have not yet produced even the classification result on the
real book.

## 11. Implementation notes for replicators

`src/data/fi2010.py` owns loading and splits. `src/models/` owns
architectures. `src/train/train.py` is the only training entry.
`configs/toy.yaml` is CPU-safe. `configs/default.yaml` is the intended
FI-2010 DeepLOB run and will silently fall back to synthetic data if the
archive is missing — which is why the CSV `source` column must be checked
before any number is copied into a paper. Tests live in
`tests/test_shapes.py` and must pass without FI-2010. CI is
`.github/workflows/ci.yml`.

The companion replication repository will consume numbers from this one.
Until FI-2010 is run, that companion can only hold literature values and
TOY smoke rows.

## 12. Conclusion of the outline

The scientific object is a comparison of inductive biases on a public LOB
classification task, with an explicit dual-split protocol and an imbalance
metric. The software exists. The empirical object does not, yet. This
report stops there.
