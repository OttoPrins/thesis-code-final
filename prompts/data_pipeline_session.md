# Data Pipeline Session Prompt

Paste this at the start of a data preparation session.

---

Read CLAUDE.md. We are working on the **data pipeline** — Stage 0 of the thesis implementation.

## Goal for this session

Build a modular, extensible data pipeline that converts raw transaction files into
PyTorch-ready tensors. Every dataset must produce the same tensor format so models
can be trained identically across all 4 datasets.

## Required output format

Every dataset pipeline must produce tensors with these shapes:

```python
X           : (N_windows, T, F)   # input sequences; T=lookback weeks, F=features
y_freq      : (N_windows,)         # next-period frequency class {0,1,2,3}
y_spend     : (N_windows,)         # next-period log-spend (log1p); 0.0 if no purchase
customer_id : (N_windows,)         # to aggregate cohort-level metrics later
mask        : (N_windows, T)       # 1 = real data, 0 = padding
# Optional:
covariates  : (N_customers, C)    # static per-customer features (Dunnhumby Extension 3 only)
```

## Pipeline stages (implement in this order)

1. **RawLoader** — reads original CSV/file format, returns clean pandas DataFrame with
   standardised columns: `[customer_id, date, transaction_amount]`
2. **Cleaner** — removes cancelled orders, negative amounts, null customer IDs
3. **WeeklyAggregator** — groups by `(customer_id, week_number)`, computes:
   - `weekly_freq`: count of transactions in that week (then discretise to {0,1,2,3+})
   - `weekly_spend`: sum of `transaction_amount` in that week
4. **TemporalSplitter** — separates calibration vs. holdout by week number
   (uses config YAML; never hardcode dates/weeks)
5. **Scaler** — fits MinMaxScaler or StandardScaler on calibration spend ONLY,
   transforms both calibration and holdout (no leakage)
6. **SequenceBuilder** — creates sliding windows of length T over each customer's
   weekly history; labels are next week's freq and spend
7. **CovariateBuilder** — (Dunnhumby only) attaches demographics + campaign features
8. **CustomerDataset** — wraps into PyTorch Dataset; includes padding + mask

## Files to implement

```
src/data/pipeline.py          ← Abstract BasePipeline class with run() method
src/data/transforms.py        ← WeeklyAggregator, TemporalSplitter, Scaler, SequenceBuilder
src/data/dataset.py           ← CustomerDataset(Dataset) + collate_fn
src/data/collate.py           ← Pads sequences to max length in batch, builds mask
src/data/datasets/cdnow.py    ← Start here; implement CDNOWLoader + CDNOWPipeline
```

## Start with CDNOW

CDNOW is simplest: two columns (`date`, `amount`), clean data, canonical split.
Get the full pipeline working for CDNOW end-to-end before touching other datasets.

Validation checks to run after CDNOW pipeline:
- Shape of X: should be (N, 52, F) where N = number of sliding windows
- y_freq distribution: check counts for each class {0,1,2,3}
- y_spend: verify log1p applied; check no negative values
- mask: verify 0s only appear at the start of sequences (left-padding or right-padding — be consistent)
- No customer appears in both calibration and holdout (impossible by construction, but verify)
- Scaler: verify it was fitted ONLY on calibration data

## Design constraints

- `BasePipeline.run(config: dict) -> (train_dataset, val_dataset, test_dataset)`
- Config passed from YAML file (see `experiments/configs/`)
- All random operations seeded via `src/utils/seed.py`
- Save processed tensors to `data/processed/<dataset>/` to avoid recomputing
- Each dataset class should be importable as:
  ```python
  from src.data.datasets.cdnow import CDNOWPipeline
  train, val, test = CDNOWPipeline().run(config)
  ```

## After CDNOW works, implement in this order:
1. UCI Online Retail II (similar structure but needs more cleaning)
2. Ta-Feng Grocery (note: may need shorter lookback window due to dense data)
3. Dunnhumby (most complex — add covariate builder last)

Please read CLAUDE.md and then start with `src/data/transforms.py` and `src/data/datasets/cdnow.py`.
Before writing any code, explain your planned class structure and I will confirm.
