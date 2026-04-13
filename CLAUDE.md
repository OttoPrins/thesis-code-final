# CLAUDE.md — Thesis Implementation: Deep Learning for CLV Prediction

Read this file at the start of every coding session. It is the single authoritative
reference for all implementation decisions.

---

## 1. Project Identity

**Research question (verbatim from proposal):**
"How does joint deep learning of transaction frequency and monetary value improve
customer lifetime value prediction in non-contractual retail?"

**Student:** Otto Prins | Erasmus University Rotterdam | Student number: 622671
**Supervisors:** Prof. Radek Karpienko (supervisor), Michel van de Velden (assessor)
**Companion writing project:** `../thesis-tf/` — LaTeX theoretical framework chapter
**Results from this repo feed directly into:** `../thesis-tf/results/` (tables and plots)

---

## 2. The Central Paper (replication target)

**Valendin, J., Reutterer, T., Platzer, M., & Kalcher, K. (2022).**
Customer base analysis with recurrent neural networks.
*International Journal of Research in Marketing*, 39(4), 988–1018.

Critical facts to remember every session:
- Published in **IJRM** — a marketing journal, NOT a machine learning conference.
- Proposes a **Base LSTM** (frequency only) and **Extended LSTM** (+ static/dynamic covariates).
- **Architecture:** Stacked LSTM layers → softmax classification head over discretised weekly counts.
- **Output:** Discretised weekly transaction count per customer: {0, 1, 2, 3+} (4 classes).
- **Training paradigm:** Self-supervised, rolling-window. The model is trained to predict the
  next week's transaction count from the full history up to that point.
- **Benchmarks evaluated against:** Pareto/NBD, Pareto/GGG (Platzer & Reutterer 2016), GPPM.
- **Key result:** Cohort-level forecast bias reduced from ~18% to ~2%; MAPE nearly halved.
- **8 empirical datasets used;** CDNOW is the primary benchmark — used for replication here.
- **Key gap the paper acknowledges:** NO monetary value prediction — only transaction counts.
- **Open-source code available:** GitHub (enables direct replication — find it before implementing).

---

## 3. Four-Stage Methodology

Following the proposal exactly:

**Stage 1 — Replication**
Reproduce the Base LSTM on CDNOW. Benchmark against Pareto/NBD, Pareto/GGG, GPPM.
Verify with: individual-level RMSE, cohort-level aggregate bias, MAPE.

**Stage 2 — Extension 1 (Joint Prediction)**
Add a regression output head predicting **log-transformed spending per period** in
parallel with the existing softmax classification head. Balance losses using
**Kendall et al. (2018) homoscedastic task-uncertainty weighting**.
Motivation: Closes the gap left by the gamma-gamma sub-model's independence assumption.

**Stage 3 — Extension 2 (Transformer Encoder)**
Implement Transformer encoder with **Time2Vec temporal embeddings** (Kazemi et al. 2019)
+ **sinusoidal positional encodings**. Evaluate under the same joint prediction framework.
NOTE: Proposal specifies Time2Vec + sinusoidal ONLY. Do NOT add tAPE or eRPE.

**Stage 4 — Extension 3 (Covariate Ablation)**
Quantify marginal contribution of **household demographics** (income, size) and
**campaign exposures** (coupon redemptions, marketing contacts) to frequency vs. spend.
Dataset: Dunnhumby "The Complete Journey" only. Use SHAP for attribution.

---

## 4. Models

### Probabilistic Benchmarks (to implement or wrap via lifetimes/pymc)

| Model | Citation | What it predicts | Key parameters |
|-------|----------|-----------------|----------------|
| Pareto/NBD | Schmittlein et al. (1987) | Frequency + latent churn | λ~Gamma(r,α), μ~Gamma(s,β) |
| BG/NBD | Fader, Hardie & Lee (2005) | Frequency + latent churn | Beta-Geometric dropout |
| Gamma-Gamma | Fader, Hardie & Lee (2005b) | Spend per transaction | z_i~Gamma(p,ν_i), ν_i~Gamma(q,γ) |
| Pareto/GGG | Platzer & Reutterer (2016) | Frequency + timing regularity | k=regularity parameter |
| GPPM | Dew & Ansari (2018) | Nonparametric propensity | Gaussian process |

**CRITICAL DISTINCTION:** Gamma-Gamma ≠ Pareto/GGG. They solve completely different problems.
- Gamma-Gamma (Fader et al. 2005b) = spend sub-model (how much spent per transaction)
- Pareto/GGG (Platzer & Reutterer 2016) = timing regularity model (when transactions happen)

### Deep Learning Models

| Model | Output heads | Temporal encoding |
|-------|-------------|-------------------|
| Base LSTM | Softmax (4-class frequency) | None (order via sequence) |
| Joint LSTM | Softmax + log-spend regression | None |
| Transformer Encoder | Softmax + log-spend regression | Time2Vec + sinusoidal PE |

**Shared architecture principle:** Frequency and spend heads share the same sequence
encoder. They must be trained jointly, not as two separate models.

---

## 5. Datasets

### CDNOW (Primary replication dataset)
- **Domain:** Music retail (CD purchases), USA, 1997–1998
- **Size:** 2,357 customers, ~69,659 transactions
- **Format:** `customer_id, date, num_cds, amount` (USD)
- **Standard split:** Calibration = first 52 weeks (Jan–Dec 1997), Holdout = next 52 weeks
- **Source:** Available from BTYD R package / Fader's Wharton page
- **Weekly aggregation:** 7-day bins from first purchase date
- **Notes:** 10% sample of the full dataset. Used in Fader et al. (2005) and Valendin et al. (2022).

### UCI Online Retail II (Generalisability check)
- **Domain:** UK e-commerce, 2009–2011
- **Size:** ~4,300 active customers after cleaning
- **Format:** `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`
- **Source:** UCI Machine Learning Repository (download directly from UCI)
- **Split:** First 18 months calibration, last 6 months holdout
- **Cleaning:** Drop cancelled invoices (InvoiceNo starts with 'C'), drop negative quantities,
  keep UK customers only (or all — decide and document), drop CustomerID = NaN

### Ta-Feng Grocery (High-frequency; expect Transformer to shine here)
- **Domain:** Grocery retail, Taiwan, 2000–2001
- **Size:** ~32,000 customers, ~817,741 transactions
- **Format:** `date, customer_id, product_id, product_subclass, amount_paid, quantity, asset`
- **Source:** Kaggle (IJCAI-15 competition data)
- **Split:** First 3 months calibration, last 1 month holdout (dense data, shorter windows work)
- **Notes:** High purchase frequency — the proposal specifically expects the Transformer
  to outperform LSTM here due to longer-range dependencies.

### Dunnhumby Complete Journey (Extension 3 primary)
- **Domain:** US grocery + coupon, 2 years
- **Size:** ~2,500 households
- **Key files:**
  - `transactions.csv` — household_id, basket_id, day, product_id, quantity, sales_value, store_id, retail_disc, trans_time, week_no, coupon_disc, coupon_match_disc
  - `hh_demographic.csv` — age_desc, marital_status_code, income_desc, homeowner_desc, hh_comp_desc, household_size_desc, kid_category_desc
  - `campaign_table.csv` — household_id, campaign, description, start_day, end_day
  - `coupon_redempt.csv` — household_id, day, coupon_upc, campaign
  - `causal_data.csv` — (optional) store-level marketing conditions
- **Source:** Dunnhumby website / Kaggle "The Complete Journey"
- **Covariates for Extension 3:**
  - Demographics: `income_desc` (encode as ordinal), `household_size_desc` (encode as ordinal)
  - Campaign exposure: coupon redemptions per week, campaign exposure flag per week
- **Split:** First 80 weeks calibration, last 4 weeks holdout (match Dunnhumby's week_no)

---

## 6. Data Pipeline Architecture

This is the **immediate priority** — implement before any model training.

### Design Principles
1. **Common interface:** Every dataset produces the same PyTorch-ready tensors.
2. **Extensible:** Adding a new dataset means creating one new file in `src/data/datasets/`.
3. **No leakage:** All normalization statistics computed on calibration set only.
4. **Configurable periods:** Calibration/holdout split determined by config YAML, not hardcoded.
5. **Covariate support:** Optional. Pipeline produces base tensors + optional covariate tensors.

### Pipeline Stages (for each dataset)

```
Raw files (data/raw/<dataset>/)
    ↓  [RawLoader] — reads original format, outputs pandas DataFrame
    ↓  [Cleaner]   — drops bad rows, normalizes column names, enforces dtypes
    ↓  [Aggregator]— bins transactions into weekly periods per customer
    ↓  [Splitter]  — separates calibration / holdout based on config
    ↓  [Scaler]    — fits MinMax/Standard scaler on calibration spend only
    ↓  [SequenceBuilder] — builds sliding-window sequences (X, y_freq, y_spend)
    ↓  [CovariateBuilder]— attaches static/dynamic covariates (optional)
    ↓  CustomerDataset  — PyTorch Dataset, returned by pipeline.build()
```

### Common Tensor Format

After the pipeline, every dataset produces:

```python
# X: (N_windows, T, F) — input sequences
# T = lookback window length (default 52 weeks)
# F = features: [weekly_freq, weekly_spend_log, time_since_first, ...optional RFM]

# y_freq: (N_windows,) — next-period transaction count, discretised {0,1,2,3+}
# y_spend: (N_windows,) — next-period total spend, log-transformed (0 if no purchase)

# covariates: (N_customers, C) or None — static per-customer features (Extension 3 only)
# customer_ids: (N_windows,) — needed to aggregate to cohort-level metrics

# mask: (N_windows, T) — 1 where sequence exists, 0 for padding (variable-length histories)
```

### File layout for `src/data/`

```
src/data/
├── __init__.py
├── pipeline.py          # Abstract BasePipeline class + run_pipeline() entry point
├── transforms.py        # WeeklyAggregator, TemporalSplitter, Scaler, SequenceBuilder
├── dataset.py           # CustomerDataset (PyTorch Dataset class)
├── collate.py           # custom collate_fn for variable-length sequences (padding)
└── datasets/
    ├── __init__.py
    ├── cdnow.py          # CDNOWLoader + CDNOWPipeline
    ├── uci_retail.py     # UCIRetailLoader + UCIRetailPipeline
    ├── tafeng.py         # TaFengLoader + TaFengPipeline
    └── dunnhumby.py      # DunnhumbyLoader + DunnhumbyPipeline (+ covariate builder)
```

### Key design decisions
- **Weekly aggregation is standard** (matches Valendin et al. 2022).
- **Spend transformation:** `log1p(spend)` — add 1 before log to handle zero-spend weeks.
- **Frequency discretisation:** {0: no purchase, 1: 1 purchase, 2: 2 purchases, 3: 3+}.
  This matches Valendin et al. (2022)'s 4-class softmax output.
- **Minimum sequence length:** Drop customers with fewer than 5 active weeks in calibration.
- **Padding:** Shorter sequences padded with zeros + mask tensor (do NOT use arbitrary fill values).
- **Lookback window T:** Configurable per dataset (default=52; Ta-Feng may use shorter windows).

---

## 7. Evaluation Protocol

**Strict temporal holdout** — no data from the holdout period is seen during training.

### Metrics

| Level | Metric | Formula / Notes |
|-------|--------|-----------------|
| Individual-level | RMSE | Root mean squared error of predicted vs. actual purchase count |
| Individual-level | MAE | Mean absolute error |
| Cohort-level | MAPE | Mean absolute percentage error of aggregated cohort predictions |
| Cohort-level | Bias | (sum(predicted) - sum(actual)) / sum(actual) × 100% |
| Monetary | MAE / RMSE | On log-transformed spend (primary) and raw spend (secondary) |
| Monetary | R² | Coefficient of determination for spend regression |

**Primary aggregate metric:** Quarterly revenue (sum over all customers × 13 weeks).
**Individual-level metrics** computed first; cohort-level aggregated from individuals.

### Comparison table structure (saves to `results/tables/`)

```
model             | dataset  | freq_rmse | freq_mape | spend_mae | spend_r2 | bias%
Pareto/NBD        | cdnow    | ...       | ...       | (N/A)     | (N/A)    | ...
BG/NBD            | cdnow    | ...       | ...       | (N/A)     | (N/A)    | ...
Gamma-Gamma*      | cdnow    | (N/A)     | (N/A)     | ...       | ...      | ...
Pareto/GGG        | cdnow    | ...       | ...       | (N/A)     | (N/A)    | ...
GPPM              | cdnow    | ...       | ...       | (N/A)     | (N/A)    | ...
Base LSTM         | cdnow    | ...       | ...       | (N/A)     | (N/A)    | ...
Joint LSTM        | cdnow    | ...       | ...       | ...       | ...      | ...
Transformer       | cdnow    | ...       | ...       | ...       | ...      | ...
```
*Gamma-Gamma is paired with Pareto/NBD or BG/NBD for spend; never listed as standalone frequency model.

---

## 8. Architecture Specifications

### LSTM (replication target — match Valendin et al. 2022 exactly)
- Input: sequence of weekly feature vectors
- Layers: 2 stacked LSTM layers (check paper for exact hidden size; ~64-128 units)
- Dropout: 0.2 between LSTM layers
- Frequency head: Linear → Softmax (4 classes: 0, 1, 2, 3+)
- Spend head (Extension 1): Linear → scalar (log-transformed spend)
- Loss: CrossEntropy (frequency) + MSE (log-spend), combined via Kendall et al. 2018

### Kendall et al. (2018) Multi-Task Loss
```
L = Σ_i [ L_i / (2σ_i²) + log(σ_i) ]
```
- σ_i = learnable task uncertainty parameter (one per task)
- Initialise σ_i = 1.0; let it optimise freely
- L_freq = CrossEntropyLoss; L_spend = MSELoss (on log-spend)

### Transformer Encoder (Extension 2)
- Input: sequence of weekly feature vectors
- Temporal encoding: Time2Vec (learnable) + sinusoidal positional encoding (fixed)
  - Time2Vec: ω·t + φ for linear component; sin(ω_k·t + φ_k) for periodic components
  - Sinusoidal PE: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)
- Encoder: Multi-head self-attention (n_heads=4 or 8) + FFN + LayerNorm + residual
- Depth: 2-3 encoder layers (keep simple; not full BERT scale)
- Prediction heads: same as Joint LSTM (Softmax + regression)
- NOTE: Do NOT use causal masking for the encoder (it sees full history at inference).
  This is an encoder-only model; sequence length = T history weeks.

---

## 9. Configuration Files

All experiments are configured via YAML in `experiments/configs/`. Never hardcode
hyperparameters in Python. Configs look like:

```yaml
# experiments/configs/lstm_joint_cdnow.yaml
dataset:
  name: cdnow
  lookback_weeks: 52
  holdout_weeks: 52
  min_active_weeks: 5
  freq_bins: [0, 1, 2, 3]  # 3 means 3+

model:
  type: lstm
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  joint: true  # enable spend head

training:
  epochs: 100
  batch_size: 256
  lr: 1e-3
  weight_decay: 1e-4
  early_stopping_patience: 10
  seed: 42

loss:
  type: kendall  # or 'fixed_weights'
  # if fixed_weights: freq_weight: 1.0, spend_weight: 0.5

output:
  results_dir: results/
  run_name: lstm_joint_cdnow_v1
```

---

## 10. Key Constraints and Non-Negotiables

| Constraint | Rule |
|------------|------|
| tAPE / eRPE | NEVER include these — proposal specifies Time2Vec + sinusoidal only |
| Pareto/GGG citation | ALWAYS Platzer & Reutterer (2016) — NEVER Abe et al. (2009) |
| Gamma-Gamma | This is the SPEND sub-model (Fader 2005b), NOT a frequency model |
| Joint learning | Frequency + spend MUST share sequence encoder; never train separately |
| Spend transform | Always log1p(spend); report metrics on both log and raw scale |
| Holdout leakage | ZERO. Fit all scalers/statistics on calibration set only |
| Valendin venue | IJRM (marketing journal) — not NeurIPS, ICML, or any CS venue |
| Null results | If extension doesn't help, report honestly — proposal frames this as informative |
| SHAP | Applied only on Dunnhumby (Extension 3), not globally |
| Transformer depth | Keep simple (2-3 layers); this is an encoder, not full BERT |

---

## 11. Libraries and Tools

```
torch>=2.0           # Core deep learning
pytorch-lightning    # Training loop management (optional but recommended)
numpy, pandas        # Data wrangling
scikit-learn         # Scalers, metrics, train/test splits
lifetimes            # Python package for Pareto/NBD and BG/NBD (pip install lifetimes)
pymc                 # For Pareto/GGG and GPPM if not available in lifetimes
shap                 # SHAP values for Extension 3
matplotlib, seaborn  # Plotting
pyyaml               # Config loading
tqdm                 # Progress bars
wandb                # (Optional) experiment tracking
```

**For probabilistic benchmarks:** The `lifetimes` Python library implements
Pareto/NBD and BG/NBD + Gamma-Gamma. Check if Pareto/GGG is available there;
if not, implement from scratch using the Platzer & Reutterer (2016) paper or
find an R-to-Python port. GPPM may require PyMC.

---

## 12. File Map

```
thesis-code/
├── CLAUDE.md                         ← This file. Read at session start.
├── prompts/
│   ├── session_start.md              ← Paste at start of every session
│   ├── data_pipeline_session.md      ← Session prompt for data work
│   ├── model_training_session.md     ← Session prompt for training
│   └── evaluation_session.md        ← Session prompt for evaluation
├── src/
│   ├── data/
│   │   ├── pipeline.py               ← Abstract BasePipeline
│   │   ├── transforms.py             ← Aggregator, Splitter, Scaler, SequenceBuilder
│   │   ├── dataset.py                ← CustomerDataset (PyTorch Dataset)
│   │   ├── collate.py                ← Padding + mask collation
│   │   └── datasets/
│   │       ├── cdnow.py              ← CDNOW loader + pipeline
│   │       ├── uci_retail.py         ← UCI loader + pipeline
│   │       ├── tafeng.py             ← Ta-Feng loader + pipeline
│   │       └── dunnhumby.py          ← Dunnhumby loader + pipeline (+ covariates)
│   ├── models/
│   │   ├── lstm.py                   ← LSTM encoder
│   │   ├── transformer.py            ← Transformer encoder + Time2Vec + sinusoidal PE
│   │   ├── heads.py                  ← FrequencyHead (softmax) + SpendHead (regression)
│   │   └── losses.py                 ← KendallMultiTaskLoss
│   ├── training/
│   │   ├── trainer.py                ← Training loop (or Lightning module)
│   │   └── callbacks.py              ← EarlyStopping, ModelCheckpoint
│   ├── evaluation/
│   │   ├── metrics.py                ← RMSE, MAE, MAPE, bias, R² implementations
│   │   ├── benchmarks.py             ← Pareto/NBD, BG/NBD, Pareto/GGG, GPPM wrappers
│   │   └── compare.py                ← Build comparison tables + plots
│   └── utils/
│       ├── config.py                 ← Load YAML configs
│       ├── seed.py                   ← Set random seeds everywhere
│       └── logging.py                ← Consistent logging setup
├── notebooks/
│   ├── 01_eda_cdnow.ipynb            ← EDA for CDNOW
│   ├── 02_eda_uci.ipynb              ← EDA for UCI Retail
│   ├── 03_eda_tafeng.ipynb           ← EDA for Ta-Feng
│   ├── 04_eda_dunnhumby.ipynb        ← EDA for Dunnhumby (include covariate distributions)
│   ├── 05_baseline_btyd.ipynb        ← Fit and evaluate probabilistic benchmarks
│   └── 06_results_comparison.ipynb  ← Final comparison tables + thesis plots
├── experiments/
│   └── configs/
│       ├── lstm_base_cdnow.yaml
│       ├── lstm_joint_cdnow.yaml
│       ├── transformer_joint_cdnow.yaml
│       ├── lstm_joint_uci.yaml
│       ├── lstm_joint_tafeng.yaml
│       ├── transformer_joint_tafeng.yaml
│       └── extension3_dunnhumby.yaml
├── results/
│   ├── tables/                       ← CSV comparison tables (thesis-ready)
│   ├── plots/                        ← Saved figures (PDF or PNG, 300 DPI)
│   └── checkpoints/                  ← Best model .pt files (gitignored)
├── data/
│   ├── raw/                          ← Original downloaded files (gitignored)
│   └── processed/                    ← Cleaned + preprocessed (gitignored for large files)
├── requirements.txt
└── README.md
```

---

## 13. Development Sequence (do this in order)

**Current priority: Data pipeline (Stage 0)**

- [ ] Download all 4 datasets to `data/raw/`
- [ ] Implement `src/data/transforms.py` (WeeklyAggregator, TemporalSplitter, Scaler, SequenceBuilder)
- [ ] Implement `src/data/dataset.py` (CustomerDataset + collate_fn)
- [ ] Implement `src/data/datasets/cdnow.py` — get CDNOW pipeline working end-to-end first
- [ ] Validate CDNOW pipeline: check shapes, inspect sequences, verify no leakage
- [ ] Implement remaining dataset pipelines (uci_retail, tafeng, dunnhumby)
- [ ] Write EDA notebooks for all 4 datasets

**Stage 1 — Replication**
- [ ] Fit Pareto/NBD and BG/NBD via `lifetimes` on CDNOW
- [ ] Fit Pareto/GGG (Platzer & Reutterer 2016) on CDNOW
- [ ] Fit GPPM (Dew & Ansari 2018) on CDNOW
- [ ] Implement Base LSTM + training loop; match Valendin et al. (2022) CDNOW results
- [ ] Compute RMSE, cohort bias, MAPE; save to `results/tables/`

**Stage 2 — Extension 1**
- [ ] Add SpendHead and KendallMultiTaskLoss
- [ ] Train Joint LSTM on CDNOW; compare to Pareto/NBD + Gamma-Gamma pipeline

**Stage 3 — Extension 2**
- [ ] Implement Time2Vec + sinusoidal PE
- [ ] Implement Transformer encoder
- [ ] Train Transformer on all 4 datasets; compare vs. LSTM

**Stage 4 — Extension 3**
- [ ] Build Dunnhumby covariate pipeline (demographics + campaign exposure features)
- [ ] Train Joint LSTM + Transformer with covariates on Dunnhumby
- [ ] Compute SHAP values; build ablation bar charts

---

## 14. Session Checklist

Before every coding session:
- [ ] Read `prompts/session_start.md` (or the specific session prompt)
- [ ] Identify which stage and task is being addressed today
- [ ] Check current git status; know what was last completed
- [ ] Verify data constraints: no holdout leakage, scaler fitted on calibration only
- [ ] After session: commit changes, save results with descriptive names
- [ ] If producing results: save to `results/` with format `{model}_{dataset}_{metric}.csv`
