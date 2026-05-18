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
- **Output:** Discretised weekly transaction count per customer (integer, clipped at some max).
- **Training paradigm:** Self-supervised, **sequence-to-sequence**. `return_sequences=True` — the
  model predicts at EVERY time step, not just the last. Loss computed across all steps.
- **Inference:** Autoregressive. Calibration history is fed in as a "seed" to warm up the LSTM's
  cell state; then future weeks are generated step-by-step, feeding each prediction back as input.
- **Benchmarks evaluated against:** Pareto/NBD, Pareto/GGG (Platzer & Reutterer 2016), GPPM.
- **Key result:** Cohort-level forecast bias reduced from ~18% to ~2%; MAPE nearly halved.
- **8 empirical datasets used;** CDNOW is the primary benchmark — used for replication here.
- **Key gap the paper acknowledges:** NO monetary value prediction — only transaction counts.
- **Open-source repo:** `../THesis DSMA/Code/rfm2lstm-main/` — contains `banking_transactions_demo.ipynb`.
  NOTE: The demo uses Czech banking data (trans.csv), NOT CDNOW. Written in **TensorFlow/Keras**,
  not PyTorch. Use it as an architectural reference, not as runnable code for this project.

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
- **Standard split:** Calibration = first 52 weeks (Jan–Dec 1997), Holdout = next 26 weeks (Jan–Jun 1998). Dataset ends at week 77; only 26 real holdout weeks available.
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
- **Split:**
  - Stages 1–3 (`lstm_base_dunnhumby`, `lstm_joint_dunnhumby`, `transformer_joint_dunnhumby`): 80 weeks calibration + 22 weeks holdout. The available Complete Journey transaction file ends at week 102, so 80 + 22 = 102 is the maximum observable window.
  - Extension 3 covariate ablation (`extension3_dunnhumby`): 80 weeks calibration + 4 weeks holdout, to match Dunnhumby's original week_no window and keep SHAP attribution on a short, dense period.

---

## 6. Data Pipeline Architecture

This section documents the implemented data pipeline architecture.

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
- **Input representation:** `week` (integer 0-51) and `transaction_count` (integer) are
  **categorical inputs**, passed through `nn.Embedding` layers — NOT raw floats.
  Embedding size heuristic from the repo: `int(max_val ** 0.5) + 1`.
  (week: emb_dim≈8; transaction count: emb_dim depends on max clipped value)
- **Spend (Extension 1 only):** log1p-transformed spend is an additional continuous input
  feature and a regression target. It is NOT part of the Base LSTM.
- **Frequency discretisation:** Clip transaction count at some maximum (check paper for exact
  value; repo clips at a value determined by the data, then uses all observed classes).
  Our config sets `freq_bins: [0, 1, 2, 3]` (3 means 3+) as the default — verify against paper.
- **Minimum sequence length:** Drop customers with fewer than 5 active weeks in calibration.
- **Sequence-to-sequence output:** `y_freq` has shape `(N, T)` — a label at every time step,
  not just the final step. The dataset must return the full target sequence, not a scalar.
- **Lookback window T:** The full calibration history is the input (no sliding window needed
  for training in the base replication). The LSTM is trained on the complete sequence.
  For Extension 1 and 2, decide whether to use full history or sliding windows.

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

Verified from the open-source repo (`banking_transactions_demo.ipynb`, TF/Keras). Our
implementation is in PyTorch but must match the architecture exactly.

**Input representation:**
- `week` — integer week-of-year (0–51), passed through an `Embedding` layer.
  Embedding size heuristic from the repo: `int(max_week ** 0.5) + 1` = 8 for max_week=52.
- `transaction_count` — integer count (clipped at some max), passed through an `Embedding` layer.
  Embedding size: `int(max_trans ** 0.5) + 1`.
- The two embeddings are concatenated along the feature dimension.

**Model:**
- `memory_units = 128` (confirmed from repo)
- `dense_units = 128` (Dense layer after LSTM, before softmax)
- Single LSTM layer with `return_sequences=True` (NOT 2 stacked layers — confirmed from repo)
- Training uses stateless LSTM; inference uses stateful LSTM (PyTorch equivalent: manage
  hidden state manually)
- Final softmax over transaction count classes

**Sequence-to-sequence training (critical):**
- The model predicts the transaction count at EVERY time step, not just the last.
- `return_sequences=True` → output shape (B, T, n_classes); loss applied at all T steps.
- This means during training, the loss is the mean CrossEntropy over all T time steps.

**Autoregressive inference (critical):**
1. Feed entire calibration history through the LSTM to warm up hidden state `(h, c)`.
2. Take the output at the last calibration step as first holdout prediction.
3. For each subsequent holdout week: feed the previous prediction + current week embedding
   into the LSTM (reusing the accumulated hidden state); sample from the softmax distribution.
4. Average multiple sampled scenarios (NO_SCENARIOS ≥ 20) to reduce sampling noise.

**For Extension 1 (joint prediction):** Add a parallel regression head on top of the same
LSTM. At each time step, the LSTM output goes to both the frequency softmax head AND the
spend regression head. Both losses are computed at every time step.

### Kendall et al. (2018) Multi-Task Loss
```
L = Σ_i [ L_i / (2σ_i²) + log(σ_i) ]
```
- σ_i = learnable log-variance parameter (one per task); use `log_var` parameterisation
- Initialise log_var = 0.0 (i.e. σ² = 1); let it optimise freely
- L_freq = mean CrossEntropyLoss over all T steps; L_spend = mean MSELoss over all T steps

### Transformer Encoder (Extension 2)
- Input: same embedding representation as LSTM (week + transaction count embeddings)
- Temporal encoding: Time2Vec (learnable) + sinusoidal positional encoding (fixed)
  - Time2Vec: ω·t + φ for linear component; sin(ω_k·t + φ_k) for periodic components
  - Sinusoidal PE: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)
- Encoder: Multi-head self-attention (n_heads=4 or 8) + FFN + LayerNorm + residual
- Depth: 2-3 encoder layers (keep simple; not full BERT scale)
- Output: also sequence-to-sequence (predict at every step) to match LSTM training paradigm
- **Use causal masking** (each position can only attend to past positions) — this is needed
  because the Transformer sees the full sequence at once during training but must not use
  future time steps. The LSTM is causal by construction; the Transformer is not.
- NOTE: Do NOT add tAPE or eRPE.

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
  early_stopping_patience: 5   # matches Valendin et al. repo (patience=5)
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
Pareto/NBD and BG/NBD + Gamma-Gamma. Pareto/GGG is available through the R
package `BTYDplus` (no dot in the package name) via `rpy2`; if that package is
not available, implement from scratch using the Platzer & Reutterer (2016)
paper or find an R-to-Python port. GPPM may require PyMC/CmdStan.

---

## 12. File Map

```
thesis-code/
├── CLAUDE.md                         ← This file. Read at session start.
├── AGENTS.md                         ← Agents SDK entry point (mirrors CLAUDE.md)
├── README.md
├── requirements.txt
├── requirements-kaggle.txt           ← Kaggle-specific dependencies
├── train.py                          ← Main training entry point
├── run_benchmarks.py                 ← Probabilistic benchmark runner
├── run_seeds.py                      ← Multi-seed experiment runner
├── run_full_analysis.sh              ← End-to-end analysis script
├── validate_pipelines.py             ← Pipeline validation utility
├── validate_final_setup.py           ← Pre-run setup validation
├── push_to_kaggle.sh                 ← Push notebook to Kaggle
├── upload_data_to_kaggle.sh          ← Upload data to Kaggle dataset
├── kernel-metadata.json              ← Kaggle kernel config
├── dataset-metadata.json             ← Kaggle dataset config
├── Empirical_Papers/                 ← 21 research PDFs + literature_matrix.xlsx
├── Resuts_Kaggle_Notebook/           ← Kaggle run outputs (checkpoints/, tables/)
├── prompts/
│   ├── session_start.md
│   ├── data_pipeline_session.md
│   ├── model_training_session.md
│   ├── evaluation_session.md
│   └── run_experiments.md
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
│   │   ├── trainer.py                ← Training loop
│   │   ├── callbacks.py              ← EarlyStopping, ModelCheckpoint
│   │   └── inference.py              ← Autoregressive inference + scenario sampling
│   ├── evaluation/
│   │   ├── metrics.py                ← RMSE, MAE, MAPE, bias, R²
│   │   ├── benchmarks.py             ← Pareto/NBD, BG/NBD, Pareto/GGG, GPPM wrappers
│   │   ├── compare.py                ← Build comparison tables + plots
│   │   ├── calibration.py            ← Calibration diagnostics
│   │   ├── significance.py           ← Statistical significance tests
│   │   ├── rolling_origin.py         ← Rolling-origin cross-validation
│   │   ├── shap_analysis.py          ← SHAP attribution (Extension 3)
│   │   ├── rescore.py                ← Re-scoring from saved arrays
│   │   └── stan/                     ← Stan models for Bayesian benchmarks
│   └── utils/
│       ├── config.py                 ← Load YAML configs
│       ├── seed.py                   ← Set random seeds everywhere
│       └── final_manifest.py         ← Finalization manifest helper
├── notebooks/
│   ├── 01_canary_cdnow.ipynb         ← CDNOW pipeline canary + quick EDA
│   ├── kaggle_runner.ipynb           ← Kaggle cloud training wrapper
│   └── 06_results_comparison.ipynb  ← Final comparison tables + thesis plots
├── tests/
│   ├── test_data_contracts.py
│   ├── test_finalization_contracts.py
│   └── test_gppm_recovery.py
├── experiments/
│   ├── configs/                      ← 29 YAML experiment configs
│   │   ├── lstm_base_{cdnow,dunnhumby,tafeng,uci}.yaml
│   │   ├── lstm_joint_{cdnow,dunnhumby,tafeng,uci}{,_v2}.yaml
│   │   ├── transformer_joint_{cdnow,dunnhumby,tafeng,uci}{,_v2}.yaml
│   │   └── extension3_{lstm,transformer}_{none,static,dynamic,full}_dunnhumby.yaml
│   ├── insights/                     ← Comparison plots
│   └── final_manifest.yaml
├── results/
│   ├── tables/                       ← JSON/CSV/NPZ result files (172 files as of 2026-05-18)
│   ├── plots/                        ← Saved thesis figures (to be populated)
│   ├── checkpoints/                  ← Best model .pt files (gitignored)
│   ├── logs/                         ← Training logs
│   └── archive/                      ← Pre-final results snapshot
├── data/
│   ├── raw/                          ← Original downloaded files (gitignored)
│   └── processed/                    ← Cleaned + preprocessed (gitignored for large files)
└── venv/                             ← Python virtual environment (gitignored)
```

---

## 13. Development Sequence

**Stage 0 — Data Pipeline (complete)**
- [x] Download all 4 datasets to `data/raw/`
- [x] Implement `src/data/transforms.py` (WeeklyAggregator, TemporalSplitter, Scaler, SequenceBuilder)
- [x] Implement `src/data/dataset.py` (CustomerDataset + collate_fn)
- [x] Implement `src/data/datasets/cdnow.py` — CDNOW pipeline working end-to-end
- [x] Validate CDNOW pipeline: shapes verified via `01_canary_cdnow.ipynb`
- [x] Implement remaining dataset pipelines (uci_retail, tafeng, dunnhumby)
- [ ] Write full EDA notebooks for UCI, Ta-Feng, Dunnhumby (not done; `01_canary_cdnow.ipynb` exists for CDNOW only)

**Stage 1 — Replication (complete)**
- [x] Fit Pareto/NBD and BG/NBD + Gamma-Gamma on all 4 datasets
- [x] Fit Pareto/GGG (Platzer & Reutterer 2016) on CDNOW
- [x] Fit GPPM (Dew & Ansari 2018) on CDNOW
- [x] Implement Base LSTM + training loop; trained on all 4 datasets (3 seeds each)
- [x] Compute RMSE, cohort bias, MAPE; saved to `results/tables/`

**Stage 2 — Extension 1 (complete)**
- [x] Add SpendHead and KendallMultiTaskLoss
- [x] Train Joint LSTM on all 4 datasets (3 seeds each)

**Stage 3 — Extension 2 (complete)**
- [x] Implement Time2Vec + sinusoidal PE
- [x] Implement Transformer encoder
- [x] Train Transformer on all 4 datasets (3 seeds each)
- [x] `comparison_all.csv` and `comparison_all.tex` generated

**Stage 4 — Extension 3 (partial)**
- [x] Build Dunnhumby covariate pipeline (demographics + campaign exposure)
- [x] Train extension3 (none + static covariates) for LSTM + Transformer, 3 seeds
- [ ] Train extension3 dynamic + full covariate variants
- [ ] Compute SHAP values for covariate attribution
- [ ] Generate thesis plots (save to `results/plots/`, 300 DPI)
- [ ] Finalize `06_results_comparison.ipynb` with all final figures

---

## 14. Session Checklist

Before every coding session:
- [ ] Read `prompts/session_start.md` (or the specific session prompt)
- [ ] Identify which stage and task is being addressed today
- [ ] Check current git status; know what was last completed
- [ ] Verify data constraints: no holdout leakage, scaler fitted on calibration only
- [ ] After session: commit changes, save results with descriptive names
- [ ] If producing results: save to `results/` with format `{model}_{dataset}_{metric}.csv`
