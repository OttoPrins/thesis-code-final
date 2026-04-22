# Run Experiments Guide

Complete workflow for reproducing all thesis results from raw data to final tables and figures.

---

## 1. Prerequisites

- Python 3.10+ with virtual environment activated
- ~10 GB free disk space (raw data + checkpoints)
- macOS (MPS) or CUDA GPU recommended; CPU works but is slow for large datasets

```bash
pip install -r requirements.txt
```

---

## 2. Raw Data Download & Placement

```
data/raw/
├── cdnow/            ← cdnow_sample.txt  (Fader Wharton page or BTYD R package)
├── UCI Retail/       ← online_retail_II.xlsx  (UCI Machine Learning Repository)
├── TaFeng/           ← ta_feng_all_months_merged.csv  (Kaggle IJCAI-15)
└── Dunnhumby datasets/
    ├── transaction_data.csv
    ├── hh_demographic.csv
    ├── campaign_table.csv
    ├── campaign_desc.csv
    └── coupon_redempt.csv     (Kaggle "The Complete Journey")
```

---

## 3. Validate Pipelines

Run the smoke test after placing raw data. Checks tensor shapes, hold-out leakage,
customer counts, and covariate shapes for Dunnhumby.

```bash
python validate_pipelines.py                   # all datasets
python validate_pipelines.py --dataset cdnow   # single dataset
```

For Dunnhumby with covariates (Extension 3), the covariate tensor should be
`(N_customers, T_total, 4)` — verify this is printed correctly.

---

## 4. Stage 0 — Probabilistic Benchmarks

```bash
python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_nbd bgnbd_gg
python run_benchmarks.py --config experiments/configs/lstm_base_uci.yaml --models pareto_nbd bgnbd_gg
python run_benchmarks.py --config experiments/configs/lstm_base_tafeng.yaml --models pareto_nbd bgnbd_gg
python run_benchmarks.py --config experiments/configs/lstm_base_dunnhumby.yaml --models pareto_nbd bgnbd_gg
```

Fits Pareto/NBD, BG/NBD + Gamma-Gamma, Pareto/GGG, and GPPM.
Results saved to `results/tables/{model}_{dataset}_metrics.json`.

---

## 5. Stage 1 — Base LSTM (Replication of Valendin et al. 2022)

Strict replication: single LSTM layer, Dense(128), softmax over frequency classes,
Monte Carlo inference with n_scenarios=30 (configurable via `inference.n_scenarios`).

```bash
python train.py --config experiments/configs/lstm_base_cdnow.yaml
python train.py --config experiments/configs/lstm_base_uci.yaml
python train.py --config experiments/configs/lstm_base_tafeng.yaml
python train.py --config experiments/configs/lstm_base_dunnhumby.yaml
```

Target on CDNOW (from Valendin et al.): cohort bias ~2%, MAPE ≈ half of Pareto/NBD.

---

## 6. Stage 2 — Joint LSTM (Extension 1: frequency + spend)

Adds a parallel log-spend regression head. Both heads share the LSTM encoder.
Loss balanced by Kendall et al. (2018) learnable homoscedastic uncertainty.
Learned task weights are logged per epoch in `*_history.json`.
Raw-currency spend metrics (`spend_mae_raw`, `spend_rmse_raw`) auto-computed.

```bash
python train.py --config experiments/configs/lstm_joint_cdnow.yaml
python train.py --config experiments/configs/lstm_joint_uci.yaml
python train.py --config experiments/configs/lstm_joint_tafeng.yaml
python train.py --config experiments/configs/lstm_joint_dunnhumby.yaml
```

The `*_metrics.json` will include: `spend_mae`, `spend_rmse`, `spend_r2`,
`spend_mae_raw`, `spend_rmse_raw`.

---

## 7. Stage 3 — Transformer Encoder (Extension 2)

Adds Time2Vec + sinusoidal PE. Autoregressive inference uses KV caching for
O(H) complexity per scenario (vs. O(H·T²) without cache). Toggle via `inference.use_kv_cache`.

```bash
python train.py --config experiments/configs/transformer_joint_cdnow.yaml
python train.py --config experiments/configs/transformer_joint_uci.yaml
python train.py --config experiments/configs/transformer_joint_tafeng.yaml
python train.py --config experiments/configs/transformer_joint_dunnhumby.yaml
```

Expect the Transformer to most outperform LSTM on Ta-Feng (high-frequency, longer
range dependencies). Report both datasets in the thesis comparison table.

---

## 8. Stage 4 — Dunnhumby Covariate Ablation (Extension 3)

Trains the joint LSTM and Transformer with time-varying household covariates:
`[income_ordinal, hh_size_ordinal, coupon_redemptions_per_week, campaign_active_flag]`.

The covariate tensor has shape `(N_customers, T_total, 4)` where
`T_total = calibration_weeks + holdout_weeks`. Static demographics are broadcast across
all weeks; campaign and coupon features vary per week. Campaign exposure in the holdout
period represents firm-controlled marketing decisions known at prediction time — this is
standard CLV planning context, not leakage.

### 8a. Train with covariates

```bash
python train.py --config experiments/configs/extension3_dunnhumby.yaml
```

Key config fields:
```yaml
dataset:
  include_covariates: true
model:
  covariate_dim: 4
  covariate_emb_dim: 8   # optional; default 8
inference:
  n_scenarios: 30
  use_kv_cache: true     # Transformer only
```

### 8b. SHAP attribution (dual-head)

Attributes over the 4 covariate features using `KernelExplainer`. Produces
separate bar charts for the frequency head and the log-spend head.

```bash
python -m src.evaluation.shap_analysis \
    --config experiments/configs/extension3_dunnhumby.yaml \
    --checkpoint results/checkpoints/extension3_dunnhumby_v1.pt \
    --n_background 100 \
    --n_explain 200 \
    --out_dir experiments/insights
```

Outputs:
- `experiments/insights/shap_freq_dunnhumby.png`
- `experiments/insights/shap_spend_dunnhumby.png`
- `experiments/insights/shap_freq_values.npy` (raw SHAP values for custom plots)

---

## 9. Aggregation & Visual Analytics

After all experiments, compile results and generate thesis-ready outputs in one command:

```bash
python -m src.evaluation.compare --all
```

Or selectively:

```bash
python -m src.evaluation.compare --latex   # → results/tables/comparison_all.tex
python -m src.evaluation.compare --plots   # → experiments/insights/comparison_*.png
python -m src.evaluation.compare --weights # → experiments/insights/kendall_weights_*.png
```

**Outputs:**
| File | Description |
|------|-------------|
| `results/tables/comparison_all.csv` | Full metrics DataFrame |
| `results/tables/comparison_all.tex` | LaTeX booktabs table (`\input{}` into thesis) |
| `experiments/insights/comparison_freq_mape.png` | Grouped bar chart — frequency MAPE |
| `experiments/insights/comparison_bias_pct.png` | Grouped bar chart — cohort bias |
| `experiments/insights/comparison_spend_mae_raw.png` | Grouped bar chart — raw spend MAE |
| `experiments/insights/kendall_weights_{run}.png` | Task weight evolution per joint run |

---

## 10. Sanity-Check Reference Numbers

| Model | Dataset | Expected freq MAPE | Expected bias % |
|-------|---------|-------------------|----------------|
| Pareto/NBD | CDNOW | ~18% | ~+15% |
| Base LSTM | CDNOW | ~9–11% | ~2% |
| Joint LSTM | CDNOW | similar freq; + spend metrics |
| Transformer | TaFeng | should beat LSTM on this dataset |

The `metrics.json` dict now includes:
- `spend_mae_raw`, `spend_rmse_raw` — original-currency spend error (joint models)
- `task_weight_freq`, `task_weight_spend` tracked per epoch in `*_history.json`

---

## 11. Troubleshooting

**`covariates` shape mismatch** — ensure the config has `include_covariates: true`
and `model.covariate_dim: 4`. The pipeline produces `(N, T_total, 4)` where
`T_total = calibration_weeks + holdout_weeks`. If the model was built without
`covariate_dim`, rebuild from scratch (don't load an old checkpoint).

**KV cache correctness check** — run the Transformer inference with
`use_kv_cache: false` in the config and compare predictions:
```python
results_cached   = autoregressive_inference_transformer(..., use_kv_cache=True)
results_uncached = autoregressive_inference_transformer(..., use_kv_cache=False)
import numpy as np
np.testing.assert_allclose(
    results_cached["pred_freq"], results_uncached["pred_freq"], atol=1e-4
)
```

**SHAP slow** — reduce `--n_explain` to 50 for a quick sanity check. For final thesis
numbers use 200 (approximately 20 minutes on CPU).

**Memory error on Ta-Feng** — reduce `batch_size` in the config (e.g., 128 instead of 256).

**Missing `spend_mae_raw` in metrics** — the `scaler` is only passed for joint models.
Check that `model.joint: true` in the config.

**Transformer training slower than LSTM** — expected; the Transformer has more parameters
and the attention mechanism is O(T²) during training (KV cache only applies at inference).
