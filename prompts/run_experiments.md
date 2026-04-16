# How to Run the Experiments

## Setup

```bash
cd thesis-code
pip install -r requirements.txt
```

---

## Experiment Matrix

Every model is run on every dataset. Run configs in order — later stages build on earlier results.

### Stage 1 — Base LSTM (frequency only, replication of Valendin et al. 2022)

```bash
python train.py --config experiments/configs/lstm_base_cdnow.yaml
python train.py --config experiments/configs/lstm_base_uci.yaml
python train.py --config experiments/configs/lstm_base_tafeng.yaml
python train.py --config experiments/configs/lstm_base_dunnhumby.yaml
```

### Stage 2 — Joint LSTM (frequency + spend, Extension 1)

```bash
python train.py --config experiments/configs/lstm_joint_cdnow.yaml
python train.py --config experiments/configs/lstm_joint_uci.yaml
python train.py --config experiments/configs/lstm_joint_tafeng.yaml
python train.py --config experiments/configs/lstm_joint_dunnhumby.yaml
```

### Stage 3 — Transformer (frequency + spend, Extension 2)

```bash
python train.py --config experiments/configs/transformer_joint_cdnow.yaml
python train.py --config experiments/configs/transformer_joint_uci.yaml
python train.py --config experiments/configs/transformer_joint_tafeng.yaml
python train.py --config experiments/configs/transformer_joint_dunnhumby.yaml
```

### Stage 4 — Covariate Ablation on Dunnhumby (Extension 3)

Blocked until `DunnhumbyPipeline.build_covariates()` is implemented.
See `src/data/datasets/dunnhumby.py:151`.

```bash
python train.py --config experiments/configs/extension3_dunnhumby.yaml
```

---

## Outputs

Each run saves three files to `results/`:

| File | Contents |
|------|----------|
| `results/checkpoints/{run_name}.pt` | Best model weights |
| `results/tables/{run_name}_metrics.json` | Holdout evaluation metrics |
| `results/tables/{run_name}_history.json` | Train/val loss per epoch |

### Key metrics to compare across models

- `freq_rmse` — individual-level RMSE of transaction count predictions
- `freq_mape` — cohort-level MAPE (paper reports ~2% for LSTM vs ~18% for Pareto/NBD)
- `bias_pct` — signed cohort bias; positive = over-prediction
- `spend_mae`, `spend_r2` — spend metrics (joint models only)

---

## Probabilistic Benchmarks (Stage 1)

Install benchmarks dependencies:
```bash
pip install lifetimes rpy2 pymc
```

Run benchmarks on each dataset:

```bash
# Pareto/NBD and BG/NBD+Gamma-Gamma (lifetimes-based, always available)
python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_nbd bgnbd_gg
python run_benchmarks.py --config experiments/configs/lstm_base_uci.yaml --models pareto_nbd bgnbd_gg
python run_benchmarks.py --config experiments/configs/lstm_base_tafeng.yaml --models pareto_nbd bgnbd_gg
python run_benchmarks.py --config experiments/configs/lstm_base_dunnhumby.yaml --models pareto_nbd bgnbd_gg

# Optional: Pareto/GGG (requires R + BTYD.plus package)
# R -e 'install.packages("BTYD.plus")'
python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_ggg

# Optional: GPPM (requires PyMC; slow on large datasets)
python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models gppm
```

**Notes:**
- `lifetimes`: Pareto/NBD, BG/NBD, Gamma-Gamma — stable and fast
- `Pareto/GGG`: Implemented via `rpy2` calling R's `BTYD.plus` package. Requires R installation.
- `GPPM`: Full PyMC implementation using GP propensity model. Slow on datasets > 500 customers; skipped for large datasets.

---

## Comparison Table

After running DL models and benchmarks, build the comparison table:

```bash
python -c "from src.evaluation.compare import build_comparison_table; build_comparison_table('results/', 'cdnow')"
python -c "from src.evaluation.compare import build_comparison_table; build_comparison_table('results/', 'uci')"
python -c "from src.evaluation.compare import build_comparison_table; build_comparison_table('results/', 'tafeng')"
python -c "from src.evaluation.compare import build_comparison_table; build_comparison_table('results/', 'dunnhumby')"
```

Outputs: `results/tables/comparison_{dataset}.csv` with all models and metrics for each dataset.

---

## Verification Checklist (after first run)

After running `lstm_base_cdnow`:

- [ ] `results/tables/lstm_base_cdnow_v1_metrics.json` exists
- [ ] `freq_rmse` and `bias_pct` are in a reasonable range (compare to Valendin et al. 2022)
- [ ] `*_history.json` shows decreasing loss that stabilizes before epoch 100
- [ ] Early stopping triggered (patience = 5) — run should stop well before epoch 100

---

## Notes

- **Holdout leakage check:** Scaler is always fitted on calibration only. Never touch holdout data during training.
- **Spend transform:** Always `log1p`. Metrics reported on log scale; raw scale available via `SpendScaler.inverse_transform_spend()`.
- **Reproducibility:** All configs use `seed: 42`.
- **Device:** Script auto-detects CUDA > MPS (Apple Silicon) > CPU.
- **Inference scenarios:** Default is 50 stochastic scenarios averaged per customer. Increase via `training.n_scenarios` in config for smoother predictions.
