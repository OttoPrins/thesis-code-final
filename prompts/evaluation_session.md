# Evaluation Session Prompt

Paste this at the start of an evaluation or results session.

---

Read CLAUDE.md. We are working on **evaluation, benchmarks, and results**.

## Goal

Produce the comparison table from CLAUDE.md Section 7 (Evaluation Protocol).
All models evaluated on all applicable datasets, all metrics computed consistently.

## Probabilistic benchmarks

Use the `lifetimes` Python library where possible:
```python
from lifetimes import BetaGeoFitter, GammaGammaFitter  # BG/NBD + Gamma-Gamma
from lifetimes import ParetoNBDFitter                   # Pareto/NBD
```

For Pareto/GGG (Platzer & Reutterer 2016):
- Check if available in `lifetimes` or a maintained fork
- If not: implement from scratch using the paper's EM algorithm or use PyMC
- Key: Pareto/GGG adds regularity parameter k to the Pareto/NBD model

For GPPM (Dew & Ansari 2018):
- Gaussian process model; likely requires PyMC or Stan
- If implementation is too complex, note it in the thesis and use available alternatives

## Metrics to compute (see CLAUDE.md Section 7)

Individual-level:
- `freq_rmse`: RMSE of predicted vs. actual purchase count over holdout
- `freq_mae`: MAE of purchase counts
- `spend_mae`: MAE of spend (in log-space and raw)
- `spend_rmse`: RMSE of spend
- `spend_r2`: R² for spend regression

Cohort-level:
- `freq_mape`: MAPE of aggregated predictions vs. aggregated actuals
- `bias_pct`: (sum_pred - sum_actual) / sum_actual × 100

Primary aggregate: quarterly revenue (sum of individual spend forecasts × 13 weeks).

## Files to implement / complete

```
src/evaluation/metrics.py     ← All metric functions, vectorised with numpy
src/evaluation/benchmarks.py  ← Fit and predict wrappers for each BTYD model
src/evaluation/compare.py     ← Build comparison DataFrame; save to results/tables/
notebooks/05_baseline_btyd.ipynb   ← Fit all BTYD benchmarks on CDNOW
notebooks/06_results_comparison.ipynb ← Final tables + thesis plots
```

## Output format

Save results as:
- `results/tables/comparison_cdnow.csv` — full comparison table for CDNOW
- `results/tables/comparison_{dataset}.csv` — one per dataset
- `results/tables/extension3_dunnhumby.csv` — covariate ablation results
- `results/plots/loss_curves_{model}_{dataset}.pdf`
- `results/plots/cohort_forecast_{model}_{dataset}.pdf`
- `results/plots/shap_summary_dunnhumby.pdf` — Extension 3 SHAP plot

## Plot style for thesis

- Use matplotlib with a clean style (seaborn-whitegrid or custom)
- Figure size: 6×4 inches (single column) or 12×4 (double column)
- Font size: 11pt to match LaTeX thesis body
- Save as PDF at 300 DPI (vector format preferred for LaTeX)
- No titles in the figure (captions go in LaTeX)

Please read CLAUDE.md, confirm the plan, and then begin.
