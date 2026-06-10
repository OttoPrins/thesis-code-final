# Extension 3 SHAP Publication Supplement

The primary result explains 701 Dunnhumby households with observed demographics,
using a disjoint empirical background of 100 households. It aggregates LSTM and
Transformer checkpoints trained with seeds 7, 42, and 2024 at 128 expected-gradient
samples.

Use `shap_extension3_summary.csv` for the thesis table and headline architecture
comparison. The beeswarm and temporal plots provide distributional and weekly detail.
The all-household file is a sensitivity analysis only: missing demographics are
zero-coded and therefore confounded with valid lowest-category values.

These are interventional model attributions conditional on each household's fixed
80-week transaction history, not causal campaign-effect estimates. Expected-gradient
additivity is approximate; see `shap_extension3_additivity.csv` and the method notes.
