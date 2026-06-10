# Extension 3 SHAP Method Notes

## Interpretation

The analysis uses `shap.GradientExplainer` to explain the full-covariate models'
first holdout-week expected frequency and conditional log1p-spend predictions.
Each household retains its own 80-week transaction history while its covariates
are compared with a shared empirical background. Results are interventional model
attributions conditional on transaction history, not causal campaign effects.

The primary sample contains only households with an observed row in
`hh_demographic.csv`. The all-household seed-42 analysis is a sensitivity check:
missing demographics are encoded as zero in the trained pipeline and are therefore
confounded with valid lowest-category values.

## Final Integration Budgets

- LSTM: 128 expected-gradient samples
- Transformer: 128 expected-gradient samples
- Background: 100 households
- Explained sample: 701 disjoint households
- Model seeds: 7, 42, and 2024

Dynamic feature contributions are summed with their signs across the 80 calibration
weeks before absolute global importance is calculated. Conditional spend values and
attributions are converted from robust-scaled model units to original log1p-spend.

## Why Earlier Attempts Failed

1. SHAP subprocess failures were previously ignored or allowed the notebook to continue.
2. `GradientExplainer` requires two-dimensional model output, while the original wrapper returned `(B,)`.
3. SHAP versions returned different nested layouts for multi-input attributions.
4. Enabling training mode for the full wrapper fixed cuDNN LSTM backward but also enabled Transformer dropout.
5. Duplicate notebook cells reran the analysis, and generic filenames let architectures overwrite each other.
6. Customer index `N // 2` was labelled the median customer without computing a median.
7. One fixed transaction history was used for every explained household.
8. Dynamic importance used `sum(abs(phi_t))`, which does not preserve grouped SHAP additivity.
9. Spend attributions were left in robust-scaled model units.
10. Missing demographics for most households were encoded identically to valid lowest categories.
11. `thesis_final_v2` contains derived outputs; model checkpoints live under `results/final_kaggle/checkpoints/`.

## Convergence Audit

```text
architecture  head  low_samples  main_samples  feature_rank_correlation  max_relative_importance_change_pp  normalized_additivity_error  requires_128_samples  selected_samples                      selection_basis
        lstm  freq           32            64                       1.0                           3.600197                     0.267815                  True               128 Prior fixed-sample convergence pilot
        lstm spend           32            64                       1.0                           0.641395                     0.172654                  True               128 Prior fixed-sample convergence pilot
 transformer  freq           32            64                       1.0                           0.975508                     0.123270                  True               128 Prior fixed-sample convergence pilot
 transformer spend           32            64                       1.0                           0.877764                     0.144118                  True               128 Prior fixed-sample convergence pilot
```

## Final Additivity Diagnostics

The table below reports the normalized residual in the approximate SHAP identity
`sum(phi) = prediction - mean(background prediction)`. The requested escalation
to 128 samples was completed for both architectures. Residuals above 0.10 remain
visible here and in `shap_extension3_additivity.csv`; they reflect Monte Carlo
approximation error and should not be described as exact additivity.

```text
architecture  seed  head  normalized_additivity_error  n_integration_samples
        lstm     7  freq                     0.105112                    128
        lstm     7 spend                     0.114308                    128
        lstm    42  freq                     0.126191                    128
        lstm    42 spend                     0.110057                    128
        lstm  2024  freq                     0.133512                    128
        lstm  2024 spend                     0.118210                    128
 transformer     7  freq                     0.115073                    128
 transformer     7 spend                     0.108339                    128
 transformer    42  freq                     0.090126                    128
 transformer    42 spend                     0.098210                    128
 transformer  2024  freq                     0.102232                    128
 transformer  2024 spend                     0.113857                    128
```
