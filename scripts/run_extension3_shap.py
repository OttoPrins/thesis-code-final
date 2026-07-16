#!/usr/bin/env python3
"""
Run and aggregate the final Extension 3 SHAP analysis.

This script reuses the trained full-covariate LSTM and Transformer checkpoints.
It does not train or modify model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.datasets import DunnhumbyPipeline
from src.evaluation.shap_analysis import (
    DYNAMIC_FEATURES,
    HEADS,
    STATIC_FEATURES,
    eligible_customer_ids,
    load_sample_manifest,
    make_disjoint_sample,
    run_output_dir,
    run_shap,
    save_sample_manifest,
)
from src.utils.config import load_config


ARCHITECTURES = ("lstm", "transformer")
SEEDS = (7, 42, 2024)
CONFIGS = {
    "lstm": PROJECT_ROOT
    / Path(
        "experiments/configs_final/extension3_lstm_full_dunnhumby_final.yaml"
    ),
    "transformer": PROJECT_ROOT
    / Path(
        "experiments/configs_final/extension3_transformer_full_dunnhumby_final.yaml"
    ),
}
RUN_PREFIXES = {
    "lstm": "extension3_lstm_full_dunnhumby_final",
    "transformer": "extension3_transformer_full_dunnhumby_final",
}
FEATURE_LABELS = {
    "income": "Income",
    "household_size": "Household size",
    "coupon_redemptions_week": "Weekly coupon redemptions",
    "campaign_active_flag": "Active campaign",
}


def _checkpoint_path(
    checkpoint_root: Path,
    architecture: str,
    seed: int,
) -> Path:
    path = (
        checkpoint_root
        / f"{RUN_PREFIXES[architecture]}_seed{seed}_sample.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {path}")
    return path


def _sample_manifest(
    *,
    config_path: Path,
    output_root: Path,
    cohort: str,
    n_background: int,
    n_explain: int,
    analysis_seed: int,
) -> Path:
    path = (
        output_root
        / "samples"
        / f"{cohort}_seed{analysis_seed}_{n_background}x{n_explain}.json"
    )
    if path.exists():
        sample = load_sample_manifest(path)
        if (
            len(sample.background_ids) == n_background
            and len(sample.explain_ids) == n_explain
            and sample.analysis_seed == analysis_seed
            and sample.cohort == cohort
        ):
            return path

    config = load_config(str(config_path))
    _, _, inference_ds, _, _ = DunnhumbyPipeline().run(config)
    ids = eligible_customer_ids(
        inference_ds,
        raw_dir=config["dataset"]["raw_dir"],
        cohort=cohort,
    )
    sample = make_disjoint_sample(
        ids,
        n_background=n_background,
        n_explain=n_explain,
        analysis_seed=analysis_seed,
        cohort=cohort,
    )
    return save_sample_manifest(sample, path)


def _run_one(
    *,
    architecture: str,
    seed: int,
    n_samples: int,
    cohort: str,
    sample_manifest: Path,
    checkpoint_root: Path,
    output_root: Path,
    device: str,
    force: bool,
) -> dict[str, Any]:
    return run_shap(
        config_path=str(CONFIGS[architecture]),
        checkpoint_path=str(
            _checkpoint_path(checkpoint_root, architecture, seed)
        ),
        sample_manifest_path=str(sample_manifest),
        output_root=str(output_root),
        n_integration_samples=n_samples,
        device_name=device,
        model_seed=seed,
        force=force,
    )


def _load_run(
    output_root: Path,
    *,
    cohort: str,
    architecture: str,
    seed: int,
    n_samples: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    run_dir = run_output_dir(
        output_root,
        cohort=cohort,
        architecture=architecture,
        seed=seed,
        n_integration_samples=n_samples,
    )
    with np.load(run_dir / "shap_values.npz", allow_pickle=False) as values:
        arrays = {key: np.asarray(values[key]) for key in values.files}
    summary = pd.read_csv(run_dir / "summary.csv")
    provenance = json.loads((run_dir / "provenance.json").read_text())
    return arrays, summary, provenance


def _relative_importance(frame: pd.DataFrame) -> pd.Series:
    total = frame["mean_abs_shap"].sum()
    if total <= 0:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    return 100.0 * frame["mean_abs_shap"] / total


def _convergence_audit(
    output_root: Path,
    *,
    architecture: str,
    low_samples: int,
    main_samples: int,
    cohort: str,
) -> tuple[pd.DataFrame, bool]:
    _, low, _ = _load_run(
        output_root,
        cohort=cohort,
        architecture=architecture,
        seed=42,
        n_samples=low_samples,
    )
    _, main, provenance = _load_run(
        output_root,
        cohort=cohort,
        architecture=architecture,
        seed=42,
        n_samples=main_samples,
    )
    rows = []
    rerun = False
    for head in HEADS:
        low_head = low[low["head"] == head].copy()
        main_head = main[main["head"] == head].copy()
        merged = low_head.merge(
            main_head,
            on=["architecture", "seed", "cohort", "head", "feature_type", "feature"],
            suffixes=("_low", "_main"),
        )
        rank_correlation = float(
            spearmanr(
                merged["mean_abs_shap_low"],
                merged["mean_abs_shap_main"],
            ).statistic
        )
        low_relative = _relative_importance(
            merged.rename(
                columns={"mean_abs_shap_low": "mean_abs_shap"}
            )
        )
        main_relative = _relative_importance(
            merged.rename(
                columns={"mean_abs_shap_main": "mean_abs_shap"}
            )
        )
        max_relative_change = float(
            np.max(np.abs(low_relative - main_relative))
        )
        additivity_error = float(
            provenance["normalized_additivity_error"][head]
        )
        head_rerun = (
            not np.isfinite(rank_correlation)
            or rank_correlation < 0.90
            or max_relative_change > 5.0
            or additivity_error > 0.10
        )
        rerun = rerun or head_rerun
        rows.append(
            {
                "architecture": architecture,
                "head": head,
                "low_samples": low_samples,
                "main_samples": main_samples,
                "feature_rank_correlation": rank_correlation,
                "max_relative_importance_change_pp": max_relative_change,
                "normalized_additivity_error": additivity_error,
                "requires_128_samples": head_rerun,
            }
        )
    return pd.DataFrame(rows), rerun


def _feature_arrays(
    arrays: dict[str, np.ndarray],
    head: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    static = arrays[f"{head}_static_shap"]
    dynamic = arrays[f"{head}_dynamic_grouped_shap"]
    names = list(STATIC_FEATURES) + list(DYNAMIC_FEATURES)
    types = ["static"] * len(STATIC_FEATURES) + ["dynamic"] * len(
        DYNAMIC_FEATURES
    )
    shap_values = np.concatenate([static, dynamic], axis=1)
    feature_values = np.concatenate(
        [
            arrays["static_feature_values"],
            arrays["dynamic_feature_totals"],
        ],
        axis=1,
    )
    return names, types, shap_values, feature_values


def aggregate_primary_runs(
    *,
    output_root: Path,
    selected_samples: dict[str, int],
    n_bootstrap: int,
    analysis_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    customer_rows: list[dict[str, Any]] = []
    rng = np.random.RandomState(analysis_seed)

    for architecture in ARCHITECTURES:
        sample_budget = selected_samples[architecture]
        runs = [
            _load_run(
                output_root,
                cohort="observed_demographics",
                architecture=architecture,
                seed=seed,
                n_samples=sample_budget,
            )
            for seed in SEEDS
        ]
        reference_ids = runs[0][0]["customer_ids"]
        for arrays, _, _ in runs[1:]:
            if not np.array_equal(reference_ids, arrays["customer_ids"]):
                raise RuntimeError(
                    f"Customer alignment failed across {architecture} seeds."
                )

        for head in HEADS:
            per_seed_values = []
            per_seed_feature_values = []
            names: list[str] = []
            types: list[str] = []
            for arrays, _, _ in runs:
                names, types, values, feature_values = _feature_arrays(
                    arrays, head
                )
                per_seed_values.append(values)
                per_seed_feature_values.append(feature_values)
            shap_stack = np.stack(per_seed_values, axis=0)
            feature_values = per_seed_feature_values[0]
            seed_importance = np.mean(np.abs(shap_stack), axis=1)
            seed_relative = (
                100.0
                * seed_importance
                / np.maximum(seed_importance.sum(axis=1, keepdims=True), 1e-12)
            )

            bootstrap_values = np.empty(
                (n_bootstrap, shap_stack.shape[-1]), dtype=np.float64
            )
            n_households = shap_stack.shape[1]
            for bootstrap_index in range(n_bootstrap):
                indices = rng.randint(0, n_households, size=n_households)
                bootstrap_values[bootstrap_index] = np.mean(
                    np.mean(np.abs(shap_stack[:, indices, :]), axis=1),
                    axis=0,
                )

            mean_importance = seed_importance.mean(axis=0)
            relative_importance = (
                100.0
                * mean_importance
                / max(float(mean_importance.sum()), 1e-12)
            )
            for feature_index, (feature, feature_type) in enumerate(
                zip(names, types)
            ):
                aggregate_rows.append(
                    {
                        "architecture": architecture,
                        "head": head,
                        "feature_type": feature_type,
                        "feature": feature,
                        "mean_abs_shap": float(
                            mean_importance[feature_index]
                        ),
                        "seed_sd": float(
                            seed_importance[:, feature_index].std(ddof=1)
                        ),
                        "bootstrap_ci_low": float(
                            np.percentile(
                                bootstrap_values[:, feature_index], 2.5
                            )
                        ),
                        "bootstrap_ci_high": float(
                            np.percentile(
                                bootstrap_values[:, feature_index], 97.5
                            )
                        ),
                        "relative_importance_pct": float(
                            relative_importance[feature_index]
                        ),
                        "relative_seed_sd_pp": float(
                            seed_relative[:, feature_index].std(ddof=1)
                        ),
                        "n_seeds": len(SEEDS),
                        "n_households": n_households,
                        "n_integration_samples": sample_budget,
                        "cohort": "observed_demographics",
                    }
                )
                for seed_index, seed in enumerate(SEEDS):
                    seed_rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "head": head,
                            "feature_type": feature_type,
                            "feature": feature,
                            "mean_abs_shap": float(
                                seed_importance[
                                    seed_index, feature_index
                                ]
                            ),
                            "relative_importance_pct": float(
                                seed_relative[
                                    seed_index, feature_index
                                ]
                            ),
                            "n_integration_samples": sample_budget,
                        }
                    )
                averaged_signed = shap_stack[
                    :, :, feature_index
                ].mean(axis=0)
                for customer_index, customer_id in enumerate(reference_ids):
                    customer_rows.append(
                        {
                            "architecture": architecture,
                            "head": head,
                            "customer_id": int(customer_id),
                            "feature_type": feature_type,
                            "feature": feature,
                            "mean_signed_shap_across_seeds": float(
                                averaged_signed[customer_index]
                            ),
                            "feature_value": float(
                                feature_values[
                                    customer_index, feature_index
                                ]
                            ),
                        }
                    )

    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(customer_rows),
    )


def _aggregate_additivity(
    output_root: Path,
    selected_samples: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            _, _, provenance = _load_run(
                output_root,
                cohort="observed_demographics",
                architecture=architecture,
                seed=seed,
                n_samples=selected_samples[architecture],
            )
            for head in HEADS:
                rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "head": head,
                        "normalized_additivity_error": provenance[
                            "normalized_additivity_error"
                        ][head],
                        "n_integration_samples": selected_samples[
                            architecture
                        ],
                    }
                )
    return pd.DataFrame(rows)


def _sensitivity_summary(
    output_root: Path,
    selected_samples: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for architecture in ARCHITECTURES:
        _, summary, provenance = _load_run(
            output_root,
            cohort="all_households",
            architecture=architecture,
            seed=42,
            n_samples=selected_samples[architecture],
        )
        for head in HEADS:
            subset = summary[summary["head"] == head].copy()
            subset["relative_importance_pct"] = _relative_importance(subset)
            subset["demographic_missingness_warning"] = (
                "All-household sensitivity only: missing demographics are "
                "encoded as zero and are confounded with valid lowest categories."
            )
            subset["normalized_additivity_error"] = provenance[
                "normalized_additivity_error"
            ][head]
            rows.extend(subset.to_dict("records"))
    return pd.DataFrame(rows)


def _plot_beeswarm(
    customer_frame: pd.DataFrame,
    *,
    architecture: str,
    output_dir: Path,
    analysis_seed: int,
) -> None:
    rng = np.random.RandomState(analysis_seed)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    features = list(FEATURE_LABELS)
    for axis, head in zip(axes, HEADS):
        subset = customer_frame[
            (customer_frame["architecture"] == architecture)
            & (customer_frame["head"] == head)
        ]
        for feature_index, feature in enumerate(features):
            feature_rows = subset[subset["feature"] == feature]
            values = feature_rows["feature_value"].to_numpy(dtype=float)
            if np.ptp(values) > 0:
                colors = (values - values.min()) / np.ptp(values)
            else:
                colors = np.zeros_like(values)
            jitter = rng.normal(0, 0.08, size=len(feature_rows))
            axis.scatter(
                feature_rows["mean_signed_shap_across_seeds"],
                feature_index + jitter,
                c=colors,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                s=18,
                alpha=0.75,
                linewidths=0,
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(features)))
        axis.set_yticklabels([FEATURE_LABELS[value] for value in features])
        axis.set_xlabel("Signed SHAP value, averaged across seeds")
        axis.set_title(
            "Frequency"
            if head == "freq"
            else "Conditional log1p-spend"
        )
    figure.suptitle(
        f"{architecture.upper()} covariate SHAP distributions\n"
        "Blue = low feature value, red = high feature value"
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"shap_beeswarm_{architecture}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _plot_temporal_heatmap(
    output_root: Path,
    *,
    architecture: str,
    sample_budget: int,
    output_dir: Path,
) -> None:
    rows = []
    labels = []
    for head in HEADS:
        seed_values = []
        for seed in SEEDS:
            arrays, _, _ = _load_run(
                output_root,
                cohort="observed_demographics",
                architecture=architecture,
                seed=seed,
                n_samples=sample_budget,
            )
            seed_values.append(
                np.mean(np.abs(arrays[f"{head}_dynamic_shap"]), axis=0)
            )
        temporal = np.mean(seed_values, axis=0)
        for feature_index, feature in enumerate(DYNAMIC_FEATURES):
            rows.append(temporal[:, feature_index])
            labels.append(
                f"{'Frequency' if head == 'freq' else 'Spend'}: "
                f"{FEATURE_LABELS[feature]}"
            )
    matrix = np.stack(rows)
    figure, axis = plt.subplots(figsize=(14, 4.2))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xlabel("Calibration week")
    axis.set_xticks(np.arange(0, matrix.shape[1], 10))
    axis.set_xticklabels(np.arange(1, matrix.shape[1] + 1, 10))
    axis.set_title(
        f"{architecture.upper()} mean absolute weekly dynamic SHAP"
    )
    figure.colorbar(image, ax=axis, label="Mean absolute SHAP")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"shap_temporal_{architecture}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _plot_architecture_comparison(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    features = list(FEATURE_LABELS)
    x = np.arange(len(features))
    width = 0.36
    colors = {"lstm": "#4C72B0", "transformer": "#DD8452"}
    for axis, head in zip(axes, HEADS):
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            subset = (
                summary[
                    (summary["architecture"] == architecture)
                    & (summary["head"] == head)
                ]
                .set_index("feature")
                .reindex(features)
            )
            offset = (architecture_index - 0.5) * width
            axis.bar(
                x + offset,
                subset["relative_importance_pct"],
                width,
                yerr=subset["relative_seed_sd_pp"],
                capsize=3,
                label=architecture.upper(),
                color=colors[architecture],
                alpha=0.9,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FEATURE_LABELS[value] for value in features],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel("Relative importance within head (%)")
        axis.set_title(
            "Frequency"
            if head == "freq"
            else "Conditional log1p-spend"
        )
        axis.legend()
    figure.suptitle(
        "Dunnhumby covariate attribution across architectures\n"
        "Mean over three seeds; error bars show seed SD"
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"shap_architecture_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_method_notes(
    path: Path,
    *,
    selected_samples: dict[str, int],
    convergence: pd.DataFrame,
    additivity: pd.DataFrame,
    n_background: int,
    n_explain: int,
) -> None:
    notes = f"""# Extension 3 SHAP Method Notes

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

- LSTM: {selected_samples['lstm']} expected-gradient samples
- Transformer: {selected_samples['transformer']} expected-gradient samples
- Background: {n_background} households
- Explained sample: {n_explain} disjoint households
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
{convergence.to_string(index=False)}
```

## Final Additivity Diagnostics

The table below reports the normalized residual in the approximate SHAP identity
`sum(phi) = prediction - mean(background prediction)`. The requested escalation
to 128 samples was completed for both architectures. Residuals above 0.10 remain
visible here and in `shap_extension3_additivity.csv`; they reflect Monte Carlo
approximation error and should not be described as exact additivity.

```text
{additivity.to_string(index=False)}
```
"""
    path.write_text(notes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all final Extension 3 SHAP checkpoints and aggregate them."
    )
    parser.add_argument(
        "--results_root", default="results/final_kaggle"
    )
    parser.add_argument("--checkpoint_root")
    parser.add_argument("--output_root")
    parser.add_argument(
        "--lstm_config", default=None,
        help="Override the LSTM full-covariate config YAML "
             "(default: the frozen configs_final path; the tuned workflow "
             "passes experiments/configs_tuned/extension3_lstm_full_dunnhumby_tuned.yaml).",
    )
    parser.add_argument(
        "--transformer_config", default=None,
        help="Override the Transformer full-covariate config YAML.",
    )
    parser.add_argument(
        "--lstm_run_prefix", default=None,
        help="Override the LSTM checkpoint run prefix "
             "(default: extension3_lstm_full_dunnhumby_final; checkpoints are "
             "expected at <checkpoint_root>/<prefix>_seed<seed>_sample.pt).",
    )
    parser.add_argument(
        "--transformer_run_prefix", default=None,
        help="Override the Transformer checkpoint run prefix.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--analysis_seed", type=int, default=42)
    parser.add_argument("--n_background", type=int, default=100)
    parser.add_argument("--n_explain", type=int, default=100)
    parser.add_argument(
        "--sensitivity_n_explain",
        type=int,
        help=(
            "Explained households for the all-household sensitivity analysis. "
            "Defaults to --n_explain."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--convergence_samples", type=int, default=32)
    parser.add_argument("--escalated_samples", type=int, default=128)
    parser.add_argument(
        "--fixed_integration_samples",
        type=int,
        help=(
            "Run every primary checkpoint directly at this integration budget. "
            "Use when a prior convergence pilot has already selected the budget."
        ),
    )
    parser.add_argument(
        "--pilot_convergence_csv",
        help=(
            "Optional prior 32-vs-64 convergence audit copied into the final "
            "outputs when --fixed_integration_samples is used."
        ),
    )
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_sensitivity", action="store_true")
    args = parser.parse_args()

    # Optional overrides so the same orchestration can score the tuned
    # full-covariate checkpoints (kaggle_tuned_runner.ipynb). All downstream
    # helpers read these module-level dicts at call time.
    if args.lstm_config:
        CONFIGS["lstm"] = Path(args.lstm_config)
    if args.transformer_config:
        CONFIGS["transformer"] = Path(args.transformer_config)
    if args.lstm_run_prefix:
        RUN_PREFIXES["lstm"] = args.lstm_run_prefix
    if args.transformer_run_prefix:
        RUN_PREFIXES["transformer"] = args.transformer_run_prefix

    results_root = Path(args.results_root)
    checkpoint_root = Path(
        args.checkpoint_root or results_root / "checkpoints"
    )
    output_root = Path(args.output_root or results_root / "shap")
    tables_dir = results_root / "tables"
    plots_dir = results_root / "plots" / "shap"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    primary_manifest = _sample_manifest(
        config_path=CONFIGS["lstm"],
        output_root=output_root,
        cohort="observed_demographics",
        n_background=args.n_background,
        n_explain=args.n_explain,
        analysis_seed=args.analysis_seed,
    )
    sensitivity_n_explain = (
        args.sensitivity_n_explain
        if args.sensitivity_n_explain is not None
        else args.n_explain
    )
    sensitivity_manifest = _sample_manifest(
        config_path=CONFIGS["lstm"],
        output_root=output_root,
        cohort="all_households",
        n_background=args.n_background,
        n_explain=sensitivity_n_explain,
        analysis_seed=args.analysis_seed,
    )

    if args.fixed_integration_samples is not None:
        fixed_samples = args.fixed_integration_samples
        if fixed_samples <= 0:
            raise ValueError("--fixed_integration_samples must be positive.")
        selected_samples = {
            architecture: fixed_samples for architecture in ARCHITECTURES
        }
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                _run_one(
                    architecture=architecture,
                    seed=seed,
                    n_samples=fixed_samples,
                    cohort="observed_demographics",
                    sample_manifest=primary_manifest,
                    checkpoint_root=checkpoint_root,
                    output_root=output_root,
                    device=args.device,
                    force=args.force,
                )
        if args.pilot_convergence_csv:
            convergence_frame = pd.read_csv(args.pilot_convergence_csv)
            required = {"architecture", "head"}
            missing = required - set(convergence_frame.columns)
            if missing:
                raise ValueError(
                    "Pilot convergence CSV is missing columns: "
                    + ", ".join(sorted(missing))
                )
            convergence_frame["selection_basis"] = (
                "Prior fixed-sample convergence pilot"
            )
        else:
            convergence_frame = pd.DataFrame(
                [
                    {
                        "architecture": architecture,
                        "head": head,
                        "selection_basis": (
                            "Fixed integration budget supplied by CLI"
                        ),
                    }
                    for architecture in ARCHITECTURES
                    for head in HEADS
                ]
            )
    else:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                _run_one(
                    architecture=architecture,
                    seed=seed,
                    n_samples=args.n_samples,
                    cohort="observed_demographics",
                    sample_manifest=primary_manifest,
                    checkpoint_root=checkpoint_root,
                    output_root=output_root,
                    device=args.device,
                    force=args.force,
                )
            _run_one(
                architecture=architecture,
                seed=42,
                n_samples=args.convergence_samples,
                cohort="observed_demographics",
                sample_manifest=primary_manifest,
                checkpoint_root=checkpoint_root,
                output_root=output_root,
                device=args.device,
                force=args.force,
            )

        convergence_frames = []
        selected_samples = {
            architecture: args.n_samples for architecture in ARCHITECTURES
        }
        for architecture in ARCHITECTURES:
            convergence, requires_escalation = _convergence_audit(
                output_root,
                architecture=architecture,
                low_samples=args.convergence_samples,
                main_samples=args.n_samples,
                cohort="observed_demographics",
            )
            convergence_frames.append(convergence)
            if requires_escalation:
                selected_samples[architecture] = args.escalated_samples
                for seed in SEEDS:
                    _run_one(
                        architecture=architecture,
                        seed=seed,
                        n_samples=args.escalated_samples,
                        cohort="observed_demographics",
                        sample_manifest=primary_manifest,
                        checkpoint_root=checkpoint_root,
                        output_root=output_root,
                        device=args.device,
                        force=args.force,
                    )
        convergence_frame = pd.concat(convergence_frames, ignore_index=True)

    convergence_frame["selected_samples"] = convergence_frame[
        "architecture"
    ].map(selected_samples)
    convergence_frame.to_csv(
        tables_dir / "shap_extension3_convergence.csv", index=False
    )

    if not args.skip_sensitivity:
        for architecture in ARCHITECTURES:
            _run_one(
                architecture=architecture,
                seed=42,
                n_samples=selected_samples[architecture],
                cohort="all_households",
                sample_manifest=sensitivity_manifest,
                checkpoint_root=checkpoint_root,
                output_root=output_root,
                device=args.device,
                force=args.force,
            )

    summary, seed_summary, customer_values = aggregate_primary_runs(
        output_root=output_root,
        selected_samples=selected_samples,
        n_bootstrap=args.n_bootstrap,
        analysis_seed=args.analysis_seed,
    )
    additivity = _aggregate_additivity(output_root, selected_samples)
    summary.to_csv(
        tables_dir / "shap_extension3_summary.csv", index=False
    )
    seed_summary.to_csv(
        tables_dir / "shap_extension3_seed_summary.csv", index=False
    )
    customer_values.to_csv(
        tables_dir / "shap_extension3_customer_values.csv.gz",
        index=False,
        compression="gzip",
    )
    additivity.to_csv(
        tables_dir / "shap_extension3_additivity.csv", index=False
    )
    if not args.skip_sensitivity:
        _sensitivity_summary(output_root, selected_samples).to_csv(
            tables_dir / "shap_extension3_sensitivity_all_households.csv",
            index=False,
        )

    for architecture in ARCHITECTURES:
        _plot_beeswarm(
            customer_values,
            architecture=architecture,
            output_dir=plots_dir,
            analysis_seed=args.analysis_seed,
        )
        _plot_temporal_heatmap(
            output_root,
            architecture=architecture,
            sample_budget=selected_samples[architecture],
            output_dir=plots_dir,
        )
    _plot_architecture_comparison(summary, plots_dir)
    _write_method_notes(
        tables_dir / "shap_extension3_method_notes.md",
        selected_samples=selected_samples,
        convergence=convergence_frame,
        additivity=additivity,
        n_background=args.n_background,
        n_explain=args.n_explain,
    )

    run_manifest = {
        "selected_integration_samples": selected_samples,
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "analysis_seed": args.analysis_seed,
        "n_background": args.n_background,
        "n_explain": args.n_explain,
        "sensitivity_n_explain": sensitivity_n_explain,
        "primary_sample_manifest": str(primary_manifest),
        "sensitivity_sample_manifest": str(sensitivity_manifest),
        "primary_cohort": "observed_demographics",
        "sensitivity_cohort": "all_households",
        "causal_warning": (
            "Interventional model attribution conditional on transaction "
            "history; not a causal campaign-effect estimate."
        ),
    }
    (tables_dir / "shap_extension3_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )
    print(
        f"[SHAP] final summary: {tables_dir / 'shap_extension3_summary.csv'}"
    )
    print(
        f"[SHAP] architecture figure: "
        f"{plots_dir / 'shap_architecture_comparison.png'}"
    )


if __name__ == "__main__":
    main()
