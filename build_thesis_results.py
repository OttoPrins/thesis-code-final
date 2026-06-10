"""
build_thesis_results.py — One-shot generator for thesis-ready figures, tables, and writing guide.

Usage:
    python build_thesis_results.py

Reads from:  results/final_kaggle/
Writes to:   results/thesis_final_v2/
             ├── figures/       F1–F9 as .pdf + .png @ 300 DPI
             ├── tables/        T1–T6 as .tex (booktabs) + .csv
             ├── supplementary/ Extension 3 SHAP diagnostics and provenance
             ├── RESULTS_WRITING_GUIDE.md
             └── ARTIFACT_MANIFEST.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.evaluation.compare import aggregate_all_results, aggregate_seeds
from src.evaluation.metrics import (
    compute_all_metrics,
    per_customer_clv,
)
from src.utils.final_manifest import load_final_manifest

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
FINAL = Path("results/final_kaggle")
OUT = Path("results/thesis_final_v2")
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
SUPPLEMENTARY = OUT / "supplementary" / "extension3_shap"

MAIN_MODELS = ["pareto_nbd", "lstm_base", "lstm_joint", "transformer_joint"]
DATASETS = ["cdnow", "uci", "tafeng", "dunnhumby"]

MODEL_LABELS = {
    "pareto_nbd":        "Pareto/NBD",
    "lstm_base":         "Base LSTM",
    "lstm_joint":        "Joint LSTM",
    "transformer_joint": "Joint Transformer",
}
MONEY_MODEL_LABELS = {
    **MODEL_LABELS,
    "pareto_nbd": "Pareto/NBD + Gamma-Gamma",
}
MONEY_MODEL_LABELS_SHORT = {
    **MODEL_LABELS,
    "pareto_nbd": "Pareto/NBD + GG",
}
DATASET_LABELS = {
    "cdnow":     "CDNOW",
    "uci":       "UCI",
    "tafeng":    "Ta-Feng",
    "dunnhumby": "Dunnhumby",
}
MODEL_COLORS = {
    "pareto_nbd":        "#888888",
    "lstm_base":         "#4C72B0",
    "lstm_joint":        "#55A868",
    "transformer_joint": "#DD8452",
}
DATASET_DISPLAY_ORDER = ["cdnow", "uci", "tafeng", "dunnhumby"]

SEEDS = [7, 42, 2024]
WEEKLY_DISCOUNT_RATE = 0.0018
F3_COMPLETE_HOLDOUT_WEEKS = {"dunnhumby": 21}

EXTENSION3_MODELS = [
    "extension3_lstm_none",
    "extension3_lstm_static",
    "extension3_lstm_dynamic",
    "extension3_lstm_full",
    "extension3_transformer_none",
    "extension3_transformer_static",
    "extension3_transformer_dynamic",
    "extension3_transformer_full",
]
EXTENSION3_VARIANTS = ["none", "static", "dynamic", "full"]
EXTENSION3_LABELS = {
    "none": "None",
    "static": "Static",
    "dynamic": "Dynamic",
    "full": "Full",
}
EXTENSION3_COLORS = {
    "static": "#4C72B0",
    "dynamic": "#DD8452",
    "full": "#55A868",
}
SHAP_CSV = FINAL / "tables" / "shap_extension3_summary.csv"
SHAP_ADDITIVITY_CSV = FINAL / "tables" / "shap_extension3_additivity.csv"
SHAP_RUN_MANIFEST = FINAL / "tables" / "shap_extension3_run_manifest.json"
SHAP_FEATURE_LABELS = {
    "income": "Income",
    "household_size": "Household size",
    "coupon_redemptions_week": "Weekly coupon redemptions",
    "campaign_active_flag": "Active campaign",
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _savefig(fig, name: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in (".png", ".pdf"):
        p = FIGURES / (name + ext)
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    print(f"  Saved {FIGURES / name}.[png,pdf]")
    return paths


def _savetable(df: pd.DataFrame, name: str, tex_content: str | None = None) -> list[Path]:
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = []
    csv_p = TABLES / (name + ".csv")
    df.to_csv(csv_p, index=False)
    paths.append(csv_p)
    if tex_content is not None:
        tex_p = TABLES / (name + ".tex")
        tex_p.write_text(tex_content)
        paths.append(tex_p)
    print(f"  Saved {TABLES / name}.[csv" + (",tex" if tex_content else "") + "]")
    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return manifest-qualified result rows and their seed aggregates."""
    df = aggregate_all_results(results_dir=str(FINAL), final_only=True)
    _validate_result_inventory(df)
    df = _rescore_monetary_against_raw_truth(df)
    seeds = aggregate_seeds(df)
    return df, seeds


def _validate_result_inventory(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No manifest-qualified result files were found.")

    expected_models = MAIN_MODELS + EXTENSION3_MODELS
    missing: list[str] = []
    for model in expected_models:
        datasets = DATASETS if model in MAIN_MODELS else ["dunnhumby"]
        for dataset in datasets:
            sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
            if model == "pareto_nbd":
                if len(sub) != 1:
                    missing.append(f"{model}/{dataset}: expected 1 benchmark, found {len(sub)}")
                continue
            found_seeds = {
                int(seed) for seed in sub["seed"].dropna().astype(int).tolist()
            }
            if found_seeds != set(SEEDS):
                missing.append(
                    f"{model}/{dataset}: expected seeds {SEEDS}, found {sorted(found_seeds)}"
                )

    if missing:
        raise RuntimeError("Incomplete publication result inventory:\n  " + "\n  ".join(missing))


def _arrays_path(run_name: str) -> Path:
    path = FINAL / "tables" / f"{run_name}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing array artifact: {path}")
    return path


def _canonical_targets(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Use direct raw benchmark aggregation as the common holdout truth."""
    with np.load(_arrays_path(f"pareto_nbd_{dataset}"), allow_pickle=False) as data:
        return (
            np.asarray(data["per_week_true_freq"], dtype=np.float64),
            np.asarray(data["per_week_true_spend"], dtype=np.float64),
        )


def _rescore_monetary_against_raw_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute monetary and CLV metrics against one raw holdout matrix per dataset.

    Deep-model artifacts store inverse-transformed spend truth, which can differ
    slightly from direct raw aggregation. Publication comparisons use the direct
    raw benchmark matrix for every monetary model while preserving predictions.
    """
    out = df.copy()
    money_models = {"pareto_nbd", "lstm_joint", "transformer_joint"}
    monetary_prefixes = ("spend_", "clv_")

    for dataset in DATASETS:
        true_freq_week, true_spend_week = _canonical_targets(dataset)
        true_freq_total = true_freq_week.sum(axis=1)
        customer_ids = np.arange(len(true_freq_total))

        row_indices = out[
            (out["dataset"] == dataset) & (out["model"].isin(money_models))
        ].index
        for idx in row_indices:
            run_name = str(out.at[idx, "run_name"])
            with np.load(_arrays_path(run_name), allow_pickle=False) as data:
                pred_freq_week = np.asarray(data["per_week_pred_freq"], dtype=np.float64)
                pred_spend_week = np.asarray(data["per_week_pred_spend"], dtype=np.float64)
                stored_true_freq = np.asarray(data["per_week_true_freq"], dtype=np.float64)

            if not np.array_equal(stored_true_freq, true_freq_week):
                raise RuntimeError(
                    f"Customer/holdout frequency alignment failed for {run_name}."
                )
            if pred_freq_week.shape != true_freq_week.shape:
                raise RuntimeError(f"Frequency shape mismatch for {run_name}.")
            if pred_spend_week.shape != true_spend_week.shape:
                raise RuntimeError(f"Spend shape mismatch for {run_name}.")

            rescored = compute_all_metrics(
                y_freq_true=true_freq_total,
                y_freq_pred=pred_freq_week.sum(axis=1),
                customer_ids=customer_ids,
                y_freq_true_per_week=true_freq_week,
                y_freq_pred_per_week=pred_freq_week,
                y_spend_true_per_week_raw=true_spend_week,
                y_spend_pred_per_week_raw=pred_spend_week,
                freq_mase_scale=out.at[idx, "freq_mase_scale"]
                if "freq_mase_scale" in out.columns and pd.notna(out.at[idx, "freq_mase_scale"])
                else None,
                spend_mase_scale=out.at[idx, "spend_mase_scale"]
                if "spend_mase_scale" in out.columns and pd.notna(out.at[idx, "spend_mase_scale"])
                else None,
                weekly_discount_rate=WEEKLY_DISCOUNT_RATE,
            )
            for key, value in rescored.items():
                if key.startswith(monetary_prefixes) and not key.startswith("_"):
                    out.at[idx, key] = value
    return out


def _load_arrays(model_prefix: str, dataset: str, seed: int | None = None) -> np.lib.npyio.NpzFile | None:
    if seed is None:
        pats = [f"{model_prefix}_{dataset}_final_seed*_sample_arrays.npz",
                f"{model_prefix}_{dataset}_arrays.npz"]
    else:
        pats = [f"{model_prefix}_{dataset}_final_seed{seed}_sample_arrays.npz",
                f"{model_prefix}_{dataset}_arrays.npz"]
    for pat in pats:
        matches = sorted((FINAL / "tables").glob(pat))
        if matches:
            return np.load(matches[0], allow_pickle=False)
    return None


def _load_all_seed_arrays(model_prefix: str, dataset: str) -> list[np.lib.npyio.NpzFile]:
    out = []
    for seed in SEEDS:
        z = _load_arrays(model_prefix, dataset, seed)
        if z is not None:
            out.append(z)
    if not out:
        z = _load_arrays(model_prefix, dataset, seed=None)
        if z is not None:
            out.append(z)
    return out


def _load_history(model_prefix: str, dataset: str, seed: int) -> dict | None:
    pats = [f"{model_prefix}_{dataset}_final_seed{seed}_sample_history.json"]
    for pat in pats:
        matches = sorted((FINAL / "tables").glob(pat))
        if matches:
            return json.loads(matches[0].read_text())
    return None


def _get_seed_metric(df: pd.DataFrame, model: str, dataset: str, metric: str) -> list[float]:
    sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
    vals = sub[metric].dropna().tolist()
    return vals


def _get_mean_std(seeds_df: pd.DataFrame, model: str, dataset: str, metric: str) -> tuple[float, float]:
    sub = seeds_df[(seeds_df["model"] == model) & (seeds_df["dataset"] == dataset)]
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    if sub.empty or mean_col not in sub.columns:
        return float("nan"), float("nan")
    row = sub.iloc[0]
    return float(row.get(mean_col, float("nan"))), float(row.get(std_col, 0.0) or 0.0)


# ---------------------------------------------------------------------------
# T1 — Frequency accuracy table
# ---------------------------------------------------------------------------
def make_table_t1(df_all: pd.DataFrame, df_seeds: pd.DataFrame) -> pd.DataFrame:
    """All 4 models × 4 datasets: freq RMSE, freq MAE, freq MAPE, bias %."""
    print("[T1] Frequency accuracy table...")
    freq_metrics = ["freq_rmse", "freq_mae", "freq_mape", "bias_pct"]

    rows = []
    for ds in DATASETS:
        for model in MAIN_MODELS:
            is_bench = model == "pareto_nbd"
            if is_bench:
                sub = df_all[(df_all["model"] == model) & (df_all["dataset"] == ds)]
                if sub.empty:
                    continue
                row_src = sub.iloc[0]
                r = {"Model": MODEL_LABELS[model], "Dataset": DATASET_LABELS[ds], "N seeds": "—"}
                for m in freq_metrics:
                    v = row_src.get(m, float("nan"))
                    r[m] = float(v) if not pd.isna(v) else float("nan")
                    r[m + "_std"] = float("nan")
            else:
                r = {"Model": MODEL_LABELS[model], "Dataset": DATASET_LABELS[ds]}
                vals_per_metric = {}
                for m in freq_metrics:
                    vals = _get_seed_metric(df_all, model, ds, m)
                    vals_per_metric[m] = vals
                    r[m] = float(np.mean(vals)) if vals else float("nan")
                    r[m + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                r["N seeds"] = max(len(v) for v in vals_per_metric.values()) if vals_per_metric else 0
            rows.append(r)

    df = pd.DataFrame(rows)

    # Build formatted CSV with mean±std
    display_rows = []
    for _, r in df.iterrows():
        dr = {"Model": r["Model"], "Dataset": r["Dataset"], "N_seeds": r["N seeds"]}
        for m in freq_metrics:
            mean, std = r[m], r.get(m + "_std", float("nan"))
            if pd.isna(mean):
                dr[m] = "—"
            elif pd.isna(std) or std == 0:
                dr[m] = f"{mean:.2f}"
            else:
                dr[m] = f"{mean:.2f} ± {std:.2f}"
        display_rows.append(dr)
    display_df = pd.DataFrame(display_rows)

    tex = _build_freq_latex(df, freq_metrics)
    _savetable(display_df, "T1_freq_accuracy", tex)
    return df


def _build_freq_latex(df: pd.DataFrame, metrics: list[str]) -> str:
    col_map = {
        "freq_rmse": "Freq RMSE",
        "freq_mae": "Freq MAE",
        "freq_mape": r"Freq MAPE\,\%",
        "bias_pct": r"Bias\,\%",
    }
    header_cols = ["Model", "Dataset"] + [col_map[m] for m in metrics]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Frequency prediction accuracy across models and datasets. "
        r"Mean $\pm$ SD over 3 seeds; Pareto/NBD is a single run.}",
        r"\label{tab:freq_accuracy}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]
    prev_ds = None
    for _, r in df.iterrows():
        if prev_ds and r["Dataset"] != prev_ds:
            lines.append(r"\midrule")
        prev_ds = r["Dataset"]
        cells = [r["Model"], r["Dataset"]]
        for m in metrics:
            mean, std = r[m], r.get(m + "_std", float("nan"))
            if pd.isna(mean):
                cells.append("—")
            elif pd.isna(std) or std == 0:
                cells.append(f"{mean:.2f}")
            else:
                cells.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T2 — Monetary and CLV table
# ---------------------------------------------------------------------------
def make_table_t2(df_all: pd.DataFrame, df_seeds: pd.DataFrame) -> pd.DataFrame:
    """spend MAE, spend R², CLV MAE, CLV Spearman, CLV decile lift."""
    print("[T2] Monetary & CLV table...")
    money_models = ["pareto_nbd", "lstm_joint", "transformer_joint"]
    money_metrics = ["spend_mae_raw", "spend_r2_log", "clv_mae", "clv_spearman", "clv_decile_lift"]

    rows = []
    for ds in DATASETS:
        for model in money_models:
            is_bench = model == "pareto_nbd"
            if is_bench:
                sub = df_all[(df_all["model"] == model) & (df_all["dataset"] == ds)]
                if sub.empty:
                    continue
                row_src = sub.iloc[0]
                r = {
                    "Model": MONEY_MODEL_LABELS[model],
                    "Dataset": DATASET_LABELS[ds],
                    "N seeds": "—",
                }
                for m in money_metrics:
                    v = row_src.get(m, float("nan"))
                    r[m] = float(v) if not pd.isna(v) else float("nan")
                    r[m + "_std"] = float("nan")
            else:
                r = {"Model": MODEL_LABELS[model], "Dataset": DATASET_LABELS[ds]}
                seed_counts = []
                for m in money_metrics:
                    vals = _get_seed_metric(df_all, model, ds, m)
                    r[m] = float(np.mean(vals)) if vals else float("nan")
                    r[m + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    seed_counts.append(len(vals))
                r["N seeds"] = max(seed_counts) if seed_counts else 0
            rows.append(r)

    df = pd.DataFrame(rows)

    display_rows = []
    for _, r in df.iterrows():
        dr = {"Model": r["Model"], "Dataset": r["Dataset"], "N_seeds": r["N seeds"]}
        for m in money_metrics:
            mean, std = r[m], r.get(m + "_std", float("nan"))
            if pd.isna(mean):
                dr[m] = "—"
            elif pd.isna(std) or std == 0:
                dr[m] = f"{mean:.3f}"
            else:
                dr[m] = f"{mean:.3f} ± {std:.3f}"
        display_rows.append(dr)
    display_df = pd.DataFrame(display_rows)

    tex = _build_money_latex(df, money_metrics)
    _savetable(display_df, "T2_monetary_clv", tex)
    return df


def _build_money_latex(df: pd.DataFrame, metrics: list[str]) -> str:
    col_map = {
        "spend_mae_raw":    r"Spend MAE (\$)",
        "spend_r2_log":     r"Spend $R^2$ (log)",
        "clv_mae":          r"CLV MAE (\$)",
        "clv_spearman":     r"CLV $\rho$",
        "clv_decile_lift":  r"Decile Lift",
    }
    header_cols = ["Model", "Dataset"] + [col_map[m] for m in metrics]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Monetary and CLV prediction across models and datasets. "
        r"Mean $\pm$ SD over 3 seeds; Pareto/NBD + Gamma--Gamma is a single fit. "
        r"All models are rescored against the same directly aggregated raw holdout spend. "
        r"Base LSTM omitted (frequency-only model).}",
        r"\label{tab:monetary_clv}",
        r"\begin{tabular}{ll" + "r" * len(metrics) + "}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]
    prev_ds = None
    for _, r in df.iterrows():
        if prev_ds and r["Dataset"] != prev_ds:
            lines.append(r"\midrule")
        prev_ds = r["Dataset"]
        cells = [r["Model"], r["Dataset"]]
        for m in metrics:
            mean, std = r[m], r.get(m + "_std", float("nan"))
            if pd.isna(mean):
                cells.append("—")
            elif pd.isna(std) or std == 0:
                cells.append(f"{mean:.3f}")
            else:
                cells.append(f"{mean:.3f} $\\pm$ {std:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T3 — Headline summary (reuse export_latex_table_aggregated)
# ---------------------------------------------------------------------------
def make_table_t3(df_all: pd.DataFrame, df_seeds: pd.DataFrame) -> None:
    """Compact summary table using the existing pipeline function."""
    print("[T3] Headline summary table (export_latex_table_aggregated)...")
    from src.evaluation.compare import export_latex_table_aggregated
    TABLES.mkdir(parents=True, exist_ok=True)
    export_latex_table_aggregated(
        df_all=df_all,
        df_seeds=df_seeds,
        out_path=str(TABLES / "T3_headline_summary.tex"),
        caption=(
            r"Holdout performance by model and dataset (mean $\pm$ SD, 3 seeds). "
            r"Pareto/NBD is a single run. "
            r"Base LSTM is frequency-only ($-$ for spend/CLV)."
        ),
        label="tab:headline_summary",
    )


# ---------------------------------------------------------------------------
# T4 — Paired bootstrap significance
# ---------------------------------------------------------------------------
def make_table_t4() -> pd.DataFrame:
    """
    Paired customer bootstrap on errors averaged over all available DL seeds.

    The same customer indices are resampled for both models. For deep models,
    each bootstrap statistic is computed per seed and then averaged, matching
    the mean-over-seeds estimand reported in the main tables. Holm-adjusted
    p-values control family-wise error across the comparisons in this table.
    """
    print("[T4] Significance tests...")
    n_resamples = 5_000

    comparisons = [
        # (model_a, model_b, metric, dataset)
        ("lstm_joint", "lstm_base",         "freq_rmse",     "cdnow"),
        ("lstm_joint", "lstm_base",         "freq_mae",      "cdnow"),
        ("lstm_joint", "pareto_nbd",        "freq_rmse",     "cdnow"),
        ("lstm_joint", "pareto_nbd",        "freq_rmse",     "dunnhumby"),
        ("lstm_joint", "pareto_nbd",        "clv_mae",       "dunnhumby"),
        ("transformer_joint", "lstm_joint", "freq_rmse",     "tafeng"),
        ("transformer_joint", "lstm_joint", "spend_mae_raw", "dunnhumby"),
        ("transformer_joint", "lstm_joint", "clv_mae",       "dunnhumby"),
        ("lstm_joint", "lstm_base",         "freq_rmse",     "uci"),
        ("transformer_joint", "lstm_joint", "freq_rmse",     "uci"),
    ]

    rows = []
    for model_a, model_b, metric, ds in comparisons:
        errors_a, seeds_a = _model_error_stack(model_a, ds, metric)
        errors_b, seeds_b = _model_error_stack(model_b, ds, metric)
        if errors_a.shape[1] != errors_b.shape[1]:
            raise RuntimeError(
                f"Customer count mismatch for {model_a} vs {model_b} / {metric} / {ds}."
            )

        result = _paired_bootstrap_seed_mean(
            errors_a,
            errors_b,
            n_resamples=n_resamples,
            seed=42,
            is_squared=metric == "freq_rmse",
        )
        rows.append({
            "Model A":      _comparison_label(model_a, metric),
            "Model B":      _comparison_label(model_b, metric),
            "Metric":       _metric_label(metric),
            "Dataset":      DATASET_LABELS.get(ds, ds),
            "Delta":        result["delta"],
            "CI_low":       result["ci_low"],
            "CI_high":      result["ci_high"],
            "p_value":      result["p_value"],
            "N seeds A":    seeds_a,
            "N seeds B":    seeds_b,
            "N customers":  result["n_customers"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  [T4] No significance results.")
        return df
    df["p_holm"] = _holm_adjust(df["p_value"].to_numpy(dtype=float))
    df["sig_holm_p<0.05"] = df["p_holm"] < 0.05

    tex = _build_sig_latex(df)
    _savetable(df, "T4_significance", tex)
    return df


def _comparison_label(model: str, metric: str) -> str:
    if model == "pareto_nbd" and metric in {"spend_mae_raw", "clv_mae"}:
        return MONEY_MODEL_LABELS[model]
    return MODEL_LABELS.get(model, model)


def _metric_label(metric: str) -> str:
    return {
        "freq_rmse": "Frequency RMSE",
        "freq_mae": "Frequency MAE",
        "spend_mae_raw": r"Spend MAE (\$)",
        "clv_mae": "CLV MAE",
    }.get(metric, metric)


def _model_error_stack(
    model: str,
    dataset: str,
    metric: str,
) -> tuple[np.ndarray, int]:
    true_freq_week, true_spend_week = _canonical_targets(dataset)
    true_freq = true_freq_week.sum(axis=1)
    true_spend = true_spend_week.sum(axis=1)
    true_clv = per_customer_clv(true_spend_week, WEEKLY_DISCOUNT_RATE)

    run_names = (
        [f"pareto_nbd_{dataset}"]
        if model == "pareto_nbd"
        else [f"{model}_{dataset}_final_seed{seed}_sample" for seed in SEEDS]
    )
    errors = []
    for run_name in run_names:
        with np.load(_arrays_path(run_name), allow_pickle=False) as data:
            stored_true_freq = np.asarray(data["per_week_true_freq"], dtype=np.float64)
            pred_freq_week = np.asarray(data["per_week_pred_freq"], dtype=np.float64)
            if not np.array_equal(stored_true_freq, true_freq_week):
                raise RuntimeError(f"Customer ordering mismatch for {run_name}.")

            if metric == "freq_rmse":
                pred = pred_freq_week.sum(axis=1)
                err = (true_freq - pred) ** 2
            elif metric == "freq_mae":
                pred = pred_freq_week.sum(axis=1)
                err = np.abs(true_freq - pred)
            elif metric == "spend_mae_raw":
                pred_spend = np.asarray(data["per_week_pred_spend"], dtype=np.float64)
                err = np.abs(true_spend - pred_spend.sum(axis=1))
            elif metric == "clv_mae":
                pred_spend = np.asarray(data["per_week_pred_spend"], dtype=np.float64)
                pred_clv = per_customer_clv(pred_spend, WEEKLY_DISCOUNT_RATE)
                err = np.abs(true_clv - pred_clv)
            else:
                raise ValueError(f"Unsupported significance metric: {metric}")
        errors.append(np.asarray(err, dtype=np.float64))
    return np.stack(errors, axis=0), len(run_names)


def _paired_bootstrap_seed_mean(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    is_squared: bool,
) -> dict[str, float | int]:
    errors_a = np.asarray(errors_a, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)
    if errors_a.ndim != 2 or errors_b.ndim != 2:
        raise ValueError("Seed-aware bootstrap expects arrays shaped (S, N).")
    if errors_a.shape[1] != errors_b.shape[1]:
        raise ValueError("Paired bootstrap requires equal customer counts.")

    def aggregate(values: np.ndarray, idx: np.ndarray) -> float:
        per_seed_mean = values[:, idx].mean(axis=1)
        if is_squared:
            per_seed_mean = np.sqrt(per_seed_mean)
        return float(per_seed_mean.mean())

    n_customers = errors_a.shape[1]
    full_idx = np.arange(n_customers)
    observed_delta = aggregate(errors_a, full_idx) - aggregate(errors_b, full_idx)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_customers, size=n_customers)
        draws[i] = aggregate(errors_a, idx) - aggregate(errors_b, idx)

    wrong_side = np.mean(draws >= 0) if observed_delta < 0 else np.mean(draws <= 0)
    return {
        "delta": observed_delta,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_value": min(1.0, float(2.0 * wrong_side)),
        "n_customers": n_customers,
        "n_resamples": n_resamples,
    }


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty_like(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted_sorted[rank] = min(1.0, running_max)
    adjusted = np.empty_like(p_values)
    adjusted[order] = adjusted_sorted
    return adjusted


def _format_p(value: float, significant: bool) -> str:
    if value < 0.001:
        return r"$<0.001^{*}$" if significant else r"$<0.001$"
    return f"{value:.3f}" + (r"$^{*}$" if significant else "")


def _build_sig_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{threeparttable}",
        r"\caption{Paired customer-bootstrap tests (5{,}000 resamples, bootstrap seed 42). "
        r"Deep-model statistics are averaged across seeds 7, 42, and 2024; "
        r"probabilistic benchmarks use one deterministic fit. "
        r"$\Delta$ = mean error of Model A $-$ mean error of Model B; "
        r"negative $\Delta$ means Model A is better. "
        r"Two-sided $p$-values are Holm-adjusted across the displayed comparisons.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Model A & Model B & Metric & Dataset & $\Delta$ & CI$_{2.5}$ & CI$_{97.5}$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Model A']} & {r['Model B']} & {r['Metric']} & {r['Dataset']} & "
            f"{r['Delta']:.4f} & {r['CI_low']:.4f} & {r['CI_high']:.4f} & "
            f"{_format_p(r['p_holm'], r['p_holm'] < 0.05)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\footnotesize",
              r"\item[$^*$] Holm-adjusted $p < 0.05$. Confidence intervals are unadjusted.",
              r"\end{tablenotes}",
              r"\end{threeparttable}",
              r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T5 — Extension 3 covariate ablation
# ---------------------------------------------------------------------------
def make_table_t5(df_all: pd.DataFrame) -> pd.DataFrame:
    print("[T5] Extension 3 covariate ablation...")
    metrics = [
        "freq_rmse",
        "freq_mape",
        "bias_pct",
        "spend_r2_log",
        "clv_spearman",
    ]
    rows = []
    for architecture in ("lstm", "transformer"):
        for variant in EXTENSION3_VARIANTS:
            model = f"extension3_{architecture}_{variant}"
            sub = df_all[(df_all["model"] == model) & (df_all["dataset"] == "dunnhumby")]
            row = {
                "Architecture": "LSTM" if architecture == "lstm" else "Transformer",
                "Covariates": EXTENSION3_LABELS[variant],
                "N seeds": int(sub["seed"].notna().sum()),
            }
            for metric in metrics:
                values = sub[metric].dropna().astype(float).to_numpy()
                row[metric] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            rows.append(row)
    df = pd.DataFrame(rows)

    display_rows = []
    for _, row in df.iterrows():
        display = {
            "Architecture": row["Architecture"],
            "Covariates": row["Covariates"],
            "N_seeds": row["N seeds"],
        }
        for metric in metrics:
            display[metric] = f"{row[metric]:.3f} ± {row[f'{metric}_std']:.3f}"
        display_rows.append(display)
    display_df = pd.DataFrame(display_rows)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Dunnhumby covariate ablation over the 80-week calibration and "
        r"4-week holdout protocol. Values are mean $\pm$ SD over three seeds.}",
        r"\label{tab:covariate_ablation}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Architecture & Covariates & Freq RMSE & Freq MAPE\,\% & Bias\,\% & Spend $R^2$ & CLV $\rho$ \\",
        r"\midrule",
    ]
    previous_architecture = None
    for _, row in df.iterrows():
        if previous_architecture and row["Architecture"] != previous_architecture:
            lines.append(r"\midrule")
        previous_architecture = row["Architecture"]
        cells = [row["Architecture"], row["Covariates"]]
        for metric in metrics:
            cells.append(f"{row[metric]:.3f} $\\pm$ {row[f'{metric}_std']:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    _savetable(display_df, "T5_covariate_ablation", "\n".join(lines))
    return df


def _extension3_seed_deltas(
    df_all: pd.DataFrame,
    architecture: str,
    variant: str,
    metric: str,
) -> np.ndarray:
    base_model = f"extension3_{architecture}_none"
    variant_model = f"extension3_{architecture}_{variant}"
    base = (
        df_all[df_all["model"] == base_model]
        .set_index("seed")[metric]
        .astype(float)
    )
    changed = (
        df_all[df_all["model"] == variant_model]
        .set_index("seed")[metric]
        .astype(float)
    )
    common = sorted(set(base.index) & set(changed.index))
    if {int(seed) for seed in common} != set(SEEDS):
        raise RuntimeError(
            f"Extension 3 seed alignment failed for {architecture}/{variant}/{metric}."
        )
    return np.asarray([changed.loc[seed] - base.loc[seed] for seed in common])


# ---------------------------------------------------------------------------
# T6 — Extension 3 SHAP attribution
# ---------------------------------------------------------------------------
def _load_shap_frame() -> pd.DataFrame:
    if not SHAP_CSV.exists():
        raise FileNotFoundError(
            f"Missing final dual-head SHAP summary: {SHAP_CSV}. "
            "Run scripts/run_extension3_shap.py with the final full-covariate checkpoints."
        )
    df = pd.read_csv(SHAP_CSV)
    required = {
        "architecture",
        "head",
        "feature_type",
        "feature",
        "mean_abs_shap",
        "seed_sd",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "relative_importance_pct",
        "relative_seed_sd_pp",
        "n_seeds",
        "n_households",
        "n_integration_samples",
        "cohort",
    }
    if not required.issubset(df.columns):
        raise RuntimeError(f"Malformed SHAP summary; missing {sorted(required - set(df.columns))}.")
    if set(df["head"]) != {"freq", "spend"}:
        raise RuntimeError("Publication SHAP summary must contain frequency and spend heads.")
    if set(df["architecture"]) != {"lstm", "transformer"}:
        raise RuntimeError("Publication SHAP summary must contain LSTM and Transformer rows.")
    if set(df["cohort"]) != {"observed_demographics"}:
        raise RuntimeError("Publication SHAP summary must use the observed-demographics cohort.")
    if set(df["n_households"]) != {701}:
        raise RuntimeError("Publication SHAP summary must contain all 701 explained households.")
    if set(df["n_seeds"]) != {3}:
        raise RuntimeError("Publication SHAP summary must aggregate seeds 7, 42, and 2024.")
    if set(df["n_integration_samples"]) != {128}:
        raise RuntimeError("Publication SHAP summary must use the escalated 128-sample budget.")
    if not SHAP_RUN_MANIFEST.exists():
        raise FileNotFoundError(f"Missing SHAP run manifest: {SHAP_RUN_MANIFEST}")
    run_manifest = json.loads(SHAP_RUN_MANIFEST.read_text())
    expected_manifest = {
        "n_background": 100,
        "n_explain": 701,
        "primary_cohort": "observed_demographics",
    }
    mismatches = {
        key: (run_manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if run_manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Publication SHAP manifest mismatch: {mismatches}")
    df = df.copy()
    df["Architecture"] = df["architecture"].map(
        {"lstm": "LSTM", "transformer": "Transformer"}
    )
    df["Head"] = df["head"].map(
        {"freq": "Frequency", "spend": "Conditional log1p-spend"}
    )
    df["Feature type"] = df["feature_type"].map(
        {"static": "Static", "dynamic": "Dynamic"}
    )
    df["Feature"] = df["feature"].map(SHAP_FEATURE_LABELS).fillna(df["feature"])
    df["Relative importance (%)"] = df["relative_importance_pct"]
    return df


def make_table_t6() -> pd.DataFrame:
    print("[T6] Dual-head SHAP attribution...")
    df = _load_shap_frame()
    additivity = pd.read_csv(SHAP_ADDITIVITY_CSV)
    additivity_low = 100 * additivity["normalized_additivity_error"].min()
    additivity_high = 100 * additivity["normalized_additivity_error"].max()
    display = df[
        [
            "Architecture",
            "Head",
            "Feature type",
            "Feature",
            "mean_abs_shap",
            "seed_sd",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "Relative importance (%)",
        ]
    ].copy()
    display = display.sort_values(
        ["Architecture", "Head", "Relative importance (%)"],
        ascending=[True, True, False],
    )
    display = display.rename(
        columns={
            "mean_abs_shap": "Mean absolute SHAP",
            "seed_sd": "Seed SD",
            "bootstrap_ci_low": "CI low",
            "bootstrap_ci_high": "CI high",
        }
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Extension 3 covariate attribution for the full-covariate LSTM "
        r"and Transformer on Dunnhumby. Values are means over three seeds; uncertainty "
        r"is the seed SD and a customer-bootstrap 95\% CI. Dynamic SHAP values are "
        r"summed with their signs over 80 calibration weeks before absolute importance. "
        f"The 100-background/701-explanation expected-gradient runs have normalized "
        f"additivity residuals of {additivity_low:.1f}--{additivity_high:.1f}\\%.}}",
        r"\label{tab:shap_attribution}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Architecture & Head & Type & Feature & Mean $|\mathrm{SHAP}|$ & Seed SD & 95\% CI & Rel.\,\% \\",
        r"\midrule",
    ]
    previous_group = None
    for _, row in display.iterrows():
        group = (row["Architecture"], row["Head"])
        if previous_group and group != previous_group:
            lines.append(r"\midrule")
        previous_group = group
        lines.append(
            f"{row['Architecture']} & {row['Head']} & {row['Feature type']} & "
            f"{row['Feature']} & {row['Mean absolute SHAP']:.5f} & "
            f"{row['Seed SD']:.5f} & "
            f"[{row['CI low']:.5f}, {row['CI high']:.5f}] & "
            f"{row['Relative importance (%)']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    _savetable(display, "T6_shap_attribution", "\n".join(lines))
    return display


# ---------------------------------------------------------------------------
# F1 — Cohort bias grouped bars
# ---------------------------------------------------------------------------
def fig_cohort_bias(df_all: pd.DataFrame) -> None:
    print("[F1] Cohort bias figure...")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)

    x = np.arange(len(MAIN_MODELS))
    width = 0.6

    for ax, ds in zip(axes, DATASET_DISPLAY_ORDER):
        bars, errs = [], []
        for model in MAIN_MODELS:
            vals = _get_seed_metric(df_all, model, ds, "bias_pct")
            bars.append(np.mean(vals) if vals else float("nan"))
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        colors = [MODEL_COLORS[m] for m in MAIN_MODELS]
        ax.bar(x, bars, width, color=colors, alpha=0.85,
               yerr=errs, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhspan(-2, 2, alpha=0.12, color="green", label="±2% band")
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MAIN_MODELS], rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_ylabel("Cohort Bias %" if ds == "cdnow" else "")

    # shared legend for reference band
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="green", alpha=0.25, label="±2% target")]
    axes[-1].legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig.suptitle("Cohort forecast bias by model and dataset\n(error bars = SD across 3 seeds)", y=1.02)
    fig.tight_layout()
    _savefig(fig, "F1_cohort_bias")


# ---------------------------------------------------------------------------
# F2 — Frequency MAPE grouped bars
# ---------------------------------------------------------------------------
def fig_freq_mape(df_all: pd.DataFrame) -> None:
    print("[F2] Frequency MAPE figure...")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)
    x = np.arange(len(MAIN_MODELS))
    width = 0.6

    for ax, ds in zip(axes, DATASET_DISPLAY_ORDER):
        bars, errs = [], []
        for model in MAIN_MODELS:
            vals = _get_seed_metric(df_all, model, ds, "freq_mape")
            bars.append(np.mean(vals) if vals else float("nan"))
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        colors = [MODEL_COLORS[m] for m in MAIN_MODELS]
        ax.bar(x, bars, width, color=colors, alpha=0.85,
               yerr=errs, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MAIN_MODELS], rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_ylabel("Freq MAPE %" if ds == "cdnow" else "")

    # Add legend for model colours
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MAIN_MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("Frequency MAPE by model and dataset (lower = better)\n(error bars = SD across 3 seeds)", y=1.02)
    fig.tight_layout()
    _savefig(fig, "F2_freq_mape")


# ---------------------------------------------------------------------------
# F3 — Cumulative cohort calibration
# ---------------------------------------------------------------------------
def _cumulative_calibration_summary(
    true_weekly: np.ndarray,
    pred_weekly_by_seed: np.ndarray,
    plot_weeks: int | None = None,
) -> dict[str, np.ndarray]:
    """Summarise cumulative predicted/actual calibration across training seeds."""
    true = np.asarray(true_weekly, dtype=np.float64)
    pred = np.asarray(pred_weekly_by_seed, dtype=np.float64)
    if true.ndim != 1:
        raise ValueError(f"true_weekly must be 1D, got shape {true.shape}.")
    if pred.ndim == 1:
        pred = pred[None, :]
    if pred.ndim != 2:
        raise ValueError(f"pred_weekly_by_seed must be 1D or 2D, got shape {pred.shape}.")
    if pred.shape[1] != true.shape[0]:
        raise ValueError(
            "Weekly truth/prediction length mismatch: "
            f"{true.shape[0]} vs {pred.shape[1]}."
        )

    source_weeks = true.shape[0]
    plotted_weeks = source_weeks if plot_weeks is None else int(plot_weeks)
    if not 1 <= plotted_weeks <= source_weeks:
        raise ValueError(
            f"plot_weeks must be between 1 and {source_weeks}, got {plotted_weeks}."
        )

    true_plot = true[:plotted_weeks]
    pred_plot = pred[:, :plotted_weeks]
    cumulative_true = np.cumsum(true_plot)
    if np.any(cumulative_true <= 0):
        raise ValueError("Cumulative actual transactions must remain positive.")

    seed_ratios = 100.0 * np.cumsum(pred_plot, axis=1) / cumulative_true[None, :]
    return {
        "weeks": np.arange(1, plotted_weeks + 1),
        "seed_ratios": seed_ratios,
        "mean": seed_ratios.mean(axis=0),
        "lower": seed_ratios.min(axis=0),
        "upper": seed_ratios.max(axis=0),
    }


def _annotate_final_calibration(
    ax: plt.Axes,
    week: int,
    value: float,
    color: str,
    y_offset_points: float,
) -> None:
    ax.annotate(
        f"{value:.0f}%",
        xy=(week, value),
        xytext=(-5, y_offset_points),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
    )


def _holdout_week_ticks(plot_weeks: int, max_ticks: int = 6) -> np.ndarray:
    """Return sparse integer ticks that always include the first and final week."""
    if plot_weeks <= max_ticks:
        return np.arange(1, plot_weeks + 1)
    return np.unique(np.rint(np.linspace(1, plot_weeks, max_ticks)).astype(int))


def fig_weekly_tracking(df_all: pd.DataFrame) -> None:
    print("[F3] Cumulative cohort calibration figure...")
    models_to_plot = ["pareto_nbd", "lstm_joint", "transformer_joint"]
    model_line_styles = {
        "pareto_nbd":        ("--", MODEL_COLORS["pareto_nbd"]),
        "lstm_joint":        ("-",  MODEL_COLORS["lstm_joint"]),
        "transformer_joint": ("-.", MODEL_COLORS["transformer_joint"]),
    }
    endpoint_offsets = {
        "cdnow": {"pareto_nbd": 9, "lstm_joint": -9, "transformer_joint": 0},
        "uci": {"pareto_nbd": 0, "lstm_joint": 9, "transformer_joint": -9},
        "tafeng": {"pareto_nbd": 9, "lstm_joint": 0, "transformer_joint": -9},
        "dunnhumby": {"pareto_nbd": 0, "lstm_joint": -9, "transformer_joint": 9},
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.8), sharey=True)
    axes_flat = axes.flatten()

    for panel_label, ax, ds in zip("abcd", axes_flat, DATASET_DISPLAY_ORDER):
        true_weekly, _ = _canonical_targets(ds)
        true_cohort_weekly = true_weekly.sum(axis=0)
        source_weeks = len(true_cohort_weekly)
        plot_weeks = F3_COMPLETE_HOLDOUT_WEEKS.get(ds, source_weeks)
        if ds == "dunnhumby" and source_weeks != 22:
            raise RuntimeError(
                "F3 expects the publication Dunnhumby artifact to retain all 22 "
                f"holdout bins, found {source_weeks}."
            )

        ax.axhline(100.0, color="black", linewidth=1.8, zorder=4, label="Perfect calibration")
        for model in models_to_plot:
            all_pred_weeks = []
            for seed in SEEDS:
                z = _load_arrays(model, ds, seed)
                if z is None:
                    continue
                if "per_week_pred_freq" not in z.files:
                    continue
                all_pred_weeks.append(z["per_week_pred_freq"].sum(axis=0))

            if not all_pred_weeks:
                z = _load_arrays(model, ds, seed=None)
                if z is not None and "per_week_pred_freq" in z.files:
                    all_pred_weeks = [z["per_week_pred_freq"].sum(axis=0)]
            if not all_pred_weeks:
                continue

            summary = _cumulative_calibration_summary(
                true_cohort_weekly,
                np.stack(all_pred_weeks, axis=0),
                plot_weeks=plot_weeks,
            )
            weeks = summary["weeks"]

            style, color = model_line_styles[model]
            ax.plot(weeks, summary["mean"], style, color=color, linewidth=1.7,
                    label=MODEL_LABELS[model])
            if summary["seed_ratios"].shape[0] > 1:
                ax.fill_between(
                    weeks,
                    summary["lower"],
                    summary["upper"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            _annotate_final_calibration(
                ax,
                int(weeks[-1]),
                float(summary["mean"][-1]),
                color,
                endpoint_offsets[ds][model],
            )

        ax.set_title(f"({panel_label}) {DATASET_LABELS[ds]}", fontsize=10.5, loc="left")
        ax.set_ylim(0, 205)
        ax.set_xlim(1, plot_weeks)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_xticks(_holdout_week_ticks(plot_weeks))
        ax.grid(axis="y", alpha=0.2)

        if ds == "dunnhumby":
            ax.text(
                0.985,
                0.04,
                "Complete weeks 1-21",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#7A5A20",
            )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.supxlabel("Holdout week", y=0.015, fontsize=10)
    fig.supylabel("Cumulative predicted / actual (%)", x=0.015, fontsize=10)
    fig.tight_layout(rect=(0.035, 0.04, 1, 0.89))
    _savefig(fig, "F3_weekly_tracking")


# ---------------------------------------------------------------------------
# F4 — CLV decile lift
# ---------------------------------------------------------------------------
def _tie_preserving_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_groups: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build approximately equal ordered groups without splitting tied scores."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    )
    ends = np.r_[starts[1:], len(sorted_score)]
    blocks = list(zip(starts, ends))

    group_true = []
    group_score = []
    group_size = []
    block_idx = 0
    groups_left = min(n_groups, len(blocks))
    observations_left = len(sorted_score)
    while block_idx < len(blocks):
        target_size = observations_left / groups_left
        group_start = blocks[block_idx][0]
        size = 0
        while block_idx < len(blocks):
            block_start, block_end = blocks[block_idx]
            block_size = block_end - block_start
            if size and abs(size - target_size) <= abs(size + block_size - target_size):
                break
            size += block_size
            block_idx += 1
            if size >= target_size:
                break
        group_end = blocks[block_idx - 1][1]
        group_true.append(float(sorted_true[group_start:group_end].mean()))
        group_score.append(float(sorted_score[group_start:group_end].mean()))
        group_size.append(group_end - group_start)
        observations_left -= group_end - group_start
        groups_left -= 1

    return (
        np.asarray(group_true),
        np.asarray(group_score),
        np.asarray(group_size),
    )


def fig_clv_decile(df_all: pd.DataFrame) -> None:
    print("[F4] CLV decile lift figure...")
    money_models = ["pareto_nbd", "lstm_joint", "transformer_joint"]

    # ---- subplot 1: CLV decile lift bars ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(DATASET_DISPLAY_ORDER))
    bar_width = 0.22
    offsets = np.linspace(-(len(money_models) - 1) / 2, (len(money_models) - 1) / 2, len(money_models)) * bar_width

    for i, model in enumerate(money_models):
        vals_mean, vals_err = [], []
        for ds in DATASET_DISPLAY_ORDER:
            vals = _get_seed_metric(df_all, model, ds, "clv_decile_lift")
            vals_mean.append(np.mean(vals) if vals else float("nan"))
            vals_err.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        ax.bar(x + offsets[i], vals_mean, bar_width, color=MODEL_COLORS[model],
               alpha=0.85, yerr=vals_err, capsize=3, label=MONEY_MODEL_LABELS_SHORT[model],
               error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Random (lift=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASET_DISPLAY_ORDER])
    ax.set_ylabel("Top-decile CLV lift")
    ax.set_title("CLV decile lift (higher = better ranking)")
    ax.legend(fontsize=8)

    # ---- subplot 2: predicted-rank calibration on Dunnhumby (best dataset) ----
    ax2 = axes[1]
    ds = "dunnhumby"
    _, true_spend_week = _canonical_targets(ds)
    true_clv = per_customer_clv(true_spend_week, WEEKLY_DISCOUNT_RATE)
    for model in money_models:
        pred_spend_weeks = []
        seeds = [None] if model == "pareto_nbd" else SEEDS
        for seed in seeds:
            z = _load_arrays(model, ds, seed=seed)
            if z is not None and "per_week_pred_spend" in z.files:
                pred_spend_weeks.append(
                    np.asarray(z["per_week_pred_spend"], dtype=np.float64)
                )
        if not pred_spend_weeks:
            continue
        pred_clv = per_customer_clv(
            np.mean(pred_spend_weeks, axis=0),
            WEEKLY_DISCOUNT_RATE,
        )
        group_actual, group_predicted, group_sizes = _tie_preserving_groups(
            true_clv,
            pred_clv,
        )
        x_group = np.arange(1, len(group_actual) + 1)
        ax2.plot(x_group, group_actual, "o-", color=MODEL_COLORS[model],
                 label=MONEY_MODEL_LABELS_SHORT[model], linewidth=1.5, markersize=5)
        if model == "pareto_nbd" and group_predicted[0] == 0:
            ax2.text(
                0.02,
                0.97,
                f"Pareto/NBD group 1: {group_sizes[0]} tied zero predictions",
                transform=ax2.transAxes,
                va="top",
                fontsize=8,
                color=MODEL_COLORS[model],
            )

    ax2.set_xlabel("Ordered predicted-CLV group (1=lowest, 10=highest; ties intact)")
    ax2.set_ylabel("Mean actual holdout CLV ($)")
    ax2.set_title("Realized CLV by ordered prediction group - Dunnhumby")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, "F4_clv_decile")


# ---------------------------------------------------------------------------
# F5 — Spend R² bars
# ---------------------------------------------------------------------------
def fig_spend_r2(df_all: pd.DataFrame) -> None:
    print("[F5] Spend R² figure...")
    money_models = ["pareto_nbd", "lstm_joint", "transformer_joint"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(DATASET_DISPLAY_ORDER))
    bar_width = 0.22
    offsets = np.linspace(-(len(money_models) - 1) / 2, (len(money_models) - 1) / 2, len(money_models)) * bar_width

    for i, model in enumerate(money_models):
        vals_mean, vals_err = [], []
        for ds in DATASET_DISPLAY_ORDER:
            vals = _get_seed_metric(df_all, model, ds, "spend_r2_log")
            vals_mean.append(np.mean(vals) if vals else float("nan"))
            vals_err.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        ax.bar(x + offsets[i], vals_mean, bar_width, color=MODEL_COLORS[model],
               alpha=0.85, yerr=vals_err, capsize=3, label=MONEY_MODEL_LABELS_SHORT[model],
               error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="R²=0 baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASET_DISPLAY_ORDER])
    ax.set_ylabel("Spend $R^2$ (log-space)")
    ax.set_title("Spend prediction $R^2$ (log-space) across datasets\n(negative = worse than mean baseline)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "F5_spend_r2")


# ---------------------------------------------------------------------------
# F6 — Kendall task-weight evolution
# ---------------------------------------------------------------------------
def fig_kendall_weights() -> None:
    print("[F6] Kendall task-weight trajectories...")
    joint_models = ["lstm_joint", "transformer_joint"]
    freq_color = "#4C72B0"
    spend_color = "#DD8452"

    fig, axes = plt.subplots(len(joint_models), len(DATASETS), figsize=(16, 7), sharey="row")

    for row_i, model in enumerate(joint_models):
        for col_j, ds in enumerate(DATASET_DISPLAY_ORDER):
            ax = axes[row_i][col_j]
            loaded_any = False
            for seed in SEEDS:
                h = _load_history(model, ds, seed)
                if h is None:
                    continue
                tw_freq = h.get("task_weight_freq", [])
                tw_spend = h.get("task_weight_spend", [])
                if not tw_freq:
                    continue
                ep = np.arange(1, len(tw_freq) + 1)
                ax.plot(ep, tw_freq, "-", color=freq_color, alpha=0.62, linewidth=1.2)
                ax.plot(ep, tw_spend, "--", color=spend_color, alpha=0.62, linewidth=1.2)
                loaded_any = True

            if col_j == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=9)
            if row_i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=10)
            if row_i == len(joint_models) - 1:
                ax.set_xlabel("Epoch")

            if not loaded_any:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="grey")
            else:
                ax.set_yscale("log")

    # Legend in first subplot
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], linestyle="-", color=freq_color, label="Task weight: frequency"),
        Line2D([0], [0], linestyle="--", color=spend_color, label="Task weight: spend"),
    ]
    axes[0][0].legend(handles=legend_handles, fontsize=8)
    fig.suptitle("Kendall task-weight evolution during training\n(each line = one seed)", y=1.01)
    fig.tight_layout()
    _savefig(fig, "F6_kendall_weights")


# ---------------------------------------------------------------------------
# F7 — Learning curves (appendix)
# ---------------------------------------------------------------------------
def fig_learning_curves() -> None:
    print("[F7] Learning curves (appendix)...")
    models_lc = ["lstm_base", "lstm_joint", "transformer_joint"]
    ds = "cdnow"  # most illustrative

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, model in zip(axes, models_lc):
        for seed in SEEDS:
            h = _load_history(model, ds, seed)
            if h is None:
                continue
            train = h.get("train_loss", [])
            val = h.get("val_loss", [])
            if not train:
                continue
            ep = np.arange(1, len(train) + 1)
            ax.plot(ep, train, "-", alpha=0.6, linewidth=1.0,
                    color=MODEL_COLORS[model], label=f"Train s={seed}")
            ax.plot(ep, val,   "--", alpha=0.6, linewidth=1.0,
                    color=MODEL_COLORS[model])
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if model == "lstm_base" else "")

    fig.suptitle("Training & validation loss curves — CDNOW (solid=train, dashed=val, each line=1 seed)")
    fig.tight_layout()
    _savefig(fig, "F7_learning_curves")


# ---------------------------------------------------------------------------
# F8 — Extension 3 Comic Ablation
# ---------------------------------------------------------------------------
def fig_covariate_ablation(df_all: pd.DataFrame) -> None:
    print("[F8] Extension 3 covariate ablation deltas...")
    variants = ["static", "dynamic", "full"]
    architectures = ["lstm", "transformer"]
    architecture_labels = {"lstm": "LSTM", "transformer": "Transformer"}
    x = np.arange(len(variants))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    metric_specs = [
        ("freq_rmse", "Change in frequency RMSE", "Lower than zero is better"),
        ("spend_r2_log", r"Change in spend $R^2$ (log)", "Higher than zero is better"),
    ]

    for ax, (metric, ylabel, subtitle) in zip(axes, metric_specs):
        for arch_idx, architecture in enumerate(architectures):
            means, stds = [], []
            for variant in variants:
                deltas = _extension3_seed_deltas(
                    df_all, architecture, variant, metric
                )
                means.append(float(deltas.mean()))
                stds.append(float(deltas.std(ddof=1)))
            offset = (arch_idx - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                alpha=0.88,
                label=architecture_labels[architecture],
                color=MODEL_COLORS[
                    "lstm_joint" if architecture == "lstm" else "transformer_joint"
                ],
                error_kw={"elinewidth": 1.1, "ecolor": "black"},
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([EXTENSION3_LABELS[v] for v in variants])
        ax.set_xlabel("Covariates added relative to none")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)

    axes[0].legend()
    fig.suptitle(
        "Dunnhumby covariate ablation: paired seed-wise change from no covariates\n"
        "(error bars = SD across three seeds)"
    )
    fig.tight_layout()
    _savefig(fig, "F8_covariate_ablation")


# ---------------------------------------------------------------------------
# F9 — Extension 3 SHAP attribution
# ---------------------------------------------------------------------------
def fig_shap_attribution() -> None:
    print("[F9] Dual-head SHAP attribution...")
    df = _load_shap_frame()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    head_specs = [
        ("freq", "Frequency"),
        ("spend", "Conditional log1p-spend"),
    ]
    architectures = ["lstm", "transformer"]
    architecture_colors = {"lstm": "#4C72B0", "transformer": "#DD8452"}
    feature_order = list(SHAP_FEATURE_LABELS)
    x = np.arange(len(feature_order))
    width = 0.36

    for ax, (head, title) in zip(axes, head_specs):
        for index, architecture in enumerate(architectures):
            sub = (
                df[
                    (df["head"] == head)
                    & (df["architecture"] == architecture)
                ]
                .set_index("feature")
                .reindex(feature_order)
            )
            offset = (index - 0.5) * width
            ax.bar(
                x + offset,
                sub["Relative importance (%)"],
                width,
                yerr=sub["relative_seed_sd_pp"],
                capsize=3,
                color=architecture_colors[architecture],
                alpha=0.9,
                label=architecture.upper(),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHAP_FEATURE_LABELS[value] for value in feature_order],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("Relative importance within head (%)")
        ax.set_title(title)
        ax.legend()
    fig.suptitle(
        "Dunnhumby covariate attribution by architecture and prediction head\n"
        "(mean over three seeds; error bars = seed SD)"
    )
    fig.tight_layout()
    _savefig(fig, "F9_shap_attribution")


# ---------------------------------------------------------------------------
# Extension 3 SHAP supplement
# ---------------------------------------------------------------------------
def copy_shap_supplement() -> None:
    print("[Supplement] Copying Extension 3 SHAP diagnostics...")
    if SUPPLEMENTARY.exists():
        shutil.rmtree(SUPPLEMENTARY)
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)
    table_names = [
        "shap_extension3_summary.csv",
        "shap_extension3_seed_summary.csv",
        "shap_extension3_customer_values.csv.gz",
        "shap_extension3_additivity.csv",
        "shap_extension3_convergence.csv",
        "shap_extension3_sensitivity_all_households.csv",
        "shap_extension3_run_manifest.json",
        "shap_extension3_method_notes.md",
    ]
    for name in table_names:
        shutil.copy2(FINAL / "tables" / name, SUPPLEMENTARY / name)

    plot_dir = SUPPLEMENTARY / "plots"
    plot_dir.mkdir(exist_ok=True)
    for source in sorted((FINAL / "plots" / "shap").glob("*")):
        if source.is_file():
            shutil.copy2(source, plot_dir / source.name)

    provenance_dir = SUPPLEMENTARY / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    for source in sorted((FINAL / "shap").glob("*/*/seed*/nsamples*/provenance.json")):
        relative = source.relative_to(FINAL / "shap")
        destination = provenance_dir / (
            "__".join(relative.parts[:-1]) + ".json"
        )
        shutil.copy2(source, destination)

    readme = """# Extension 3 SHAP Publication Supplement

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
"""
    (SUPPLEMENTARY / "README.md").write_text(readme)


# ---------------------------------------------------------------------------
# Writing guide
# ---------------------------------------------------------------------------
def make_writing_guide(df_all: pd.DataFrame, df_seeds: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    print("[Guide] Writing RESULTS_WRITING_GUIDE.md...")

    def _m(model: str, ds: str, metric: str, digits: int = 2) -> str:
        vals = _get_seed_metric(df_all, model, ds, metric)
        if not vals:
            return "N/A"
        mu = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        if std == 0 or len(vals) == 1:
            return f"{mu:.{digits}f}"
        return f"{mu:.{digits}f} ± {std:.{digits}f}"

    def _b(model: str, ds: str, metric: str) -> str:
        sub = df_all[(df_all["model"] == model) & (df_all["dataset"] == ds)]
        if sub.empty:
            return "N/A"
        v = sub.iloc[0].get(metric)
        return f"{v:.3f}" if not pd.isna(v) else "N/A"

    # Pull key numbers
    pnbd_cdnow_mape = _b("pareto_nbd", "cdnow", "freq_mape")
    pnbd_cdnow_bias = _b("pareto_nbd", "cdnow", "bias_pct")
    pnbd_cdnow_r2   = _b("pareto_nbd", "cdnow", "spend_r2_log")
    pnbd_cdnow_rho  = _b("pareto_nbd", "cdnow", "clv_spearman")

    base_cdnow_mape = _m("lstm_base", "cdnow", "freq_mape")
    jlstm_cdnow_mape = _m("lstm_joint", "cdnow", "freq_mape")
    tformer_cdnow_mape = _m("transformer_joint", "cdnow", "freq_mape")
    tformer_cdnow_bias = _m("transformer_joint", "cdnow", "bias_pct")

    pnbd_dunn_bias = _b("pareto_nbd", "dunnhumby", "bias_pct")
    jlstm_dunn_bias = _m("lstm_joint", "dunnhumby", "bias_pct")
    tformer_dunn_bias = _m("transformer_joint", "dunnhumby", "bias_pct")
    jlstm_dunn_r2 = _m("lstm_joint", "dunnhumby", "spend_r2_log")
    tformer_dunn_r2 = _m("transformer_joint", "dunnhumby", "spend_r2_log")
    jlstm_dunn_rho = _m("lstm_joint", "dunnhumby", "clv_spearman")
    tformer_dunn_rho = _m("transformer_joint", "dunnhumby", "clv_spearman")

    jlstm_uci_mape = _m("lstm_joint", "uci", "freq_mape")
    pnbd_uci_mape = _b("pareto_nbd", "uci", "freq_mape")
    tformer_tafeng_rmse = _m("transformer_joint", "tafeng", "freq_rmse")
    jlstm_tafeng_rmse = _m("lstm_joint", "tafeng", "freq_rmse")
    pnbd_tafeng_rmse = _b("pareto_nbd", "tafeng", "freq_rmse")

    lstm_static_freq_delta = _extension3_seed_deltas(
        df_all, "lstm", "static", "freq_rmse"
    ).mean()
    lstm_static_spend_delta = _extension3_seed_deltas(
        df_all, "lstm", "static", "spend_r2_log"
    ).mean()
    transformer_static_spend_delta = _extension3_seed_deltas(
        df_all, "transformer", "static", "spend_r2_log"
    ).mean()
    shap_df = _load_shap_frame()
    shap_additivity = pd.read_csv(SHAP_ADDITIVITY_CSV)
    shap_additivity_low = 100 * shap_additivity["normalized_additivity_error"].min()
    shap_additivity_high = 100 * shap_additivity["normalized_additivity_error"].max()
    shap_top = {}
    for architecture in ("lstm", "transformer"):
        for head in ("freq", "spend"):
            row = shap_df[
                (shap_df["architecture"] == architecture)
                & (shap_df["head"] == head)
            ].sort_values("Relative importance (%)", ascending=False).iloc[0]
            shap_top[(architecture, head)] = (
                row["Feature"],
                row["Relative importance (%)"],
            )

    guide = f"""# Results Section Writing Guide
*Auto-generated by build_thesis_results.py from manifest-qualified files in `{FINAL}`.*
*Update by re-running the script. Do not edit numbers by hand.*

---

## Figure and Table Index

| ID | Filename | Use in thesis |
|----|----------|--------------|
| T1 | T1_freq_accuracy.csv / .tex | Section 5.1–5.2: full frequency accuracy table |
| T2 | T2_monetary_clv.csv / .tex | Section 5.3: spend + CLV table |
| T3 | T3_headline_summary.tex (+ _freq, _spend) | Section 5.0 or summary: compact overview |
| T4 | T4_significance.csv / .tex | Section 5.6: statistical significance |
| T5 | T5_covariate_ablation.csv / .tex | Section 5.5: Extension 3 ablation |
| T6 | T6_shap_attribution.csv / .tex | Section 5.5: dual-head SHAP attribution |
| F1 | F1_cohort_bias.png/.pdf | Section 5.2: bias comparison (headline) |
| F2 | F2_freq_mape.png/.pdf | Section 5.2: MAPE comparison |
| F3 | F3_weekly_tracking.png/.pdf | Section 5.1 + 5.2: Cumulative calibration over the holdout |
| F4 | F4_clv_decile.png/.pdf | Section 5.3: CLV ranking quality |
| F5 | F5_spend_r2.png/.pdf | Section 5.3: spend point-accuracy |
| F6 | F6_kendall_weights.png/.pdf | Section 5.3 or methodology appendix |
| F7 | F7_learning_curves.png/.pdf | Appendix: convergence evidence |
| F8 | F8_covariate_ablation.png/.pdf | Section 5.5: marginal covariate effects |
| F9 | F9_shap_attribution.png/.pdf | Section 5.5: attribution by prediction head |

---

## Recommended Narrative Structure

### 5.0 Experimental Setup (half page)
- **Datasets:** CDNOW master cohort (23,570 customers, 39+39 week split), UCI (78+26 weeks),
  Ta-Feng (12+5 weeks), Dunnhumby (80+22 weeks).
- **Models:** Pareto/NBD for frequency, paired with Gamma-Gamma for monetary outcomes;
  Base LSTM (frequency only, replication),
  Joint LSTM (frequency + spend, Extension 1), Joint Transformer (Extension 2).
- **Extension 3:** none/static/dynamic/full covariate variants on Dunnhumby using an 80+4
  week protocol, followed by dual-head SHAP on the full-covariate LSTM and Transformer.
- **Training protocol:** 3 seeds (7, 42, 2024); autoregressive inference, 30 sampled scenarios.
- **Metrics:** Valendin MAPE, cohort bias %, individual RMSE/MAE, spend R² (log-space), CLV
  Spearman ρ, CLV decile lift (McCarthy & Fader, 2018).
- **Comparability:** monetary and CLV metrics are rescored against the same directly aggregated
  raw holdout-spend matrices for every model.
- **Dunnhumby coverage:** the 22nd holdout bin contains four observed days (days 708-711)
  because the public transaction file ends on day 711. Headline tables retain the fixed 80+22
  protocol; F3 ends at week 21 so its calibration trajectories compare complete weeks only.

---

### 5.1 Replication on CDNOW (≈ 1 page)

**Claim:** On sparse CDNOW data, Pareto/NBD remains the strongest model on aggregate frequency
MAPE, while the Joint LSTM improves calibration relative to the Base LSTM.

**Key numbers to cite:**
| Model | Freq MAPE | Bias % | Spend R² | CLV ρ |
|-------|----------|--------|---------|-------|
| Pareto/NBD + Gamma-Gamma | {pnbd_cdnow_mape}% | {pnbd_cdnow_bias}% | {pnbd_cdnow_r2} | {pnbd_cdnow_rho} |
| Base LSTM | {base_cdnow_mape}% | — | N/A | N/A |
| Joint LSTM | {jlstm_cdnow_mape}% | — | see T2 | see T2 |
| Joint Transformer | {tformer_cdnow_mape}% (bias {tformer_cdnow_bias}%) | — | see T2 | see T2 |

**To write:**
- Pareto/NBD achieves MAPE of {pnbd_cdnow_mape}% with bias {pnbd_cdnow_bias}%. The Base LSTM
  has a higher MAPE of {base_cdnow_mape}%, so the replication does not reproduce a universal
  deep-learning advantage on this sparse panel.
- The Joint LSTM reduces MAPE relative to the Base LSTM ({jlstm_cdnow_mape}% versus
  {base_cdnow_mape}%) and substantially improves cohort bias, but individual RMSE is worse.
  Report this as a metric-dependent trade-off rather than a uniform gain.
- The Joint Transformer shows the largest miscalibration on CDNOW (bias {tformer_cdnow_bias}%,
  with substantial seed variation).
- Spend R² is negative for both joint deep models. Their CLV ranking remains positive but below
  the Pareto/NBD + Gamma-Gamma benchmark on CDNOW.

**Key figures/tables:** F1 (CDNOW panel), F3 (CDNOW weekly tracking), T1 (rows for CDNOW).

---

### 5.2 Frequency across datasets: the density effect (≈ 1 page)

**Claim:** The relative performance of sequence models improves on denser retail panels,
especially UCI and Dunnhumby, but the pattern is not uniform across every metric.

**Key numbers to cite:**
| Dataset | Pareto/NBD MAPE | Joint LSTM MAPE | Pareto/NBD bias | DL bias |
|---------|----------------|-----------------|----------------|---------|
| CDNOW | {pnbd_cdnow_mape}% | {jlstm_cdnow_mape}% | {pnbd_cdnow_bias}% | see T1 |
| UCI | {pnbd_uci_mape}% | {jlstm_uci_mape}% | see T1 | see T1 |
| Ta-Feng | see T1 | see T1 | see T1 | see T1 |
| Dunnhumby | see T1 | see T1 | {pnbd_dunn_bias}% | {jlstm_dunn_bias}% |

**To write:**
- On UCI, the Joint LSTM reduces MAPE:
  Joint LSTM achieves {jlstm_uci_mape}% on UCI vs Pareto/NBD's {pnbd_uci_mape}%.
- On Ta-Feng (high-frequency grocery), the Joint Transformer achieves RMSE {tformer_tafeng_rmse}
  vs Joint LSTM {jlstm_tafeng_rmse} and Pareto/NBD {pnbd_tafeng_rmse}, supporting, but not
  proving, the proposed density-dependent Transformer advantage.
- On Dunnhumby, Pareto/NBD underpredicts aggregate frequency by {pnbd_dunn_bias}%. Joint LSTM and
  Transformer maintain near-zero bias ({jlstm_dunn_bias}% and {tformer_dunn_bias}% respectively),
  providing the strongest evidence for learned sequence representations in this study.
- The complete-week Dunnhumby sensitivity (weeks 1-21) reaches bias/MAPE of
  -74.66%/74.66% for Pareto/NBD, -1.15%/3.16% for the Joint LSTM, and
  +0.29%/3.89% for the Joint Transformer. The partial final bin therefore does not drive
  the substantive result.

**Key figures/tables:** F1 (all 4 panels), F2, F3 (all panels), T1.

---

### 5.3 Joint monetary prediction and CLV ranking (≈ 1 page)

**Claim:** Joint learning enables monetary and CLV outputs, with frequency trade-offs that vary
by dataset; CLV ranking can remain useful even where spend point accuracy is weak.

**Key numbers to cite (Dunnhumby, best dataset):**
| Model | Spend R² (log) | CLV ρ | CLV decile lift |
|-------|---------------|-------|----------------|
| Pareto/NBD + Gamma-Gamma | {_b("pareto_nbd", "dunnhumby", "spend_r2_log")} | {_b("pareto_nbd", "dunnhumby", "clv_spearman")} | see T2 |
| Joint LSTM | {jlstm_dunn_r2} | {jlstm_dunn_rho} | see T2 |
| Joint Transformer | {tformer_dunn_r2} | {tformer_dunn_rho} | see T2 |

**To write:**
- On Dunnhumby, the Joint LSTM achieves spend R² = {jlstm_dunn_r2} (log-space) and CLV
  Spearman ρ = {jlstm_dunn_rho}; the Joint Transformer achieves R² = {tformer_dunn_r2}
  and ρ = {tformer_dunn_rho}. Pareto/NBD + Gamma-Gamma achieves ρ =
  {_b("pareto_nbd", "dunnhumby", "clv_spearman")} at R² =
  {_b("pareto_nbd", "dunnhumby", "spend_r2_log")}.
- On CDNOW, UCI, and Ta-Feng, spend R² is negative for DL models too — point-spend
  prediction is generally hard at the individual level. Report CLV ranking alongside point error
  rather than using ranking as a substitute for calibration.
- Kendall weights vary by orders of magnitude on some datasets (F6). Treat them as optimization
  diagnostics, not direct measures of economic feature importance.
- T4 shows that the frequency effect of adding the spend head is dataset- and metric-dependent.

**Key figures/tables:** T2, F4, F5, F6.

---

### 5.4 Transformer vs LSTM (≈ 0.5 page)

**Claim:** The Transformer performs best on selected dense-data outcomes and worst on sparse
CDNOW, offering qualified support for the density-dependency hypothesis.

**To write:**
- On Ta-Feng, the Transformer achieves RMSE {tformer_tafeng_rmse} vs LSTM {jlstm_tafeng_rmse}
  and the lowest absolute cohort bias among the evaluated models.
- On Dunnhumby, the Transformer edges the LSTM on spend R² ({tformer_dunn_r2} vs {jlstm_dunn_r2})
  and CLV ρ ({tformer_dunn_rho} vs {jlstm_dunn_rho}).
- On CDNOW, the Transformer has the worst frequency MAPE ({tformer_cdnow_mape}%) and highest
  seed variance. This is consistent with, but does not identify, a calibration-data limitation.
- **Limitation:** time complexity of the Transformer is O(T²) vs O(T) for the LSTM; for
  long sequences (Dunnhumby 80 weeks) this is still tractable but would scale poorly.

**Key figures/tables:** T1, F1, F2.

---

### 5.5 Covariate ablation and attribution (≈ 1 page)

**Claim:** Dunnhumby demographics and campaign variables provide small, architecture-specific
gains rather than a uniform improvement.

- Static covariates change LSTM frequency RMSE by {lstm_static_freq_delta:+.3f} and spend R² by
  {lstm_static_spend_delta:+.3f} relative to no covariates.
- Static covariates change Transformer spend R² by {transformer_static_spend_delta:+.3f}, while
  frequency effects are small and mixed across variants.
- For the LSTM, the largest normalized contribution is
  **{shap_top[("lstm", "freq")][0]}** ({shap_top[("lstm", "freq")][1]:.1f}%)
  for frequency and **{shap_top[("lstm", "spend")][0]}**
  ({shap_top[("lstm", "spend")][1]:.1f}%) for conditional log-spend.
- For the Transformer, the corresponding largest contributions are
  **{shap_top[("transformer", "freq")][0]}**
  ({shap_top[("transformer", "freq")][1]:.1f}%) and
  **{shap_top[("transformer", "spend")][0]}**
  ({shap_top[("transformer", "spend")][1]:.1f}%).
- SHAP magnitudes are interventional model attributions conditional on each household's
  fixed transaction history. They do not identify causal campaign effects.
- The primary SHAP cohort uses 100 disjoint background households and all remaining
  701 households with observed demographics, across seeds 7, 42, and 2024.
- Expected-gradient additivity is approximate: normalized residuals range from
  {shap_additivity_low:.1f}% to {shap_additivity_high:.1f}% after escalation to 128 samples.

**Key figures/tables:** T5, T6, F8, F9.

---

### 5.6 Statistical significance (≈ 0.5 page)

Refer to T4. Key claims to test:
- Joint LSTM vs Base LSTM on frequency (CDNOW, UCI) — does the spend head hurt frequency?
- Joint LSTM vs Pareto/NBD on Dunnhumby — is the DL advantage significant?
- Transformer vs Joint LSTM on spend/CLV on Dunnhumby.

**Reporting template:** "A paired customer bootstrap (5,000 resamples), averaging each
deep model over three training seeds, finds Δ = X.XX (95% CI [Y.YY, Z.ZZ],
Holm-adjusted p = W.WW) for Joint LSTM versus Base LSTM on CDNOW frequency RMSE."

---

### 5.7 Synthesis and limitations (0.5 page)

**Where DL wins:**
1. Long-horizon Dunnhumby: Pareto/NBD substantially underpredicts; DL remains near aggregate calibration.
2. Medium-density datasets (UCI, Ta-Feng): DL improves frequency MAPE over Pareto/NBD.
3. CLV ranking is strong across all datasets (ρ 0.39–0.82, lift 3–6×).

**Where DL does not clearly win:**
1. Sparse CDNOW: Pareto/NBD matches or beats DL on frequency MAPE; spend R² is negative.
2. Point-spend prediction is hard at the individual level on all except Dunnhumby.

**Acknowledged limitations:**
- Single probabilistic benchmark (Pareto/NBD): BG/NBD, Pareto/GGG, and GPPM are discussed in
  the methods chapter but were not included in the final comparative run.
- Dunnhumby's final holdout bin is a four-day partial week. Headline tables retain the
  pre-specified 80+22 protocol, while F3 reports complete weeks 1-21 as a sensitivity view.
- Pareto/NBD + Gamma-Gamma assigns zero predicted CLV to 369 Dunnhumby customers. F4 keeps
  that tie intact, so its ordered groups are approximately rather than exactly equal-sized.
- Three seeds per deep configuration; bootstrap inference captures customer sampling uncertainty
  after seed averaging, not uncertainty over the population of possible training seeds.
- SHAP attribution is associational and conditional on each household's fixed transaction history.

---

## Numbers quick-reference (paste into writing)

### CDNOW
- Pareto/NBD + Gamma-Gamma: MAPE={pnbd_cdnow_mape}%, bias={pnbd_cdnow_bias}%, spend R²={pnbd_cdnow_r2}, CLV ρ={pnbd_cdnow_rho}
- Base LSTM: MAPE={base_cdnow_mape}%
- Joint LSTM: MAPE={jlstm_cdnow_mape}%
- Transformer: MAPE={tformer_cdnow_mape}%, bias={tformer_cdnow_bias}%

### Dunnhumby
- Pareto/NBD: bias={pnbd_dunn_bias}% (substantial aggregate underprediction)
- Joint LSTM: bias={jlstm_dunn_bias}%, spend R²={jlstm_dunn_r2}, CLV ρ={jlstm_dunn_rho}
- Transformer: bias={tformer_dunn_bias}%, spend R²={tformer_dunn_r2}, CLV ρ={tformer_dunn_rho}

### UCI
- Pareto/NBD MAPE: {pnbd_uci_mape}%, Joint LSTM MAPE: {jlstm_uci_mape}%

### Ta-Feng
- RMSE: Pareto/NBD={pnbd_tafeng_rmse}, Joint LSTM={jlstm_tafeng_rmse}, Transformer={tformer_tafeng_rmse}

---

*Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} from {FINAL}*
"""

    (OUT / "RESULTS_WRITING_GUIDE.md").write_text(guide)
    print(f"  Saved {OUT / 'RESULTS_WRITING_GUIDE.md'}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(df_all: pd.DataFrame) -> Path:
    print("[Manifest] Recording provenance and checksums...")
    final_manifest = load_final_manifest() or {}
    source_files = []
    for run_name in sorted(df_all["run_name"].astype(str).unique()):
        for suffix in ("_metrics.json", "_arrays.npz"):
            path = FINAL / "tables" / f"{run_name}{suffix}"
            if path.exists():
                source_files.append({
                    "path": str(path),
                    "sha256": _sha256(path),
                })
    for path in (
        SHAP_CSV,
        FINAL / "tables" / "shap_extension3_summary.csv",
        FINAL / "tables" / "shap_extension3_seed_summary.csv",
        FINAL / "tables" / "shap_extension3_additivity.csv",
        FINAL / "tables" / "shap_extension3_convergence.csv",
        FINAL / "tables" / "shap_extension3_run_manifest.json",
        FINAL / "tables" / "shap_extension3_method_notes.md",
        FINAL / "tables" / "shap_extension3_sensitivity_all_households.csv",
        FINAL / "tables" / "shap_extension3_customer_values.csv.gz",
    ):
        if path.exists():
            source_files.append({"path": str(path), "sha256": _sha256(path)})

    output_files = []
    for path in sorted(
        list(TABLES.glob("*"))
        + list(FIGURES.glob("*"))
        + list(SUPPLEMENTARY.rglob("*"))
    ):
        if path.is_file():
            output_files.append({
                "path": str(path.relative_to(OUT)),
                "sha256": _sha256(path),
            })
    guide_path = OUT / "RESULTS_WRITING_GUIDE.md"
    if guide_path.exists():
        output_files.append({
            "path": str(guide_path.relative_to(OUT)),
            "sha256": _sha256(guide_path),
        })

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_dir": str(FINAL),
        "output_dir": str(OUT),
        "final_manifest_version": final_manifest.get("version"),
        "final_manifest_status": final_manifest.get("results_status"),
        "result_rows": int(len(df_all)),
        "deep_learning_seeds": SEEDS,
        "monetary_ground_truth": (
            "Direct raw holdout aggregation from each dataset's Pareto/NBD array artifact; "
            "all monetary models rescored against this common truth."
        ),
        "significance_method": (
            "Paired customer bootstrap with 5,000 resamples; deep-model statistics "
            "averaged across seeds; Holm adjustment across displayed tests."
        ),
        "source_files": source_files,
        "output_files": output_files,
    }
    path = OUT / "ARTIFACT_MANIFEST.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    print("=== Loading data from", FINAL, "===")
    df_all, df_seeds = load_data()
    print(f"  {len(df_all)} runs loaded; {df_all['model'].unique().tolist()}")

    print("\n=== Tables ===")
    make_table_t1(df_all, df_seeds)
    make_table_t2(df_all, df_seeds)
    make_table_t3(df_all, df_seeds)
    sig_df = make_table_t4()
    make_table_t5(df_all)
    make_table_t6()

    print("\n=== Figures ===")
    fig_cohort_bias(df_all)
    fig_freq_mape(df_all)
    fig_weekly_tracking(df_all)
    fig_clv_decile(df_all)
    fig_spend_r2(df_all)
    fig_kendall_weights()
    fig_learning_curves()
    fig_covariate_ablation(df_all)
    fig_shap_attribution()

    print("\n=== Supplementary materials ===")
    copy_shap_supplement()

    print("\n=== Writing guide ===")
    make_writing_guide(df_all, df_seeds, sig_df)
    artifact_manifest = write_artifact_manifest(df_all)

    print("\n=== Manifest ===")
    written = sorted(
        list(TABLES.glob("*"))
        + list(FIGURES.glob("*"))
        + list(SUPPLEMENTARY.rglob("*"))
        + [OUT / "RESULTS_WRITING_GUIDE.md", artifact_manifest]
    )
    for p in written:
        print(f"  {p.relative_to(OUT)}")
    print(f"\nDone. {len(written)} files written to {OUT}/")


if __name__ == "__main__":
    main()
