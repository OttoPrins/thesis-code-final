"""
Build comparison tables and visual analytics from all benchmark and DL model results.

Functions:
    parse_run_name              — split run_name into (canonical_model, dataset, version, seed, mode)
    build_comparison_table      — read *_metrics.json files for one dataset → DataFrame + CSV
    build_all_comparison_tables — run build_comparison_table for all datasets
    aggregate_all_results       — scan results/ for all *_metrics.json → combined DataFrame
    aggregate_seeds             — group by (model, dataset, mode) and report mean ± std
    export_latex_table          — export DataFrame as publication-ready LaTeX table
    plot_model_comparison_bars  — grouped bar chart of primary metrics across models/datasets
    plot_kendall_weight_evolution— line plot of Kendall task weights across training epochs

CLI (python -m src.evaluation.compare):
    --latex   exports comparison_all.tex to results/tables/
    --plots   generates bar chart and weight-evolution plots to experiments/insights/
    --seeds   aggregates seeded runs with mean ± std into comparison_seeds.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.final_manifest import load_final_manifest, result_matches_manifest

logger = logging.getLogger(__name__)

# run_name patterns we care about, in order of specificity:
#   <model>_<dataset>_v<N>_seed<S>_<mode>
#   <model>_<dataset>_v<N>_seed<S>
#   <model>_<dataset>_v<N>
#   <model>_<dataset>
# where <model> ∈ {lstm_base, lstm_joint, transformer_joint, extension3, ...}
# and <dataset> ∈ {cdnow, uci, tafeng, dunnhumby}
_DATASETS = ("cdnow", "uci", "tafeng", "dunnhumby")
_MODE_RE = re.compile(r"_(sample|expected)$")
_SEED_RE = re.compile(r"_seed(\d+)$")
_VERSION_RE = re.compile(r"_(v\d+|final)$")

# Canonical model order for table rows
_MODEL_ORDER = [
    "pareto_nbd",
    "bgnbd_gg",
    "pareto_ggg",
    "gppm",
    "gamma_poisson",
    "lstm_base",
    "lstm_joint",
    "transformer_joint",
    "extension3_lstm",
    "extension3_transformer",
]

_PRIMARY_METRICS = ["freq_mape", "bias_pct", "spend_mae_raw", "clv_mae", "clv_spearman", "clv_decile_lift"]
_META_KEYS = {
    "arrays_file",
    "final_manifest_version",
    "final_manifest_config_hash",
    "final_manifest_actual_config_hash",
    "final_manifest_run_name",
    "final_manifest_benchmark_name",
    "final_manifest_config",
    "final_manifest_seed",
    "final_manifest_epochs",
    "final_manifest_inference_mode",
    "final_manifest_n_scenarios",
}

_BENCHMARK_MODELS = {"pareto_nbd", "bgnbd_gg", "pareto_ggg", "gppm", "gamma_poisson"}


def _sort_key(model_name: str) -> int:
    try:
        return _MODEL_ORDER.index(model_name)
    except ValueError:
        return len(_MODEL_ORDER)


def parse_run_name(stem: str) -> Tuple[str, str, Optional[str], Optional[int], Optional[str]]:
    """
    Decompose a run_name like 'lstm_joint_cdnow_v2_seed42_expected' into
    (canonical_model, dataset, version, seed, mode).

    Returns tuple where missing components are None / 'unknown'. Stripping order:
        1. trailing _<mode>
        2. trailing _seed<N>
        3. trailing _v<N>
        4. trailing _<dataset>
    """
    s = stem
    mode = None
    m = _MODE_RE.search(s)
    if m:
        mode = m.group(1)
        s = s[:m.start()]
    seed = None
    m = _SEED_RE.search(s)
    if m:
        seed = int(m.group(1))
        s = s[:m.start()]
    version = None
    m = _VERSION_RE.search(s)
    if m:
        version = m.group(0).lstrip("_")
        s = s[:m.start()]
    dataset = "unknown"
    for ds in _DATASETS:
        if s.endswith(f"_{ds}"):
            dataset = ds
            s = s[: -(len(ds) + 1)]
            break
    canonical_model = s
    return canonical_model, dataset, version, seed, mode


def _metric_stem(path: str | Path) -> str:
    return Path(path).stem.removesuffix("_metrics")


def _filter_metric_files(
    metrics_files: List[str],
    *,
    final_only: bool = True,
    include_expected: bool = False,
) -> List[str]:
    def deep_artifacts_present(fpath: str, metrics: dict) -> tuple[bool, str]:
        stem = _metric_stem(fpath)
        metrics_path = Path(fpath)
        arrays_file = metrics.get("arrays_file")
        if not arrays_file:
            return False, "deep-learning result lacks arrays_file sidecar"
        arrays_path = metrics_path.with_name(arrays_file)
        if not arrays_path.exists():
            return False, f"array sidecar is missing: {arrays_path.name}"
        checkpoint_path = metrics_path.parent.parent / "checkpoints" / f"{stem}.pt"
        if not checkpoint_path.exists():
            return False, f"checkpoint is missing: {checkpoint_path}"
        return True, "ok"

    def diagnostics_valid(fpath: str) -> bool:
        try:
            with open(fpath) as f:
                metrics = json.load(f)
        except Exception:
            return False
        if metrics.get("benchmark_valid") is False:
            logger.info(
                "Skipping %s: benchmark diagnostics marked this run invalid.",
                Path(fpath).name,
            )
            return False
        stem = _metric_stem(fpath)
        is_gppm = metrics.get("model") == "gppm" or stem.startswith("gppm_")
        if is_gppm and metrics.get("benchmark_valid") is not True:
            logger.info(
                "Skipping %s: GPPM result lacks passing Stan diagnostics; regenerate after repairs.",
                Path(fpath).name,
            )
            return False
        is_benchmark = (
            "final_manifest_benchmark_name" in metrics
            or metrics.get("model") in _BENCHMARK_MODELS
        )
        if metrics.get("run_valid") is False:
            logger.info(
                "Skipping %s: run validity checks failed (%s).",
                Path(fpath).name,
                metrics.get("run_invalid_reason", "no reason recorded"),
            )
            return False
        if not is_benchmark and metrics.get("run_valid") is not True:
            logger.info(
                "Skipping %s: deep-learning result lacks run_valid=True; regenerate after repairs.",
                Path(fpath).name,
            )
            return False
        return True

    if not final_only:
        return [fpath for fpath in metrics_files if diagnostics_valid(fpath)]

    manifest = load_final_manifest()
    if manifest is None:
        logger.warning("Final manifest not found; no final-only results will be included.")
        return []

    out: List[str] = []
    for fpath in metrics_files:
        stem = _metric_stem(fpath)
        if not diagnostics_valid(fpath):
            continue
        try:
            with open(fpath) as f:
                metrics = json.load(f)
        except Exception:
            continue
        ok, reason = result_matches_manifest(
            stem,
            metrics,
            manifest,
            include_expected=include_expected,
            include_unseeded=True,
        )
        if not ok:
            logger.info(
                "Skipping %s: %s.",
                Path(fpath).name,
                reason,
            )
            continue
        if stem not in set(manifest.get("benchmark_run_names", [])):
            artifacts_ok, artifact_reason = deep_artifacts_present(fpath, metrics)
            if not artifacts_ok:
                logger.info(
                    "Skipping %s: %s.",
                    Path(fpath).name,
                    artifact_reason,
                )
                continue
        out.append(fpath)
    return out


def build_comparison_table(
    results_dir: str,
    dataset: str,
    *,
    final_only: bool = True,
    include_expected: bool = False,
) -> pd.DataFrame:
    """
    Read all *_{dataset}*_metrics.json files and assemble a comparison DataFrame.
    Saves to results/tables/comparison_{dataset}.csv.

    Args:
        results_dir: Path to results directory (typically "results/")
        dataset:     Dataset name (e.g., "cdnow", "uci", "tafeng", "dunnhumby")

    Returns:
        DataFrame with columns:
            model, dataset, freq_rmse, freq_mae, freq_mape, bias_pct,
            spend_mae_log, spend_rmse_log, spend_r2_log, spend_mae_raw, spend_rmse_raw
    """
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"

    if not tables_dir.exists():
        logger.warning(f"Results directory {tables_dir} does not exist")
        return pd.DataFrame()

    import glob as _glob
    pattern = str(tables_dir / f"*{dataset}*metrics.json")
    metrics_files = sorted(_glob.glob(pattern))
    metrics_files = [f for f in metrics_files if not Path(f).name.startswith("comparison_")]
    metrics_files = _filter_metric_files(
        metrics_files,
        final_only=final_only,
        include_expected=include_expected,
    )

    if not metrics_files:
        logger.warning(f"No metrics files found for dataset {dataset!r} in {tables_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(metrics_files)} metrics files for {dataset}")

    rows = []
    for metrics_file in metrics_files:
        with open(metrics_file) as f:
            metrics = json.load(f)

        if "model" in metrics:
            model_name = metrics["model"]
        else:
            stem = Path(metrics_file).stem.removesuffix("_metrics")
            model_name, _, _, _, _ = parse_run_name(stem)

        row = {
            "model": model_name,
            "dataset": dataset,
            "freq_rmse":       metrics.get("freq_rmse", np.nan),
            "freq_mae":        metrics.get("freq_mae", np.nan),
            "freq_mape":       metrics.get("freq_mape", np.nan),
            "bias_pct":        metrics.get("bias_pct", np.nan),
            "freq_weekly_mape": metrics.get("freq_weekly_mape", np.nan),
            "freq_weekly_bias_pct": metrics.get("freq_weekly_bias_pct", np.nan),
            "freq_mase":       metrics.get("freq_mase", np.nan),
            "freq_normalized_gini": metrics.get("freq_normalized_gini", np.nan),
            "spend_mae_log":   metrics.get("spend_mae_log", np.nan),
            "spend_rmse_log":  metrics.get("spend_rmse_log", np.nan),
            "spend_r2_log":    metrics.get("spend_r2_log", np.nan),
            "spend_mae_raw":   metrics.get("spend_mae_raw", np.nan),
            "spend_rmse_raw":  metrics.get("spend_rmse_raw", np.nan),
            "spend_weekly_mape": metrics.get("spend_weekly_mape", np.nan),
            "spend_weekly_bias_pct": metrics.get("spend_weekly_bias_pct", np.nan),
            "spend_mase":      metrics.get("spend_mase", np.nan),
            "spend_normalized_gini": metrics.get("spend_normalized_gini", np.nan),
            "clv_mae":         metrics.get("clv_mae", np.nan),
            "clv_rmse":        metrics.get("clv_rmse", np.nan),
            "clv_spearman":    metrics.get("clv_spearman", np.nan),
            "clv_decile_lift": metrics.get("clv_decile_lift", np.nan),
            "clv_normalized_gini": metrics.get("clv_normalized_gini", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["_sort"] = df["model"].apply(_sort_key)
    df = df.sort_values("_sort").drop("_sort", axis=1).reset_index(drop=True)

    output_path = tables_dir / f"comparison_{dataset}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table: {output_path}")
    return df


def build_all_comparison_tables(
    results_dir: str,
    datasets: Optional[List[str]] = None,
    *,
    final_only: bool = True,
    include_expected: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build comparison tables for all datasets (or specified subset)."""
    if datasets is None:
        datasets = ["cdnow", "uci", "tafeng", "dunnhumby"]

    tables = {}
    for dataset in datasets:
        logger.info(f"Building comparison table for {dataset} ...")
        df = build_comparison_table(
            results_dir,
            dataset,
            final_only=final_only,
            include_expected=include_expected,
        )
        if not df.empty:
            tables[dataset] = df
    return tables


def aggregate_all_results(
    results_dir: str = "results/",
    *,
    final_only: bool = True,
    include_expected: bool = False,
) -> pd.DataFrame:
    """
    Scan results/tables/ for all *_metrics.json files and compile into one DataFrame.

    Each file becomes one row; canonical_model / dataset / version / seed / mode
    are parsed from the run_name via parse_run_name. Includes all metric columns
    (log-space and raw-currency).

    Returns:
        DataFrame with columns:
            model, dataset, version, seed, mode, + all available metric columns
    """
    tables_dir = Path(results_dir) / "tables"
    if not tables_dir.exists():
        logger.warning(f"Results directory {tables_dir} does not exist")
        return pd.DataFrame()

    import glob as _glob
    all_files = sorted(_glob.glob(str(tables_dir / "*_metrics.json")))
    all_files = [f for f in all_files if not Path(f).name.startswith("comparison_")]
    all_files = _filter_metric_files(
        all_files,
        final_only=final_only,
        include_expected=include_expected,
    )

    if not all_files:
        logger.warning(f"No *_metrics.json files found in {tables_dir}")
        return pd.DataFrame()

    rows = []
    for fpath in all_files:
        with open(fpath) as f:
            metrics = json.load(f)

        stem = Path(fpath).stem.removesuffix("_metrics")
        canonical_model, dataset, version, seed, mode = parse_run_name(stem)
        # Benchmarks store dataset/model directly in the JSON; prefer those.
        if "dataset" in metrics:
            dataset = metrics["dataset"]
        if "model" in metrics:
            canonical_model = metrics["model"]

        row = {
            "model": canonical_model,
            "dataset": dataset,
            "version": version,
            "seed": seed,
            "mode": mode,
        }
        row.update({
            k: v for k, v in metrics.items()
            if k not in ("model", "dataset") and k not in _META_KEYS
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df["_sort"] = df["model"].apply(_sort_key)
    df = df.sort_values(["dataset", "_sort"]).drop("_sort", axis=1).reset_index(drop=True)
    return df


def aggregate_seeds(
    df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Group seeded runs by (model, dataset, mode, version) and report mean ± std.

    For benchmarks (no seed column), the row passes through unchanged with
    mean = value and std = NaN.

    Args:
        df:          DataFrame from aggregate_all_results.
        metric_cols: Which metric columns to aggregate. Defaults to the union of
                     numeric columns minus identifying ones.

    Returns:
        DataFrame with columns: model, dataset, mode, version, n_seeds, and for
        each metric two columns "<metric>_mean" and "<metric>_std".
    """
    if df.empty:
        return df

    if metric_cols is None:
        ignore = {"model", "dataset", "version", "seed", "mode"}
        metric_cols = [
            c for c in df.columns
            if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
        ]

    group_keys = ["model", "dataset", "mode", "version"]
    rows = []
    # Use dropna=False so benchmark rows (mode/version/seed all NaN) still group.
    for keys, sub in df.groupby(group_keys, dropna=False):
        out = dict(zip(group_keys, keys))
        out["n_seeds"] = int(sub["seed"].notna().sum()) if "seed" in sub.columns else 0
        for col in metric_cols:
            vals = sub[col].dropna().astype(float)
            if vals.empty:
                out[f"{col}_mean"] = float("nan")
                out[f"{col}_std"] = float("nan")
            else:
                out[f"{col}_mean"] = float(vals.mean())
                out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(out)

    agg = pd.DataFrame(rows)
    agg["_sort"] = agg["model"].apply(_sort_key)
    agg = agg.sort_values(["dataset", "_sort"]).drop("_sort", axis=1).reset_index(drop=True)
    return agg


def export_latex_table(
    df: pd.DataFrame,
    out_path: str = "results/tables/comparison_all.tex",
    caption: str = "Model comparison across datasets",
    label: str = "tab:model_comparison",
) -> None:
    """
    Export DataFrame to a publication-ready booktabs LaTeX table.

    Selects and renames key columns for thesis output. Saves the .tex file
    directly (not a full document — use \\input{} to embed in thesis).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols_order = [
        "model", "dataset",
        "freq_rmse", "freq_mape", "bias_pct",
        "spend_mae_log", "spend_mae_raw", "spend_r2_log",
        "clv_mae", "clv_spearman", "clv_decile_lift",
    ]
    rename_map = {
        "model": "Model",
        "dataset": "Dataset",
        "freq_rmse": "Freq RMSE",
        "freq_mape": "Freq MAPE \\%",
        "bias_pct": "Bias \\%",
        "spend_mae_log": "Spend MAE (log)",
        "spend_mae_raw": "Spend MAE (\\$)",
        "spend_r2_log": "Spend $R^2$ (log)",
        "clv_mae": "CLV MAE (\\$)",
        "clv_spearman": "CLV Spearman",
        "clv_decile_lift": "CLV Decile Lift",
    }

    # Keep only available columns
    available = [c for c in cols_order if c in df.columns]
    sub = df[available].rename(columns=rename_map)

    latex_str = sub.to_latex(
        index=False,
        float_format="%.3f",
        na_rep="—",
        escape=False,
        caption=caption,
        label=label,
    )

    # Upgrade to booktabs (pandas may not add booktabs rules by default in older versions)
    latex_str = latex_str.replace("\\begin{tabular}", "\\begin{tabular}")
    if "\\toprule" not in latex_str:
        latex_str = latex_str.replace("\\hline\n", "\\toprule\n", 1)
        latex_str = latex_str.replace("\n\\hline\n", "\n\\midrule\n", 1)
        latex_str = latex_str.replace("\n\\hline\n", "\n\\bottomrule\n", 1)

    out_path.write_text(latex_str)
    print(f"LaTeX table saved: {out_path}")


def plot_model_comparison_bars(
    df: pd.DataFrame,
    out_dir: str = "experiments/insights",
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Grouped bar chart of primary metrics across models, coloured by dataset.
    Saves one figure per metric to experiments/insights/.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    if metrics is None:
        metrics = _PRIMARY_METRICS

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric!r} not in DataFrame; skipping.")
            continue

        plot_df = df[["model", "dataset", metric]].dropna(subset=[metric])
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        datasets = sorted(plot_df["dataset"].unique())
        ordered = [m for m in _MODEL_ORDER if m in plot_df["model"].values]
        extra = [m for m in plot_df["model"].unique() if m not in _MODEL_ORDER]
        models = list(dict.fromkeys(ordered + extra))

        x = np.arange(len(models))
        width = 0.8 / max(len(datasets), 1)
        palette = sns.color_palette("tab10", n_colors=len(datasets))

        for i, ds in enumerate(datasets):
            vals = []
            for m in models:
                row = plot_df[(plot_df["model"] == m) & (plot_df["dataset"] == ds)]
                vals.append(float(row[metric].values[0]) if not row.empty else float("nan"))
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=ds, color=palette[i])

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Model comparison — {metric}")
        ax.legend(title="Dataset")
        fig.tight_layout()

        fig_path = out_path / f"comparison_{metric}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {fig_path}")


def plot_kendall_weight_evolution(
    history_files: Optional[List[str]] = None,
    results_dir: str = "results/",
    out_dir: str = "experiments/insights",
) -> None:
    """
    Line plot of Kendall multi-task weight evolution across training epochs.
    One line per joint model run found in results/tables/*_history.json.
    """
    import glob as _glob
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if history_files is None:
        tables_dir = Path(results_dir) / "tables"
        history_files = sorted(_glob.glob(str(tables_dir / "*_history.json")))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for fpath in history_files:
        with open(fpath) as f:
            history = json.load(f)

        if "task_weight_freq" not in history:
            continue  # base model — no Kendall weights

        run_name = Path(fpath).stem.removesuffix("_history")
        epochs = list(range(1, len(history["task_weight_freq"]) + 1))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history["task_weight_freq"], label="freq weight (1/2σ²_freq)",
                linewidth=1.8, color="#4C72B0")
        ax.plot(epochs, history["task_weight_spend"], label="spend weight (1/2σ²_spend)",
                linewidth=1.8, color="#DD8452", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Task weight")
        ax.set_title(f"Kendall uncertainty weights — {run_name}")
        ax.legend()
        fig.tight_layout()

        fig_path = out_path / f"kendall_weights_{run_name}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results and generate thesis figures.")
    parser.add_argument("--results_dir", default="results/", help="Path to results directory")
    parser.add_argument("--latex", action="store_true", help="Export LaTeX comparison table")
    parser.add_argument("--plots", action="store_true", help="Generate comparison bar charts")
    parser.add_argument("--weights", action="store_true",
                        help="Generate Kendall weight evolution plots")
    parser.add_argument("--seeds", action="store_true",
                        help="Aggregate seeded runs (mean ± std) into comparison_seeds.csv")
    parser.add_argument("--include_exploratory", action="store_true",
                        help="Include non-manifest or pre-final metrics files")
    parser.add_argument("--include_expected", action="store_true",
                        help="Include expected-mode diagnostic runs")
    parser.add_argument("--all", action="store_true", help="Run all output types")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    do_all = args.all
    df = aggregate_all_results(
        args.results_dir,
        final_only=not args.include_exploratory,
        include_expected=args.include_expected,
    )

    if df.empty:
        print("No results found. Run train.py experiments first.")
    else:
        print(f"Aggregated {len(df)} model results.")

        if do_all or args.latex:
            export_latex_table(df, out_path="results/tables/comparison_all.tex")

        if do_all or args.plots:
            plot_model_comparison_bars(df)

        if do_all or args.weights:
            plot_kendall_weight_evolution(results_dir=args.results_dir)

        if do_all or args.seeds:
            seeds_df = aggregate_seeds(df)
            out_path = Path(args.results_dir) / "tables" / "comparison_seeds.csv"
            seeds_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

        # Always save the aggregated CSV
        df.to_csv(Path(args.results_dir) / "tables" / "comparison_all.csv", index=False)
        print(f"Saved: {Path(args.results_dir) / 'tables' / 'comparison_all.csv'}")
