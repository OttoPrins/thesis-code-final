"""
Build comparison tables from all benchmark and DL model results.

Aggregates metrics JSONs into a CSV for thesis reporting.
"""

import json
import logging
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_comparison_table(results_dir: str, dataset: str) -> pd.DataFrame:
    """
    Read all *_{dataset}_metrics.json files and assemble a comparison DataFrame.

    Saves to results/tables/comparison_{dataset}.csv

    Args:
        results_dir: Path to results directory (typically "results/")
        dataset: Dataset name (e.g., "cdnow", "uci", "tafeng", "dunnhumby")

    Returns:
        DataFrame with columns:
            model, dataset, freq_rmse, freq_mae, freq_mape, bias_pct,
            spend_mae, spend_rmse, spend_r2
    """
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"

    if not tables_dir.exists():
        logger.warning(f"Results directory {tables_dir} does not exist")
        return pd.DataFrame()

    # Find all metrics files for this dataset.
    # Benchmark files: "{model}_{dataset}_metrics.json"   e.g. pareto_nbd_cdnow_metrics.json
    # DL model files:  "{run_name}_metrics.json"          e.g. lstm_base_cdnow_v1_metrics.json
    # Both contain the dataset name, so we match on that.
    pattern = str(tables_dir / f"*{dataset}*metrics.json")
    metrics_files = sorted(glob(pattern))

    # Exclude the comparison CSV itself if it ended up with a .json extension (shouldn't happen)
    metrics_files = [f for f in metrics_files if not Path(f).name.startswith("comparison_")]

    if not metrics_files:
        logger.warning(f"No metrics files found for dataset {dataset!r} in {tables_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(metrics_files)} metrics files for {dataset}")

    rows = []
    for metrics_file in metrics_files:
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Parse model name from filename.
        # Strategy: strip trailing "_metrics" suffix, then remove the dataset portion
        # to recover the model identifier.
        # e.g. "pareto_nbd_cdnow_metrics"     → strip "_metrics" → "pareto_nbd_cdnow"
        #                                      → remove trailing "_{dataset}" → "pareto_nbd"
        # e.g. "lstm_base_cdnow_v1_metrics"   → strip "_metrics" → "lstm_base_cdnow_v1"
        #                                      → remove first occurrence of "_{dataset}_" → "lstm_base_v1"
        # Use "model" key from JSON if available (benchmark runner writes it; DL runner does not).
        if "model" in metrics:
            model_name = metrics["model"]
        else:
            stem = Path(metrics_file).stem  # e.g. "lstm_base_cdnow_v1_metrics"
            stem = stem.removesuffix("_metrics")  # e.g. "lstm_base_cdnow_v1"
            # Remove the dataset portion; it can appear mid-name (DL) or at end (benchmark)
            model_name = stem.replace(f"_{dataset}_", "_").rstrip(f"_{dataset}")

        row = {
            "model": model_name,
            "dataset": dataset,
            "freq_rmse": metrics.get("freq_rmse", np.nan),
            "freq_mae": metrics.get("freq_mae", np.nan),
            "freq_mape": metrics.get("freq_mape", np.nan),
            "bias_pct": metrics.get("bias_pct", np.nan),
            "spend_mae": metrics.get("spend_mae", np.nan),
            "spend_rmse": metrics.get("spend_rmse", np.nan),
            "spend_r2": metrics.get("spend_r2", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by model name (probabilistic benchmarks first, then DL models)
    benchmark_models = [
        "pareto_nbd",
        "bgnbd_gg",
        "pareto_ggg",
        "gppm",
        "lstm_base",
        "lstm_joint",
        "transformer_joint",
    ]
    # Create a sort key that respects the predefined order
    def sort_key(model_name):
        try:
            return benchmark_models.index(model_name)
        except ValueError:
            # Unknown model; put at end
            return len(benchmark_models)

    df["sort_key"] = df["model"].apply(sort_key)
    df = df.sort_values("sort_key").drop("sort_key", axis=1).reset_index(drop=True)

    # Save to CSV
    output_path = tables_dir / f"comparison_{dataset}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table: {output_path}")

    return df


def build_all_comparison_tables(results_dir: str, datasets: list = None) -> dict:
    """
    Build comparison tables for all datasets (or specified subset).

    Args:
        results_dir: Path to results directory
        datasets: List of dataset names (default: all standard datasets)

    Returns:
        Dict mapping dataset name to comparison DataFrame
    """
    if datasets is None:
        datasets = ["cdnow", "uci", "tafeng", "dunnhumby"]

    tables = {}
    for dataset in datasets:
        logger.info(f"Building comparison table for {dataset}...")
        df = build_comparison_table(results_dir, dataset)
        if not df.empty:
            tables[dataset] = df

    return tables


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        build_comparison_table("results/", dataset)
    else:
        build_all_comparison_tables("results/")
