"""
Benchmark runner for probabilistic CLV models.

Fits and evaluates Pareto/NBD, BG/NBD+Gamma-Gamma, Pareto/GGG, and GPPM
on all datasets using the same config files as the DL experiments.

Usage:
    python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_nbd bgnbd_gg
    python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_nbd bgnbd_gg pareto_ggg gppm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.datasets import (
    CDNOWPipeline,
    DunnhumbyPipeline,
    TaFengPipeline,
    UCIRetailPipeline,
)
from src.evaluation.benchmarks import get_benchmark_model
from src.evaluation.metrics import compute_all_metrics
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PIPELINES = {
    "cdnow": CDNOWPipeline,
    "uci": UCIRetailPipeline,
    "tafeng": TaFengPipeline,
    "dunnhumby": DunnhumbyPipeline,
}


def main():
    parser = argparse.ArgumentParser(
        description="Fit and evaluate probabilistic benchmarks on CLV datasets."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pareto_nbd", "bgnbd_gg"],
        help="List of benchmark models to fit (pareto_nbd, bgnbd_gg, pareto_ggg, gppm).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    output_cfg = config.get("output", {})

    dataset_name = dataset_cfg["name"]
    holdout_weeks = dataset_cfg["holdout_weeks"]

    print(f"Config: {args.config}")
    print(f"Dataset: {dataset_name}")
    print(f"Models: {', '.join(args.models)}")

    # Reproducibility
    set_seed(config.get("training", {}).get("seed", 42))

    # Load data pipeline and get RFM summary
    if dataset_name not in PIPELINES:
        print(f"Unknown dataset: {dataset_name!r}. Choose from: {list(PIPELINES)}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Loading {dataset_name} dataset...")
    pipeline = PIPELINES[dataset_name]()
    rfm_calib, rfm_holdout = pipeline.get_rfm_summary(config)

    logger.info(f"Calibration RFM: {len(rfm_calib)} customers")
    logger.info(f"Holdout ground truth: {len(rfm_holdout)} customers")

    # Verify customer ordering
    assert np.array_equal(
        rfm_calib["customer_id"].values, rfm_holdout["customer_id"].values
    ), "Customer ID mismatch between calibration and holdout"

    customer_ids = rfm_calib["customer_id"].values

    # Results table
    results_dir = Path(output_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Fit and evaluate each model
    results_list = []
    for model_name in args.models:
        print(f"\n--- {model_name.upper()} ---")
        try:
            model = get_benchmark_model(model_name)
            model.fit(rfm_calib)

            # Predict frequency
            pred_freq = model.predict_freq(rfm_calib, holdout_weeks)
            true_freq = rfm_holdout["actual_freq"].values

            # Predict spend (optional; None for frequency-only models)
            pred_spend = model.predict_spend(rfm_calib, holdout_weeks)
            true_spend = rfm_holdout["actual_spend"].values if pred_spend is not None else None

            # Log-transform spend if available
            if true_spend is not None and pred_spend is not None:
                pred_spend_log = np.log1p(pred_spend)
                true_spend_log = np.log1p(true_spend)
            else:
                pred_spend_log = None
                true_spend_log = None

            # Compute metrics
            metrics = compute_all_metrics(
                y_freq_true=true_freq.astype(np.float32),
                y_freq_pred=pred_freq,
                y_spend_true_log=true_spend_log,
                y_spend_pred_log=pred_spend_log,
                customer_ids=customer_ids,
            )

            # Add model name and dataset
            metrics["model"] = model_name
            metrics["dataset"] = dataset_name

            # Save metrics
            metrics_path = tables_dir / f"{model_name}_{dataset_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics: {metrics_path}")

            # Print metrics
            print(f"Metrics for {model_name}:")
            for k, v in metrics.items():
                if k not in ("model", "dataset"):
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

            results_list.append(metrics)

        except Exception as e:
            logger.error(f"Failed to fit {model_name}: {e}", exc_info=True)
            print(f"ERROR: {model_name} failed: {e}")

    # Print summary table
    if results_list:
        print("\n=== Summary ===")
        print(f"{'Model':<15} {'Freq RMSE':<12} {'Freq MAPE':<12} {'Bias %':<10} {'Spend MAE':<12} {'Spend R²':<10}")
        print("-" * 75)
        for res in results_list:
            model = res["model"]
            freq_rmse = f"{res.get('freq_rmse', np.nan):.4f}"
            freq_mape = f"{res.get('freq_mape', np.nan):.4f}"
            bias_pct = f"{res.get('bias_pct', np.nan):.2f}"
            spend_mae = f"{res.get('spend_mae', np.nan):.4f}"
            spend_r2 = f"{res.get('spend_r2', np.nan):.4f}"
            print(f"{model:<15} {freq_rmse:<12} {freq_mape:<12} {bias_pct:<10} {spend_mae:<12} {spend_r2:<10}")


if __name__ == "__main__":
    main()
