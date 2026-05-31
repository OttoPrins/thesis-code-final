"""
Benchmark runner for probabilistic CLV models.

Fits and evaluates Pareto/NBD, BG/NBD+Gamma-Gamma, Pareto/GGG, and GPPM
on all datasets using the same config files as the DL experiments.

Usage:
    python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_nbd bgnbd_gg
    python run_benchmarks.py --config experiments/configs/lstm_base_cdnow.yaml --models pareto_nbd bgnbd_gg pareto_ggg gppm
"""

import argparse
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
from src.data.transforms import TemporalSplitter, WeeklyAggregator
from src.evaluation.benchmarks import get_benchmark_model
from src.evaluation.metrics import compute_all_metrics, save_metrics_with_artifacts
from src.utils.config import load_config
from src.utils.final_manifest import attach_manifest_metadata
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


def _calibration_weekly_counts_for_gppm(pipeline, config: dict):
    """Rebuild weekly calibration counts for the Stan-backed CDNOW GPPM."""
    import pandas as pd

    dataset_cfg = config["dataset"]
    pipeline._current_dataset_cfg = dataset_cfg
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    clean_df = pipeline.clean(pipeline.load_raw(raw_dir))

    cohort_end = dataset_cfg.get("cohort_end_date")
    if cohort_end is not None:
        cohort_end = pd.to_datetime(cohort_end)
        first_purchase = clean_df.groupby("customer_id")["date"].min()
        cohort_ids = first_purchase[first_purchase <= cohort_end].index
        clean_df = clean_df[clean_df["customer_id"].isin(cohort_ids)].copy()

    sample_n = dataset_cfg.get("sample_n")
    if sample_n is not None:
        all_custs = np.sort(clean_df["customer_id"].unique())
        if len(all_custs) > sample_n:
            rng_samp = np.random.RandomState(dataset_cfg.get("sample_seed", 0))
            sampled = rng_samp.choice(all_custs, size=sample_n, replace=False)
            clean_df = clean_df[clean_df["customer_id"].isin(sampled)].copy()

    agg = WeeklyAggregator().fit_transform(
        clean_df,
        origin_date=dataset_cfg.get("origin_date"),
    )
    calib, _ = TemporalSplitter(
        dataset_cfg["calibration_weeks"],
        dataset_cfg["holdout_weeks"],
    ).split(agg)
    return calib


def _holdout_weekly_matrices(pipeline, config: dict, customer_ids: np.ndarray):
    """
    Build true holdout weekly frequency and raw-spend matrices, shape (N, H).

    Rows are aligned to `customer_ids` (the calibration RFM order); columns are
    holdout weeks 0..H-1 (the splitter numbers holdout weeks from calibration_weeks,
    so we subtract it). These give benchmarks the per-week ground truth needed to
    score Valendin's weekly MAPE and per-customer CLV on the same basis as the LSTM.
    """
    import pandas as pd

    dataset_cfg = config["dataset"]
    pipeline._current_dataset_cfg = dataset_cfg
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    clean_df = pipeline.clean(pipeline.load_raw(raw_dir))

    cohort_end = dataset_cfg.get("cohort_end_date")
    if cohort_end is not None:
        cohort_end = pd.to_datetime(cohort_end)
        first_purchase = clean_df.groupby("customer_id")["date"].min()
        cohort_ids = first_purchase[first_purchase <= cohort_end].index
        clean_df = clean_df[clean_df["customer_id"].isin(cohort_ids)].copy()

    sample_n = dataset_cfg.get("sample_n")
    if sample_n is not None:
        all_custs = np.sort(clean_df["customer_id"].unique())
        if len(all_custs) > sample_n:
            rng_samp = np.random.RandomState(dataset_cfg.get("sample_seed", 0))
            sampled = rng_samp.choice(all_custs, size=sample_n, replace=False)
            clean_df = clean_df[clean_df["customer_id"].isin(sampled)].copy()

    agg = WeeklyAggregator().fit_transform(
        clean_df, origin_date=dataset_cfg.get("origin_date")
    )
    calib_weeks = int(dataset_cfg["calibration_weeks"])
    holdout_weeks = int(dataset_cfg["holdout_weeks"])
    _, holdout = TemporalSplitter(calib_weeks, holdout_weeks).split(agg)

    cid_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    N, H = len(customer_ids), holdout_weeks
    freq_m = np.zeros((N, H), dtype=np.float32)
    spend_m = np.zeros((N, H), dtype=np.float32)

    holdout = holdout[holdout["customer_id"].isin(cid_to_idx)]
    if not holdout.empty:
        rows = holdout["customer_id"].map(cid_to_idx).values.astype(int)
        cols = holdout["week"].values.astype(int) - calib_weeks
        valid = (cols >= 0) & (cols < H)
        freq_m[rows[valid], cols[valid]] = holdout["weekly_freq"].values.astype(np.float32)[valid]
        spend_m[rows[valid], cols[valid]] = holdout["weekly_spend"].values.astype(np.float32)[valid]
    return freq_m, spend_m


def main():
    parser = argparse.ArgumentParser(
        description="Fit and evaluate probabilistic benchmarks on CLV datasets."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pareto_nbd", "bgnbd_gg"],
        help=(
            "List of benchmark models to fit "
            "(pareto_nbd, bgnbd_gg, pareto_ggg, gppm, gamma_poisson)."
        ),
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
            if model_name == "gppm":
                if dataset_name != "cdnow":
                    raise ValueError(
                        "True GPPM is restricted to CDNOW replication in the final "
                        "thesis protocol. Use `gamma_poisson` for an explicit "
                        "non-thesis diagnostic on other datasets."
                    )
                calib_weekly = _calibration_weekly_counts_for_gppm(pipeline, config)
                model.fit_weekly_counts(
                    calib_weekly_df=calib_weekly,
                    customer_ids=customer_ids,
                    calibration_weeks=dataset_cfg["calibration_weeks"],
                    holdout_weeks=holdout_weeks,
                )
            else:
                model.fit(rfm_calib)

            # Predict frequency (per-customer holdout total, for individual-level RMSE)
            pred_freq = model.predict_freq(rfm_calib, holdout_weeks)
            true_freq = rfm_holdout["actual_freq"].values

            # True holdout weekly matrices (N, H), aligned to the calibration RFM order.
            true_freq_per_week, true_spend_per_week_raw = _holdout_weekly_matrices(
                pipeline, config, customer_ids
            )
            # Benchmark per-week expected transactions (cumulative differencing) so the
            # SAME Valendin weekly MAPE is computed for benchmarks and the LSTM.
            pred_freq_per_week = model.predict_freq_per_week(rfm_calib, holdout_weeks)

            freq_week_kwargs = {
                "y_freq_true_per_week": true_freq_per_week,
                "y_freq_pred_per_week": pred_freq_per_week,
            }

            # Predict spend per week (raw currency). Frequency-only models return None.
            # Passing per-week raw matrices (not just totals) unlocks per-customer CLV.
            pred_spend_per_week = model.predict_spend_per_week(rfm_calib, holdout_weeks)
            spend_kwargs = {}
            if pred_spend_per_week is not None:
                spend_kwargs["y_spend_true_per_week_raw"] = true_spend_per_week_raw
                spend_kwargs["y_spend_pred_per_week_raw"] = pred_spend_per_week.astype(np.float32)

            # Compute metrics
            metrics = compute_all_metrics(
                y_freq_true=true_freq.astype(np.float32),
                y_freq_pred=pred_freq,
                customer_ids=customer_ids,
                weekly_discount_rate=config.get("evaluation", {}).get(
                    "weekly_discount_rate", 0.0
                ),
                **freq_week_kwargs,
                **spend_kwargs,
            )
            diagnostics = getattr(model, "diagnostics", None)
            if diagnostics:
                metrics.update(diagnostics)

            # Add model name and dataset
            metrics["model"] = model_name
            metrics["dataset"] = dataset_name
            protocol_name = dataset_cfg.get(
                "protocol_name",
                f"{dataset_name}_{dataset_cfg['calibration_weeks']}x{dataset_cfg['holdout_weeks']}",
            )
            metrics["protocol_name"] = str(protocol_name)
            metrics["cohort_size"] = int(len(customer_ids))
            metrics["calibration_weeks"] = int(dataset_cfg["calibration_weeks"])
            metrics["holdout_weeks"] = int(dataset_cfg["holdout_weeks"])
            metrics["inference_mode"] = "benchmark_expected_total"
            metrics["n_scenarios"] = 0
            benchmark_run_name = f"{model_name}_{dataset_name}"
            attach_manifest_metadata(
                metrics,
                config_path=args.config,
                config=config,
                run_name=benchmark_run_name,
                benchmark_name=benchmark_run_name,
            )

            # Save metrics
            metrics_path = tables_dir / f"{benchmark_run_name}_metrics.json"
            _, arrays_path = save_metrics_with_artifacts(metrics, metrics_path)
            logger.info(f"Saved metrics: {metrics_path}")
            if arrays_path is not None:
                logger.info(f"Saved array artifacts: {arrays_path}")

            # Print metrics
            print(f"Metrics for {model_name}:")
            for k, v in metrics.items():
                if (
                    k in ("model", "dataset")
                    or k.startswith("_")
                    or k.startswith("final_manifest_")
                ):
                    continue
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

            results_list.append(metrics)

        except Exception as e:
            logger.error(f"Failed to fit {model_name}: {e}", exc_info=True)
            print(f"ERROR: {model_name} failed: {e}")

    # Print summary table
    if results_list:
        print("\n=== Summary ===")
        print(f"{'Model':<15} {'Freq RMSE':<12} {'Freq MAPE':<12} {'Bias %':<10} {'Spend MAE raw':<15} {'Spend R² log':<14}")
        print("-" * 82)
        for res in results_list:
            model = res["model"]
            freq_rmse = f"{res.get('freq_rmse', np.nan):.4f}"
            freq_mape = f"{res.get('freq_mape', np.nan):.4f}"
            bias_pct = f"{res.get('bias_pct', np.nan):.2f}"
            spend_mae_raw = f"{res.get('spend_mae_raw', np.nan):.4f}"
            spend_r2_log = f"{res.get('spend_r2_log', np.nan):.4f}"
            print(f"{model:<15} {freq_rmse:<12} {freq_mape:<12} {bias_pct:<10} {spend_mae_raw:<15} {spend_r2_log:<14}")


if __name__ == "__main__":
    main()
