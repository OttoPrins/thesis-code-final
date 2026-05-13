"""
Pipeline validation / smoke-test script.

Runs each dataset pipeline end-to-end and prints tensor shapes + basic
sanity checks. Run this before training to confirm all raw files are in
place and the pipelines produce the expected output format.

Usage:
    python validate_pipelines.py              # all 4 datasets
    python validate_pipelines.py --dataset cdnow
"""

import argparse
import os
import sys
import traceback

import numpy as np

from src.data.datasets import (
    CDNOWPipeline,
    DunnhumbyPipeline,
    TaFengPipeline,
    UCIRetailPipeline,
)
from src.utils.config import load_config

DATASET_CONFIGS = {
    "cdnow":     "experiments/configs/lstm_base_cdnow.yaml",
    "uci":       "experiments/configs/lstm_base_uci.yaml",
    "tafeng":    "experiments/configs/lstm_base_tafeng.yaml",
    "dunnhumby": "experiments/configs/lstm_base_dunnhumby.yaml",
}

PIPELINES = {
    "cdnow":     CDNOWPipeline,
    "uci":       UCIRetailPipeline,
    "tafeng":    TaFengPipeline,
    "dunnhumby": DunnhumbyPipeline,
}


def validate_dataset(name: str) -> bool:
    config_path = DATASET_CONFIGS[name]
    print(f"\n{'='*60}")
    print(f"  {name.upper()}  ({config_path})")
    print(f"{'='*60}")

    try:
        config = load_config(config_path)
        if os.environ.get("KAGGLE_ENV") == "1":
            from src.utils.config import apply_kaggle_overrides
            apply_kaggle_overrides(config)
        pipeline = PIPELINES[name]()
        train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)
    except Exception:
        print(f"  [FAIL] Pipeline raised an exception:")
        traceback.print_exc()
        return False

    calib_weeks = config["dataset"]["calibration_weeks"]
    holdout_weeks = config["dataset"]["holdout_weeks"]

    # --- Customer counts ---
    print(f"  Customers — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"inference: {len(inference_ds)}")

    # --- Tensor shapes from first batch ---
    sample = train_ds[0]
    week_shape = sample["week"].shape
    freq_shape = sample["y_freq"].shape
    spend_shape = sample["y_spend"].shape
    print(f"  Sequence length (T-1): {week_shape[0]}  "
          f"(expected {calib_weeks - 1})")
    print(f"  week: {week_shape}, y_freq: {freq_shape}, "
          f"y_spend: {spend_shape}")

    # --- Spend range ---
    all_spend = np.array([train_ds[i]["spend"].numpy()
                          for i in range(min(100, len(train_ds)))])
    print(f"  spend_input range (first 100 customers): "
          f"[{all_spend.min():.3f}, {all_spend.max():.3f}]")

    # --- Holdout ground truth ---
    N = len(holdout_gt["customer_ids"])
    H = holdout_gt["freq"].shape[1]
    print(f"  Holdout GT — customers: {N}, weeks: {H} (expected {holdout_weeks})")
    print(f"  Holdout total_freq — mean: {holdout_gt['total_freq'].mean():.2f}, "
          f"max: {holdout_gt['total_freq'].max()}")

    # --- Leakage check: holdout weeks must all be >= calib_weeks ---
    ok = True
    if H != holdout_weeks:
        print(f"  [WARN] Holdout weeks mismatch: got {H}, expected {holdout_weeks}")
        ok = False

    if week_shape[0] != calib_weeks - 1:
        print(f"  [WARN] Sequence length mismatch: got {week_shape[0]}, "
              f"expected {calib_weeks - 1} (calib_weeks - 1)")
        ok = False

    if len(inference_ds) != N:
        print(f"  [WARN] inference_ds size ({len(inference_ds)}) != "
              f"holdout_gt customers ({N})")
        ok = False

    if ok:
        print(f"  [OK] All checks passed.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate data pipelines.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="Validate a single dataset (default: all).",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    results = {}
    for name in datasets:
        results[name] = validate_dataset(name)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {name}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
