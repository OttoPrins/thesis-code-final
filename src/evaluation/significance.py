"""
Paired bootstrap significance tests for between-model comparisons.

Usage (CLI):
    python -m src.evaluation.significance \
        --metrics_dir results/tables \
        --pairs lstm_joint_cdnow:lstm_base_cdnow \
                transformer_joint_cdnow:lstm_joint_cdnow \
                lstm_joint_cdnow:pareto_nbd_cdnow \
                lstm_joint_cdnow:bgnbd_gg_cdnow \
                transformer_joint_tafeng:lstm_joint_tafeng \
        --metrics freq_rmse spend_mae_raw clv_mae

Outputs results/tables/significance_tests.csv with columns:
    model_a, model_b, metric, delta, ci_low, ci_high, p_value, n_customers, n_resamples

The test resamples *customers* (not time steps) with replacement, computes the
difference in mean per-customer error for each bootstrap draw, and derives the
95% CI and two-sided p-value from the empirical distribution.

The per-customer error arrays are stored in each model's sidecar
    `*_arrays.npz` file under keys:
    per_customer_freq_se, per_customer_freq_ae, per_customer_spend_ae,
    per_customer_clv_ae.
Legacy metrics JSONs with embedded arrays are still supported as a fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.evaluation.metrics import metrics_arrays_path
from src.utils.final_manifest import load_final_manifest, result_matches_manifest


# Map CLI metric names to JSON keys
_METRIC_TO_KEY = {
    "freq_rmse": "_per_customer_freq_se",   # bootstrap on SE → diff in RMSE
    "freq_mae": "_per_customer_freq_ae",
    "spend_mae": "_per_customer_spend_ae",      # backwards-compatible alias
    "spend_mae_raw": "_per_customer_spend_ae",
    "clv_mae": "_per_customer_clv_ae",
}

# Whether the array stores squared errors (need sqrt of mean) or absolute errors
_METRIC_IS_SE = {
    "freq_rmse": True,
    "freq_mae": False,
    "spend_mae": False,
    "spend_mae_raw": False,
    "clv_mae": False,
}


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    is_squared: bool = False,
) -> dict:
    """
    Paired bootstrap test: does model A make smaller errors than model B?

    Resamples (errors_a[i], errors_b[i]) pairs with replacement to estimate
    the sampling distribution of mean(errors_a) - mean(errors_b).

    Args:
        errors_a:   (N,) per-customer error magnitudes for model A
        errors_b:   (N,) per-customer error magnitudes for model B
        n_resamples: number of bootstrap draws
        seed:       RNG seed for reproducibility
        is_squared: if True, aggregate by sqrt(mean(x)) instead of mean(x)
                    (used when the array stores squared errors for RMSE)

    Returns:
        dict with keys: delta, ci_low, ci_high, p_value, n_customers, n_resamples
        where delta = agg(errors_a) - agg(errors_b)
        Negative delta → model A is better (lower error).
    """
    errors_a = np.asarray(errors_a, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)
    assert len(errors_a) == len(errors_b), "Arrays must have equal length"
    N = len(errors_a)

    agg = (lambda x: float(np.sqrt(x.mean()))) if is_squared else (lambda x: float(x.mean()))
    observed_delta = agg(errors_a) - agg(errors_b)

    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        bootstrap_deltas[i] = agg(errors_a[idx]) - agg(errors_b[idx])

    ci_low = float(np.percentile(bootstrap_deltas, 2.5))
    ci_high = float(np.percentile(bootstrap_deltas, 97.5))
    # Two-sided p-value: fraction of bootstrap draws on the wrong side of zero
    p_value = float(np.mean(bootstrap_deltas >= 0) if observed_delta < 0
                    else np.mean(bootstrap_deltas <= 0))
    p_value = min(1.0, 2.0 * p_value)  # two-sided

    return {
        "delta": observed_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_customers": N,
        "n_resamples": n_resamples,
    }


def _latest_metrics_file(metrics_dir: Path, run_prefix: str) -> Optional[Path]:
    return _select_metrics_file(metrics_dir, run_prefix)


def _select_metrics_file(
    metrics_dir: Path,
    run_prefix: str,
    seed_filter: Optional[int] = None,
    mode_filter: Optional[str] = None,
) -> Optional[Path]:
    """
    Find the most recent metrics JSON for a run prefix.

    Looks for files matching <run_prefix>*_metrics.json, then picks the one
    whose run_name ends with the highest version/seed suffix. This handles both
    single-seed files (lstm_base_cdnow_v3_metrics.json) and seed-averaged
    comparisons. Returns None if nothing found.
    """
    candidates = sorted(metrics_dir.glob(f"{run_prefix}*_metrics.json"))
    if not candidates:
        return None
    manifest = load_final_manifest()
    if manifest is not None:
        final_candidates = []
        for path in candidates:
            stem = path.stem.removesuffix("_metrics")
            try:
                with open(path) as f:
                    metrics = json.load(f)
            except Exception:
                continue
            ok, _ = result_matches_manifest(
                stem,
                metrics,
                manifest,
                include_expected=False,
            )
            if ok:
                final_candidates.append(path)
        candidates = final_candidates
        if not candidates:
            return None

    # For final thesis tables we compare matched seeded DL runs. Keep benchmark
    # rows usable: if no candidate has the requested seed/mode, fall back to the
    # manifest-valid benchmark candidate instead of dropping it.
    if seed_filter is not None:
        seeded = [
            path for path in candidates
            if f"_seed{seed_filter}" in path.stem.removesuffix("_metrics")
        ]
        if seeded:
            candidates = seeded
    if mode_filter is not None:
        mode_matched = [
            path for path in candidates
            if path.stem.removesuffix("_metrics").endswith(f"_{mode_filter}")
        ]
        if mode_matched:
            candidates = mode_matched

    # Prefer seed-aggregated or the highest numbered version
    return candidates[-1]


def _load_per_customer_array(metrics_path: Path, key: str) -> Optional[np.ndarray]:
    artifact_key = key.lstrip("_")
    arrays_path = metrics_arrays_path(metrics_path)
    if arrays_path.exists():
        with np.load(arrays_path) as data:
            if artifact_key in data:
                return np.asarray(data[artifact_key], dtype=np.float64)

    with open(metrics_path) as f:
        data = json.load(f)
    arr = data.get(key)
    if arr is None:
        arr = data.get(artifact_key)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64)


def run_significance_tests(
    metrics_dir: Path,
    pairs: list[tuple[str, str]],
    metrics: list[str],
    n_resamples: int = 10_000,
    seed: int = 42,
    seed_filter: Optional[int] = None,
    mode_filter: Optional[str] = None,
) -> list[dict]:
    """
    Run paired bootstrap for all (model_a, model_b, metric) combinations.

    Args:
        metrics_dir: directory containing *_metrics.json files
        pairs:       list of (run_prefix_a, run_prefix_b) tuples
        metrics:     list of metric names (from _METRIC_TO_KEY keys)
        n_resamples: bootstrap draws
        seed:        RNG seed

    Returns:
        list of result dicts, one per (pair, metric) combination
    """
    results = []
    for (prefix_a, prefix_b) in pairs:
        path_a = _select_metrics_file(metrics_dir, prefix_a, seed_filter, mode_filter)
        path_b = _select_metrics_file(metrics_dir, prefix_b, seed_filter, mode_filter)

        if path_a is None:
            print(f"[significance] WARNING: no metrics file for {prefix_a!r} — skipping")
            continue
        if path_b is None:
            print(f"[significance] WARNING: no metrics file for {prefix_b!r} — skipping")
            continue

        for metric in metrics:
            key = _METRIC_TO_KEY.get(metric)
            if key is None:
                print(f"[significance] Unknown metric {metric!r} — skipping")
                continue

            arr_a = _load_per_customer_array(path_a, key)
            arr_b = _load_per_customer_array(path_b, key)

            if arr_a is None:
                print(f"[significance] {prefix_a}: missing {key!r} — skipping {metric}")
                continue
            if arr_b is None:
                print(f"[significance] {prefix_b}: missing {key!r} — skipping {metric}")
                continue
            if len(arr_a) != len(arr_b):
                print(
                    f"[significance] Customer count mismatch: "
                    f"{prefix_a}({len(arr_a)}) vs {prefix_b}({len(arr_b)}) — skipping {metric}"
                )
                continue

            is_sq = _METRIC_IS_SE.get(metric, False)
            res = paired_bootstrap(arr_a, arr_b, n_resamples=n_resamples, seed=seed, is_squared=is_sq)
            sig = "YES" if res["ci_high"] < 0 or res["ci_low"] > 0 else "no"
            print(
                f"  {prefix_a} vs {prefix_b} | {metric}: "
                f"Δ={res['delta']:+.4f}  95% CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
                f"  p={res['p_value']:.4f}  sig={sig}"
            )
            results.append({
                "model_a": prefix_a,
                "model_b": prefix_b,
                "file_a": path_a.name,
                "file_b": path_b.name,
                "metric": metric,
                **res,
            })
    return results


def _save_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        print("[significance] No results to save.")
        return
    fieldnames = ["model_a", "model_b", "file_a", "file_b", "metric", "delta", "ci_low", "ci_high",
                  "p_value", "n_customers", "n_resamples"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[significance] Results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap significance tests.")
    parser.add_argument(
        "--metrics_dir", type=Path, default=Path("results/tables"),
        help="Directory containing *_metrics.json files"
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="Pairs as 'prefix_a:prefix_b'. Example: lstm_joint_cdnow:lstm_base_cdnow"
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["freq_rmse", "spend_mae", "clv_mae"],
        help="Metrics to test. Choices: freq_rmse, spend_mae, clv_mae"
    )
    parser.add_argument("--n_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed_filter", type=int, default=None,
        help=(
            "Prefer metrics files containing this training seed. "
            "Benchmarks without a seed suffix are kept as fallbacks."
        ),
    )
    parser.add_argument(
        "--mode_filter", choices=["sample", "expected"], default=None,
        help=(
            "Prefer metrics files with this inference mode suffix. "
            "Benchmarks without a mode suffix are kept as fallbacks."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path. Defaults to <metrics_dir>/significance_tests.csv"
    )
    args = parser.parse_args()

    pairs = []
    for p in args.pairs:
        parts = p.split(":")
        if len(parts) != 2:
            parser.error(f"Pair must be 'prefix_a:prefix_b', got: {p!r}")
        pairs.append((parts[0], parts[1]))

    out_path = args.out or (args.metrics_dir / "significance_tests.csv")

    print(f"Running paired bootstrap ({args.n_resamples} resamples) ...")
    results = run_significance_tests(
        metrics_dir=args.metrics_dir,
        pairs=pairs,
        metrics=args.metrics,
        n_resamples=args.n_resamples,
        seed=args.seed,
        seed_filter=args.seed_filter,
        mode_filter=args.mode_filter,
    )
    _save_csv(results, out_path)


if __name__ == "__main__":
    main()
