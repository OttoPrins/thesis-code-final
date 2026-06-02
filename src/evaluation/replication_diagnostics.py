"""Diagnostics for the Valendin CDNOW replication audit.

This module is deliberately diagnostic-only. It reads saved metrics/array
sidecars and writes tables/plots that explain seed instability and aggregate
underprediction; it does not alter metrics and does not promote any repair.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import metrics_arrays_path
from src.evaluation.replication_report import (
    DEFAULT_GOVERNANCE_PATH,
    load_replication_decision,
    metrics_path,
)


METRIC_COLUMNS = ("freq_rmse", "bias_pct", "freq_mape")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _history_path(results_dir: str | Path, run_name: str) -> Path:
    return Path(results_dir) / "tables" / f"{run_name}_history.json"


def _safe_pct(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / abs(denominator) * 100.0)


def _seed_from_run_name(run_name: str) -> int | None:
    match = re.search(r"_seed(\d+)_", run_name)
    return int(match.group(1)) if match else None


def _aggregate_week_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "per_week_true_freq" not in arrays or "per_week_pred_freq" not in arrays:
        raise KeyError("arrays must contain per_week_true_freq and per_week_pred_freq")
    true = np.asarray(arrays["per_week_true_freq"], dtype=np.float64)
    pred = np.asarray(arrays["per_week_pred_freq"], dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"per-week true/pred shape mismatch: {true.shape} vs {pred.shape}")
    if true.ndim == 2:
        return true.sum(axis=0), pred.sum(axis=0)
    if true.ndim == 1:
        return true, pred
    raise ValueError(f"per-week arrays must be 1D or 2D, got ndim={true.ndim}")


def _window_summary(true: np.ndarray, pred: np.ndarray, start: int, stop: int) -> dict[str, float]:
    t = true[start:stop]
    p = pred[start:stop]
    residual = p - t
    total_true = float(t.sum())
    return {
        "mape": _safe_pct(float(np.abs(residual).sum()), total_true),
        "bias_pct": _safe_pct(float(residual.sum()), total_true),
        "true_total": total_true,
        "pred_total": float(p.sum()),
    }


def _default_run_specs(gov: dict[str, Any], include_rescore_runs: bool = True) -> list[dict[str, str]]:
    evidence = gov["final_evidence_set"]
    seeds = [int(seed) for seed in evidence["seeds"]]
    mode = str(evidence["inference_mode"])
    strict = gov["official_baseline"]["run_name"]
    paper = gov["paper_consistent_variant"]["run_name"]

    specs: list[dict[str, str]] = []
    for family, base in [
        ("strict_demo_pure", strict),
        ("non_promoted_paper_finetune", paper),
    ]:
        for seed in seeds:
            specs.append({
                "family": family,
                "run_name": f"{base}_seed{seed}_{mode}",
                "label": f"{family} seed {seed}",
            })
        specs.append({
            "family": family,
            "run_name": f"{base}_ensemble_{mode}",
            "label": f"{family} seed ensemble",
        })

    for rejected in gov.get("rejected_ablations", []):
        specs.append({
            "family": "rejected_ablation",
            "run_name": f"{rejected['run_name']}_seed42_sample",
            "label": f"rejected {rejected['run_name']} seed 42",
        })

    for sensitivity in gov.get("diagnostic_sensitivities", []):
        base = sensitivity["run_name"]
        for seed in seeds:
            specs.append({
                "family": "diagnostic_sensitivity",
                "run_name": f"{base}_seed{seed}_{mode}",
                "label": f"{base} seed {seed}",
            })

    for ablation in gov.get("forecast_quality_ablations", []):
        base = ablation["run_name"]
        role = ablation.get("role", "forecast_quality_sensitivity")
        for seed in seeds:
            specs.append({
                "family": role,
                "run_name": f"{base}_seed{seed}_{mode}",
                "label": f"{base} seed {seed}",
            })
        specs.append({
            "family": role,
            "run_name": f"{base}_ensemble_{mode}",
            "label": f"{base} seed ensemble",
        })

    if include_rescore_runs:
        for seed in seeds:
            specs.append({
                "family": "diagnostic_rescore_expected",
                "run_name": f"{strict}_seed{seed}_expected_rescore",
                "label": f"strict expected rescore seed {seed}",
            })
            specs.append({
                "family": "diagnostic_rescore_sample100",
                "run_name": f"{strict}_seed{seed}_sample100_rescore",
                "label": f"strict sample100 rescore seed {seed}",
            })
        for ablation in gov.get("forecast_quality_ablations", []):
            if "expected_rescore" not in set(ablation.get("modes", [])):
                continue
            base = ablation["run_name"]
            for seed in seeds:
                specs.append({
                    "family": "diagnostic_rescore_expected",
                    "run_name": f"{base}_seed{seed}_expected_rescore",
                    "label": f"{base} expected rescore seed {seed}",
                })
                specs.append({
                    "family": "diagnostic_rescore_sample100",
                    "run_name": f"{base}_seed{seed}_sample100_rescore",
                    "label": f"{base} sample100 rescore seed {seed}",
                })
    return specs


def build_seed_diagnostics(
    *,
    results_dir: str | Path = "results",
    governance_path: str | Path = DEFAULT_GOVERNANCE_PATH,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Build seed-level validation-loss vs holdout-metric diagnostics."""
    gov = load_replication_decision(governance_path)
    rows: list[dict[str, Any]] = []
    for spec in _default_run_specs(gov, include_rescore_runs=False):
        run_name = spec["run_name"]
        metrics = _load_json(metrics_path(results_dir, run_name))
        if metrics is None:
            continue
        history = _load_json(_history_path(results_dir, run_name)) or {}
        val_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)
        train_loss = np.asarray(history.get("train_loss", []), dtype=np.float64)
        row = {
            "family": spec["family"],
            "label": spec["label"],
            "run_name": run_name,
            "seed": _seed_from_run_name(run_name),
            "status": "complete" if metrics.get("run_valid", True) else "invalid",
            "epochs_completed": metrics.get("epochs_completed", len(val_loss)),
            "best_val_loss": float(np.nanmin(val_loss)) if val_loss.size else np.nan,
            "final_val_loss": float(val_loss[-1]) if val_loss.size else np.nan,
            "best_train_loss": float(np.nanmin(train_loss)) if train_loss.size else np.nan,
            "repair_attempted": bool(metrics.get("repair_attempted", False)),
            "diagnostic_only": bool(metrics.get("diagnostic_only", False)),
        }
        for col in METRIC_COLUMNS:
            row[col] = metrics.get(col, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    if write_outputs:
        out = Path(results_dir) / "tables" / "valendin_replication_seed_diagnostics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def build_per_week_diagnostics(
    *,
    results_dir: str | Path = "results",
    governance_path: str | Path = DEFAULT_GOVERNANCE_PATH,
    run_names: list[str] | None = None,
    make_plots: bool = True,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-week residual and concentration diagnostics from sidecar arrays."""
    gov = load_replication_decision(governance_path)
    specs = _default_run_specs(gov)
    if run_names is not None:
        wanted = set(run_names)
        specs = [spec for spec in specs if spec["run_name"] in wanted]

    week_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_name = spec["run_name"]
        metrics_file = metrics_path(results_dir, run_name)
        arrays = _load_npz(metrics_arrays_path(metrics_file))
        if arrays is None:
            continue
        try:
            true, pred = _aggregate_week_arrays(arrays)
        except KeyError:
            continue

        residual = pred - true
        cum_true = np.cumsum(true)
        cum_pred = np.cumsum(pred)
        for i, (a, p, r, ca, cp) in enumerate(zip(true, pred, residual, cum_true, cum_pred), start=1):
            week_rows.append({
                "family": spec["family"],
                "label": spec["label"],
                "run_name": run_name,
                "holdout_week": i,
                "actual_freq": float(a),
                "pred_freq": float(p),
                "residual": float(r),
                "abs_residual": float(abs(r)),
                "cumulative_actual_freq": float(ca),
                "cumulative_pred_freq": float(cp),
                "cumulative_bias_pct": _safe_pct(float(cp - ca), float(ca)),
            })

        first_stop = min(13, len(true))
        first = _window_summary(true, pred, 0, first_stop)
        later = _window_summary(true, pred, first_stop, len(true))
        full = _window_summary(true, pred, 0, len(true))
        first_abs = float(np.abs(residual[:first_stop]).sum())
        full_abs = float(np.abs(residual).sum())
        summary_rows.append({
            "family": spec["family"],
            "label": spec["label"],
            "run_name": run_name,
            "n_weeks": int(len(true)),
            "first_13_mape": first["mape"],
            "first_13_bias_pct": first["bias_pct"],
            "later_mape": later["mape"],
            "later_bias_pct": later["bias_pct"],
            "horizon_mape": full["mape"],
            "horizon_bias_pct": full["bias_pct"],
            "first_13_abs_error_share": (
                float(first_abs / full_abs) if full_abs > 0 else np.nan
            ),
            "actual_total": full["true_total"],
            "pred_total": full["pred_total"],
        })

    week_df = pd.DataFrame(week_rows)
    summary_df = pd.DataFrame(summary_rows)
    if write_outputs:
        tables_dir = Path(results_dir) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        week_df.to_csv(tables_dir / "valendin_replication_per_week_diagnostics.csv", index=False)
        summary_df.to_csv(tables_dir / "valendin_replication_per_week_summary.csv", index=False)
    if make_plots and not week_df.empty:
        _write_per_week_plots(week_df, Path(results_dir) / "plots" / "valendin_replication")
    return week_df, summary_df


def _write_per_week_plots(week_df: pd.DataFrame, plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    for run_name, group in week_df.groupby("run_name", sort=False):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
        x = group["holdout_week"].to_numpy()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, group["actual_freq"], label="actual", linewidth=2)
        ax.plot(x, group["pred_freq"], label="predicted", linewidth=2)
        ax.set_xlabel("Holdout week")
        ax.set_ylabel("Cohort transactions")
        ax.set_title(run_name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{safe}_actual_vs_pred.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(x, group["residual"], width=0.85)
        ax.set_xlabel("Holdout week")
        ax.set_ylabel("Predicted - actual")
        ax.set_title(f"{run_name} residuals")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{safe}_residuals.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, group["cumulative_actual_freq"], label="actual cumulative", linewidth=2)
        ax.plot(x, group["cumulative_pred_freq"], label="predicted cumulative", linewidth=2)
        ax.set_xlabel("Holdout week")
        ax.set_ylabel("Cumulative transactions")
        ax.set_title(f"{run_name} cumulative totals")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{safe}_cumulative.png", dpi=180)
        plt.close(fig)


def build_replication_diagnostics(
    *,
    results_dir: str | Path = "results",
    governance_path: str | Path = DEFAULT_GOVERNANCE_PATH,
    make_plots: bool = True,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    seed_df = build_seed_diagnostics(
        results_dir=results_dir,
        governance_path=governance_path,
        write_outputs=write_outputs,
    )
    week_df, summary_df = build_per_week_diagnostics(
        results_dir=results_dir,
        governance_path=governance_path,
        make_plots=make_plots,
        write_outputs=write_outputs,
    )
    return {"seed": seed_df, "per_week": week_df, "summary": summary_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Valendin replication diagnostics.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--governance", default=str(DEFAULT_GOVERNANCE_PATH))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    outputs = build_replication_diagnostics(
        results_dir=args.results_dir,
        governance_path=args.governance,
        make_plots=not args.no_plots,
        write_outputs=not args.no_write,
    )
    print("Valendin replication diagnostics")
    for name, df in outputs.items():
        print(f"  {name}: {len(df)} row(s)")


if __name__ == "__main__":
    main()
