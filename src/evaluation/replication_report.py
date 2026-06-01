"""Final Valendin CDNOW replication interpretation table.

This module turns the frozen replication decision file into a compact table for
the thesis: paper target, local benchmark context, strict Base LSTM seeds,
paper-finetune seeds, and seed ensembles. It can also build seed ensembles from
saved per-week prediction arrays, which is useful after Kaggle run artifacts are
copied back but checkpoint inference is not being rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_all_metrics, metrics_arrays_path, save_metrics_with_artifacts


DEFAULT_GOVERNANCE_PATH = Path(
    "experiments/configs/governance/valendin_replication_decision.yaml"
)
METRIC_COLUMNS = ("freq_rmse", "bias_pct", "freq_mape")


def load_replication_decision(path: str | Path = DEFAULT_GOVERNANCE_PATH) -> dict[str, Any]:
    """Load the frozen replication decision/governance file."""
    with open(path) as f:
        return yaml.safe_load(f)


def metrics_path(results_dir: str | Path, run_name: str) -> Path:
    return Path(results_dir) / "tables" / f"{run_name}_metrics.json"


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays_path = metrics_arrays_path(path)
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing arrays sidecar for {path.name}: {arrays_path}")
    with np.load(arrays_path) as data:
        return {name: data[name] for name in data.files}


def _base_row(label: str, *, role: str, run_name: str = "", note: str = "") -> dict[str, Any]:
    return {
        "label": label,
        "role": role,
        "status": "missing",
        "run_name": run_name,
        "n_runs": 0,
        "freq_rmse": np.nan,
        "freq_rmse_std": np.nan,
        "bias_pct": np.nan,
        "bias_pct_std": np.nan,
        "freq_mape": np.nan,
        "freq_mape_std": np.nan,
        "cohort_size": np.nan,
        "protocol_name": "",
        "note": note,
    }


def _row_from_metrics(
    label: str,
    metrics: dict[str, Any] | None,
    *,
    role: str,
    run_name: str,
    note: str = "",
) -> dict[str, Any]:
    row = _base_row(label, role=role, run_name=run_name, note=note)
    if metrics is None:
        return row
    row["status"] = "complete" if metrics.get("run_valid", True) else "invalid"
    row["n_runs"] = 1
    for col in METRIC_COLUMNS:
        row[col] = metrics.get(col, np.nan)
    row["cohort_size"] = metrics.get("cohort_size", np.nan)
    row["protocol_name"] = metrics.get("protocol_name", "")
    return row


def _seed_run_names(base_run_name: str, seeds: list[int], mode: str) -> list[str]:
    return [f"{base_run_name}_seed{seed}_{mode}" for seed in seeds]


def _row_from_seed_mean(
    label: str,
    metrics_by_run: list[tuple[str, dict[str, Any] | None]],
    *,
    role: str,
    note: str = "",
) -> dict[str, Any]:
    available = [(run, metrics) for run, metrics in metrics_by_run if metrics is not None]
    row = _base_row(label, role=role, note=note)
    row["run_name"] = ",".join(run for run, _ in metrics_by_run)
    row["n_runs"] = len(available)
    if not available:
        return row

    row["status"] = "complete" if len(available) == len(metrics_by_run) else "partial"
    for col in METRIC_COLUMNS:
        vals = np.asarray(
            [metrics.get(col, np.nan) for _, metrics in available],
            dtype=np.float64,
        )
        row[col] = float(np.nanmean(vals))
        row[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    first = available[0][1]
    row["cohort_size"] = first.get("cohort_size", np.nan)
    row["protocol_name"] = first.get("protocol_name", "")
    return row


def _assert_same_true_arrays(reference: np.ndarray, candidate: np.ndarray, run_name: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate):
        raise ValueError(f"True per-week frequency array mismatch for {run_name}")


def build_array_ensemble(
    *,
    results_dir: str | Path,
    base_run_name: str,
    seeds: list[int],
    mode: str,
    output_run_name: str,
    overwrite: bool = False,
) -> Path:
    """Average saved per-week seed predictions and score the ensemble."""
    out_path = metrics_path(results_dir, output_run_name)
    if out_path.exists() and not overwrite:
        return out_path

    source_runs = _seed_run_names(base_run_name, seeds, mode)
    source_metrics: list[dict[str, Any]] = []
    pred_weeks: list[np.ndarray] = []
    true_week: np.ndarray | None = None
    for run_name in source_runs:
        path = metrics_path(results_dir, run_name)
        metrics = _load_metrics(path)
        if metrics is None:
            raise FileNotFoundError(f"Missing seed metrics for array ensemble: {path}")
        arrays = _load_arrays(path)
        if "per_week_true_freq" not in arrays or "per_week_pred_freq" not in arrays:
            raise KeyError(f"{path.name} lacks per_week_true_freq/per_week_pred_freq arrays")
        if true_week is None:
            true_week = np.asarray(arrays["per_week_true_freq"], dtype=np.float32)
        else:
            _assert_same_true_arrays(
                true_week,
                np.asarray(arrays["per_week_true_freq"], dtype=np.float32),
                run_name,
            )
        pred_weeks.append(np.asarray(arrays["per_week_pred_freq"], dtype=np.float32))
        source_metrics.append(metrics)

    assert true_week is not None
    pred_week = np.mean(np.stack(pred_weeks, axis=0), axis=0).astype(np.float32)
    true_total = true_week.sum(axis=1).astype(np.float32)
    pred_total = pred_week.sum(axis=1).astype(np.float32)
    freq_mase_scales = [
        float(m["freq_mase_scale"])
        for m in source_metrics
        if m.get("freq_mase_scale") is not None and np.isfinite(float(m["freq_mase_scale"]))
    ]
    freq_mase_scale = float(np.mean(freq_mase_scales)) if freq_mase_scales else None

    metrics = compute_all_metrics(
        y_freq_true=true_total,
        y_freq_pred=pred_total,
        y_freq_true_per_week=true_week,
        y_freq_pred_per_week=pred_week,
        customer_ids=np.arange(true_week.shape[0]),
        freq_mase_scale=freq_mase_scale,
    )
    first = source_metrics[0]
    metrics.update({
        "model": first.get("model", "lstm_base") + "_ensemble",
        "dataset": first.get("dataset", "cdnow"),
        "protocol_name": first.get("protocol_name", ""),
        "cohort_size": first.get("cohort_size", true_week.shape[0]),
        "calibration_weeks": first.get("calibration_weeks"),
        "holdout_weeks": first.get("holdout_weeks", true_week.shape[1]),
        "inference_mode": mode,
        "n_scenarios": first.get("n_scenarios"),
        "ensemble_total_scenarios": int(
            sum(int(m.get("n_scenarios", 0) or 0) for m in source_metrics)
        ),
        "ensemble_seeds": seeds,
        "ensemble_source": "saved_prediction_arrays",
        "ensemble_source_runs": ",".join(source_runs),
        "run_valid": all(bool(m.get("run_valid", True)) for m in source_metrics),
        "run_invalid_reason": "",
        "run_warning": "seed ensemble built from saved prediction arrays",
    })
    save_metrics_with_artifacts(metrics, out_path)
    return out_path


def build_replication_interpretation_table(
    *,
    results_dir: str | Path = "results",
    governance_path: str | Path = DEFAULT_GOVERNANCE_PATH,
    build_array_ensembles: bool = False,
    overwrite_ensembles: bool = False,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Build the frozen Valendin replication interpretation table."""
    gov = load_replication_decision(governance_path)
    report_cfg = gov["report"]
    evidence = gov["final_evidence_set"]
    seeds = [int(seed) for seed in evidence["seeds"]]
    mode = str(evidence["inference_mode"])

    rows: list[dict[str, Any]] = []
    ref = report_cfg["reference"]
    ref_row = _base_row(
        ref["label"],
        role="paper_reference",
        note=ref.get("note", ""),
    )
    ref_row.update({
        "status": "reference",
        "n_runs": 0,
        "freq_rmse": ref.get("freq_rmse", np.nan),
        "bias_pct": ref.get("bias_pct", np.nan),
        "freq_mape": ref.get("freq_mape", np.nan),
        "protocol_name": "paper_cdnow",
    })
    rows.append(ref_row)

    for spec in report_cfg["rows"]:
        kind = spec["kind"]
        label = spec["label"]
        note = spec.get("note", "")
        if kind == "metrics":
            run_name = spec["run_name"]
            rows.append(_row_from_metrics(
                label,
                _load_metrics(metrics_path(results_dir, run_name)),
                role="single_run_or_benchmark",
                run_name=run_name,
                note=note,
            ))
        elif kind == "seed_mean":
            base_run = spec["base_run_name"]
            run_names = _seed_run_names(base_run, seeds, mode)
            metrics_by_run = [
                (run_name, _load_metrics(metrics_path(results_dir, run_name)))
                for run_name in run_names
            ]
            rows.append(_row_from_seed_mean(
                label,
                metrics_by_run,
                role="seed_mean",
                note=note,
            ))
        elif kind == "ensemble":
            run_name = spec["run_name"]
            if build_array_ensembles and not metrics_path(results_dir, run_name).exists():
                build_array_ensemble(
                    results_dir=results_dir,
                    base_run_name=spec["base_run_name"],
                    seeds=seeds,
                    mode=mode,
                    output_run_name=run_name,
                    overwrite=overwrite_ensembles,
                )
            rows.append(_row_from_metrics(
                label,
                _load_metrics(metrics_path(results_dir, run_name)),
                role="seed_ensemble",
                run_name=run_name,
                note=note,
            ))
        else:
            raise ValueError(f"Unknown report row kind: {kind!r}")

    for rejected in gov.get("rejected_ablations", []):
        run_name = f"{rejected['run_name']}_seed42_sample"
        rows.append(_row_from_metrics(
            f"Rejected ablation: {rejected['run_name']}",
            _load_metrics(metrics_path(results_dir, run_name)),
            role="rejected_ablation",
            run_name=run_name,
            note=rejected.get("reason", ""),
        ))
        rows[-1]["status"] = (
            "rejected" if rows[-1]["status"] != "missing" else "missing_rejected"
        )

    df = pd.DataFrame(rows)
    if write_outputs:
        csv_path = Path(report_cfg["output_csv"])
        md_path = Path(report_cfg["output_markdown"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        md_path.write_text(_to_markdown(df), encoding="utf-8")
    return df


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if np.isnan(value) else f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    text = str(value)
    return text.replace("\n", " ")


def _to_markdown(df: pd.DataFrame) -> str:
    cols = [
        "label",
        "status",
        "n_runs",
        "freq_rmse",
        "freq_rmse_std",
        "bias_pct",
        "bias_pct_std",
        "freq_mape",
        "freq_mape_std",
        "note",
    ]
    lines = [
        "# Valendin CDNOW Replication Interpretation",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Valendin replication interpretation table.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--governance", default=str(DEFAULT_GOVERNANCE_PATH))
    parser.add_argument("--build-array-ensembles", action="store_true")
    parser.add_argument("--overwrite-ensembles", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    df = build_replication_interpretation_table(
        results_dir=args.results_dir,
        governance_path=args.governance,
        build_array_ensembles=args.build_array_ensembles,
        overwrite_ensembles=args.overwrite_ensembles,
        write_outputs=not args.no_write,
    )
    print(df[["label", "status", "n_runs", "freq_rmse", "bias_pct", "freq_mape"]].to_string(index=False))


if __name__ == "__main__":
    main()
