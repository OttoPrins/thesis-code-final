"""
Evaluation metrics for frequency and spend predictions.

Frequency metrics (individual-level):
    freq_rmse: RMSE of predicted vs actual purchase count
    freq_mae:  MAE of purchase counts

Cohort-level metrics:
    freq_mape:  MAPE of aggregated cohort predictions vs actuals
    bias_pct:   Percentage bias = (sum_pred - sum_actual) / sum_actual * 100

Spend aggregation (critical):
    The model predicts per-week spend in raw log1p space. To obtain
    holdout-period total spend in raw currency we must inverse-transform each
    week individually (expm1) and then sum the raw-currency values. Summing
    log-space values directly is mathematically incoherent (sum of logs ≠ log
    of sum) and produces meaningless raw-currency results after inversion.

Spend metrics (raw currency — primary reporting):
    spend_mae_raw:  MAE on per-customer holdout-period total (sum of per-week raw spend)
    spend_rmse_raw: RMSE on the same

Spend metrics (log1p of raw total — secondary, for distributional comparison):
    spend_mae_log:  MAE on log1p of the raw total
    spend_rmse_log: RMSE on log1p of the raw total
    spend_r2_log:   R² on log1p of the raw total

Primary aggregate: full holdout-period revenue per customer (sum over H weeks).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats


def freq_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def freq_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def freq_mape(y_true_agg: float, y_pred_agg: float) -> float:
    """Cohort-level MAPE. Inputs are scalar aggregated totals."""
    if y_true_agg == 0:
        return float("nan")
    return float(abs(y_pred_agg - y_true_agg) / abs(y_true_agg) * 100)


def bias_pct(y_true_agg: float, y_pred_agg: float) -> float:
    """Percentage bias (signed). Positive = over-prediction."""
    if y_true_agg == 0:
        return float("nan")
    return float((y_pred_agg - y_true_agg) / abs(y_true_agg) * 100)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def aggregate_cohort(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    customer_ids: np.ndarray,
) -> dict:
    """Aggregate individual-level predictions to cohort level."""
    total_true = float(y_true.sum())
    total_pred = float(y_pred.sum())
    return {
        "total_true": total_true,
        "total_pred": total_pred,
        "mape": freq_mape(total_true, total_pred),
        "bias_pct": bias_pct(total_true, total_pred),
        "n_customers": len(np.unique(customer_ids)),
    }


def compute_smearing_factor(
    true_log1p: np.ndarray,
    pred_log1p: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate Duan's smearing factor from validation residuals.

    Computes S = mean(exp(true_log1p - pred_log1p)) over active validation steps.
    Apply as: corrected_spend = raw_spend * S, where raw_spend = expm1(pred_log1p).

    This corrects the downward bias from Jensen's Inequality: E[expm1(Z)] > expm1(E[Z]),
    so naively inverting predicted log1p spend systematically under-predicts revenue.

    Args:
        true_log1p: (N, ...) array of true log1p spend values (e.g. validation y_spend)
        pred_log1p: (N, ...) matching predicted log1p spend
        mask:       optional boolean/float array of same shape; only masked-True
                    positions contribute (e.g. active weeks where y_freq > 0).

    Returns:
        S: float smearing factor (>= 1 for right-skewed spend distributions)
    """
    residuals = np.asarray(true_log1p, dtype=np.float64) - np.asarray(pred_log1p, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        residuals = residuals.ravel()[m]
    else:
        residuals = residuals.ravel()
    if residuals.size == 0:
        return 1.0
    return float(np.mean(np.exp(residuals)))


def per_customer_clv(
    raw_spend_per_week: np.ndarray,
    weekly_discount_rate: float = 0.0,
) -> np.ndarray:
    """
    Net present value of per-customer weekly revenue over the holdout horizon.

    CLV_i = Σ_h  raw_spend_per_week[i, h] / (1 + r)^h

    Args:
        raw_spend_per_week: (N, H) raw-currency spend per customer per week
                            (should already have activity weighting applied so
                            inactive weeks contribute zero).
        weekly_discount_rate: r ≈ 0.0018 for 10% annual (CDNOW/UCI/Dunnhumby);
                              0.0 for short Ta-Feng / Extension 3 holdouts.

    Returns:
        (N,) float32 discounted CLV per customer.
    """
    raw = np.asarray(raw_spend_per_week, dtype=np.float64)
    N, H = raw.shape
    discount = (1.0 / (1.0 + weekly_discount_rate)) ** np.arange(H, dtype=np.float64)
    return (raw * discount[np.newaxis, :]).sum(axis=1).astype(np.float32)


def clv_spearman(true_clv: np.ndarray, pred_clv: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual per-customer CLV."""
    r, _ = stats.spearmanr(true_clv, pred_clv)
    return float(r)


def clv_decile_lift(true_clv: np.ndarray, pred_clv: np.ndarray) -> float:
    """
    McCarthy & Fader (2018) decile-lift metric.

    Identifies the top-decile of customers by *predicted* CLV, then computes
    their mean *actual* CLV as a multiple of the population mean actual CLV.
    A lift > 1 means the model successfully rank-orders high-value customers.
    """
    true_clv = np.asarray(true_clv, dtype=np.float64)
    pred_clv = np.asarray(pred_clv, dtype=np.float64)
    n = len(pred_clv)
    top_n = max(1, int(np.ceil(n * 0.1)))
    top_idx = np.argsort(pred_clv)[-top_n:]
    mean_all = true_clv.mean()
    if mean_all == 0:
        return float("nan")
    return float(true_clv[top_idx].mean() / mean_all)


def _spend_per_week_raw_matrix(
    per_week_scaled_log: np.ndarray,
    scaler,
    activity_weights: Optional[np.ndarray] = None,
    smearing_factor: Optional[float] = None,
) -> np.ndarray:
    """
    Convert per-week scaled log-spend to raw-currency (N, H) matrix.

    This is the building block used by both aggregate_spend_to_raw_total (which
    sums to produce (N,) totals) and per_customer_clv (which applies discounting
    before summing).  Keeping it separate avoids duplicated inverse-transform
    logic when both the sum and the matrix are needed.
    """
    N, H = per_week_scaled_log.shape
    raw = scaler.inverse_transform_spend(per_week_scaled_log.reshape(-1))
    raw = raw.reshape(N, H)
    raw = np.clip(raw, 0.0, None)
    if smearing_factor is not None and smearing_factor != 1.0:
        raw = raw * float(smearing_factor)
    if activity_weights is not None:
        raw = raw * np.asarray(activity_weights, dtype=raw.dtype)
    return raw


def aggregate_spend_to_raw_total(
    per_week_scaled_log: np.ndarray,
    scaler,
    activity_weights: Optional[np.ndarray] = None,
    smearing_factor: Optional[float] = None,
) -> np.ndarray:
    """
    Convert a per-week scaled log-spend array to per-customer raw-currency totals.

    Inverse-transforms each week individually (unscale → expm1) and sums across
    the holdout horizon. This is the only mathematically meaningful way to go
    from the per-week log-scaled prediction space to business-interpretable
    total spend.

    Args:
        per_week_scaled_log: (N, H) per-week scaled log1p spend
        scaler:              fitted SpendScaler
        activity_weights:    optional (N, H) array — when provided, raw per-week
                             spend is multiplied by this weight before summing.
                             Used by deep models so a week the model predicts as
                             inactive contributes zero raw spend (sampling: 0/1
                             indicator; expected-value: P(freq>0)).
        smearing_factor:     optional Duan's smearing correction (computed from
                             validation residuals via compute_smearing_factor).
                             Applied to predictions only — not to ground-truth paths.
                             Corrects the downward bias from Jensen's Inequality.

    Returns:
        (N,) raw-currency holdout totals
    """
    raw = _spend_per_week_raw_matrix(
        per_week_scaled_log, scaler,
        activity_weights=activity_weights,
        smearing_factor=smearing_factor,
    )
    return raw.sum(axis=1).astype(np.float32)


def compute_all_metrics(
    y_freq_true: np.ndarray,
    y_freq_pred: np.ndarray,
    y_spend_true_raw_total: Optional[np.ndarray] = None,
    y_spend_pred_raw_total: Optional[np.ndarray] = None,
    y_spend_true_per_week: Optional[np.ndarray] = None,
    y_spend_pred_per_week: Optional[np.ndarray] = None,
    customer_ids: Optional[np.ndarray] = None,
    scaler=None,
    pred_activity_per_week: Optional[np.ndarray] = None,
    true_activity_per_week: Optional[np.ndarray] = None,
    smearing_factor: Optional[float] = None,
    weekly_discount_rate: float = 0.0,
) -> dict:
    """
    Compute all evaluation metrics and return as a flat dict.

    Spend can be supplied in one of two forms:
        (a) Raw per-customer holdout totals (benchmarks):
            pass y_spend_{true,pred}_raw_total in raw currency units.
        (b) Per-week scaled log-spend arrays (deep models):
            pass y_spend_{true,pred}_per_week of shape (N, H) together with
            a fitted `scaler`; this function inverse-transforms each week and
            sums to get the raw total.

    Args:
        y_freq_true:             (N,) true total transaction counts (unclipped)
        y_freq_pred:             (N,) predicted total transaction counts
        y_spend_true_raw_total:  (N,) raw-currency holdout totals — ground truth
        y_spend_pred_raw_total:  (N,) raw-currency holdout totals — prediction
        y_spend_true_per_week:   (N, H) per-week scaled log spend — ground truth
        y_spend_pred_per_week:   (N, H) per-week scaled log spend — prediction
        customer_ids:            (N,) customer identifiers (for cohort aggregation)
        scaler:                  fitted SpendScaler (required when per-week arrays given)
        smearing_factor:         Duan's smearing correction factor from validation
                                 residuals (applied to predictions only). Omit or pass
                                 None to disable. Compute with compute_smearing_factor().
    """
    metrics = {
        "freq_rmse": freq_rmse(y_freq_true, y_freq_pred),
        "freq_mae": freq_mae(y_freq_true, y_freq_pred),
    }
    if customer_ids is not None:
        cohort = aggregate_cohort(y_freq_true, y_freq_pred, customer_ids)
        metrics["freq_mape"] = cohort["mape"]
        metrics["bias_pct"] = cohort["bias_pct"]

    # Resolve spend inputs to raw-currency totals
    true_raw_total: Optional[np.ndarray] = None
    pred_raw_total: Optional[np.ndarray] = None

    # Per-week raw matrices (needed for CLV; computed once when per-week data available)
    true_raw_matrix: Optional[np.ndarray] = None
    pred_raw_matrix: Optional[np.ndarray] = None

    if y_spend_true_raw_total is not None and y_spend_pred_raw_total is not None:
        true_raw_total = np.asarray(y_spend_true_raw_total, dtype=np.float32)
        pred_raw_total = np.asarray(y_spend_pred_raw_total, dtype=np.float32)
    elif y_spend_true_per_week is not None and y_spend_pred_per_week is not None:
        if scaler is None:
            raise ValueError("scaler is required when per-week spend arrays are provided")
        # Ground truth: weights are the actual binary activity indicator (so the
        # tiny non-zero spend the inverse transform produces from log1p(0)≈0 doesn't
        # leak into "active" totals).
        true_weights = (
            true_activity_per_week.astype(np.float32)
            if true_activity_per_week is not None else None
        )
        # Build raw per-week matrices for both truth and prediction
        true_raw_matrix = _spend_per_week_raw_matrix(
            y_spend_true_per_week, scaler, activity_weights=true_weights,
        )
        pred_raw_matrix = _spend_per_week_raw_matrix(
            y_spend_pred_per_week, scaler,
            activity_weights=pred_activity_per_week,
            smearing_factor=smearing_factor,
        )
        true_raw_total = true_raw_matrix.sum(axis=1).astype(np.float32)
        pred_raw_total = pred_raw_matrix.sum(axis=1).astype(np.float32)

    if true_raw_total is not None and pred_raw_total is not None:
        if smearing_factor is not None:
            metrics["smearing_factor"] = float(smearing_factor)
        metrics["spend_mae_raw"] = _mae(true_raw_total, pred_raw_total)
        metrics["spend_rmse_raw"] = _rmse(true_raw_total, pred_raw_total)

        # log1p of raw total — keeps a log-scale view without the sum-of-logs bug
        true_log = np.log1p(true_raw_total)
        pred_log = np.log1p(pred_raw_total)
        metrics["spend_mae_log"] = _mae(true_log, pred_log)
        metrics["spend_rmse_log"] = _rmse(true_log, pred_log)
        metrics["spend_r2_log"] = _r2(true_log, pred_log)

        # Cohort-level spend bias (signed, as % of true total revenue)
        total_true = float(true_raw_total.sum())
        total_pred = float(pred_raw_total.sum())
        metrics["spend_bias_pct"] = bias_pct(total_true, total_pred)

    # Per-customer CLV: only computable when we have per-week matrices
    # (joint deep models only; base LSTM and BTYD benchmarks produce scalar totals).
    if true_raw_matrix is not None and pred_raw_matrix is not None:
        true_clv = per_customer_clv(true_raw_matrix, weekly_discount_rate)
        pred_clv = per_customer_clv(pred_raw_matrix, weekly_discount_rate)
        metrics["clv_mae"] = float(np.mean(np.abs(true_clv - pred_clv)))
        metrics["clv_rmse"] = float(np.sqrt(np.mean((true_clv - pred_clv) ** 2)))
        metrics["clv_spearman"] = clv_spearman(true_clv, pred_clv)
        metrics["clv_decile_lift"] = clv_decile_lift(true_clv, pred_clv)
        metrics["clv_total_true"] = float(true_clv.sum())
        metrics["clv_total_pred"] = float(pred_clv.sum())
        # Save per-customer arrays for paired bootstrap significance testing.
        # Stored as lists so they survive json.dump without extra serialisation.
        metrics["_per_customer_freq_se"] = ((y_freq_true - y_freq_pred) ** 2).tolist()
        metrics["_per_customer_freq_ae"] = np.abs(y_freq_true - y_freq_pred).tolist()
        metrics["_per_customer_spend_ae"] = np.abs(true_raw_total - pred_raw_total).tolist()
        metrics["_per_customer_clv_ae"] = np.abs(true_clv - pred_clv).tolist()
    elif true_raw_total is not None and pred_raw_total is not None:
        # Benchmark models: save per-customer arrays for freq/spend bootstrap
        # (CLV is skipped — no per-week decomposition available).
        metrics["_per_customer_freq_se"] = ((y_freq_true - y_freq_pred) ** 2).tolist()
        metrics["_per_customer_freq_ae"] = np.abs(y_freq_true - y_freq_pred).tolist()
        metrics["_per_customer_spend_ae"] = np.abs(true_raw_total - pred_raw_total).tolist()
    elif true_raw_total is None:
        # Frequency-only models (Base LSTM, Pareto/NBD, Pareto/GGG, GPPM)
        metrics["_per_customer_freq_se"] = ((y_freq_true - y_freq_pred) ** 2).tolist()
        metrics["_per_customer_freq_ae"] = np.abs(y_freq_true - y_freq_pred).tolist()

    return metrics


def split_metric_artifacts(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Split a metrics dict into scalar JSON fields and array artifacts.

    Final thesis comparison tables should read scalar metrics only. Per-customer
    arrays used for paired bootstrap tests are written separately to `.npz` so
    JSON files remain clean, small, and table-safe.
    """
    scalar_metrics: dict[str, Any] = {}
    array_artifacts: dict[str, np.ndarray] = {}

    for key, value in metrics.items():
        if isinstance(value, np.generic):
            scalar_metrics[key] = value.item()
            continue

        if isinstance(value, np.ndarray) or key.startswith("_") or isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype.kind in {"b", "i", "u", "f", "c"} and arr.ndim >= 1:
                artifact_key = key.lstrip("_")
                array_artifacts[artifact_key] = arr
                continue

        if value is None or isinstance(value, (str, bool, int, float)):
            scalar_metrics[key] = value
            continue

        # Last-resort scalar conversion for rare numpy/pandas scalar-like objects.
        try:
            if np.isscalar(value):
                scalar_metrics[key] = value.item() if hasattr(value, "item") else value
                continue
        except Exception:
            pass

        raise TypeError(
            f"Metric {key!r} is not JSON-scalar and is not a numeric array artifact: "
            f"{type(value).__name__}"
        )

    return scalar_metrics, array_artifacts


def metrics_arrays_path(metrics_path: str | Path) -> Path:
    """Return the sidecar `.npz` path for a `*_metrics.json` file."""
    path = Path(metrics_path)
    if path.name.endswith("_metrics.json"):
        return path.with_name(path.name.replace("_metrics.json", "_arrays.npz"))
    return path.with_suffix(".npz")


def save_metrics_with_artifacts(metrics: dict[str, Any], metrics_path: str | Path) -> tuple[Path, Optional[Path]]:
    """
    Save scalar metrics to JSON and numeric arrays to a sidecar `.npz`.

    Returns:
        (metrics_path, arrays_path_or_none)
    """
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    scalar_metrics, array_artifacts = split_metric_artifacts(metrics)

    arrays_path: Optional[Path] = None
    if array_artifacts:
        arrays_path = metrics_arrays_path(metrics_path)
        np.savez_compressed(arrays_path, **array_artifacts)
        scalar_metrics["arrays_file"] = arrays_path.name

    with open(metrics_path, "w") as f:
        json.dump(scalar_metrics, f, indent=2)

    return metrics_path, arrays_path
