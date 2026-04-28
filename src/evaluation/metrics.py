"""
Evaluation metrics for frequency and spend predictions.

Frequency metrics (individual-level):
    freq_rmse: RMSE of predicted vs actual purchase count
    freq_mae:  MAE of purchase counts

Cohort-level metrics:
    freq_mape:  MAPE of aggregated cohort predictions vs actuals
    bias_pct:   Percentage bias = (sum_pred - sum_actual) / sum_actual * 100

Spend aggregation (critical):
    The model predicts per-week spend in a MinMax-scaled log1p space. To obtain
    holdout-period total spend in raw currency we must inverse-transform each
    week individually and then sum the raw-currency values. Summing the scaled
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
from typing import Optional

import numpy as np


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


def aggregate_spend_to_raw_total(
    per_week_scaled_log: np.ndarray,
    scaler,
    activity_weights: Optional[np.ndarray] = None,
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

    Returns:
        (N,) raw-currency holdout totals
    """
    N, H = per_week_scaled_log.shape
    raw = scaler.inverse_transform_spend(per_week_scaled_log.reshape(-1))
    raw = raw.reshape(N, H)
    # Floor small negatives from numerical error in inverse MinMax
    raw = np.clip(raw, 0.0, None)
    if activity_weights is not None:
        raw = raw * np.asarray(activity_weights, dtype=raw.dtype)
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
        true_raw_total = aggregate_spend_to_raw_total(
            y_spend_true_per_week, scaler, activity_weights=true_weights,
        )
        pred_raw_total = aggregate_spend_to_raw_total(
            y_spend_pred_per_week, scaler, activity_weights=pred_activity_per_week,
        )

    if true_raw_total is not None and pred_raw_total is not None:
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

    return metrics
