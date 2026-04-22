"""
Evaluation metrics for frequency and spend predictions.

Frequency metrics (individual-level):
    freq_rmse: RMSE of predicted vs actual purchase count
    freq_mae:  MAE of purchase counts

Cohort-level metrics:
    freq_mape:  MAPE of aggregated cohort predictions vs actuals
    bias_pct:   Percentage bias = (sum_pred - sum_actual) / sum_actual * 100

Spend metrics (log-space — primary reporting):
    spend_mae:  MAE on log-scaled predictions
    spend_rmse: RMSE on log-scaled predictions
    spend_r2:   R² for spend regression

Spend metrics (raw currency — secondary reporting):
    spend_mae_raw:  MAE after inverse-transforming to original currency
    spend_rmse_raw: RMSE after inverse-transforming to original currency

Primary aggregate: quarterly_revenue = sum of per-customer spend over 13-week window.
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


def spend_mae(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true_log - y_pred_log)))


def spend_rmse(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))


def spend_r2(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - np.mean(y_true_log)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def spend_mae_raw(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    scaler,
) -> float:
    """MAE in original currency units after inverse-transforming from log-scaled space."""
    y_true_raw = scaler.inverse_transform_spend(y_true_log)
    y_pred_raw = scaler.inverse_transform_spend(y_pred_log)
    return float(np.mean(np.abs(y_true_raw - y_pred_raw)))


def spend_rmse_raw(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    scaler,
) -> float:
    """RMSE in original currency units after inverse-transforming from log-scaled space."""
    y_true_raw = scaler.inverse_transform_spend(y_true_log)
    y_pred_raw = scaler.inverse_transform_spend(y_pred_log)
    return float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))


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


def compute_all_metrics(
    y_freq_true: np.ndarray,
    y_freq_pred: np.ndarray,
    y_spend_true_log: Optional[np.ndarray] = None,
    y_spend_pred_log: Optional[np.ndarray] = None,
    customer_ids: Optional[np.ndarray] = None,
    scaler=None,
) -> dict:
    """
    Compute all evaluation metrics and return as a flat dict.

    Args:
        y_freq_true:       (N,) true transaction counts
        y_freq_pred:       (N,) predicted transaction counts
        y_spend_true_log:  (N,) true log-scaled spend (optional; joint models only)
        y_spend_pred_log:  (N,) predicted log-scaled spend (optional)
        customer_ids:      (N,) customer identifiers (for cohort aggregation)
        scaler:            fitted SpendScaler instance; if provided together with spend
                           arrays, computes raw-currency MAE and RMSE as well
    """
    metrics = {
        "freq_rmse": freq_rmse(y_freq_true, y_freq_pred),
        "freq_mae": freq_mae(y_freq_true, y_freq_pred),
    }
    if customer_ids is not None:
        cohort = aggregate_cohort(y_freq_true, y_freq_pred, customer_ids)
        metrics["freq_mape"] = cohort["mape"]
        metrics["bias_pct"] = cohort["bias_pct"]

    if y_spend_true_log is not None and y_spend_pred_log is not None:
        metrics["spend_mae"] = spend_mae(y_spend_true_log, y_spend_pred_log)
        metrics["spend_rmse"] = spend_rmse(y_spend_true_log, y_spend_pred_log)
        metrics["spend_r2"] = spend_r2(y_spend_true_log, y_spend_pred_log)
        if scaler is not None:
            metrics["spend_mae_raw"] = spend_mae_raw(
                y_spend_true_log, y_spend_pred_log, scaler
            )
            metrics["spend_rmse_raw"] = spend_rmse_raw(
                y_spend_true_log, y_spend_pred_log, scaler
            )

    return metrics
