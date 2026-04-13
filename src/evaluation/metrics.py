"""
Evaluation metrics for frequency and spend predictions.

Frequency metrics (individual-level):
    freq_rmse: RMSE of predicted vs actual purchase count
    freq_mae:  MAE of purchase counts

Cohort-level metrics:
    freq_mape:  MAPE of aggregated cohort predictions vs actuals
    bias_pct:   Percentage bias = (sum_pred - sum_actual) / sum_actual * 100

Spend metrics:
    spend_mae:  MAE on log-space predictions
    spend_rmse: RMSE on log-space predictions
    spend_r2:   R² for spend regression
    spend_mae_raw: MAE after inverse transform (raw currency)

Primary aggregate: quarterly_revenue = sum of per-customer spend over 13-week window.
"""

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


def aggregate_cohort(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    customer_ids: np.ndarray,
) -> dict:
    """
    Aggregate individual-level predictions to cohort level.
    Returns cohort totals for computing MAPE and bias.
    """
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
    y_spend_true_log: np.ndarray = None,
    y_spend_pred_log: np.ndarray = None,
    customer_ids: np.ndarray = None,
) -> dict:
    """Compute all metrics and return as a flat dict."""
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

    return metrics
