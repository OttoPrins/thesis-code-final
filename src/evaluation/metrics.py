"""
Evaluation metrics for frequency and spend predictions.

Frequency metrics (individual-level):
    freq_rmse: RMSE of predicted vs actual purchase count
    freq_mae:  MAE of purchase counts

Cohort-level metrics:
    freq_mape: Valendin et al. (2022) aggregate weekly MAPE:
               100 * sum_t |A_t - P_t| / sum_t A_t
    freq_horizon_abs_bias_pct:
               Absolute full-horizon aggregate error. This is the historical
               scalar "MAPE" implementation retained under an explicit name.
    bias_pct:  Signed full-horizon aggregate bias:
               100 * (sum_t P_t - sum_t A_t) / sum_t A_t

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


METRIC_DEFINITION_VERSION = "valendin_aggregate_mape_v1"


def freq_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def freq_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def horizon_abs_bias_pct(y_true_agg: float, y_pred_agg: float) -> float:
    """Absolute full-horizon aggregate error as a percent of true total."""
    if y_true_agg == 0:
        return float("nan")
    return float(abs(y_pred_agg - y_true_agg) / abs(y_true_agg) * 100)


def bias_pct(y_true_agg: float, y_pred_agg: float) -> float:
    """Percentage bias (signed). Positive = over-prediction."""
    if y_true_agg == 0:
        return float("nan")
    return float((y_pred_agg - y_true_agg) / abs(y_true_agg) * 100)


def freq_mape(y_true_per_week: np.ndarray, y_pred_per_week: np.ndarray) -> float:
    """
    Valendin et al. (2022) aggregate weekly MAPE.

    The paper scores the cohort-level weekly transaction flow, not a scalar
    holdout total and not the mean of week-by-week percentage errors:

        100 * sum_t |A_t - P_t| / sum_t A_t

    Inputs may be per-customer weekly matrices (N, H) or already aggregated
    weekly vectors (H,). In both cases the denominator is total actual
    aggregate transactions over the scored horizon.
    """
    true = np.asarray(y_true_per_week, dtype=np.float64)
    pred = np.asarray(y_pred_per_week, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(
            f"Valendin MAPE true/pred shape mismatch: {true.shape} vs {pred.shape}"
        )
    if true.ndim == 2:
        true = true.sum(axis=0)
        pred = pred.sum(axis=0)
    elif true.ndim != 1:
        raise ValueError(f"Valendin MAPE expects 1D or 2D arrays, got ndim={true.ndim}")
    total_true = float(true.sum())
    if total_true == 0:
        return float("nan")
    return float(np.abs(true - pred).sum() / abs(total_true) * 100)


def per_week_aggregate_metrics(
    y_true_per_week: np.ndarray,
    y_pred_per_week: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """
    Aggregate all customers by week and score the resulting cohort time series.

    This complements full-horizon cohort bias: a model can have acceptable total
    holdout bias while missing the timing of purchases within the horizon.
    Valendin MAPE uses total actual aggregate transactions as denominator; the
    old mean of week-level percentage errors is retained with an explicit name.
    """
    true = np.asarray(y_true_per_week, dtype=np.float64)
    pred = np.asarray(y_pred_per_week, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(
            f"Per-week true/pred arrays must have the same shape, got {true.shape} and {pred.shape}"
        )
    if true.ndim != 2:
        raise ValueError(f"Per-week arrays must be 2D (N, H), got ndim={true.ndim}")

    true_agg = true.sum(axis=0)
    pred_agg = pred.sum(axis=0)
    abs_err = np.abs(pred_agg - true_agg)
    valid = np.abs(true_agg) > 1e-12

    out = {
        f"{prefix}_weekly_mae": float(abs_err.mean()),
        f"{prefix}_weekly_total_true": float(true_agg.sum()),
        f"{prefix}_weekly_total_pred": float(pred_agg.sum()),
        f"{prefix}_valendin_mape": freq_mape(true_agg, pred_agg),
        f"{prefix}_horizon_abs_bias_pct": horizon_abs_bias_pct(
            float(true_agg.sum()),
            float(pred_agg.sum()),
        ),
        f"{prefix}_horizon_bias_pct": bias_pct(
            float(true_agg.sum()),
            float(pred_agg.sum()),
        ),
    }
    if valid.any():
        pct_err = (pred_agg[valid] - true_agg[valid]) / np.abs(true_agg[valid])
        out[f"{prefix}_weekly_mean_pct_mape"] = float(np.mean(np.abs(pct_err)) * 100)
        out[f"{prefix}_weekly_mean_pct_bias_pct"] = float(np.mean(pct_err) * 100)
        # Volume-weighted tracking error: scale the SUMMED mismatch by the SUMMED
        # absolute truth on the valid slice. Equal weight per week (unweighted mean)
        # inflates metrics on dead weeks; this formulation matches the CBA tracking-
        # error convention used by Valendin et al. (2022) for aggregate reporting.
        true_valid_total = float(np.abs(true_agg[valid]).sum())
        if true_valid_total > 0:
            out[f"{prefix}_weekly_bias_pct"] = float(
                (pred_agg[valid] - true_agg[valid]).sum() / true_valid_total * 100
            )
            out[f"{prefix}_weekly_mape"] = float(
                np.abs(pred_agg[valid] - true_agg[valid]).sum() / true_valid_total * 100
            )
        else:
            out[f"{prefix}_weekly_bias_pct"] = float("nan")
            out[f"{prefix}_weekly_mape"] = float("nan")
    else:
        out[f"{prefix}_weekly_mean_pct_mape"] = float("nan")
        out[f"{prefix}_weekly_mean_pct_bias_pct"] = float("nan")
        out[f"{prefix}_weekly_bias_pct"] = float("nan")
        out[f"{prefix}_weekly_mape"] = float("nan")
    return out


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: Optional[float] = None,
    seasonal_period: int = 1,
) -> float:
    """
    Mean absolute scaled error.

    If `scale` is omitted, the denominator is the mean absolute one-step naive
    change within `y_true` along its last axis. For holdout-only matrices this is
    a diagnostic scaled MAE; final rolling-origin claims should pass a calibration
    denominator explicitly.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"MASE true/pred shape mismatch: {true.shape} vs {pred.shape}")
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be >= 1")

    mae = np.mean(np.abs(true - pred))
    if scale is None:
        scale = mase_scale(true, seasonal_period=seasonal_period)
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(mae / scale)


def mase_scale(y_train: np.ndarray, seasonal_period: int = 1) -> float:
    """
    In-sample naive MAE denominator for MASE.

    Hyndman & Koehler define the denominator on the training period, not the
    holdout. Callers should pass this value into `mase(..., scale=...)` whenever
    calibration trajectories are available.
    """
    train = np.asarray(y_train, dtype=np.float64)
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be >= 1")
    if train.shape[-1] <= seasonal_period:
        return float("nan")
    diffs = np.abs(train[..., seasonal_period:] - train[..., :-seasonal_period])
    scale = float(np.mean(diffs))
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


def _gini(actual: np.ndarray, score: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if actual.shape != score.shape:
        raise ValueError(f"Gini actual/score shape mismatch: {actual.shape} vs {score.shape}")
    finite = np.isfinite(actual) & np.isfinite(score)
    actual = actual[finite]
    score = score[finite]
    n = actual.size
    if n == 0 or np.sum(actual) <= 0:
        return float("nan")
    order = np.lexsort((np.arange(n), score))
    actual_sorted = actual[order]
    cumulative = np.cumsum(actual_sorted)
    return float((n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n)


def normalized_gini(actual: np.ndarray, score: np.ndarray) -> float:
    """Gini(score) divided by the perfect-model Gini(actual)."""
    denom = _gini(actual, actual)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float(_gini(actual, score) / denom)


def decile_calibration_arrays(
    actual: np.ndarray,
    score: np.ndarray,
    prefix: str,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Decile calibration table ordered by predicted score, top decile first.

    Returned as numeric arrays so save_metrics_with_artifacts writes them to the
    `.npz` sidecar rather than bloating the scalar JSON metrics.
    """
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if actual.shape != score.shape:
        raise ValueError(
            f"Decile calibration actual/score shape mismatch: {actual.shape} vs {score.shape}"
        )
    finite = np.isfinite(actual) & np.isfinite(score)
    actual = actual[finite]
    score = score[finite]
    if actual.size == 0:
        empty = np.array([], dtype=np.float64)
        return {
            f"_{prefix}_decile": empty,
            f"_{prefix}_decile_n": empty,
            f"_{prefix}_decile_actual_mean": empty,
            f"_{prefix}_decile_pred_mean": empty,
            f"_{prefix}_decile_actual_total": empty,
            f"_{prefix}_decile_pred_total": empty,
            f"_{prefix}_decile_actual_lift": empty,
        }

    order = np.argsort(score)[::-1]
    groups = np.array_split(order, min(n_bins, actual.size))
    pop_mean = actual.mean()
    rows = []
    for i, idx in enumerate(groups, start=1):
        actual_mean = float(actual[idx].mean())
        pred_mean = float(score[idx].mean())
        rows.append((
            i,
            len(idx),
            actual_mean,
            pred_mean,
            float(actual[idx].sum()),
            float(score[idx].sum()),
            float(actual_mean / pop_mean) if pop_mean != 0 else float("nan"),
        ))
    arr = np.asarray(rows, dtype=np.float64)
    return {
        f"_{prefix}_decile": arr[:, 0],
        f"_{prefix}_decile_n": arr[:, 1],
        f"_{prefix}_decile_actual_mean": arr[:, 2],
        f"_{prefix}_decile_pred_mean": arr[:, 3],
        f"_{prefix}_decile_actual_total": arr[:, 4],
        f"_{prefix}_decile_pred_total": arr[:, 5],
        f"_{prefix}_decile_actual_lift": arr[:, 6],
    }


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
        "horizon_abs_bias_pct": horizon_abs_bias_pct(total_true, total_pred),
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


def apply_literature_activity_gate(
    y_spend_pred_per_week: np.ndarray,
    pred_activity_per_week: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Zero-mask predicted per-week spend on weeks the model predicts as inactive.

    Deep models emit tiny log-space values on dormant weeks; expm1 of those values
    leaks fractional currency into spend_mae_raw even when the activity head was
    correct. This gate removes the leak so "Literature Lens" metrics match the CBA
    reporting convention (only active-week spend counts toward tracking error).

    Does NOT modify compute_all_metrics — call explicitly in re-scoring scripts to
    keep the Valuation Lens (ungated, for DCF stress-testing) intact.

    Args:
        y_spend_pred_per_week: (N, H) raw-currency or log-space spend predictions.
        pred_activity_per_week: (N, H) model activity probability P(freq > 0).
        threshold: weeks with activity probability < threshold are zero-masked.

    Returns:
        (N, H) zero-masked copy of y_spend_pred_per_week.
    """
    spend = np.asarray(y_spend_pred_per_week, dtype=np.float64)
    activity = np.asarray(pred_activity_per_week, dtype=np.float64)
    if spend.shape != activity.shape:
        raise ValueError(
            f"apply_literature_activity_gate: spend/activity shape mismatch: "
            f"{spend.shape} vs {activity.shape}"
        )
    gate = (activity >= threshold).astype(spend.dtype)
    return spend * gate


def _spend_per_week_raw_matrix(
    per_week_scaled_log: np.ndarray,
    scaler,
    activity_weights: Optional[np.ndarray] = None,
    smearing_factor: Optional[float] = None,
    calibration_factor: float = 1.0,
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
    if calibration_factor != 1.0:
        raw = raw * float(calibration_factor)
    if activity_weights is not None:
        raw = raw * np.asarray(activity_weights, dtype=raw.dtype)
    return raw


def aggregate_spend_to_raw_total(
    per_week_scaled_log: np.ndarray,
    scaler,
    activity_weights: Optional[np.ndarray] = None,
    smearing_factor: Optional[float] = None,
    calibration_factor: float = 1.0,
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
        calibration_factor=calibration_factor,
    )
    return raw.sum(axis=1).astype(np.float32)


def compute_all_metrics(
    y_freq_true: np.ndarray,
    y_freq_pred: np.ndarray,
    y_freq_true_per_week: Optional[np.ndarray] = None,
    y_freq_pred_per_week: Optional[np.ndarray] = None,
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
    freq_mase_scale: Optional[float] = None,
    spend_mase_scale: Optional[float] = None,
    freq_calibration_factor: float = 1.0,
    spend_calibration_factor: float = 1.0,
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
        y_freq_true_per_week:    (N, H) true per-week transaction counts
        y_freq_pred_per_week:    (N, H) predicted per-week transaction counts
        y_spend_true_raw_total:  (N,) raw-currency holdout totals — ground truth
        y_spend_pred_raw_total:  (N,) raw-currency holdout totals — prediction
        y_spend_true_per_week:   (N, H) per-week scaled log spend — ground truth
        y_spend_pred_per_week:   (N, H) per-week scaled log spend — prediction
        customer_ids:            (N,) customer identifiers (for cohort aggregation)
        scaler:                  fitted SpendScaler (required when per-week arrays given)
        smearing_factor:         Duan's smearing correction factor from validation
                                 residuals (applied to predictions only). Omit or pass
                                 None to disable. Compute with compute_smearing_factor().
        freq_mase_scale:         optional in-sample naive MAE denominator for frequency MASE
        spend_mase_scale:        optional in-sample naive MAE denominator for raw-spend MASE
    """
    y_freq_true = np.asarray(y_freq_true, dtype=np.float32)
    y_freq_pred = np.asarray(y_freq_pred, dtype=np.float32) * float(freq_calibration_factor)

    metrics = {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "freq_rmse": freq_rmse(y_freq_true, y_freq_pred),
        "freq_mae": freq_mae(y_freq_true, y_freq_pred),
        "freq_normalized_gini": normalized_gini(y_freq_true, y_freq_pred),
    }
    if freq_calibration_factor != 1.0:
        metrics["freq_calibration_factor"] = float(freq_calibration_factor)
    if customer_ids is not None:
        cohort = aggregate_cohort(y_freq_true, y_freq_pred, customer_ids)
        metrics["freq_horizon_abs_bias_pct"] = cohort["horizon_abs_bias_pct"]
        metrics["bias_pct"] = cohort["bias_pct"]
    metrics.update(decile_calibration_arrays(y_freq_true, y_freq_pred, "freq"))

    if y_freq_true_per_week is not None and y_freq_pred_per_week is not None:
        true_freq_week = np.asarray(y_freq_true_per_week, dtype=np.float32)
        pred_freq_week = (
            np.asarray(y_freq_pred_per_week, dtype=np.float32)
            * float(freq_calibration_factor)
        )
        metrics.update(per_week_aggregate_metrics(true_freq_week, pred_freq_week, "freq"))
        metrics["freq_mape"] = metrics["freq_valendin_mape"]
        metrics["freq_mase"] = mase(true_freq_week, pred_freq_week, scale=freq_mase_scale)
        if freq_mase_scale is not None:
            metrics["freq_mase_scale"] = float(freq_mase_scale)
        metrics["_per_week_true_freq"] = true_freq_week
        metrics["_per_week_pred_freq"] = pred_freq_week

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
            calibration_factor=spend_calibration_factor,
        )
        true_raw_total = true_raw_matrix.sum(axis=1).astype(np.float32)
        pred_raw_total = pred_raw_matrix.sum(axis=1).astype(np.float32)

    if true_raw_total is not None and pred_raw_total is not None:
        if smearing_factor is not None:
            metrics["smearing_factor"] = float(smearing_factor)
        if spend_calibration_factor != 1.0:
            metrics["spend_calibration_factor"] = float(spend_calibration_factor)
        metrics["spend_mae_raw"] = _mae(true_raw_total, pred_raw_total)
        metrics["spend_rmse_raw"] = _rmse(true_raw_total, pred_raw_total)
        metrics["spend_normalized_gini"] = normalized_gini(true_raw_total, pred_raw_total)
        metrics.update(decile_calibration_arrays(true_raw_total, pred_raw_total, "spend"))

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

    if true_raw_matrix is not None and pred_raw_matrix is not None:
        metrics.update(per_week_aggregate_metrics(true_raw_matrix, pred_raw_matrix, "spend"))
        metrics["spend_mase"] = mase(true_raw_matrix, pred_raw_matrix, scale=spend_mase_scale)
        if spend_mase_scale is not None:
            metrics["spend_mase_scale"] = float(spend_mase_scale)
        metrics["_per_week_true_spend"] = true_raw_matrix
        metrics["_per_week_pred_spend"] = pred_raw_matrix

    # Per-customer CLV: only computable when we have per-week matrices
    # (joint deep models only; base LSTM and BTYD benchmarks produce scalar totals).
    if true_raw_matrix is not None and pred_raw_matrix is not None:
        true_clv = per_customer_clv(true_raw_matrix, weekly_discount_rate)
        pred_clv = per_customer_clv(pred_raw_matrix, weekly_discount_rate)
        metrics["clv_mae"] = float(np.mean(np.abs(true_clv - pred_clv)))
        metrics["clv_rmse"] = float(np.sqrt(np.mean((true_clv - pred_clv) ** 2)))
        metrics["clv_spearman"] = clv_spearman(true_clv, pred_clv)
        metrics["clv_decile_lift"] = clv_decile_lift(true_clv, pred_clv)
        metrics["clv_normalized_gini"] = normalized_gini(true_clv, pred_clv)
        metrics["clv_total_true"] = float(true_clv.sum())
        metrics["clv_total_pred"] = float(pred_clv.sum())
        metrics.update(decile_calibration_arrays(true_clv, pred_clv, "clv"))
        # Save per-customer arrays for paired bootstrap significance testing.
        # Stored as lists so they survive json.dump without extra serialisation.
        metrics["_per_customer_freq_se"] = ((y_freq_true - y_freq_pred) ** 2).tolist()
        metrics["_per_customer_freq_ae"] = np.abs(y_freq_true - y_freq_pred).tolist()
        metrics["_per_customer_spend_ae"] = np.abs(true_raw_total - pred_raw_total).tolist()
        metrics["_per_customer_clv_ae"] = np.abs(true_clv - pred_clv).tolist()
        metrics["_per_customer_true_freq"] = y_freq_true.tolist()
        metrics["_per_customer_pred_freq"] = y_freq_pred.tolist()
        metrics["_per_customer_true_spend"] = true_raw_total.tolist()
        metrics["_per_customer_pred_spend"] = pred_raw_total.tolist()
        metrics["_per_customer_true_clv"] = true_clv.tolist()
        metrics["_per_customer_pred_clv"] = pred_clv.tolist()
    elif true_raw_total is not None and pred_raw_total is not None:
        # Benchmark models: save per-customer arrays for freq/spend bootstrap
        # (CLV is skipped — no per-week decomposition available).
        metrics["_per_customer_freq_se"] = ((y_freq_true - y_freq_pred) ** 2).tolist()
        metrics["_per_customer_freq_ae"] = np.abs(y_freq_true - y_freq_pred).tolist()
        metrics["_per_customer_spend_ae"] = np.abs(true_raw_total - pred_raw_total).tolist()
        metrics["_per_customer_true_freq"] = y_freq_true.tolist()
        metrics["_per_customer_pred_freq"] = y_freq_pred.tolist()
        metrics["_per_customer_true_spend"] = true_raw_total.tolist()
        metrics["_per_customer_pred_spend"] = pred_raw_total.tolist()
    elif true_raw_total is None:
        # Frequency-only models (Base LSTM, Pareto/NBD, Pareto/GGG, GPPM)
        metrics["_per_customer_freq_se"] = ((y_freq_true - y_freq_pred) ** 2).tolist()
        metrics["_per_customer_freq_ae"] = np.abs(y_freq_true - y_freq_pred).tolist()
        metrics["_per_customer_true_freq"] = y_freq_true.tolist()
        metrics["_per_customer_pred_freq"] = y_freq_pred.tolist()

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
