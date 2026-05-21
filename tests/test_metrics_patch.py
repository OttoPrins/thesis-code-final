"""CPU smoke tests for metric repair patch.

Volume-weighted tracking error + literature activity gate.
Synthetic 10x4 matrices only — no dataset I/O, no torch, no scaler.

Run with:  pytest tests/test_metrics_patch.py -v
"""
import numpy as np
import pytest

from src.evaluation.metrics import (
    apply_literature_activity_gate,
    per_week_aggregate_metrics,
)


def _baseline():
    """Deterministic 10x4 fixture: high-volume peak in week 0, dead weeks 1-3."""
    true = np.zeros((10, 4), dtype=np.float32)
    true[:, 0] = 5.0
    return true


def test_volume_weighted_keys_present():
    true = _baseline()
    out = per_week_aggregate_metrics(true, true.copy(), "freq")
    assert "freq_weekly_bias_pct" in out
    assert "freq_weekly_mape" in out
    # Deprecated unweighted keys must still be emitted (additive contract)
    assert "freq_weekly_mean_pct_mape" in out
    assert "freq_weekly_mean_pct_bias_pct" in out


def test_perfect_prediction_zero_error():
    true = _baseline()
    out = per_week_aggregate_metrics(true, true.copy(), "freq")
    assert out["freq_weekly_bias_pct"] == pytest.approx(0.0)
    assert out["freq_weekly_mape"] == pytest.approx(0.0)


def test_one_week_timing_shift_volume_weighted():
    """Peak shifts from week 0 to week 1.

    true_agg = [50, 0, 0, 0]; pred_agg = [0, 50, 0, 0].
    Only week 0 is a valid week (|true_agg| > 0).
    volume-weighted bias  = (0 - 50) / 50 * 100 = -100%
    volume-weighted mape  = |0 - 50| / 50 * 100 = 100%
    unweighted mean bias  = (0 - 50) / 50 * 100 = -100% (single valid week — same)
    """
    true = _baseline()
    pred = np.zeros_like(true)
    pred[:, 1] = 5.0
    out = per_week_aggregate_metrics(true, pred, "freq")
    assert out["freq_weekly_bias_pct"] == pytest.approx(-100.0)
    assert out["freq_weekly_mape"] == pytest.approx(100.0)


def test_dead_week_all_zero_truth_no_nan_no_inf():
    """All-zero truth must not trigger division-by-zero in any branch."""
    true = np.zeros((10, 4), dtype=np.float32)
    pred = np.ones((10, 4), dtype=np.float32) * 0.02  # fractional leak
    out = per_week_aggregate_metrics(true, pred, "freq")
    assert np.isnan(out["freq_weekly_bias_pct"])
    assert np.isnan(out["freq_weekly_mape"])
    assert not np.isinf(out["freq_weekly_bias_pct"])
    assert not np.isinf(out["freq_weekly_mape"])


def test_volume_weighting_differs_from_unweighted_on_mixed_weeks():
    """Two-customer, two-week case where unweighted and volume-weighted diverge.

    true_agg = [100, 1]; pred_agg = [90, 2]
    valid weeks: both (true > 0)
    unweighted bias = mean([(90-100)/100, (2-1)/1]) = mean([-0.1, 1.0]) = 0.45 → 45%
    volume-weighted bias = (90-100 + 2-1) / (100 + 1) = -9/101 → ~-8.91%
    """
    true = np.array([[50, 0.5], [50, 0.5]], dtype=np.float32)  # agg=[100, 1]
    pred = np.array([[45, 1.0], [45, 1.0]], dtype=np.float32)  # agg=[90, 2]
    out = per_week_aggregate_metrics(true, pred, "freq")
    # Volume-weighted: (-10 + 1) / 101 * 100 ≈ -8.91%
    assert out["freq_weekly_bias_pct"] == pytest.approx(-9 / 101 * 100, abs=1e-3)
    # Unweighted (deprecated): mean([-0.1, 1.0]) * 100 = 45%
    assert out["freq_weekly_mean_pct_bias_pct"] == pytest.approx(45.0, abs=1e-3)


def test_literature_gate_zeros_below_threshold():
    spend = np.full((10, 4), 0.02, dtype=np.float32)
    activity = np.zeros((10, 4), dtype=np.float32)
    activity[:, 0] = 0.9  # only week 0 active
    gated = apply_literature_activity_gate(spend, activity, threshold=0.5)
    assert gated.shape == spend.shape
    assert np.allclose(gated[:, 0], 0.02)
    assert np.all(gated[:, 1:] == 0.0)


def test_literature_gate_threshold_boundary():
    """At-threshold passes; below-threshold is masked."""
    spend = np.ones((10, 4), dtype=np.float32)
    activity = np.tile([0.49, 0.50, 0.51, 1.00], (10, 1)).astype(np.float32)
    gated = apply_literature_activity_gate(spend, activity, threshold=0.5)
    assert np.all(gated[:, 0] == 0.0)   # 0.49 < 0.5 → masked
    assert np.all(gated[:, 1] == 1.0)   # 0.50 >= 0.5 → kept
    assert np.all(gated[:, 2] == 1.0)
    assert np.all(gated[:, 3] == 1.0)


def test_literature_gate_shape_mismatch_raises():
    spend = np.zeros((10, 4))
    bad_activity = np.zeros((10, 3))
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_literature_activity_gate(spend, bad_activity)


def test_literature_gate_no_mutation_of_input():
    spend = np.ones((10, 4), dtype=np.float32)
    activity = np.zeros((10, 4), dtype=np.float32)
    original = spend.copy()
    apply_literature_activity_gate(spend, activity)
    assert np.array_equal(spend, original)
