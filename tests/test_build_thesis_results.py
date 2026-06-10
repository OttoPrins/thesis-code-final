import numpy as np
import pytest

from build_thesis_results import (
    F3_COMPLETE_HOLDOUT_WEEKS,
    _cumulative_calibration_summary,
    _holdout_week_ticks,
)


def test_cumulative_calibration_summary_uses_seed_range_and_complete_week_cutoff():
    true_weekly = np.full(22, 10.0)
    pred_weekly_by_seed = np.vstack(
        [
            np.full(22, 9.0),
            np.full(22, 10.0),
            np.full(22, 11.0),
        ]
    )

    summary = _cumulative_calibration_summary(
        true_weekly,
        pred_weekly_by_seed,
        plot_weeks=F3_COMPLETE_HOLDOUT_WEEKS["dunnhumby"],
    )

    assert true_weekly.shape == (22,)
    assert pred_weekly_by_seed.shape == (3, 22)
    assert summary["weeks"].tolist() == list(range(1, 22))
    assert summary["seed_ratios"].shape == (3, 21)
    assert summary["mean"][-1] == pytest.approx(100.0)
    assert summary["lower"][-1] == pytest.approx(90.0)
    assert summary["upper"][-1] == pytest.approx(110.0)


def test_cumulative_calibration_summary_rejects_misaligned_weekly_arrays():
    with pytest.raises(ValueError, match="length mismatch"):
        _cumulative_calibration_summary(
            np.ones(4),
            np.ones((3, 5)),
        )


def test_holdout_week_ticks_include_both_horizon_endpoints():
    assert _holdout_week_ticks(5).tolist() == [1, 2, 3, 4, 5]
    ticks = _holdout_week_ticks(39)
    assert ticks[0] == 1
    assert ticks[-1] == 39
    assert len(ticks) <= 6
