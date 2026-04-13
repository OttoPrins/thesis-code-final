"""
Shared data transforms applied identically across all datasets.

Pipeline:
    WeeklyAggregator → TemporalSplitter → Scaler → SequenceBuilder

Design constraints (from CLAUDE.md):
    - Weekly aggregation matches Valendin et al. (2022)
    - Frequency discretised to {0, 1, 2, 3+} (4 classes; 3 means 3 or more)
    - Spend always log1p-transformed
    - Scaler fitted on calibration set ONLY (no holdout leakage)
    - Sequences padded with zeros; mask marks padding positions
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class WeeklyAggregator:
    """
    Aggregates transaction-level data into weekly customer summaries.

    Input DataFrame columns: [customer_id, date, transaction_amount]
    Output DataFrame columns: [customer_id, week, weekly_freq, weekly_spend]

    Week numbering: starts at 0 from the first transaction in the dataset.
    """

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        df["week"] = ((df["date"] - min_date).dt.days // 7).astype(int)

        agg = (
            df.groupby(["customer_id", "week"])
            .agg(weekly_freq=("transaction_amount", "count"),
                 weekly_spend=("transaction_amount", "sum"))
            .reset_index()
        )
        return agg

    @staticmethod
    def discretise_freq(freq_series: pd.Series, bins: list = None) -> pd.Series:
        """Map raw frequency to class label {0, 1, 2, 3} where 3 means 3+."""
        if bins is None:
            bins = [0, 1, 2, 3]
        return freq_series.clip(upper=bins[-1]).astype(int)


class TemporalSplitter:
    """
    Splits aggregated weekly data into calibration and holdout by week number.

    Calibration: [0, calibration_weeks)
    Holdout: [calibration_weeks, calibration_weeks + holdout_weeks)

    Note: The scaler MUST be fitted only on calibration data before transforming holdout.
    """

    def __init__(self, calibration_weeks: int, holdout_weeks: int):
        self.calibration_weeks = calibration_weeks
        self.holdout_weeks = holdout_weeks

    def split(self, df: pd.DataFrame):
        calib = df[df["week"] < self.calibration_weeks].copy()
        holdout = df[
            (df["week"] >= self.calibration_weeks) &
            (df["week"] < self.calibration_weeks + self.holdout_weeks)
        ].copy()
        return calib, holdout


class SpendScaler:
    """
    Applies log1p to spend, then optionally MinMax-scales to [0, 1].

    Fitted on calibration set only. Applied to both calibration and holdout.
    """

    def __init__(self, scale: bool = True):
        self.scale = scale
        self._scaler = MinMaxScaler() if scale else None

    def fit_transform(self, spend: np.ndarray) -> np.ndarray:
        log_spend = np.log1p(spend)
        if self.scale:
            log_spend = self._scaler.fit_transform(log_spend.reshape(-1, 1)).flatten()
        return log_spend

    def transform(self, spend: np.ndarray) -> np.ndarray:
        log_spend = np.log1p(spend)
        if self.scale and self._scaler is not None:
            log_spend = self._scaler.transform(log_spend.reshape(-1, 1)).flatten()
        return log_spend

    def inverse_transform_spend(self, log_spend: np.ndarray) -> np.ndarray:
        if self.scale and self._scaler is not None:
            log_spend = self._scaler.inverse_transform(log_spend.reshape(-1, 1)).flatten()
        return np.expm1(log_spend)


class SequenceBuilder:
    """
    Builds (X, y_freq, y_spend, customer_id) sliding-window tensors.

    For each customer, slides a window of length T over their weekly history.
    Label is the week AFTER the window (next-step prediction).

    Input: calibration DataFrame with columns [customer_id, week, weekly_freq, weekly_spend]
           (spend already log-transformed by SpendScaler)
    Output: numpy arrays ready for CustomerDataset.
    """

    def __init__(self, lookback: int = 52, min_active_weeks: int = 5,
                 freq_bins: list = None):
        self.lookback = lookback
        self.min_active_weeks = min_active_weeks
        self.freq_bins = freq_bins or [0, 1, 2, 3]

    def build(self, df: pd.DataFrame):
        """
        Returns:
            X           : (N, T, 2) — [weekly_freq_norm, log_spend] per week
            y_freq      : (N,)       — next-period frequency class {0,1,2,3}
            y_spend     : (N,)       — next-period log-spend
            customer_ids: (N,)       — customer_id for each window
            mask        : (N, T)     — 1=real data, 0=padding
        """
        # TODO: implement sliding window construction
        # Hint: for each customer, create a dense week grid (0..max_week),
        # fill missing weeks with (freq=0, spend=0), then slide window.
        raise NotImplementedError(
            "Implement SequenceBuilder.build() — see CLAUDE.md Section 6 for spec."
        )
