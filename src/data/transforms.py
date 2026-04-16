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
    Builds full-history, teacher-forced sequences for seq-to-seq training.

    Matches Valendin et al. (2022) reference implementation:
    - One sequence per customer covering the entire calibration period.
    - Dense weekly grid: inactive weeks filled with (freq=0, spend=0.0).
    - Teacher-forcing shift: input = steps 0..T-2, target = steps 1..T-1.
    - Full unshifted sequence stored as "seed" for autoregressive inference.

    Input: calibration DataFrame with columns [customer_id, week, weekly_freq, weekly_spend]
           (weekly_spend already log-transformed by SpendScaler as 'log_spend')
    Output: dict of numpy arrays ready for CustomerDataset.
    """

    def __init__(self, calibration_weeks: int = 52, min_active_weeks: int = 5,
                 freq_bins: list = None):
        self.calibration_weeks = calibration_weeks
        self.min_active_weeks = min_active_weeks
        self.freq_bins = freq_bins or [0, 1, 2, 3]
        self.max_trans = self.freq_bins[-1]  # e.g. 3 for [0,1,2,3]

    def build(self, df: pd.DataFrame) -> dict:
        """
        Build training sequences from aggregated calibration data.

        Args:
            df: DataFrame with columns [customer_id, week, weekly_freq, log_spend]

        Returns dict with keys:
            week_input   : (N, T-1) int32   — week indices for input steps
            trans_input  : (N, T-1) int32   — transaction counts (teacher forcing)
            spend_input  : (N, T-1) float32 — log-spend at each input step
            y_freq       : (N, T-1) int32   — target freq class, shifted +1
            y_spend      : (N, T-1) float32 — target log-spend, shifted +1
            customer_ids : (N,)     int64   — customer identifiers
            mask         : (N, T-1) float32 — 1=real, 0=padding
            seed_week    : (N, T)   int32   — full calibration weeks (inference)
            seed_trans   : (N, T)   int32   — full calibration trans (inference)
            seed_spend   : (N, T)   float32 — full calibration spend (inference)
            max_trans    : int              — clipping value used
        """
        T = self.calibration_weeks
        spend_col = "log_spend" if "log_spend" in df.columns else "weekly_spend"

        # Build dense grid: every customer gets a row for every week 0..T-1
        customers = df["customer_id"].unique()

        # Count active weeks per customer and filter
        active_counts = (
            df[df["weekly_freq"] > 0]
            .groupby("customer_id")["week"]
            .nunique()
        )
        valid_customers = active_counts[
            active_counts >= self.min_active_weeks
        ].index
        customers = np.intersect1d(customers, valid_customers)

        # Pre-allocate arrays
        N = len(customers)
        full_weeks = np.tile(np.arange(T, dtype=np.int32), (N, 1))  # (N, T)
        full_trans = np.zeros((N, T), dtype=np.int32)
        full_spend = np.zeros((N, T), dtype=np.float32)

        # Fill in actual transaction data per customer
        # Index df by (customer_id, week) for fast lookup
        indexed = df.set_index(["customer_id", "week"])

        for i, cid in enumerate(customers):
            if cid not in indexed.index.get_level_values(0):
                continue
            cust_data = indexed.loc[cid]
            if isinstance(cust_data, pd.Series):
                # Single week — wrap into DataFrame
                cust_data = cust_data.to_frame().T
            for week_val, row in cust_data.iterrows():
                if 0 <= week_val < T:
                    freq = int(row["weekly_freq"])
                    full_trans[i, week_val] = min(freq, self.max_trans)
                    full_spend[i, week_val] = float(row[spend_col])

        # Teacher-forcing shift: input = [0..T-2], target = [1..T-1]
        week_input = full_weeks[:, :-1]       # (N, T-1)
        trans_input = full_trans[:, :-1]       # (N, T-1)
        spend_input = full_spend[:, :-1]       # (N, T-1)
        y_freq = full_trans[:, 1:]             # (N, T-1)
        y_spend = full_spend[:, 1:]            # (N, T-1)

        # Mask: all ones for fixed calibration period (every week is a valid observation)
        mask = np.ones((N, T - 1), dtype=np.float32)

        customer_ids = np.array(customers, dtype=np.int64)

        return {
            "week_input": week_input,
            "trans_input": trans_input,
            "spend_input": spend_input,
            "y_freq": y_freq,
            "y_spend": y_spend,
            "customer_ids": customer_ids,
            "mask": mask,
            "seed_week": full_weeks,
            "seed_trans": full_trans,
            "seed_spend": full_spend,
            "max_trans": self.max_trans,
        }
