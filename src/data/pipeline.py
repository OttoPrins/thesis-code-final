"""
Abstract data pipeline. Each dataset implements this interface.

Usage:
    from src.data.datasets.cdnow import CDNOWPipeline
    train_ds, val_ds, test_ds = CDNOWPipeline().run(config)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from src.data.transforms import WeeklyAggregator, TemporalSplitter


class BasePipeline(ABC):
    """
    Base class for all dataset pipelines.

    Subclasses override load_raw(), clean(), and optionally build_covariates().
    The run() method orchestrates the full pipeline using shared transforms.
    """

    def run(self, config: dict) -> Tuple:
        """
        Full pipeline: raw data → PyTorch datasets + holdout ground truth.

        Args:
            config: Loaded YAML config dict (see experiments/configs/).

        Returns:
            (train_ds, val_ds, inference_ds, holdout_ground_truth, scaler)

            train_ds:             CustomerDataset for training (no seed sequences)
            val_ds:               CustomerDataset for validation (with seed for inference)
            inference_ds:         CustomerDataset with all customers + seeds (for holdout eval)
            holdout_ground_truth: dict with customer_ids, freq, spend arrays for holdout period
            scaler:               fitted SpendScaler (for inverse-transforming predictions)
        """
        raise NotImplementedError

    @abstractmethod
    def load_raw(self, raw_dir: str):
        """Read original files from data/raw/<dataset>/. Return a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def clean(self, df):
        """Remove bad rows, enforce dtypes, standardise column names.

        Output columns must be: [customer_id, date, transaction_amount]
        """
        raise NotImplementedError

    def build_covariates(self, df):
        """Optional. Build static per-customer covariate matrix.

        Only implemented by DunnhumbyPipeline for Extension 3.
        Returns None by default.
        """
        return None

    def get_rfm_summary(self, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produce RFM summary for probabilistic benchmarks.
        Runs the pipeline up to temporal split (no SequenceBuilder).

        Returns:
            (rfm_calib, rfm_holdout) sorted by customer_id with identical customer sets.

            rfm_calib columns:
              customer_id, frequency, recency, T, monetary_value, litt
            rfm_holdout columns:
              customer_id, actual_freq, actual_spend

        Customers in calibration who do not appear in holdout get actual_freq=0,
        actual_spend=0.0. This preserves the N-customer alignment.
        """
        raw_dir = config["dataset"].get("raw_dir", "data/raw")
        raw_df = self.load_raw(raw_dir)
        clean_df = self.clean(raw_df)

        # Weekly aggregation
        agg = WeeklyAggregator()
        agg_df = agg.fit_transform(clean_df)

        # Temporal split
        calib_weeks = config["dataset"]["calibration_weeks"]
        holdout_weeks = config["dataset"]["holdout_weeks"]
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib_df, holdout_df = splitter.split(agg_df)

        # Compute RFM from calibration (vectorised)
        rfm_calib = self._compute_rfm_calib(calib_df, calib_weeks)

        # Compute holdout ground truth; zero-fill for customers absent in holdout
        rfm_holdout = self._compute_rfm_holdout(holdout_df, rfm_calib["customer_id"])

        # Both sorted by customer_id for guaranteed alignment
        rfm_calib = rfm_calib.sort_values("customer_id").reset_index(drop=True)
        rfm_holdout = rfm_holdout.sort_values("customer_id").reset_index(drop=True)

        assert np.array_equal(
            rfm_calib["customer_id"].values, rfm_holdout["customer_id"].values
        ), "Bug in get_rfm_summary: customer_id sets differ after zero-fill"

        return rfm_calib, rfm_holdout

    @staticmethod
    def _compute_rfm_calib(calib_df: pd.DataFrame, calib_weeks: int) -> pd.DataFrame:
        """
        Vectorised RFM computation from calibration weekly data.

        lifetimes conventions (all in weeks):
          frequency     = sum(weekly_freq) - 1  (repeat transactions, excluding first)
          recency       = last_active_week - first_active_week
          T             = (calib_weeks - 1) - first_active_week
          monetary_value = total_spend / total_transactions  (average spend per txn)
          litt           = log(recency / frequency) if frequency > 0 else 0.0
                           (approximation of mean log inter-transaction time for BTYD.plus)
        """
        agg = calib_df.groupby("customer_id").agg(
            first_week=("week", "min"),
            last_week=("week", "max"),
            total_freq=("weekly_freq", "sum"),
            total_spend=("weekly_spend", "sum"),
        ).reset_index()

        agg["frequency"] = (agg["total_freq"] - 1).clip(lower=0).astype(int)
        agg["recency"] = agg["last_week"] - agg["first_week"]
        agg["T"] = (calib_weeks - 1) - agg["first_week"]
        agg["monetary_value"] = np.where(
            agg["total_freq"] > 0,
            agg["total_spend"] / agg["total_freq"],
            0.0,
        )
        # litt: log of mean inter-transaction time; approximated from aggregate data.
        # For customers with >= 2 purchases: litt = log(recency / frequency).
        # For one-timers (frequency == 0): litt = 0.0 (no inter-transaction times).
        # This matches the approximation used in BTYD.plus documentation examples.
        agg["litt"] = np.where(
            agg["frequency"] > 0,
            np.log(np.where(agg["recency"] > 0, agg["recency"], 1.0) / agg["frequency"].clip(lower=1)),
            0.0,
        )

        return agg[["customer_id", "frequency", "recency", "T", "monetary_value", "litt"]]

    @staticmethod
    def _compute_rfm_holdout(
        holdout_df: pd.DataFrame,
        calib_customer_ids: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute actual holdout transactions and spend per customer.
        Zero-fills calibration customers absent from holdout (they bought 0 times).
        """
        if len(holdout_df) > 0:
            holdout_agg = (
                holdout_df.groupby("customer_id")
                .agg(actual_freq=("weekly_freq", "sum"), actual_spend=("weekly_spend", "sum"))
                .reset_index()
            )
        else:
            holdout_agg = pd.DataFrame(
                columns=["customer_id", "actual_freq", "actual_spend"]
            )

        # Left-join from calibration customers so every calibration customer has a row
        base = pd.DataFrame({"customer_id": calib_customer_ids})
        merged = base.merge(holdout_agg, on="customer_id", how="left")
        merged["actual_freq"] = merged["actual_freq"].fillna(0).astype(int)
        merged["actual_spend"] = merged["actual_spend"].fillna(0.0).astype(float)
        return merged[["customer_id", "actual_freq", "actual_spend"]]


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)
