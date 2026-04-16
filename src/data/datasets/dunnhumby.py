"""
Dunnhumby 'The Complete Journey' dataset pipeline.

Dataset: US grocery + coupon data, 2 years, ~2,500 households.
         Primary dataset for Extension 3 (covariate ablation).

Source: Dunnhumby website or Kaggle
        https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey
Expected raw files in data/raw/dunnhumby/:
    - transaction_data.csv
    - hh_demographic.csv
    - campaign_table.csv
    - coupon_redempt.csv

Covariates for Extension 3:
    Demographics: income_desc (ordinal encode), household_size_desc (ordinal encode)
    Campaign exposure: coupon_redemptions_per_week (count), campaign_active_flag (binary)

See CLAUDE.md Section 5 for full dataset description.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from src.data.pipeline import BasePipeline
from src.data.transforms import WeeklyAggregator, TemporalSplitter, SpendScaler, SequenceBuilder
from src.data.dataset import CustomerDataset

# Ordinal encoding order for income (low → high)
INCOME_ORDER = [
    "Under 15K", "15-24K", "25-34K", "35-49K",
    "50-74K", "75-99K", "100-124K", "125-149K",
    "150-174K", "175-199K", "200-249K", "250K+"
]

# Ordinal encoding for household size (small → large)
HOUSEHOLD_SIZE_ORDER = ["1", "2", "3", "4", "5+"]


class DunnhumbyPipeline(BasePipeline):

    def run(self, config: dict):
        """
        Full pipeline: raw Dunnhumby data → PyTorch datasets.

        Returns:
            train_ds:             CustomerDataset for training
            val_ds:               CustomerDataset for validation (with seed for inference)
            inference_ds:         CustomerDataset with all customers + seeds (for holdout inference)
            holdout_ground_truth: dict mapping customer_id → (freq_array, spend_array) for holdout
            scaler:               fitted SpendScaler
        """
        dataset_cfg = config["dataset"]
        raw_dir = dataset_cfg.get("raw_dir", "data/raw/Dunnhumby datasets")
        calib_weeks = dataset_cfg["calibration_weeks"]
        holdout_weeks = dataset_cfg["holdout_weeks"]
        min_active = dataset_cfg.get("min_active_weeks", 5)
        val_fraction = dataset_cfg.get("val_fraction", 0.1)
        freq_bins = dataset_cfg.get("freq_bins", [0, 1, 2, 3])
        include_covariates = dataset_cfg.get("include_covariates", False)
        seed = config.get("training", {}).get("seed", 42)

        df = self.load_raw(raw_dir)
        df = self.clean(df)

        agg = WeeklyAggregator().fit_transform(df)
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib, holdout = splitter.split(agg)

        scaler = SpendScaler(scale=True)
        calib = calib.copy()
        calib["log_spend"] = scaler.fit_transform(calib["weekly_spend"].values)

        builder = SequenceBuilder(
            calibration_weeks=calib_weeks,
            min_active_weeks=min_active,
            freq_bins=freq_bins,
        )
        data = builder.build(calib)

        # Build covariates if requested (Extension 3)
        covariates = None
        if include_covariates:
            covariates = self.build_covariates(
                raw_dir, data["customer_ids"], calib_weeks
            )

        n = len(data["customer_ids"])
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_data = _subset_data(data, train_idx)
        val_data = _subset_data(data, val_idx)

        # Add covariates if they exist
        if covariates is not None:
            train_data["covariates"] = covariates[train_idx]
            val_data["covariates"] = covariates[val_idx]
            data["covariates"] = covariates

        train_ds = CustomerDataset(train_data, include_seed=False)
        val_ds = CustomerDataset(val_data, include_seed=True)
        inference_ds = CustomerDataset(data, include_seed=True)

        holdout = holdout.copy()
        holdout["log_spend"] = scaler.transform(holdout["weekly_spend"].values)
        holdout_ground_truth = self._build_holdout_ground_truth(
            holdout, data["customer_ids"], calib_weeks, holdout_weeks,
            max_trans=builder.max_trans,
        )

        return train_ds, val_ds, inference_ds, holdout_ground_truth, scaler

    def load_raw(self, raw_dir: str) -> pd.DataFrame:
        """Load Dunnhumby transaction data from CSV."""
        candidates = [
            "transaction_data.csv",
            "transactions.csv",
        ]
        path = None
        for name in candidates:
            candidate = os.path.join(raw_dir, name)
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                f"Dunnhumby transaction_data.csv not found in {raw_dir}. "
                f"Tried: {candidates}. "
                f"Download from: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey"
            )
        return pd.read_csv(path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Actual column is 'household_key', not 'household_id'
        df = df.dropna(subset=["household_key", "SALES_VALUE"])
        df = df[df["SALES_VALUE"] > 0]
        # Dunnhumby uses 'DAY' (integer day number); convert to pseudo-date
        df["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["DAY"], unit="D")
        df["customer_id"] = df["household_key"].astype(int)
        df["transaction_amount"] = df["SALES_VALUE"].astype(float)
        return df[["customer_id", "date", "transaction_amount"]]

    def build_covariates(
        self, raw_dir: str, calibration_customer_ids: np.ndarray,
        calibration_weeks: int
    ) -> np.ndarray:
        """
        Build static covariate matrix for Extension 3.

        Returns: numpy array of shape (N_customers, C)
        Covariates: [income_encoded, household_size_encoded,
                     coupon_redemptions_total_in_calib,
                     campaign_exposure_count_in_calib]
        """
        # TODO: Encode demographics (ordinal) + aggregate coupon redemptions per customer
        # over calibration period only. Return numpy array.
        raise NotImplementedError(
            "Implement DunnhumbyPipeline.build_covariates() for Extension 3. "
            "See CLAUDE.md Section 5 for covariate spec."
        )

    @staticmethod
    def _build_holdout_ground_truth(
        holdout_df: pd.DataFrame,
        calibration_customer_ids: np.ndarray,
        calib_weeks: int,
        holdout_weeks: int,
        max_trans: int,
    ) -> dict:
        """Build holdout ground truth arrays for evaluation."""
        N = len(calibration_customer_ids)
        H = holdout_weeks
        freq = np.zeros((N, H), dtype=np.int32)
        spend = np.zeros((N, H), dtype=np.float32)

        cid_to_idx = {cid: i for i, cid in enumerate(calibration_customer_ids)}

        for _, row in holdout_df.iterrows():
            cid = row["customer_id"]
            if cid not in cid_to_idx:
                continue
            idx = cid_to_idx[cid]
            week_offset = int(row["week"]) - calib_weeks
            if 0 <= week_offset < H:
                freq[idx, week_offset] = min(int(row["weekly_freq"]), max_trans)
                spend[idx, week_offset] = float(row["log_spend"])

        return {
            "customer_ids": calibration_customer_ids.copy(),
            "freq": freq,
            "spend": spend,
            "total_freq": freq.sum(axis=1).astype(np.int32),
            "total_spend": spend.sum(axis=1).astype(np.float32),
        }


def _subset_data(data: dict, indices: np.ndarray) -> dict:
    """Subset a SequenceBuilder output dict by row indices."""
    return {
        k: v[indices] if isinstance(v, np.ndarray) else v
        for k, v in data.items()
    }
