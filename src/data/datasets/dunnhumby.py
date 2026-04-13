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
        raw_dir = config.get("raw_dir", "data/raw/dunnhumby")
        calib_weeks = config["dataset"]["calibration_weeks"]
        holdout_weeks = config["dataset"]["holdout_weeks"]
        lookback = config["dataset"]["lookback_weeks"]
        min_active = config["dataset"].get("min_active_weeks", 5)
        val_fraction = config["dataset"].get("val_fraction", 0.1)
        include_covariates = config["dataset"].get("include_covariates", False)

        df = self.load_raw(raw_dir)
        df = self.clean(df)

        agg = WeeklyAggregator().fit_transform(df)
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib, holdout = splitter.split(agg)

        scaler = SpendScaler(scale=True)
        calib = calib.copy()
        holdout = holdout.copy()
        calib["log_spend"] = scaler.fit_transform(calib["weekly_spend"].values)
        holdout["log_spend"] = scaler.transform(holdout["weekly_spend"].values)

        builder = SequenceBuilder(lookback=lookback, min_active_weeks=min_active)
        X, y_freq, y_spend, cids, mask = builder.build(calib)

        covariates = None
        if include_covariates:
            covariates = self.build_covariates(raw_dir, calib_weeks)

        n = len(X)
        n_val = int(n * val_fraction)
        idx = list(range(n))

        train_ds = CustomerDataset(X[idx[:-n_val]], y_freq[idx[:-n_val]],
                                   y_spend[idx[:-n_val]], cids[idx[:-n_val]],
                                   mask[idx[:-n_val]], covariates)
        val_ds = CustomerDataset(X[idx[-n_val:]], y_freq[idx[-n_val:]],
                                 y_spend[idx[-n_val:]], cids[idx[-n_val:]],
                                 mask[idx[-n_val:]], covariates)

        X_h, y_freq_h, y_spend_h, cids_h, mask_h = builder.build(holdout)
        test_ds = CustomerDataset(X_h, y_freq_h, y_spend_h, cids_h, mask_h, covariates)

        return train_ds, val_ds, test_ds

    def load_raw(self, raw_dir: str) -> pd.DataFrame:
        path = os.path.join(raw_dir, "transaction_data.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dunnhumby transaction_data.csv not found at {path}. "
                "Download from: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey"
            )
        return pd.read_csv(path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["household_id", "sales_value"])
        df = df[df["sales_value"] > 0]
        # Dunnhumby uses 'day' (integer day number); convert to pseudo-date
        df["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["day"], unit="D")
        df["customer_id"] = df["household_id"].astype(int)
        df["transaction_amount"] = df["sales_value"].astype(float)
        return df[["customer_id", "date", "transaction_amount"]]

    def build_covariates(self, raw_dir: str, calibration_weeks: int) -> np.ndarray:
        """
        Build static covariate matrix for Extension 3.

        Returns: numpy array of shape (N_customers, C)
        Covariates: [income_encoded, household_size_encoded,
                     coupon_redemptions_total_in_calib,
                     campaign_exposure_count_in_calib]
        """
        # Load demographic data
        demo_path = os.path.join(raw_dir, "hh_demographic.csv")
        demo = pd.read_csv(demo_path) if os.path.exists(demo_path) else None

        # Load coupon redemptions
        coupon_path = os.path.join(raw_dir, "coupon_redempt.csv")
        coupons = pd.read_csv(coupon_path) if os.path.exists(coupon_path) else None

        # TODO: Encode demographics (ordinal) + aggregate coupon redemptions per customer
        # over calibration period only. Return numpy array.
        raise NotImplementedError(
            "Implement DunnhumbyPipeline.build_covariates() for Extension 3. "
            "See CLAUDE.md Section 5 for covariate spec."
        )
