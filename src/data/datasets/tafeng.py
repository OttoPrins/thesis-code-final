"""
Ta-Feng Grocery dataset pipeline.

Dataset: Taiwanese grocery retail, 2000-2001.
         ~32,000 customers, ~817,741 transactions.
         High-frequency dataset — the proposal expects Transformer to outperform LSTM here.

Source: Kaggle (IJCAI-15 competition)
        https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset
Expected raw file: data/raw/tafeng/ta_feng_all_months_merged.csv

Format: date, customer_id, age_group, pin_code, product_subclass, product_id,
        amount, asset, sales_price
where: amount = number of items, sales_price = transaction value

See CLAUDE.md Section 5 for full dataset description.
"""

from __future__ import annotations
import os
import pandas as pd
from src.data.pipeline import BasePipeline
from src.data.transforms import WeeklyAggregator, TemporalSplitter, SpendScaler, SequenceBuilder
from src.data.dataset import CustomerDataset


class TaFengPipeline(BasePipeline):

    def run(self, config: dict):
        raw_dir = config.get("raw_dir", "data/raw/tafeng")
        calib_weeks = config["dataset"]["calibration_weeks"]
        holdout_weeks = config["dataset"]["holdout_weeks"]
        lookback = config["dataset"]["lookback_weeks"]
        min_active = config["dataset"].get("min_active_weeks", 5)
        val_fraction = config["dataset"].get("val_fraction", 0.1)

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

        n = len(X)
        n_val = int(n * val_fraction)
        idx = list(range(n))

        train_ds = CustomerDataset(X[idx[:-n_val]], y_freq[idx[:-n_val]],
                                   y_spend[idx[:-n_val]], cids[idx[:-n_val]],
                                   mask[idx[:-n_val]])
        val_ds = CustomerDataset(X[idx[-n_val:]], y_freq[idx[-n_val:]],
                                 y_spend[idx[-n_val:]], cids[idx[-n_val:]],
                                 mask[idx[-n_val:]])

        X_h, y_freq_h, y_spend_h, cids_h, mask_h = builder.build(holdout)
        test_ds = CustomerDataset(X_h, y_freq_h, y_spend_h, cids_h, mask_h)

        return train_ds, val_ds, test_ds

    def load_raw(self, raw_dir: str) -> pd.DataFrame:
        path = os.path.join(raw_dir, "ta_feng_all_months_merged.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Ta-Feng data not found at {path}. "
                "Download from: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset"
            )
        return pd.read_csv(path)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().upper() for c in df.columns]
        df = df.dropna(subset=["CUSTOMER_ID", "SALES_PRICE"])
        df = df[df["SALES_PRICE"] > 0]
        df["transaction_amount"] = df["SALES_PRICE"].astype(float)
        df["customer_id"] = df["CUSTOMER_ID"].astype(int)
        df["date"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df[["customer_id", "date", "transaction_amount"]]
