"""
UCI Online Retail II dataset pipeline.

Dataset: UK e-commerce transactions, 2009-2011.
         ~4,300 active customers after cleaning.

Source: UCI Machine Learning Repository
        https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
Expected raw file: data/raw/uci_retail/online_retail_II.xlsx (two sheets: 2009-2010, 2010-2011)

Cleaning rules:
    - Drop rows where CustomerID is NaN
    - Drop cancelled invoices (InvoiceNo starts with 'C')
    - Drop rows with Quantity <= 0 or UnitPrice <= 0
    - transaction_amount = Quantity * UnitPrice

See CLAUDE.md Section 5 for full dataset description.
"""

from __future__ import annotations
import os
import pandas as pd
from src.data.pipeline import BasePipeline
from src.data.transforms import WeeklyAggregator, TemporalSplitter, SpendScaler, SequenceBuilder
from src.data.dataset import CustomerDataset


class UCIRetailPipeline(BasePipeline):

    def run(self, config: dict):
        raw_dir = config.get("raw_dir", "data/raw/uci_retail")
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
        path_xlsx = os.path.join(raw_dir, "online_retail_II.xlsx")
        if not os.path.exists(path_xlsx):
            raise FileNotFoundError(
                f"UCI Online Retail II not found at {path_xlsx}. "
                "Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II"
            )
        # File has two sheets; load both and concatenate
        df1 = pd.read_excel(path_xlsx, sheet_name="Year 2009-2010", engine="openpyxl")
        df2 = pd.read_excel(path_xlsx, sheet_name="Year 2010-2011", engine="openpyxl")
        return pd.concat([df1, df2], ignore_index=True)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["Customer ID"])
        df = df[~df["Invoice"].astype(str).str.startswith("C")]
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
        df["transaction_amount"] = df["Quantity"] * df["Price"]
        df["customer_id"] = df["Customer ID"].astype(int)
        df["date"] = pd.to_datetime(df["InvoiceDate"])
        return df[["customer_id", "date", "transaction_amount"]]
