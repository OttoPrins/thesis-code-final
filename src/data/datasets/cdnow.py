"""
CDNOW dataset pipeline.

Dataset: CDNOW music retail transactions, 1997-1998.
         2,357 customers, ~69,659 transactions.
         Standard split: calibration=52 weeks, holdout=52 weeks.

Source: BTYD R package or Fader's Wharton page.
Expected raw file: data/raw/cdnow/CDNOW_sample.txt (or .csv)

Format: space/tab-separated
    Column 1: customer_id (integer)
    Column 2: date (YYYYMMDD or similar)
    Column 3: num_cds (count)
    Column 4: amount (USD float)

See CLAUDE.md Section 5 for full dataset description.
"""

from __future__ import annotations
import os
import pandas as pd
from src.data.pipeline import BasePipeline
from src.data.transforms import WeeklyAggregator, TemporalSplitter, SpendScaler, SequenceBuilder
from src.data.dataset import CustomerDataset


class CDNOWPipeline(BasePipeline):

    def run(self, config: dict):
        raw_dir = config.get("raw_dir", "data/raw/cdnow")
        calib_weeks = config["dataset"]["calibration_weeks"]
        holdout_weeks = config["dataset"]["holdout_weeks"]
        lookback = config["dataset"]["lookback_weeks"]
        min_active = config["dataset"].get("min_active_weeks", 5)
        val_fraction = config["dataset"].get("val_fraction", 0.1)

        # Stage 1: Load and clean
        df = self.load_raw(raw_dir)
        df = self.clean(df)

        # Stage 2: Aggregate to weekly level
        agg = WeeklyAggregator().fit_transform(df)

        # Stage 3: Temporal split
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib, holdout = splitter.split(agg)

        # Stage 4: Fit scaler on calibration, transform both
        scaler = SpendScaler(scale=True)
        calib = calib.copy()
        holdout = holdout.copy()
        calib["log_spend"] = scaler.fit_transform(calib["weekly_spend"].values)
        holdout["log_spend"] = scaler.transform(holdout["weekly_spend"].values)

        # Stage 5: Build sequences
        builder = SequenceBuilder(lookback=lookback, min_active_weeks=min_active)
        X, y_freq, y_spend, cids, mask = builder.build(calib)

        # Stage 6: Train/val split (within calibration)
        n = len(X)
        n_val = int(n * val_fraction)
        idx = list(range(n))

        train_ds = CustomerDataset(X[idx[:-n_val]], y_freq[idx[:-n_val]],
                                   y_spend[idx[:-n_val]], cids[idx[:-n_val]],
                                   mask[idx[:-n_val]])
        val_ds = CustomerDataset(X[idx[-n_val:]], y_freq[idx[-n_val:]],
                                 y_spend[idx[-n_val:]], cids[idx[-n_val:]],
                                 mask[idx[-n_val:]])

        # Holdout dataset
        X_h, y_freq_h, y_spend_h, cids_h, mask_h = builder.build(holdout)
        test_ds = CustomerDataset(X_h, y_freq_h, y_spend_h, cids_h, mask_h)

        return train_ds, val_ds, test_ds

    def load_raw(self, raw_dir: str) -> pd.DataFrame:
        """
        Load CDNOW raw file. Try common file names.
        Expected columns after loading: customer_id, date, num_cds, amount
        """
        candidates = [
            "CDNOW_sample.txt",
            "CDNOW_sample.csv",
            "cdnow.csv",
            "cdnow.txt",
        ]
        path = None
        for name in candidates:
            candidate = os.path.join(raw_dir, name)
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                f"CDNOW raw file not found in {raw_dir}. "
                f"Download from BTYD R package or Fader's Wharton page."
            )

        # CDNOW format is typically whitespace-separated, no header
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=["customer_id", "date", "num_cds", "amount"])
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d",
                                    errors="coerce")
        df = df.dropna(subset=["date", "customer_id", "amount"])
        df = df[df["amount"] > 0]
        df = df.rename(columns={"amount": "transaction_amount"})
        df["customer_id"] = df["customer_id"].astype(int)
        return df[["customer_id", "date", "transaction_amount"]]
