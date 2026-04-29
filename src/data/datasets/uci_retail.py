"""
UCI Online Retail II dataset pipeline.

Dataset: UK e-commerce transactions, 2009-2011.
         ~4,300 active customers after cleaning.

Source: UCI Machine Learning Repository
        https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
Expected raw file: data/raw/Online Retail.csv

Cleaning rules:
    - Drop rows where Customer ID is NaN
    - Drop cancelled invoices (Invoice starts with 'C')
    - Drop rows with Quantity <= 0 or Price <= 0
    - transaction_amount = Quantity * Price

See CLAUDE.md Section 5 for full dataset description.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from src.data.pipeline import BasePipeline
from src.data.transforms import (
    WeeklyAggregator,
    TemporalSplitter,
    SpendScaler,
    SequenceBuilder,
    resolve_freq_bins,
)
from src.data.dataset import CustomerDataset


class UCIRetailPipeline(BasePipeline):

    def run(self, config: dict):
        """
        Full pipeline: raw UCI data → PyTorch datasets.

        Returns:
            train_ds:             CustomerDataset for training
            val_ds:               CustomerDataset for validation (with seed for inference)
            inference_ds:         CustomerDataset with all customers + seeds (for holdout inference)
            holdout_ground_truth: dict mapping customer_id → (freq_array, spend_array) for holdout
            scaler:               fitted SpendScaler
        """
        dataset_cfg = config["dataset"]
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        calib_weeks = dataset_cfg["calibration_weeks"]
        holdout_weeks = dataset_cfg["holdout_weeks"]
        min_active = dataset_cfg.get("min_active_weeks", 5)
        val_fraction = dataset_cfg.get("val_fraction", 0.1)
        seed = config.get("training", {}).get("seed", 42)

        df = self.load_raw(raw_dir)
        df = self.clean(df)

        agg = WeeklyAggregator().fit_transform(df)
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib, holdout = splitter.split(agg)

        scaler = SpendScaler()
        calib = calib.copy()
        calib["log_spend"] = scaler.fit_transform(calib["weekly_spend"].values)

        freq_bins = resolve_freq_bins(dataset_cfg, calib["weekly_freq"].values)
        config.setdefault("model", {})["max_trans"] = freq_bins[-1]
        builder = SequenceBuilder(
            calibration_weeks=calib_weeks,
            min_active_weeks=min_active,
            freq_bins=freq_bins,
        )
        data = builder.build(calib)

        n = len(data["customer_ids"])
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_data = _subset_data(data, train_idx)
        val_data = _subset_data(data, val_idx)

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
        """Load UCI Online Retail II from CSV."""
        candidates = [
            "Online Retail.csv",
            "Online_Retail.csv",
            "online_retail.csv",
        ]
        path = None
        for name in candidates:
            candidate = os.path.join(raw_dir, name)
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                f"UCI Online Retail II not found in {raw_dir}. "
                f"Tried: {candidates}. "
                f"Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II"
            )
        # File uses semicolon separator; UnitPrice uses comma as decimal
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["CustomerID"])
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        # UnitPrice stored as European decimal (e.g. "2,55") — normalize to float
        df["UnitPrice"] = (
            df["UnitPrice"].astype(str).str.replace(",", ".", regex=False).astype(float)
        )
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["transaction_amount"] = df["Quantity"].astype(float) * df["UnitPrice"]
        df["customer_id"] = df["CustomerID"].astype(int)
        df["date"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        # Aggregate to invoice level so weekly_freq counts shopping trips, not line items
        df = (
            df.groupby(["customer_id", "InvoiceNo", "date"])
            .agg(transaction_amount=("transaction_amount", "sum"))
            .reset_index()
        )
        return df[["customer_id", "date", "transaction_amount"]]

    @staticmethod
    def _build_holdout_ground_truth(
        holdout_df: pd.DataFrame,
        calibration_customer_ids: np.ndarray,
        calib_weeks: int,
        holdout_weeks: int,
        max_trans: int,
    ) -> dict:
        """Build holdout ground truth arrays for evaluation.

        Stores both clipped freq (for per-step comparison) and unclipped
        raw_freq (for aggregate metrics). total_freq uses unclipped counts
        for fair comparison with probabilistic benchmarks.
        """
        N = len(calibration_customer_ids)
        H = holdout_weeks
        freq = np.zeros((N, H), dtype=np.int32)
        raw_freq = np.zeros((N, H), dtype=np.int32)
        spend = np.zeros((N, H), dtype=np.float32)

        # Vectorized fill using advanced indexing
        cid_to_idx = {cid: i for i, cid in enumerate(calibration_customer_ids)}

        df = holdout_df.copy()
        df["_row"] = df["customer_id"].map(cid_to_idx)
        df["_col"] = (df["week"].astype(int) - calib_weeks)
        df = df.dropna(subset=["_row"])
        df = df[(df["_col"] >= 0) & (df["_col"] < H)]

        row_idx = df["_row"].values.astype(int)
        col_idx = df["_col"].values.astype(int)
        raw_vals = df["weekly_freq"].values.astype(np.int32)

        raw_freq[row_idx, col_idx] = raw_vals
        freq[row_idx, col_idx] = np.minimum(raw_vals, max_trans)
        spend[row_idx, col_idx] = df["log_spend"].values.astype(np.float32)

        return {
            "customer_ids": calibration_customer_ids.copy(),
            "freq": freq,
            "raw_freq": raw_freq,
            "spend": spend,
            "total_freq": raw_freq.sum(axis=1).astype(np.int32),
        }


def _subset_data(data: dict, indices: np.ndarray) -> dict:
    """Subset a SequenceBuilder output dict by row indices."""
    return {
        k: v[indices] if isinstance(v, np.ndarray) else v
        for k, v in data.items()
    }
