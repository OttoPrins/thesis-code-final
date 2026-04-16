"""
Ta-Feng Grocery dataset pipeline.

Dataset: Taiwanese grocery retail, 2000-2001.
         ~32,000 customers, ~817,741 transactions.
         High-frequency dataset — the proposal expects Transformer to outperform LSTM here.

Source: Kaggle (IJCAI-15 competition)
        https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset
Expected raw file: data/raw/ta_feng_all_months_merged.csv

Format: date, customer_id, age_group, pin_code, product_subclass, product_id,
        amount, asset, sales_price
where: amount = number of items, sales_price = transaction value

See CLAUDE.md Section 5 for full dataset description.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from src.data.pipeline import BasePipeline
from src.data.transforms import WeeklyAggregator, TemporalSplitter, SpendScaler, SequenceBuilder
from src.data.dataset import CustomerDataset


class TaFengPipeline(BasePipeline):

    def run(self, config: dict):
        """
        Full pipeline: raw Ta-Feng data → PyTorch datasets.

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
        freq_bins = dataset_cfg.get("freq_bins", [0, 1, 2, 3])
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
        """Load Ta-Feng data from CSV."""
        candidates = [
            "ta_feng_all_months_merged.csv",
            "ta_feng.csv",
            "tafeng.csv",
        ]
        path = None
        for name in candidates:
            candidate = os.path.join(raw_dir, name)
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                f"Ta-Feng data not found in {raw_dir}. "
                f"Tried: {candidates}. "
                f"Download from: https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset"
            )
        # BOM present — utf-8-sig strips it cleanly
        return pd.read_csv(path, encoding="utf-8-sig")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Strip whitespace and upper-case all column names (handles any BOM remnants)
        df.columns = [c.strip().lstrip("\ufeff").upper() for c in df.columns]
        df = df.dropna(subset=["CUSTOMER_ID", "SALES_PRICE"])
        df["SALES_PRICE"] = pd.to_numeric(df["SALES_PRICE"], errors="coerce")
        df = df[df["SALES_PRICE"] > 0]
        df["transaction_amount"] = df["SALES_PRICE"].astype(float)
        df["customer_id"] = df["CUSTOMER_ID"].astype(int)
        # Column is TRANSACTION_DT, not DATE
        df["date"] = pd.to_datetime(df["TRANSACTION_DT"], errors="coerce")
        df = df.dropna(subset=["date"])
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
            "total_spend": spend.sum(axis=1).astype(np.float32),
        }


def _subset_data(data: dict, indices: np.ndarray) -> dict:
    """Subset a SequenceBuilder output dict by row indices."""
    return {
        k: v[indices] if isinstance(v, np.ndarray) else v
        for k, v in data.items()
    }
