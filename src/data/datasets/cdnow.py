"""
CDNOW dataset pipeline.

Dataset: CDNOW music retail transactions, 1997-1998.
         2,357 customers, ~69,659 transactions.
         Standard split: calibration=52 weeks, holdout=52 weeks.

Source: BTYD R package or Fader's Wharton page.
Expected raw file: data/raw/CDNOW_master.txt (whitespace-separated, no header)

Format:
    Column 1: customer_id (integer)
    Column 2: date (YYYYMMDD)
    Column 3: num_cds (count)
    Column 4: amount (USD float)

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


class CDNOWPipeline(BasePipeline):

    def run(self, config: dict):
        """
        Full pipeline: raw CDNOW data → PyTorch datasets.

        Returns:
            train_ds:             CustomerDataset for training
            val_ds:               CustomerDataset for validation (with seed for inference)
            inference_ds:         CustomerDataset with all customers + seeds (for holdout inference)
            holdout_ground_truth: dict mapping customer_id → (freq_array, spend_array) for holdout
            scaler:               fitted SpendScaler (needed for inverse-transforming predictions)
        """
        dataset_cfg = config["dataset"]
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        calib_weeks = dataset_cfg["calibration_weeks"]
        holdout_weeks = dataset_cfg["holdout_weeks"]
        min_active = dataset_cfg.get("min_active_weeks", 5)
        val_fraction = dataset_cfg.get("val_fraction", 0.1)
        seed = config.get("training", {}).get("seed", 42)

        # Stage 1: Load and clean
        df = self.load_raw(raw_dir)
        df = self.clean(df)

        # Stage 2: Aggregate to weekly level
        agg = WeeklyAggregator().fit_transform(df)

        # Stage 3: Temporal split
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib, holdout = splitter.split(agg)

        # Stage 4: Scale spend (fit on calibration only — no leakage)
        scaler = SpendScaler()
        calib = calib.copy()
        calib["log_spend"] = scaler.fit_transform(calib["weekly_spend"].values)

        # Stage 5: Resolve freq bins from data (or YAML override) and build sequences
        freq_bins = resolve_freq_bins(dataset_cfg, calib["weekly_freq"].values)
        config.setdefault("model", {})["max_trans"] = freq_bins[-1]
        builder = SequenceBuilder(
            calibration_weeks=calib_weeks,
            min_active_weeks=min_active,
            freq_bins=freq_bins,
        )
        data = builder.build(calib)

        # Stage 6: Train/val split by customer (shuffled, seeded)
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

        # Stage 7: Build holdout ground truth for evaluation
        holdout = holdout.copy()
        holdout["log_spend"] = scaler.transform(holdout["weekly_spend"].values)
        holdout_ground_truth = self._build_holdout_ground_truth(
            holdout, data["customer_ids"], calib_weeks, holdout_weeks,
            max_trans=builder.max_trans,
        )

        return train_ds, val_ds, inference_ds, holdout_ground_truth, scaler

    def load_raw(self, raw_dir: str) -> pd.DataFrame:
        """Load CDNOW raw file. Searches common filenames in raw_dir."""
        candidates = [
            "CDNOW_master.txt",
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
                f"Tried: {candidates}. "
                f"Download from BTYD R package or Fader's Wharton page."
            )

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

    @staticmethod
    def _build_holdout_ground_truth(
        holdout_df: pd.DataFrame,
        calibration_customer_ids: np.ndarray,
        calib_weeks: int,
        holdout_weeks: int,
        max_trans: int,
    ) -> dict:
        """
        Build holdout ground truth arrays for evaluation.

        Returns dict with:
            customer_ids : (N,)     int64   — same order as calibration
            freq         : (N, H)   int32   — weekly transaction count (clipped at max_trans)
            raw_freq     : (N, H)   int32   — weekly transaction count (unclipped)
            spend        : (N, H)   float32 — per-week scaled log-spend (for DL eval)
            total_freq   : (N,)     int32   — total unclipped transactions in holdout
        where H = holdout_weeks.

        Note: total_freq uses unclipped raw counts for fair comparison with
        probabilistic benchmarks (which predict unclipped counts).
        Total spend in raw currency is derived downstream by inverse-transforming
        `spend` per week and summing (see aggregate_spend_to_raw_total).
        """
        N = len(calibration_customer_ids)
        H = holdout_weeks
        freq = np.zeros((N, H), dtype=np.int32)
        raw_freq = np.zeros((N, H), dtype=np.int32)
        spend = np.zeros((N, H), dtype=np.float32)

        # Vectorized fill using advanced indexing
        cid_to_idx = {cid: i for i, cid in enumerate(calibration_customer_ids)}

        # Filter to valid holdout rows
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
