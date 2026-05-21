"""
CDNOW dataset pipeline.

Dataset: CDNOW music retail transactions, 1997-1998.
         Canonical academic sample: 2,357 customers, 6,919 transactions.
         Master file: 23,570 customers, 69,659 transactions.
         Standard final split: calibration=52 weeks, holdout=26 weeks.

Source: BTYD R package or Fader's Wharton page.
Expected raw file: data/raw/CDNOW_sample.txt (whitespace-separated, no header)
Fallback raw file: data/raw/CDNOW_master.txt

Formats:
    Canonical sample (5 columns):
    Column 1: master_customer_id (integer)
    Column 2: customer_id        (canonical sample id, 1..2357)
    Column 3: date               (YYYYMMDD)
    Column 4: num_cds            (count)
    Column 5: amount             (USD float)

    Master file (4 columns):
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

        # Stage 1: Load and clean
        self._current_dataset_cfg = dataset_cfg
        df = self.load_raw(raw_dir)
        df = self.clean(df)

        # Q1-1997 cohort restriction (Fader & Hardie canonical split)
        cohort_end = dataset_cfg.get("cohort_end_date")
        if cohort_end is not None:
            cohort_end = pd.to_datetime(cohort_end)
            first_purchase = df.groupby("customer_id")["date"].min()
            cohort_ids = first_purchase[first_purchase <= cohort_end].index
            df = df[df["customer_id"].isin(cohort_ids)].copy()
            print(f"  Cohort filter <= {cohort_end.date()}: {len(cohort_ids)} customers")

        # Optional reproducible sub-sample (matches Fader & Hardie 10% academic sample)
        sample_n = dataset_cfg.get("sample_n")
        if sample_n is not None:
            all_custs = np.sort(df["customer_id"].unique())
            if len(all_custs) > sample_n:
                rng_samp = np.random.RandomState(dataset_cfg.get("sample_seed", 0))
                sampled = rng_samp.choice(all_custs, size=sample_n, replace=False)
                df = df[df["customer_id"].isin(sampled)].copy()
                print(f"  Sub-sample {sample_n}/{len(all_custs)} customers (seed={dataset_cfg.get('sample_seed', 0)})")

        # Stage 2: Aggregate to weekly level
        origin_date = dataset_cfg.get("origin_date")
        agg = WeeklyAggregator().fit_transform(df, origin_date=origin_date)

        # Stage 3: Temporal split
        splitter = TemporalSplitter(calib_weeks, holdout_weeks)
        calib, holdout = splitter.split(agg)

        # Stage 4: Scale spend (fit on calibration only — no leakage)
        scaler = SpendScaler(method=dataset_cfg.get("spend_scaler", "log"))
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
        train_idx, val_idx = self.train_val_indices(n, val_fraction, dataset_cfg)

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
        """Load CDNOW raw file, supporting both sample and master formats."""
        cfg = getattr(self, "_current_dataset_cfg", {})
        requested = cfg.get("raw_file")
        if requested:
            path = os.path.join(raw_dir, requested)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Configured CDNOW raw_file not found: {path}"
                )
        else:
            path = None
        candidates = [
            "CDNOW_sample.txt",
            "CDNOW_sample.csv",
            "CDNOW_master.txt",
            "cdnow.csv",
            "cdnow.txt",
        ]
        if path is None:
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

        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[1] == 5:
            df.columns = [
                "master_customer_id",
                "customer_id",
                "date",
                "num_cds",
                "amount",
            ]
        elif df.shape[1] == 4:
            df.columns = ["customer_id", "date", "num_cds", "amount"]
            if cfg.get("prefer_sample_file", False):
                sample_path = os.path.join(raw_dir, "CDNOW_sample.txt")
                raise ValueError(
                    "Final CDNOW config requires the canonical 2,357-customer "
                    f"sample, but loaded a 4-column master file at {path}. "
                    f"Place CDNOW_sample.txt in {raw_dir} (expected {sample_path}) "
                    "or set prefer_sample_file: false for an explicit diagnostic run."
                )
        else:
            raise ValueError(
                f"Unsupported CDNOW file format at {path}: expected 4 or 5 "
                f"whitespace-separated columns, got {df.shape[1]}."
            )
        df.attrs["source_path"] = path
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
