"""
UCI Online Retail II dataset pipeline.

Dataset: UK e-commerce transactions, 2009-2011.
         ~4,300 active customers after cleaning.

Source: UCI Machine Learning Repository
        https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
Expected final raw file: data/raw/online_retail_II.xlsx (or equivalent CSV export)

Cleaning rules:
    - Drop rows where Customer ID / CustomerID is NaN
    - Drop cancelled invoices (Invoice / InvoiceNo starts with 'C')
    - Drop rows with Quantity <= 0 or Price / UnitPrice <= 0
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

        self._current_dataset_cfg = dataset_cfg
        df = self.load_raw(raw_dir)
        df = self.clean(df)
        self.validate_clean_transactions(df, raw_dir)

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
        """Load UCI Online Retail II from Excel or CSV."""
        candidates = [
            "online_retail_II.xlsx",
            "Online Retail II.xlsx",
            "online_retail_ii.xlsx",
            "Online_Retail_II.xlsx",
            "online_retail_II.csv",
            "Online Retail II.csv",
            "online_retail_ii.csv",
            "Online_Retail_II.csv",
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
        return self._read_raw_file(path)

    @staticmethod
    def _read_raw_file(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".xls", ".xlsx"}:
            sheets = pd.read_excel(path, sheet_name=None)
            df = pd.concat(sheets.values(), ignore_index=True)
        else:
            # Online Retail II CSV exports are commonly comma-separated, while the
            # one-year Online Retail CSV in this repo is semicolon-separated with
            # European decimals. `sep=None` lets pandas sniff both safely.
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        df.attrs["source_path"] = path
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        source_path = df.attrs.get("source_path")
        df = self._normalise_columns(df)
        df = df.dropna(subset=["customer_id_raw"])
        df = df[~df["invoice"].astype(str).str.startswith("C")]
        df["price"] = self._to_float(df["price"])
        df["quantity"] = self._to_float(df["quantity"])
        df = df[(df["quantity"] > 0) & (df["price"] > 0)]
        df["transaction_amount"] = df["quantity"].astype(float) * df["price"]
        df["customer_id"] = df["customer_id_raw"].astype(float).astype(int)
        df["date"] = pd.to_datetime(df["invoice_date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        # Aggregate to invoice level so weekly_freq counts shopping trips, not line items
        df = (
            df.groupby(["customer_id", "invoice", "date"])
            .agg(transaction_amount=("transaction_amount", "sum"))
            .reset_index()
        )
        out = df[["customer_id", "date", "transaction_amount"]]
        out.attrs["source_path"] = source_path
        return out

    @staticmethod
    def _to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for col in df.columns:
            key = str(col).strip().lower().replace(" ", "").replace("_", "")
            if key in {"invoice", "invoiceno"}:
                rename[col] = "invoice"
            elif key == "invoicedate":
                rename[col] = "invoice_date"
            elif key in {"customerid", "customer"}:
                rename[col] = "customer_id_raw"
            elif key in {"price", "unitprice"}:
                rename[col] = "price"
            elif key == "quantity":
                rename[col] = "quantity"
        out = df.rename(columns=rename)
        required = {"invoice", "invoice_date", "customer_id_raw", "quantity", "price"}
        missing = sorted(required - set(out.columns))
        if missing:
            raise ValueError(
                "UCI Online Retail II file is missing required columns after "
                f"normalisation: {missing}. Expected either UCI II columns "
                "(Invoice, InvoiceDate, Customer ID, Quantity, Price) or the "
                "one-year Online Retail aliases (InvoiceNo, UnitPrice, CustomerID)."
            )
        return out

    def _validate_online_retail_ii_identity(self, clean_df: pd.DataFrame, raw_dir: str) -> None:
        cfg = getattr(self, "_current_dataset_cfg", {})
        if not cfg.get("require_online_retail_ii", False):
            return
        if clean_df.empty:
            raise ValueError("UCI Online Retail II validation failed: cleaned file is empty.")

        min_date = clean_df["date"].min()
        max_date = clean_df["date"].max()
        span_days = int((max_date - min_date).days)
        looks_like_ii = (
            min_date.year <= 2009
            and max_date.year >= 2011
            and span_days >= 700
        )
        if not looks_like_ii:
            source = getattr(clean_df, "attrs", {}).get("source_path", "unknown source")
            raise ValueError(
                "Final UCI runs require the two-year UCI Online Retail II data "
                "(2009-12 to 2011-12). The loaded file appears to cover "
                f"{min_date.date()} to {max_date.date()} ({span_days} days), "
                f"from {source}. The one-year 'Online Retail.csv' file is rejected "
                f"for final thesis runs. Place Online Retail II in {raw_dir} or "
                "set require_online_retail_ii: false for an explicit diagnostic run."
            )

    def validate_clean_transactions(self, clean_df: pd.DataFrame, raw_dir: str) -> None:
        self._validate_online_retail_ii_identity(clean_df, raw_dir)

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
