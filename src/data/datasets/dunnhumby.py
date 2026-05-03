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
from src.data.transforms import (
    WeeklyAggregator,
    TemporalSplitter,
    SpendScaler,
    SequenceBuilder,
    resolve_freq_bins,
)
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
        include_covariates = dataset_cfg.get("include_covariates", False)
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

        # Build covariates if requested (Extension 3)
        static_cov = None
        dynamic_cov = None
        if include_covariates:
            static_cov, dynamic_cov = self.build_covariates(
                raw_dir, data["customer_ids"], calib_weeks, holdout_weeks
            )

        n = len(data["customer_ids"])
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_data = _subset_data(data, train_idx)
        val_data = _subset_data(data, val_idx)

        # Attach separated static and dynamic covariates
        if static_cov is not None:
            for dset, idx in [(train_data, train_idx), (val_data, val_idx), (data, slice(None))]:
                dset["static_covariates"] = static_cov[idx]
                dset["dynamic_covariates"] = dynamic_cov[idx]
            for d in (train_data, val_data, data):
                d["static_feature_names"] = ["income", "household_size"]
                d["dynamic_feature_names"] = ["coupon_redemptions_week", "campaign_active_flag"]

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
        # Raw data has one row per product per basket. Aggregate to basket level
        # (one row per shopping trip) so that weekly_freq counts basket visits,
        # not product scans. This is the correct unit for Pareto/NBD and CLV models.
        df = (
            df.groupby(["customer_id", "BASKET_ID", "date"])["transaction_amount"]
            .sum()
            .reset_index()
        )
        return df[["customer_id", "date", "transaction_amount"]]

    def build_covariates(
        self,
        raw_dir: str,
        calibration_customer_ids: np.ndarray,
        calibration_weeks: int,
        holdout_weeks: int,
    ):
        """
        Build covariate tensors for Extension 3.

        Returns a tuple (static_covariates, dynamic_covariates):
            static_covariates:  (N, 2) — [income_encoded, hh_size_encoded]
                                Constant per customer; broadcast over T at model input.
            dynamic_covariates: (N, T_total, 2) — [coupon_redemptions_per_week,
                                campaign_active_flag]. T_total = calib + holdout.

        Note: campaign exposure in the holdout period represents firm-controlled
        marketing decisions that are known at prediction time — including them is
        consistent with CLV forecasting in a marketing planning context, not leakage.
        """
        T_total = calibration_weeks + holdout_weeks
        N = len(calibration_customer_ids)
        customer_ids_int = calibration_customer_ids.astype(int)

        # --- Demographics (static, will be broadcast across T) ---
        demo = pd.read_csv(os.path.join(raw_dir, "hh_demographic.csv"))
        demo["household_key"] = demo["household_key"].astype(int)

        income_enc = OrdinalEncoder(
            categories=[INCOME_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        demo["income_encoded"] = income_enc.fit_transform(
            demo[["INCOME_DESC"]]
        ).flatten()

        size_enc = OrdinalEncoder(
            categories=[HOUSEHOLD_SIZE_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        demo["hh_size_encoded"] = size_enc.fit_transform(
            demo[["HOUSEHOLD_SIZE_DESC"]]
        ).flatten()

        demo_lookup = (
            pd.DataFrame({"household_key": customer_ids_int})
            .merge(demo[["household_key", "income_encoded", "hh_size_encoded"]],
                   on="household_key", how="left")
            .fillna(0)
        )
        # Normalise ordinals to [0, 1] so they sit on the same scale as the
        # binary campaign flag and count-based coupon feature. Divisors come
        # from the hardcoded category lists (no data dependence ⇒ no leakage).
        income_max = float(len(INCOME_ORDER) - 1)           # 11
        hh_size_max = float(len(HOUSEHOLD_SIZE_ORDER) - 1)  # 4
        income_vals = (
            demo_lookup["income_encoded"].values.astype(np.float32) / income_max
        )
        hh_size_vals = (
            demo_lookup["hh_size_encoded"].values.astype(np.float32) / hh_size_max
        )

        # --- Coupon redemptions per week (time-varying) ---
        coup = pd.read_csv(os.path.join(raw_dir, "coupon_redempt.csv"))
        coup["household_key"] = coup["household_key"].astype(int)
        coup["week"] = coup["DAY"] // 7

        cid_to_idx = {cid: i for i, cid in enumerate(customer_ids_int)}
        coupon_grid = np.zeros((N, T_total), dtype=np.float32)
        coup_in_scope = coup[coup["week"] < T_total]
        coup_in_scope = coup_in_scope[coup_in_scope["household_key"].isin(cid_to_idx)]
        coup_agg = (
            coup_in_scope.groupby(["household_key", "week"])
            .size()
            .reset_index(name="count")
        )
        for _, row in coup_agg.iterrows():
            hh = int(row["household_key"])
            wk = int(row["week"])
            if hh in cid_to_idx and 0 <= wk < T_total:
                coupon_grid[cid_to_idx[hh], wk] = row["count"]

        # --- Campaign active flag per week (time-varying) ---
        camp_table = pd.read_csv(os.path.join(raw_dir, "campaign_table.csv"))
        camp_desc = pd.read_csv(os.path.join(raw_dir, "campaign_desc.csv"))
        camp_table["household_key"] = camp_table["household_key"].astype(int)

        # Join campaigns with their date ranges
        camp_merged = camp_table.merge(
            camp_desc[["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN", how="left"
        )
        camp_merged = camp_merged.dropna(subset=["START_DAY", "END_DAY"])
        camp_merged["start_week"] = (camp_merged["START_DAY"] // 7).astype(int)
        camp_merged["end_week"] = (camp_merged["END_DAY"] // 7).astype(int)

        campaign_grid = np.zeros((N, T_total), dtype=np.float32)
        camp_in_scope = camp_merged[camp_merged["household_key"].isin(cid_to_idx)]
        for _, row in camp_in_scope.iterrows():
            hh = int(row["household_key"])
            if hh not in cid_to_idx:
                continue
            cust_idx = cid_to_idx[hh]
            w_start = max(0, int(row["start_week"]))
            w_end = min(T_total - 1, int(row["end_week"]))
            if w_start <= w_end:
                campaign_grid[cust_idx, w_start: w_end + 1] = 1.0

        # --- Assemble separated tensors ---
        # Static: (N, 2) — income and household size (constant per customer)
        static_covariates = np.stack([income_vals, hh_size_vals], axis=1)  # (N, 2)

        # Dynamic: (N, T_total, 2) — coupon redemptions and campaign activity
        dynamic_covariates = np.stack([coupon_grid, campaign_grid], axis=-1)  # (N, T_total, 2)

        return static_covariates, dynamic_covariates

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
