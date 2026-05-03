"""
Shared data transforms applied identically across all datasets.

Pipeline:
    WeeklyAggregator → TemporalSplitter → Scaler → SequenceBuilder

Design constraints (from CLAUDE.md):
    - Weekly aggregation matches Valendin et al. (2022)
    - Frequency discretised to {0, 1, ..., cap} where the cap is the 99th
      percentile of nonzero weekly counts on the calibration set (Valendin's
      "use all observed classes" rule, made data-driven)
    - Spend always log1p-transformed (no further scaling — see SpendScaler)
    - Scaler fitted on calibration set ONLY (no holdout leakage)
    - Sequences padded with zeros; mask marks padding positions
    - min_active_weeks defaults to 1: every customer that ever transacted in
      calibration is kept. This preserves the long tail of one-time / infrequent
      buyers that BTYD (and the deep models that replace it) need to learn
      latent churn from.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_freq_cap(weekly_freq: np.ndarray, q: float = 0.99) -> int:
    """
    Pick the frequency-class cap from calibration weekly counts.

    Cap = ceil( q-quantile of nonzero weekly counts ), floored at 1.
    Using only nonzero weeks avoids the long tail of "this customer didn't shop
    that week" zeros dominating the quantile.
    """
    nonzero = weekly_freq[weekly_freq > 0]
    if nonzero.size == 0:
        return 1
    cap = int(np.ceil(np.quantile(nonzero, q)))
    return max(1, cap)


def resolve_freq_bins(dataset_cfg: dict, calib_weekly_freq: np.ndarray) -> list:
    """
    Resolve the freq_bins list from a dataset config + calibration data.

    Precedence:
        1. Explicit `freq_bins: [0, 1, ..., k]` in YAML — used as-is.
        2. Explicit `freq_cap: <int>` in YAML — bins become [0..cap].
        3. `freq_cap: auto` (or unspecified) — cap derived from calibration via
           compute_freq_cap (q=0.99 by default; override with `freq_cap_quantile`).

    Logs the chosen cap for traceability.
    """
    bins = dataset_cfg.get("freq_bins")
    if bins is not None:
        print(f"[pipeline] freq_bins from config: {bins}")
        return list(bins)

    cap_setting = dataset_cfg.get("freq_cap", "auto")
    if isinstance(cap_setting, int):
        cap = cap_setting
        source = "explicit"
    else:
        q = float(dataset_cfg.get("freq_cap_quantile", 0.99))
        cap = compute_freq_cap(np.asarray(calib_weekly_freq), q=q)
        source = f"auto q={q}"
    bins = list(range(cap + 1))
    print(f"[pipeline] freq_cap={cap} ({source}) → freq_bins={bins[:4]}{'...' if len(bins)>4 else ''} (n_classes={len(bins)})")
    return bins


def compute_delta_t(trans_grid: np.ndarray) -> np.ndarray:
    """
    Per-step elapsed time since the most recent purchase.

    Args:
        trans_grid: (N, T) integer transaction counts on a dense weekly grid.

    Returns:
        delta_t: (N, T) float32 — at each step t, weeks since the most recent
                 purchase observed at or before t. Zero before the customer's
                 first purchase (sentinel "no history yet"). Zero on a step
                 where the customer just transacted.

    BTYD models (Pareto/NBD, Pareto/GGG) infer churn from inter-transaction
    times — i.e., elapsed time, not absolute calendar position. Feeding this
    feature to attention-based models gives them the same regularity signal
    that their probabilistic counterparts rely on.
    """
    N, T = trans_grid.shape
    delta_t = np.zeros((N, T), dtype=np.float32)
    weeks_since = np.zeros(N, dtype=np.float32)
    has_purchased = np.zeros(N, dtype=bool)
    for t in range(T):
        purchased_now = trans_grid[:, t] > 0
        weeks_since = np.where(purchased_now, 0.0, weeks_since + 1.0)
        has_purchased = has_purchased | purchased_now
        delta_t[:, t] = np.where(has_purchased, weeks_since, 0.0)
    return delta_t


class WeeklyAggregator:
    """
    Aggregates transaction-level data into weekly customer summaries.

    Input DataFrame columns: [customer_id, date, transaction_amount]
    Output DataFrame columns: [customer_id, week, weekly_freq, weekly_spend]

    Week numbering: starts at 0 from the first transaction in the dataset.
    """

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        df["week"] = ((df["date"] - min_date).dt.days // 7).astype(int)

        agg = (
            df.groupby(["customer_id", "week"])
            .agg(weekly_freq=("transaction_amount", "count"),
                 weekly_spend=("transaction_amount", "sum"))
            .reset_index()
        )
        return agg

    @staticmethod
    def discretise_freq(freq_series: pd.Series, bins: list = None) -> pd.Series:
        """Map raw frequency to class label {0, 1, ..., bins[-1]} (top class is 'k+')."""
        if bins is None:
            bins = [0, 1, 2, 3]
        return freq_series.clip(upper=bins[-1]).astype(int)


class TemporalSplitter:
    """
    Splits aggregated weekly data into calibration and holdout by week number.

    Calibration: [0, calibration_weeks)
    Holdout: [calibration_weeks, calibration_weeks + holdout_weeks)

    Note: The scaler MUST be fitted only on calibration data before transforming holdout.
    """

    def __init__(self, calibration_weeks: int, holdout_weeks: int):
        self.calibration_weeks = calibration_weeks
        self.holdout_weeks = holdout_weeks

    def split(self, df: pd.DataFrame):
        calib = df[df["week"] < self.calibration_weeks].copy()
        holdout = df[
            (df["week"] >= self.calibration_weeks) &
            (df["week"] < self.calibration_weeks + self.holdout_weeks)
        ].copy()
        return calib, holdout


class SpendScaler:
    """
    Applies log1p to spend. No further scaling.

    Why no MinMaxScaler: with Kendall homoscedastic uncertainty weighting
    (Kendall et al. 2018), the per-task variances σ_i² are learnable
    parameters that adapt to the natural scale of each loss. If we crush
    log-spend into [0, 1], the MSE loss becomes orders of magnitude smaller
    than the cross-entropy loss; log_var_spend then has to compensate by
    becoming very negative, but the gradient signal flowing into the spend
    head is correspondingly tiny. Working in raw log-space lets the
    uncertainty parameters do their job on losses with comparable scales.

    log1p (rather than plain log) handles the log(0) on inactive weeks where
    weekly_spend = 0; spend masking in the trainer ensures inactive-week
    targets do not contribute to the spend loss.

    Fitted on calibration set only (kept stateless to make leakage impossible).
    """

    def __init__(self, scale: bool = False):
        # `scale` is retained for API compatibility but is now a no-op.
        # Explicitly forbid the historical MinMax behaviour: it interacts badly
        # with Kendall's homoscedastic loss weighting.
        if scale:
            raise ValueError(
                "SpendScaler(scale=True) is deprecated. Pass scale=False (default); "
                "raw log1p-spend keeps the spend MSE on a scale comparable to the "
                "frequency CrossEntropy, which is what Kendall task uncertainty "
                "weighting assumes."
            )

    def fit_transform(self, spend: np.ndarray) -> np.ndarray:
        return np.log1p(spend).astype(np.float32)

    def transform(self, spend: np.ndarray) -> np.ndarray:
        return np.log1p(spend).astype(np.float32)

    def inverse_transform_spend(self, log_spend: np.ndarray) -> np.ndarray:
        return np.expm1(log_spend)


class SequenceBuilder:
    """
    Builds full-history, teacher-forced sequences for seq-to-seq training.

    Matches Valendin et al. (2022) reference implementation:
    - One sequence per customer covering the entire calibration period.
    - Dense weekly grid: inactive weeks filled with (freq=0, spend=0.0).
    - Teacher-forcing shift: input = steps 0..T-2, target = steps 1..T-1.
    - Full unshifted sequence stored as "seed" for autoregressive inference.

    Long-tail policy:
    - min_active_weeks defaults to 1. Customers with as few as one purchase in
      calibration are kept. Filtering low-frequency buyers introduces survivorship
      bias and removes precisely the customers BTYD models rely on to learn
      latent churn signals.

    Input: calibration DataFrame with columns [customer_id, week, weekly_freq, weekly_spend]
           (weekly_spend already log-transformed by SpendScaler as 'log_spend')
    Output: dict of numpy arrays ready for CustomerDataset (also includes delta_t,
            the per-step elapsed-time signal used by the Transformer's Time2Vec).
    """

    def __init__(self, calibration_weeks: int = 52, min_active_weeks: int = 1,
                 freq_bins: list = None):
        self.calibration_weeks = calibration_weeks
        self.min_active_weeks = min_active_weeks
        self.freq_bins = freq_bins or [0, 1, 2, 3]
        self.max_trans = self.freq_bins[-1]  # e.g. 3 for [0,1,2,3]

    def build(self, df: pd.DataFrame) -> dict:
        """
        Build training sequences from aggregated calibration data.

        Args:
            df: DataFrame with columns [customer_id, week, weekly_freq, log_spend]

        Returns dict with keys:
            week_input    : (N, T-1) int32   — week indices for input steps
            trans_input   : (N, T-1) int32   — transaction counts (teacher forcing)
            spend_input   : (N, T-1) float32 — log-spend at each input step
            delta_t_input : (N, T-1) float32 — weeks since last purchase at each input step
            y_freq        : (N, T-1) int32   — target freq class, shifted +1
            y_spend       : (N, T-1) float32 — target log-spend, shifted +1
            customer_ids  : (N,)     int64   — customer identifiers
            mask          : (N, T-1) float32 — 1=real, 0=padding
            seed_week     : (N, T)   int32   — full calibration weeks (inference)
            seed_trans    : (N, T)   int32   — full calibration trans (inference)
            seed_spend    : (N, T)   float32 — full calibration spend (inference)
            seed_delta_t  : (N, T)   float32 — full calibration delta_t (inference seed)
            max_trans     : int              — clipping value used
        """
        T = self.calibration_weeks
        spend_col = "log_spend" if "log_spend" in df.columns else "weekly_spend"

        # Filter to valid weeks only
        df_valid = df[(df["week"] >= 0) & (df["week"] < T)].copy()

        # Count active weeks per customer and apply the long-tail-friendly threshold.
        # min_active_weeks=1 keeps every customer with at least one transaction in
        # the calibration window — required for BTYD-style latent churn modelling.
        active_counts = (
            df_valid[df_valid["weekly_freq"] > 0]
            .groupby("customer_id")["week"]
            .nunique()
        )
        valid_customers = active_counts[
            active_counts >= self.min_active_weeks
        ].index

        # Keep only valid customers and sort for deterministic ordering
        customers = np.sort(np.intersect1d(df_valid["customer_id"].unique(), valid_customers))
        N = len(customers)

        # Pre-allocate arrays (dense grid: every customer gets T weeks, default 0)
        full_weeks = np.tile(np.arange(T, dtype=np.int32), (N, 1))  # (N, T)
        full_trans = np.zeros((N, T), dtype=np.int32)
        full_spend = np.zeros((N, T), dtype=np.float32)

        # Vectorized fill: map customer_id → row index, then use advanced indexing
        cid_to_row = {cid: i for i, cid in enumerate(customers)}
        df_fill = df_valid[df_valid["customer_id"].isin(customers)].copy()
        row_idx = df_fill["customer_id"].map(cid_to_row).values.astype(int)
        col_idx = df_fill["week"].values.astype(int)

        full_trans[row_idx, col_idx] = np.minimum(
            df_fill["weekly_freq"].values.astype(np.int32), self.max_trans
        )
        full_spend[row_idx, col_idx] = df_fill[spend_col].values.astype(np.float32)

        # Per-step elapsed-time signal — computed AFTER the dense grid is filled
        # so the values reflect actual customer behaviour over the full window.
        full_delta_t = compute_delta_t(full_trans)  # (N, T) float32

        # Teacher-forcing shift: input = [0..T-2], target = [1..T-1]
        week_input = full_weeks[:, :-1]       # (N, T-1)
        trans_input = full_trans[:, :-1]      # (N, T-1)
        spend_input = full_spend[:, :-1]      # (N, T-1)
        delta_t_input = full_delta_t[:, :-1]  # (N, T-1)
        y_freq = full_trans[:, 1:]            # (N, T-1)
        y_spend = full_spend[:, 1:]           # (N, T-1)

        # Mask = 1 only for target steps strictly after the customer's first active week.
        # This matches BTYD's T₀ conditioning: a customer does not exist in the cohort
        # until their first purchase, so loss is not computed on the unborn period.
        first_week_series = (
            df_valid[df_valid["weekly_freq"] > 0]
            .groupby("customer_id")["week"]
            .min()
        )
        first_week_idx = first_week_series.reindex(customers).values.astype(np.int32)  # (N,)
        target_steps = np.arange(1, T, dtype=np.int32)  # (T-1,)
        mask = (target_steps[None, :] > first_week_idx[:, None]).astype(np.float32)

        customer_ids = np.array(customers, dtype=np.int64)

        return {
            "week_input": week_input,
            "trans_input": trans_input,
            "spend_input": spend_input,
            "delta_t_input": delta_t_input,
            "y_freq": y_freq,
            "y_spend": y_spend,
            "customer_ids": customer_ids,
            "mask": mask,
            "seed_week": full_weeks,
            "seed_trans": full_trans,
            "seed_spend": full_spend,
            "seed_delta_t": full_delta_t,
            "max_trans": self.max_trans,
        }
