"""
PyTorch Dataset for customer transaction sequences (seq-to-seq format).

Stores separate integer tensors for week and transaction count (for embedding layers),
plus continuous spend, and sequence-level targets at every time step.

Matches the input format expected by LSTMModel and TransformerModel:
    - week:  (B, T) integer week indices → nn.Embedding
    - position: (B, T) absolute sequence positions → Transformer sinusoidal PE
    - trans: (B, T) integer transaction counts → nn.Embedding

Optional covariate support (Extension 3 — Dunnhumby only):
    Static covariates (e.g. income, household size) are stored as (N, S) vectors —
    constant per customer, no time dimension.
    Dynamic covariates (e.g. coupon redemptions, campaign flag) are stored as
    (N, T_total, D) tensors covering both calibration and holdout weeks, so the
    autoregressive inference loop can slice holdout steps individually.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomerDataset(Dataset):
    """
    PyTorch Dataset for seq-to-seq customer transaction sequences.

    Tensor shapes per sample (training, include_seed=False):
        week        : (T-1,)  int   — week indices for input steps
        position    : (T-1,)  int   — absolute sequence positions for PE
        trans       : (T-1,)  int   — transaction counts (teacher forcing input)
        spend       : (T-1,)  float — log-spend at each input step
        delta_t     : (T-1,)  float — weeks since last purchase at each step
        y_freq      : (T-1,)  int   — target frequency class at each step
        y_spend     : (T-1,)  float — target log-spend at each step
        mask        : (T-1,)  float — 1=real data, 0=padding
        customer_id : ()      int   — customer identifier

    Optional (Extension 3 — Dunnhumby only):
        static_covariates  : (S,)        float — static per-customer features (no time dim)
        dynamic_covariates : (T-1, D) or (T_total, D) float — time-varying features;
                             training slice = (T-1, D), inference = full (T_total, D)

    Optional (for autoregressive inference, include_seed=True):
        seed_week    : (T,)    int   — full calibration week sequence
        seed_position: (T,)    int   — full calibration position sequence
        seed_trans   : (T,)    int   — full calibration transaction sequence
        seed_spend   : (T,)    float — full calibration spend sequence
        seed_delta_t : (T,)    float — full calibration elapsed-time sequence
    """

    def __init__(self, data: dict, include_seed: bool = False):
        """
        Args:
            data: dict from SequenceBuilder.build() with keys:
                  week_input, trans_input, spend_input,
                  y_freq, y_spend, customer_ids, mask
                  Optionally: seed_week, seed_trans, seed_spend,
                              static_covariates, dynamic_covariates
            include_seed: if True, also store seed sequences (for inference)
        """
        self.week = torch.tensor(data["week_input"], dtype=torch.long)    # (N, T-1)
        if "position_input" in data:
            self.position = torch.tensor(data["position_input"], dtype=torch.long)
        else:
            self.position = torch.arange(self.week.shape[1]).repeat(self.week.shape[0], 1)
        self.trans = torch.tensor(data["trans_input"], dtype=torch.long)
        self.spend = torch.tensor(data["spend_input"], dtype=torch.float32)
        # delta_t (elapsed weeks since last purchase) is a recent addition; older
        # cached pickles may not include it, so fall back to zeros to stay
        # backwards-compatible.
        if "delta_t_input" in data:
            self.delta_t = torch.tensor(data["delta_t_input"], dtype=torch.float32)
        else:
            self.delta_t = torch.zeros_like(self.spend)
        self.y_freq = torch.tensor(data["y_freq"], dtype=torch.long)
        self.y_spend = torch.tensor(data["y_spend"], dtype=torch.float32)
        self.customer_ids = torch.tensor(data["customer_ids"], dtype=torch.long)
        self.mask = torch.tensor(data["mask"], dtype=torch.float32)
        self.max_trans = data["max_trans"]
        self.include_seed = include_seed

        # T-1 = input sequence length per sample (used for dynamic covariate slicing)
        self._T_train = self.week.shape[1]

        # Seed sequences (for autoregressive inference)
        if include_seed and "seed_week" in data:
            self.seed_week = torch.tensor(data["seed_week"], dtype=torch.long)
            if "seed_position" in data:
                self.seed_position = torch.tensor(data["seed_position"], dtype=torch.long)
            else:
                self.seed_position = torch.arange(self.seed_week.shape[1]).repeat(self.seed_week.shape[0], 1)
            self.seed_trans = torch.tensor(data["seed_trans"], dtype=torch.long)
            self.seed_spend = torch.tensor(data["seed_spend"], dtype=torch.float32)
            if "seed_delta_t" in data:
                self.seed_delta_t = torch.tensor(data["seed_delta_t"], dtype=torch.float32)
            else:
                self.seed_delta_t = torch.zeros_like(self.seed_spend)
        else:
            self.seed_week = None
            self.seed_position = None
            self.seed_delta_t = None

        # Static covariates: (N, S) — constant per customer, no time dimension
        self.static_covariates: Optional[torch.Tensor] = None
        self.static_feature_names: Optional[List[str]] = None
        if "static_covariates" in data and data["static_covariates"] is not None:
            sc = data["static_covariates"]
            self.static_covariates = torch.tensor(
                sc if isinstance(sc, np.ndarray) else np.array(sc),
                dtype=torch.float32,
            )  # (N, S)
        if "static_feature_names" in data:
            self.static_feature_names = data["static_feature_names"]

        # Dynamic covariates: (N, T_total, D) — time-varying, covers calib + holdout
        self.dynamic_covariates: Optional[torch.Tensor] = None
        self.dynamic_feature_names: Optional[List[str]] = None
        if "dynamic_covariates" in data and data["dynamic_covariates"] is not None:
            dc = data["dynamic_covariates"]
            self.dynamic_covariates = torch.tensor(
                dc if isinstance(dc, np.ndarray) else np.array(dc),
                dtype=torch.float32,
            )  # (N, T_total, D)
        if "dynamic_feature_names" in data:
            self.dynamic_feature_names = data["dynamic_feature_names"]

    def __len__(self):
        return len(self.week)

    def __getitem__(self, idx):
        item = {
            "week": self.week[idx],
            "position": self.position[idx],
            "trans": self.trans[idx],
            "spend": self.spend[idx],
            "delta_t": self.delta_t[idx],
            "y_freq": self.y_freq[idx],
            "y_spend": self.y_spend[idx],
            "customer_id": self.customer_ids[idx],
            "mask": self.mask[idx],
        }
        if self.seed_week is not None:
            item["seed_week"] = self.seed_week[idx]
            if self.seed_position is not None:
                item["seed_position"] = self.seed_position[idx]
            item["seed_trans"] = self.seed_trans[idx]
            item["seed_spend"] = self.seed_spend[idx]
            if self.seed_delta_t is not None:
                item["seed_delta_t"] = self.seed_delta_t[idx]

        # Static covariates: always (S,) — same for training and inference
        if self.static_covariates is not None:
            item["static_covariates"] = self.static_covariates[idx]  # (S,)

        # Dynamic covariates: training → input-aligned slice (T-1, D);
        # inference → full trajectory (T_total, D) for holdout step slicing
        if self.dynamic_covariates is not None:
            if self.include_seed:
                item["dynamic_covariates"] = self.dynamic_covariates[idx]           # (T_total, D)
            else:
                item["dynamic_covariates"] = self.dynamic_covariates[idx, :self._T_train, :]  # (T-1, D)

        return item
