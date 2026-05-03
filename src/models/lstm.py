"""
LSTM model for customer transaction sequence modelling.

Replication target: Valendin et al. (2022) Base LSTM.
Published in IJRM — NOT a machine learning conference paper.

Architecture verified from the open-source repo (banking_transactions_demo.ipynb, TF/Keras).
This is a PyTorch re-implementation matching that architecture exactly.

KEY FACTS from the repo (non-negotiable):
    - Inputs: week (int, 0-51) and transaction_count (int) — both EMBEDDED, not raw floats
    - Embedding size heuristic: int(max_val ** 0.5) + 1
    - memory_units = 128 (SINGLE LSTM layer — verified, NOT 2 stacked)
    - dense_units = 128 (Dense layer after LSTM, before softmax)
    - return_sequences=True — sequence-to-sequence, predicts at every time step
    - Training: stateless; Inference: stateful (manage hidden state manually)
    - Loss: mean CrossEntropy over all T time steps
    - num_layers is FIXED at 1 to match the rfm2lstm reference implementation.

INFERENCE PROCEDURE (autoregressive):
    1. Feed calibration history through LSTM to warm up (h, c) state.
    2. Use last output as first holdout prediction (sample from softmax).
    3. For each subsequent holdout step: feed previous prediction + week embedding,
       carry (h, c) forward, sample from new softmax.
    4. Average NO_SCENARIOS >= 20 sampled futures to reduce noise.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import math


def embedding_size(max_val: int) -> int:
    """Heuristic from Valendin et al. repo: int(max_val ** 0.5) + 1."""
    return int(math.sqrt(max_val)) + 1


class LSTMModel(nn.Module):
    """
    Full Base LSTM model: embeddings → LSTM → Dense → Softmax.
    Sequence-to-sequence: predicts at every time step.

    Architecture matches Valendin et al. (2022) / rfm2lstm exactly:
        - Single LSTM layer (num_layers=1, fixed)
        - Dense layer (128 units) between LSTM output and prediction heads

    Args:
        max_week:         Maximum week index (52 for weekly; embedding vocab = max_week + 1)
        max_trans:        Maximum transaction count after clipping (embedding vocab = max_trans + 1)
        memory_units:     LSTM hidden size (128, from repo)
        dense_units:      Dense layer size (128, from repo)
        n_classes:        Number of frequency classes (= max_trans + 1)
        dropout:          Dropout on dense layer output
        joint:            Extension 1: enable spend regression head
        static_cov_dim:   Number of static covariate features (e.g. income, hh_size).
                          These are constant per customer and broadcast across T.
        dynamic_cov_dim:  Number of time-varying covariate features (e.g. coupons, campaigns).
        cov_emb_dim:      Projection dimension for each covariate type (default 8).
    """

    def __init__(
        self,
        max_week: int = 52,
        max_trans: int = 6,
        memory_units: int = 128,
        dense_units: int = 128,
        dropout: float = 0.0,
        joint: bool = False,
        static_cov_dim: int = 0,
        dynamic_cov_dim: int = 0,
        cov_emb_dim: int = 8,
    ):
        super().__init__()
        self.joint = joint
        self.max_trans = max_trans
        self.static_cov_dim = static_cov_dim
        self.dynamic_cov_dim = dynamic_cov_dim
        n_classes = max_trans + 1

        # Embedding layers (categorical inputs — not raw floats)
        week_emb_dim = embedding_size(max_week)     # ≈ 8 for max_week=52
        trans_emb_dim = embedding_size(max_trans)

        self.embed_week = nn.Embedding(max_week + 1, week_emb_dim)
        self.embed_trans = nn.Embedding(max_trans + 1, trans_emb_dim)

        lstm_input_dim = week_emb_dim + trans_emb_dim

        # Static covariate projection (Extension 3): constant per customer, broadcast across T
        self.static_proj: Optional[nn.Linear] = None
        if static_cov_dim > 0:
            self.static_proj = nn.Linear(static_cov_dim, cov_emb_dim)
            lstm_input_dim += cov_emb_dim

        # Dynamic covariate projection (Extension 3): time-varying per customer
        self.dynamic_proj: Optional[nn.Linear] = None
        if dynamic_cov_dim > 0:
            self.dynamic_proj = nn.Linear(dynamic_cov_dim, cov_emb_dim)
            lstm_input_dim += cov_emb_dim

        # Single LSTM layer — fixed at num_layers=1 to match Valendin et al. (2022)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=memory_units,
            num_layers=1,
            batch_first=True,
        )

        # Dense layer after LSTM (matches rfm2lstm Dense(128) before output heads)
        self.dense = nn.Linear(memory_units, dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Frequency head: softmax over n_classes (output logits; apply CE loss directly)
        self.freq_head = nn.Linear(dense_units, n_classes)

        # Spend head (Extension 1 only): scalar regression
        if joint:
            self.spend_head = nn.Linear(dense_units, 1)

    def forward(
        self,
        week: torch.Tensor,                              # (B, T) integer week indices
        trans: torch.Tensor,                             # (B, T) integer transaction counts
        hidden: Optional[Tuple] = None,                  # (h_0, c_0) for stateful inference
        static_covariates: Optional[torch.Tensor] = None,   # (B, S) static per-customer
        dynamic_covariates: Optional[torch.Tensor] = None,  # (B, T, D) time-varying
    ):
        """
        Sequence-to-sequence forward pass.

        During TRAINING:
            week  = full calibration sequence of week indices (B, T)
            trans = full calibration sequence, teacher-forced (B, T)
            hidden = None (stateless; reset each batch)

        During INFERENCE (autoregressive):
            week  = single time step (B, 1)
            trans = previous prediction (B, 1)
            hidden = (h, c) from previous step (carry forward)

        Covariate shapes:
            static_covariates:  (B, S) — projected and broadcast across all T positions
            dynamic_covariates: (B, T, D) — projected and concatenated per time step

        Returns:
            freq_logits: (B, T, n_classes)
            log_spend:   (B, T) — only if joint=True
            hidden:      (h_n, c_n)
        """
        B, T = week.shape

        e_week = self.embed_week(week.clamp(0, self.embed_week.num_embeddings - 1))
        e_trans = self.embed_trans(trans.clamp(0, self.max_trans))

        x = torch.cat([e_week, e_trans], dim=-1)  # (B, T, week_emb + trans_emb)

        # Static covariates: project (B, S) → (B, cov_emb_dim), expand across T
        if self.static_proj is not None and static_covariates is not None:
            s_emb = self.static_proj(static_covariates)          # (B, cov_emb_dim)
            s_emb = s_emb.unsqueeze(1).expand(-1, T, -1)        # (B, T, cov_emb_dim)
            x = torch.cat([x, s_emb], dim=-1)

        # Dynamic covariates: project (B, T, D) → (B, T, cov_emb_dim), concatenate
        if self.dynamic_proj is not None and dynamic_covariates is not None:
            d_emb = self.dynamic_proj(dynamic_covariates)        # (B, T, cov_emb_dim)
            x = torch.cat([x, d_emb], dim=-1)

        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (B, T, memory_units)

        h = self.dropout(self.relu(self.dense(lstm_out)))  # (B, T, dense_units)

        freq_logits = self.freq_head(h)  # (B, T, n_classes)

        if self.joint:
            log_spend = self.spend_head(h).squeeze(-1)  # (B, T)
            return freq_logits, log_spend, hidden

        return freq_logits, hidden
