"""
LSTM model for customer transaction sequence modelling.

Replication target: Valendin et al. (2022) Base LSTM.
Published in IJRM — NOT a machine learning conference paper.

Architecture verified from the open-source repo (banking_transactions_demo.ipynb, TF/Keras).
This is a PyTorch re-implementation matching that architecture.

KEY FACTS from the repo:
    - Inputs: week (int, 0-51) and transaction_count (int) — both EMBEDDED, not raw floats
    - Embedding size heuristic: int(max_val ** 0.5) + 1
    - memory_units = 128 (single LSTM layer, NOT 2 stacked)
    - dense_units = 128 (Dense layer after LSTM, before softmax)
    - return_sequences=True — sequence-to-sequence, predicts at every time step
    - Training: stateless; Inference: stateful (manage hidden state manually)
    - Loss: mean CrossEntropy over all T time steps

INFERENCE PROCEDURE (autoregressive):
    1. Feed calibration history through LSTM to warm up (h, c) state.
    2. Use last output as first holdout prediction (sample from softmax).
    3. For each subsequent holdout step: feed previous prediction + week embedding,
       carry (h, c) forward, sample from new softmax.
    4. Average NO_SCENARIOS ≥ 20 sampled futures to reduce noise.
"""

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

    Args:
        max_week:       Maximum week index (52 for weekly; embedding vocab size = max_week + 1)
        max_trans:      Maximum transaction count after clipping (embedding vocab size = max_trans + 1)
        memory_units:   LSTM hidden size (128, from repo)
        dense_units:    Dense layer size (128, from repo)
        n_classes:      Number of frequency classes (= max_trans + 1)
        dropout:        Dropout on dense layer output (not in original, add conservatively)

    For Extension 1 (joint prediction), set joint=True to enable the spend regression head.
    """

    def __init__(
        self,
        max_week: int = 52,
        max_trans: int = 6,
        memory_units: int = 128,
        dense_units: int = 128,
        dropout: float = 0.0,
        joint: bool = False,      # Extension 1: enable spend head
    ):
        super().__init__()
        self.joint = joint
        self.max_trans = max_trans
        n_classes = max_trans + 1  # 0, 1, ..., max_trans

        # Embedding layers (categorical inputs — not raw floats)
        week_emb_dim = embedding_size(max_week)     # ≈ 8 for max_week=52
        trans_emb_dim = embedding_size(max_trans)   # depends on max_trans

        self.embed_week = nn.Embedding(max_week + 1, week_emb_dim)
        self.embed_trans = nn.Embedding(max_trans + 1, trans_emb_dim)

        lstm_input_dim = week_emb_dim + trans_emb_dim

        # Single LSTM layer (return_sequences=True equivalent: return all outputs)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=memory_units,
            num_layers=1,
            batch_first=True,
        )

        # Dense layer after LSTM
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
        week: torch.Tensor,         # (B, T) — integer week indices
        trans: torch.Tensor,        # (B, T) — integer transaction counts (previous step or padded)
        hidden=None,                # optional: (h_0, c_0) for stateful inference
    ):
        """
        Sequence-to-sequence forward pass.

        During TRAINING:
            week  = full calibration sequence of week indices (B, T)
            trans = full calibration sequence of transaction counts, SHIFTED by 1
                    (teacher-forced: feed ground-truth previous count, predict next)
            hidden = None (stateless; reset each batch)

        During INFERENCE (autoregressive):
            week  = single time step (B, 1)
            trans = previous prediction (B, 1)
            hidden = (h, c) from previous step (carry forward)

        Returns:
            freq_logits: (B, T, n_classes) — one distribution per time step
            log_spend:   (B, T) — only if joint=True
            hidden:      (h_n, c_n) — final hidden state (needed for autoregressive inference)
        """
        # Embed categorical inputs
        e_week = self.embed_week(week.clamp(0, self.embed_week.num_embeddings - 1))   # (B, T, week_emb_dim)
        e_trans = self.embed_trans(trans.clamp(0, self.max_trans))                     # (B, T, trans_emb_dim)

        x = torch.cat([e_week, e_trans], dim=-1)  # (B, T, lstm_input_dim)

        # LSTM — returns all time steps
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (B, T, memory_units)

        # Dense layer applied at every time step
        h = self.dropout(self.relu(self.dense(lstm_out)))  # (B, T, dense_units)

        # Frequency head
        freq_logits = self.freq_head(h)  # (B, T, n_classes)

        if self.joint:
            log_spend = self.spend_head(h).squeeze(-1)  # (B, T)
            return freq_logits, log_spend, hidden

        return freq_logits, hidden
