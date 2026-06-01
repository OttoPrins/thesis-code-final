"""
LSTM model for customer transaction sequence modelling.

Replication target: Valendin et al. (2022) Base LSTM.
Published in IJRM — NOT a machine learning conference paper.

Architecture verified from the open-source repo (banking_transactions_demo.ipynb, TF/Keras).
This is a PyTorch re-implementation matching that architecture exactly.

KEY FACTS from the repo (non-negotiable):
    - Inputs: week (int, 0-51) and transaction_count (int) — both EMBEDDED, not raw floats
    - Joint extensions also receive log-spend as a continuous input feature
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

from src.models.heads import make_spend_head


def embedding_size(max_val: int, *, mode: str = "max_index") -> int:
    """Valendin embedding-size heuristic, with explicit max-index/cardinality modes."""
    mode = str(mode).lower()
    if mode == "max_index":
        basis = int(max_val)
    elif mode == "keras_cardinality":
        basis = int(max_val) + 1
    else:
        raise ValueError(
            "embedding_size mode must be 'max_index' or 'keras_cardinality', "
            f"got {mode!r}"
        )
    return int(math.sqrt(basis)) + 1


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
        state_feature_dim:Number of causal state features (e.g. delta_t) projected into
                          the shared encoder for v2 joint models.
        spend_head:       "regression" for the original scalar spend head, or
                          "hurdle_lognormal" for v2 spend_mu/spend_log_var output.
        regression_head_hidden: Optional hidden size for a one-hidden-layer spend head.
        dense_activation: Activation after the Dense layer. Use "linear" for the
                          Valendin/Keras replication path; default keeps existing runs.
        keras_initialization: If true, initialize embeddings, LSTM, Dense and heads
                              to mirror Keras defaults used by the banking demo.
        embedding_size_mode: Whether to apply the embedding heuristic to the max
                             observed index ("max_index") or vocabulary cardinality
                             ("keras_cardinality"). The latter matches Keras input_dim.
        keras_lstm_init_mode: "gatewise" keeps the previous PyTorch approximation;
                              "whole_matrix" matches Keras initializers applied to
                              concatenated LSTM kernels.
        freeze_lstm_bias_hh: If true, zero and freeze PyTorch's redundant recurrent
                             LSTM bias so the layer behaves like Keras' single bias.
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
        state_feature_dim: int = 0,
        spend_head: str = "regression",
        regression_head_hidden: int | None = None,
        dense_activation: str = "relu",
        keras_initialization: bool = False,
        embedding_size_mode: str = "max_index",
        keras_lstm_init_mode: str = "gatewise",
        freeze_lstm_bias_hh: bool = False,
    ):
        super().__init__()
        self.joint = joint
        self.max_trans = max_trans
        self.static_cov_dim = static_cov_dim
        self.dynamic_cov_dim = dynamic_cov_dim
        self.state_feature_dim = state_feature_dim
        self.spend_head_type = spend_head
        if spend_head not in {"regression", "hurdle_lognormal"}:
            raise ValueError("spend_head must be 'regression' or 'hurdle_lognormal'")
        dense_activation = dense_activation.lower()
        if dense_activation not in {"relu", "linear", "identity", "none"}:
            raise ValueError(
                "dense_activation must be one of 'relu', 'linear', 'identity', or 'none'"
            )
        self.dense_activation_name = dense_activation
        embedding_size_mode = str(embedding_size_mode).lower()
        if embedding_size_mode not in {"max_index", "keras_cardinality"}:
            raise ValueError(
                "embedding_size_mode must be 'max_index' or 'keras_cardinality'"
            )
        keras_lstm_init_mode = str(keras_lstm_init_mode).lower()
        if keras_lstm_init_mode not in {"gatewise", "whole_matrix"}:
            raise ValueError(
                "keras_lstm_init_mode must be 'gatewise' or 'whole_matrix'"
            )
        self.embedding_size_mode = embedding_size_mode
        self.keras_lstm_init_mode = keras_lstm_init_mode
        self.freeze_lstm_bias_hh = bool(freeze_lstm_bias_hh)
        n_classes = max_trans + 1

        # Embedding layers (categorical inputs — not raw floats)
        week_emb_dim = embedding_size(max_week, mode=embedding_size_mode)
        trans_emb_dim = embedding_size(max_trans, mode=embedding_size_mode)

        self.embed_week = nn.Embedding(max_week + 1, week_emb_dim)
        self.embed_trans = nn.Embedding(max_trans + 1, trans_emb_dim)

        lstm_input_dim = week_emb_dim + trans_emb_dim

        # Joint extensions condition both heads on prior log-spend history.
        self.spend_proj: Optional[nn.Linear] = None
        if joint:
            self.spend_proj = nn.Linear(1, cov_emb_dim)
            lstm_input_dim += cov_emb_dim

        self.state_proj: Optional[nn.Linear] = None
        if state_feature_dim > 0:
            self.state_proj = nn.Linear(state_feature_dim, cov_emb_dim)
            lstm_input_dim += cov_emb_dim

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
        self.dense_activation = (
            nn.ReLU()
            if dense_activation == "relu"
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        # Frequency head: softmax over n_classes (output logits; apply CE loss directly)
        self.freq_head = nn.Linear(dense_units, n_classes)

        # Spend head (Extension 1 only): scalar regression
        if joint:
            spend_out_dim = 2 if spend_head == "hurdle_lognormal" else 1
            self.spend_head = make_spend_head(
                dense_units,
                spend_out_dim,
                hidden_dim=regression_head_hidden,
            )

        if keras_initialization:
            self._reset_parameters_keras(lstm_init_mode=keras_lstm_init_mode)
        if self.freeze_lstm_bias_hh:
            self._freeze_recurrent_lstm_bias()

    def _reset_parameters_keras(self, *, lstm_init_mode: str = "gatewise") -> None:
        """
        Approximate Keras defaults used in the Valendin banking demo.

        Keras Embedding defaults to Uniform(-0.05, 0.05), Dense uses
        Glorot/Xavier uniform with zero bias, and LSTM uses Glorot input
        kernels, orthogonal recurrent kernels, zero biases, plus unit forget
        bias. PyTorch stores LSTM gates concatenated; "gatewise" initializes
        slices separately, while "whole_matrix" initializes the concatenated
        matrices the way Keras does.
        """

        def init_linear(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.uniform_(self.embed_week.weight, -0.05, 0.05)
        nn.init.uniform_(self.embed_trans.weight, -0.05, 0.05)

        init_linear(self.dense)
        init_linear(self.freq_head)
        for module in [
            self.spend_proj,
            self.state_proj,
            self.static_proj,
            self.dynamic_proj,
        ]:
            if module is not None:
                init_linear(module)
        if self.joint and hasattr(self, "spend_head"):
            self.spend_head.apply(init_linear)

        hidden_size = self.lstm.hidden_size
        lstm_init_mode = str(lstm_init_mode).lower()
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                if lstm_init_mode == "whole_matrix":
                    nn.init.xavier_uniform_(param)
                else:
                    for gate in param.chunk(4, dim=0):
                        nn.init.xavier_uniform_(gate)
            elif "weight_hh" in name:
                if lstm_init_mode == "whole_matrix":
                    # Keras initializes recurrent_kernel with shape
                    # (hidden, 4 * hidden); PyTorch stores its transpose.
                    keras_shape = param.new_empty(hidden_size, 4 * hidden_size)
                    nn.init.orthogonal_(keras_shape)
                    param.data.copy_(keras_shape.t())
                else:
                    for gate in param.chunk(4, dim=0):
                        nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name:
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)

    def _freeze_recurrent_lstm_bias(self) -> None:
        """Emulate Keras' single LSTM bias by disabling PyTorch's bias_hh term."""
        for name, param in self.lstm.named_parameters():
            if "bias_hh" in name:
                with torch.no_grad():
                    param.zero_()
                param.requires_grad_(False)

    def forward(
        self,
        week: torch.Tensor,                              # (B, T) integer week indices
        trans: torch.Tensor,                             # (B, T) integer transaction counts
        hidden: Optional[Tuple] = None,                  # (h_0, c_0) for stateful inference
        spend: Optional[torch.Tensor] = None,            # (B, T) scaled log-spend history
        state_features: Optional[torch.Tensor] = None,   # (B, T, S) causal state features
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
            spend:             (B, T) — projected continuous log-spend input for joint models
            state_features:    (B, T, S) — projected causal state inputs for v2 models

        Returns:
            freq_logits: (B, T, n_classes)
            log_spend:   (B, T) — only if joint=True
            hidden:      (h_n, c_n)
        """
        B, T = week.shape

        e_week = self.embed_week(week.clamp(0, self.embed_week.num_embeddings - 1))
        e_trans = self.embed_trans(trans.clamp(0, self.max_trans))

        x = torch.cat([e_week, e_trans], dim=-1)  # (B, T, week_emb + trans_emb)

        # Joint spend input: previous/current log-spend history is part of the shared encoder.
        if self.spend_proj is not None:
            if spend is None:
                spend = torch.zeros((B, T), dtype=e_week.dtype, device=week.device)
            spend_emb = self.spend_proj(spend.to(dtype=e_week.dtype).unsqueeze(-1))
            x = torch.cat([x, spend_emb], dim=-1)

        if self.state_proj is not None:
            if state_features is None:
                state_features = torch.zeros(
                    (B, T, self.state_feature_dim),
                    dtype=e_week.dtype,
                    device=week.device,
                )
            state_emb = self.state_proj(state_features.to(dtype=e_week.dtype))
            x = torch.cat([x, state_emb], dim=-1)

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

        h = self.dropout(self.dense_activation(self.dense(lstm_out)))  # (B, T, dense_units)

        freq_logits = self.freq_head(h)  # (B, T, n_classes)

        if self.joint:
            spend_out = self.spend_head(h)
            if self.spend_head_type == "hurdle_lognormal":
                spend_mu = spend_out[..., 0]
                spend_log_var = spend_out[..., 1]
                return freq_logits, spend_mu, spend_log_var, hidden
            log_spend = spend_out.squeeze(-1)  # (B, T)
            return freq_logits, log_spend, hidden

        return freq_logits, hidden
