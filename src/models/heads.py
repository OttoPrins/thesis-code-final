"""
Prediction heads shared across LSTM and Transformer encoders.

Both heads are applied element-wise at every time step of the encoder output
(seq2seq training paradigm, Valendin et al. 2022): input is (B, T, D), not a
pooled (B, D) representation.

FrequencyHead: Linear → logits over {0, 1, 2, 3+} (CE loss applied outside).
SpendHead:     Linear → scalar log1p-spend prediction per time step.

Currently these classes are kept as library primitives; LSTMModel and
TransformerModel inline the equivalent `nn.Linear` for clarity.
"""

import torch
import torch.nn as nn


class FrequencyHead(nn.Module):
    """
    Softmax head for discretised weekly transaction count, applied per-step.
    Default n_classes=4 corresponds to {0, 1, 2, 3+}.
    """

    def __init__(self, input_dim: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — per-step encoder output
        Returns:
            logits: (B, T, n_classes) — raw logits (apply CrossEntropyLoss directly)
        """
        return self.fc(x)


class SpendHead(nn.Module):
    """
    Regression head for log-transformed spend, applied per-step.
    Output is a scalar per time step (unconstrained).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — per-step encoder output
        Returns:
            log_spend: (B, T) — predicted log-spend per step
        """
        return self.fc(x).squeeze(-1)
