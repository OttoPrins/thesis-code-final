"""
Prediction heads shared across LSTM and Transformer encoders.

FrequencyHead: Linear → Softmax over {0, 1, 2, 3+} (4 classes)
    Matches Valendin et al. (2022)'s discretised weekly count output.

SpendHead: Linear → scalar
    Predicts log-transformed spend for the next period (Extension 1 and 2).
    Target is log1p(weekly_spend); output is unconstrained (can be negative for log-scale).
"""

import torch
import torch.nn as nn


class FrequencyHead(nn.Module):
    """
    4-class softmax head for discretised weekly transaction count.
    Classes: 0 (no purchase), 1 (1 purchase), 2 (2 purchases), 3 (3+).
    """

    def __init__(self, input_dim: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) — encoder output (last hidden state or pooled representation)
        Returns:
            logits: (B, n_classes) — raw logits (apply CrossEntropyLoss directly)
        """
        return self.fc(x)


class SpendHead(nn.Module):
    """
    Regression head for log-transformed spend prediction.
    Output is a scalar per sample (unconstrained).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) — encoder output
        Returns:
            log_spend: (B,) — predicted log-spend
        """
        return self.fc(x).squeeze(-1)
