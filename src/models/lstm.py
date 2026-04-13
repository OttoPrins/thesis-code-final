"""
LSTM encoder for customer transaction sequence modelling.

Replication target: Valendin et al. (2022) Base LSTM and Extended LSTM.
Published in IJRM — NOT a machine learning conference paper.

Architecture:
    - Stacked LSTM layers (default: 2 layers, 128 hidden units)
    - Dropout between layers
    - Takes last hidden state as sequence representation
    - Attach FrequencyHead and optionally SpendHead from src/models/heads.py

Note: This module is the encoder only. The full model (encoder + heads + loss)
is assembled in src/training/trainer.py or a Lightning module.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Stacked LSTM encoder. Returns last-step hidden state.

    Args:
        input_dim:   Number of features per time step (F in X shape [B, T, F])
        hidden_size: LSTM hidden dimension
        num_layers:  Number of stacked LSTM layers
        dropout:     Dropout probability between LSTM layers (0 for single layer)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x    : (B, T, F) — padded input sequences
            mask : (B, T)    — 1=real, 0=padding (used to extract last real hidden state)
        Returns:
            h_last: (B, hidden_size) — hidden state at the last real time step
        """
        output, (h_n, c_n) = self.lstm(x)  # output: (B, T, H)

        if mask is not None:
            # Get index of last real time step per sequence
            lengths = mask.sum(dim=1).long()  # (B,)
            lengths = lengths.clamp(min=1) - 1  # 0-indexed last real step
            # Gather last real hidden state
            idx = lengths.view(-1, 1, 1).expand(-1, 1, output.size(-1))
            h_last = output.gather(1, idx).squeeze(1)  # (B, H)
        else:
            h_last = output[:, -1, :]  # Fall back to last step if no mask

        return h_last
