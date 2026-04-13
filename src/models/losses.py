"""
Multi-task loss functions.

KendallMultiTaskLoss implements the homoscedastic uncertainty weighting from:
    Kendall, A., Gal, Y., & Cipolla, R. (2018).
    Multi-task learning using uncertainty to weigh losses in scene geometry and semantics.
    CVPR 2018.

Formula:
    L = Σ_i [ L_i / (2σ_i²) + log(σ_i) ]

where σ_i are learnable scalar uncertainty parameters, one per task.
Initialised to 1.0. The log(σ_i) term acts as a regulariser to prevent σ_i → ∞.

Usage:
    loss_fn = KendallMultiTaskLoss(n_tasks=2)
    total_loss = loss_fn([freq_loss, spend_loss])
"""

import torch
import torch.nn as nn


class KendallMultiTaskLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task learning.
    Automatically balances CrossEntropy (frequency) and MSE (log-spend) losses.
    """

    def __init__(self, n_tasks: int = 2):
        super().__init__()
        # log(σ²) parameterisation for numerical stability (σ² = exp(log_var))
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List of per-task scalar loss tensors [L_freq, L_spend]
        Returns:
            total_loss: Scalar combined loss
        """
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    @property
    def task_weights(self):
        """Return current effective task weights (1 / 2σ²) for logging."""
        return torch.exp(-self.log_vars).detach()
