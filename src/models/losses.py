"""
Multi-task loss functions.

KendallMultiTaskLoss implements the homoscedastic uncertainty weighting from:
    Kendall, A., Gal, Y., & Cipolla, R. (2018).
    Multi-task learning using uncertainty to weigh losses in scene geometry and semantics.
    CVPR 2018.

Canonical form (Equation 7 in Kendall et al. 2018), with log_var = log(σ²):
    L = Σ_i [ 0.5 · exp(-log_var_i) · L_i + 0.5 · log_var_i ]

The 0.5 factors are the canonical constants; omitting them multiplies the loss
by a scalar (equivalent to a doubled learning rate) and makes results
non-comparable to the published formulation.

The log_var term acts as a regulariser preventing σ_i → ∞.

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

    freq_logvar_max: optional upper bound on the frequency task's log_var.
        Prevents the frequency head from being down-weighted to near-zero by the
        spend loss during joint training (multi-task interference).  A value of
        2.0 keeps the frequency precision above exp(-2)/2 ≈ 0.07 — meaningful
        gradient flows to the frequency head throughout training.
        Set to None (default) for unconstrained behaviour.
    """

    def __init__(self, n_tasks: int = 2, freq_logvar_max: float | None = None):
        super().__init__()
        # log(σ²) parameterisation for numerical stability (σ² = exp(log_var))
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.freq_logvar_max = freq_logvar_max

    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List of per-task scalar loss tensors [L_freq, L_spend]
        Returns:
            total_loss: Scalar combined loss
        """
        total = torch.zeros((), device=losses[0].device, dtype=losses[0].dtype)
        for i, loss in enumerate(losses):
            # Clamp guards against exp(-log_var) overflow if log_var drifts
            log_var = self.log_vars[i].clamp(-10.0, 10.0)
            # Apply optional upper bound on the frequency task's log_var (task 0)
            # to prevent the frequency head from being starved of gradient signal.
            if i == 0 and self.freq_logvar_max is not None:
                log_var = log_var.clamp(max=float(self.freq_logvar_max))
            precision = torch.exp(-log_var)
            total = total + 0.5 * precision * loss + 0.5 * log_var
        return total

    @property
    def task_weights(self):
        """Return current effective task weights (1 / 2σ²) for logging."""
        return (0.5 * torch.exp(-self.log_vars)).detach()

    def freeze_log_vars(self):
        """Freeze log_vars at zero (equal task weighting) for warm-up epochs."""
        self.log_vars.requires_grad_(False)
        with torch.no_grad():
            self.log_vars.zero_()

    def unfreeze_log_vars(self):
        """Unfreeze log_vars so Kendall weighting adapts dynamically."""
        self.log_vars.requires_grad_(True)
