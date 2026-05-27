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
import torch.nn.functional as F


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

    spend_logvar_max: symmetric upper bound on the spend task's log_var (task 1).
        Without it the optimiser can drive log_var_spend toward the global +10
        clamp (precision exp(-10) ≈ 4.5e-5), silencing the spend head — the
        observed cause of negative spend R² on CDNOW/Ta-Feng/UCI. A value of 2.0
        floors spend precision at exp(-2)/2 ≈ 0.07. None (default) = unconstrained.
    """

    def __init__(
        self,
        n_tasks: int = 2,
        freq_logvar_max: float | None = None,
        spend_logvar_max: float | None = None,
        freq_logvar_min: float | None = None,
        spend_logvar_min: float | None = None,
    ):
        super().__init__()
        # log(σ²) parameterisation for numerical stability (σ² = exp(log_var))
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.freq_logvar_max = freq_logvar_max
        self.spend_logvar_max = spend_logvar_max
        # Lower bounds prevent the optimizer from driving log_var negative,
        # which would make the 0.5·log_var regularizer term dominate and cause
        # total Kendall loss to go negative → model collapses to zero predictions.
        # Setting min=0.0 guarantees each Kendall term ≥ 0 (since L_i ≥ 0).
        self.freq_logvar_min = freq_logvar_min
        self.spend_logvar_min = spend_logvar_min

    def _effective_log_var(self, i: int) -> torch.Tensor:
        """Per-task log_var with the global and per-task bounds applied.

        Used by both forward (loss/gradient) and task_weights (logging) so the
        reported weights always match the weights that actually drive training.
        """
        # Clamp guards against exp(-log_var) overflow if log_var drifts
        log_var = self.log_vars[i].clamp(-10.0, 10.0)
        # Optional upper bound on the frequency task (task 0) to prevent the
        # frequency head from being starved of gradient signal.
        if i == 0 and self.freq_logvar_max is not None:
            log_var = log_var.clamp(max=float(self.freq_logvar_max))
        if i == 0 and self.freq_logvar_min is not None:
            log_var = log_var.clamp(min=float(self.freq_logvar_min))
        # Symmetric bounds on the spend task (task 1).
        if i == 1 and self.spend_logvar_max is not None:
            log_var = log_var.clamp(max=float(self.spend_logvar_max))
        if i == 1 and self.spend_logvar_min is not None:
            log_var = log_var.clamp(min=float(self.spend_logvar_min))
        return log_var

    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List of per-task scalar loss tensors [L_freq, L_spend]
        Returns:
            total_loss: Scalar combined loss
        """
        total = torch.zeros((), device=losses[0].device, dtype=losses[0].dtype)
        for i, loss in enumerate(losses):
            log_var = self._effective_log_var(i)
            precision = torch.exp(-log_var)
            total = total + 0.5 * precision * loss + 0.5 * log_var
        return total

    @property
    def task_weights(self):
        """Return current effective task weights (1 / 2σ²) for logging.

        Mirrors the clamping in forward so logged weights reflect the weights
        that actually drive training.
        """
        eff = torch.stack([self._effective_log_var(i) for i in range(self.log_vars.numel())])
        return (0.5 * torch.exp(-eff)).detach()

    def freeze_log_vars(self):
        """Freeze log_vars at zero (equal task weighting) for warm-up epochs."""
        self.log_vars.requires_grad_(False)
        with torch.no_grad():
            self.log_vars.zero_()

    def unfreeze_log_vars(self):
        """Unfreeze log_vars so Kendall weighting adapts dynamically."""
        self.log_vars.requires_grad_(True)


class MetricAlignedFrequencyLoss(nn.Module):
    """
    Frequency loss for the prediction-repair protocol.

    The original replication objective is masked cross entropy over clipped count
    classes. That is paper-faithful, but it is a weak proxy for the metrics that
    decide whether a CLV forecast is useful: active-week detection, expected
    transaction volume, and aggregate cohort bias. This module keeps CE as the
    backbone and optionally adds those metric-aligned terms.
    """

    def __init__(
        self,
        config: dict | None = None,
        class_values: torch.Tensor | None = None,
    ):
        super().__init__()
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.ce_weight = float(cfg.get("ce_weight", 1.0))
        self.activity_bce_weight = float(cfg.get("activity_bce_weight", 0.0))
        self.count_huber_weight = float(cfg.get("count_huber_weight", 0.0))
        self.aggregate_bias_weight = float(cfg.get("aggregate_bias_weight", 0.0))
        self.temporal_tail_weight = float(cfg.get("temporal_tail_weight", 0.0))
        self.temporal_tail_power = float(cfg.get("temporal_tail_power", 1.0))
        self.class_weights = cfg.get("class_weights", None)
        self.class_weight_min = float(cfg.get("class_weight_min", 0.25))
        self.class_weight_max = float(cfg.get("class_weight_max", 4.0))
        self.huber_beta = float(cfg.get("huber_beta", 1.0))
        self.aggregate_bias_beta = float(cfg.get("aggregate_bias_beta", 0.25))
        self.eps = float(cfg.get("eps", 1e-6))

        if class_values is not None:
            self.register_buffer("class_values", class_values.detach().float())
        else:
            self.class_values = None

    def _class_values(self, logits: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        if self.class_values is not None and self.class_values.numel() == n_classes:
            return self.class_values.to(device=logits.device, dtype=logits.dtype)
        return torch.arange(n_classes, device=logits.device, dtype=logits.dtype)

    def _temporal_weights(self, mask: torch.Tensor) -> torch.Tensor:
        if self.temporal_tail_weight <= 0.0:
            return mask
        _, T = mask.shape
        pos = torch.linspace(
            1.0 / max(T, 1),
            1.0,
            T,
            device=mask.device,
            dtype=mask.dtype,
        )
        tail = pos.clamp_min(0.0).pow(self.temporal_tail_power)
        strength = max(0.0, min(1.0, self.temporal_tail_weight))
        weights = ((1.0 - strength) + strength * tail).unsqueeze(0) * mask
        denom = weights.sum().clamp(min=self.eps)
        # Normalize so enabling tail weighting changes emphasis, not loss scale.
        return weights * (mask.sum().clamp(min=self.eps) / denom)

    def _dynamic_class_weights(
        self,
        y_freq: torch.Tensor,
        mask: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor | None:
        if self.class_weights is None:
            return None
        if isinstance(self.class_weights, str):
            if self.class_weights.lower() != "auto":
                raise ValueError("frequency class_weights must be 'auto', a list, or null")
            counts = torch.zeros(n_classes, device=y_freq.device, dtype=torch.float32)
            valid_y = y_freq[mask.bool()].clamp(0, n_classes - 1)
            if valid_y.numel() == 0:
                return torch.ones(n_classes, device=y_freq.device, dtype=torch.float32)
            counts.scatter_add_(0, valid_y, torch.ones_like(valid_y, dtype=torch.float32))
            inv = (counts.sum() / counts.clamp(min=1.0)).sqrt()
            inv = inv / inv.mean().clamp(min=self.eps)
            return inv.clamp(self.class_weight_min, self.class_weight_max)

        weights = torch.as_tensor(
            self.class_weights,
            device=y_freq.device,
            dtype=torch.float32,
        )
        if weights.numel() != n_classes:
            raise ValueError(
                f"frequency class_weights has {weights.numel()} entries, "
                f"but logits have {n_classes} classes"
            )
        return weights

    @staticmethod
    def _reduce(per_step: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
        return (per_step * weights).sum() / weights.sum().clamp(min=eps)

    def forward(
        self,
        logits: torch.Tensor,
        y_freq: torch.Tensor,
        mask: torch.Tensor,
        y_freq_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, T, n_classes = logits.shape
        weights = self._temporal_weights(mask.to(dtype=logits.dtype))
        class_weights = self._dynamic_class_weights(y_freq, mask, n_classes)
        if class_weights is not None:
            class_weights = class_weights.to(dtype=logits.dtype)
        ce = F.cross_entropy(
            logits.reshape(-1, n_classes),
            y_freq.reshape(-1),
            weight=class_weights,
            reduction="none",
        ).reshape(B, T)
        ce_loss = self._reduce(ce, weights, self.eps)
        total = self.ce_weight * ce_loss

        metrics = {
            "freq_ce_loss": float(ce_loss.detach()),
        }
        if not self.enabled:
            return total, metrics

        target_count = (
            y_freq_raw.to(device=logits.device, dtype=logits.dtype)
            if y_freq_raw is not None
            else y_freq.to(device=logits.device, dtype=logits.dtype)
        )
        probs = torch.softmax(logits, dim=-1)
        class_values = self._class_values(logits)
        expected_count = (probs * class_values.view(1, 1, -1)).sum(dim=-1)

        if self.activity_bce_weight > 0.0:
            active_target = (target_count > 0).to(dtype=logits.dtype)
            # Clamp prevents -inf when the model is very confident about class 0
            # (inactive), which would make BCE produce +inf for active customers
            # and propagate NaN gradients back through the LSTM.
            active_logit = (
                torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
            ).clamp(-50.0, 50.0)
            activity = F.binary_cross_entropy_with_logits(
                active_logit,
                active_target,
                reduction="none",
            )
            activity_loss = self._reduce(activity, weights, self.eps)
            total = total + self.activity_bce_weight * activity_loss
            metrics["freq_activity_bce_loss"] = float(activity_loss.detach())

        if self.count_huber_weight > 0.0:
            count_huber = F.smooth_l1_loss(
                expected_count,
                target_count,
                beta=self.huber_beta,
                reduction="none",
            )
            count_huber_loss = self._reduce(count_huber, weights, self.eps)
            total = total + self.count_huber_weight * count_huber_loss
            metrics["freq_count_huber_loss"] = float(count_huber_loss.detach())

        if self.aggregate_bias_weight > 0.0:
            pred_total = (expected_count * weights).sum()
            true_total = (target_count * weights).sum()
            rel_bias = (pred_total - true_total) / true_total.abs().clamp(min=1.0)
            aggregate_loss = F.smooth_l1_loss(
                rel_bias,
                torch.zeros_like(rel_bias),
                beta=self.aggregate_bias_beta,
                reduction="mean",
            )
            total = total + self.aggregate_bias_weight * aggregate_loss
            metrics["freq_aggregate_bias_loss"] = float(aggregate_loss.detach())
            metrics["freq_batch_bias_pct"] = float((100.0 * rel_bias).detach())

        metrics["freq_loss"] = float(total.detach())
        return total, metrics
