"""
Training loop for LSTM and Transformer CLV models.

Handles sequence-to-sequence training with masked cross-entropy loss.
Supports both base (frequency only) and joint (frequency + spend) models.
Supports optional covariate features (Extension 3 — Dunnhumby only).

Architecture:
    - LSTMModel.forward(week, trans, hidden=None, spend=None,
                        static_covariates=None, dynamic_covariates=None)
      → (freq_logits, hidden) or (freq_logits, log_spend, hidden) when joint=True
    - TransformerModel.forward(week, trans, spend=None, padding_mask=None,
                               static_covariates=None, dynamic_covariates=None)
      → freq_logits or (freq_logits, log_spend) when joint=True

Loss:
    - Base: masked mean CrossEntropy over all T time steps
    - Joint: Kendall homoscedastic uncertainty weighting of CE + MSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel


class Trainer:
    """
    Training loop for seq-to-seq CLV models.

    Args:
        model:            LSTMModel or TransformerModel instance
        optimizer:        Torch optimizer (Adam recommended)
        device:           torch.device
        joint:            Whether model has a spend regression head
        multi_task_loss:  KendallMultiTaskLoss instance (required if joint=True)
        max_grad_norm:    Gradient clipping threshold
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        joint: bool = False,
        multi_task_loss: nn.Module = None,
        max_grad_norm: float = 1.0,
        kendall_warmup_epochs: int = 5,
        spend_loss: str = "mse",
        scheduler=None,
        restore_best_checkpoint: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.joint = joint
        self.multi_task_loss = multi_task_loss
        self.max_grad_norm = max_grad_norm
        self.kendall_warmup_epochs = kendall_warmup_epochs
        self.spend_loss = spend_loss
        self.scheduler = scheduler
        self.restore_best_checkpoint = restore_best_checkpoint
        self.current_epoch: int = 0

    def _named_optimized_parameters(self):
        """Yield trainable parameters owned by the optimizer path."""
        yield from self.model.named_parameters()
        if self.multi_task_loss is not None:
            for name, param in self.multi_task_loss.named_parameters():
                yield f"multi_task_loss.{name}", param

    def _optimized_parameters(self) -> list[nn.Parameter]:
        return [
            param for _, param in self._named_optimized_parameters()
            if param.requires_grad
        ]

    @staticmethod
    def _assert_finite_loss(loss: torch.Tensor, stage: str) -> None:
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite {stage} loss: {loss.item()}")

    def _assert_finite_gradients(self) -> None:
        bad_params = []
        for name, param in self._named_optimized_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all().item():
                bad_params.append(name)
        if bad_params:
            shown = ", ".join(bad_params[:8])
            suffix = "" if len(bad_params) <= 8 else f", ... ({len(bad_params)} total)"
            raise FloatingPointError(f"Non-finite gradients in {shown}{suffix}")

    def _forward(self, batch: dict):
        """
        Run forward pass, normalising different return signatures.

        Returns:
            freq_logits:    (B, T, n_classes)
            spend_mu:       (B, T) or None
            spend_log_var:  (B, T) or None for hurdle/lognormal heads
        """
        week = batch["week"].to(self.device)
        position = batch.get("position")
        if position is not None:
            position = position.to(self.device)
        trans = batch["trans"].to(self.device)
        spend = batch.get("spend")
        if spend is not None:
            spend = spend.to(self.device)
        state_features = batch.get("state_features")
        if state_features is not None:
            state_features = state_features.to(self.device)
        mask = batch["mask"].to(self.device)

        # Covariates are optional (Extension 3 — Dunnhumby only)
        static_cov = None
        dynamic_cov = None
        if "static_covariates" in batch and batch["static_covariates"] is not None:
            static_cov = batch["static_covariates"].to(self.device)   # (B, S)
        if "dynamic_covariates" in batch and batch["dynamic_covariates"] is not None:
            dynamic_cov = batch["dynamic_covariates"].to(self.device)  # (B, T, D)

        if isinstance(self.model, LSTMModel):
            if self.joint:
                out = self.model(
                    week, trans,
                    spend=spend,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                )
                if len(out) == 4:
                    freq_logits, spend_mu, spend_log_var, _ = out
                else:
                    freq_logits, spend_mu, _ = out
                    spend_log_var = None
            else:
                freq_logits, _ = self.model(
                    week, trans,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                )
                spend_mu = None
                spend_log_var = None
        elif isinstance(self.model, TransformerModel):
            # Pass elapsed-time feature so Time2Vec learns inter-transaction
            # regularities (the BTYD signal) rather than absolute calendar position.
            delta_t = batch.get("delta_t")
            if delta_t is not None:
                delta_t = delta_t.to(self.device)
            # The training `mask` is a loss mask for valid customer-period targets.
            # Dense weekly grids are not padded sequences, so do not reuse it as an
            # attention padding mask. Only pass a true padding mask when a future
            # dataset/collate path provides one explicitly.
            padding_mask = batch.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            out = self.model(
                week, trans,
                spend=spend,
                state_features=state_features,
                position=position,
                padding_mask=padding_mask,
                static_covariates=static_cov,
                dynamic_covariates=dynamic_cov,
                delta_t=delta_t,
            )
            if self.joint:
                if len(out) == 3:
                    freq_logits, spend_mu, spend_log_var = out
                else:
                    freq_logits, spend_mu = out
                    spend_log_var = None
            else:
                freq_logits = out
                spend_mu = None
                spend_log_var = None
        else:
            raise TypeError(f"Unknown model type: {type(self.model)}")

        return freq_logits, spend_mu, spend_log_var

    def _compute_spend_loss(
        self,
        spend_mu: torch.Tensor,
        y_spend: torch.Tensor,
        mask: torch.Tensor,
        active_mask: torch.Tensor,
        spend_log_var: torch.Tensor | None,
    ) -> torch.Tensor:
        spend_head_type = getattr(self.model, "spend_head_type", "regression")
        if spend_head_type == "hurdle_lognormal":
            positive_mask = (active_mask * mask).to(dtype=spend_mu.dtype)
            denom = positive_mask.sum().clamp(min=1.0)
            if self.spend_loss == "huber":
                per_step = F.smooth_l1_loss(spend_mu, y_spend, reduction="none")
            else:
                if spend_log_var is None:
                    spend_log_var = torch.zeros_like(spend_mu)
                log_var = spend_log_var.clamp(-8.0, 8.0)
                per_step = 0.5 * (torch.exp(-log_var) * (y_spend - spend_mu) ** 2 + log_var)
            return (per_step * positive_mask).sum() / denom

        mse_per_step = F.mse_loss(spend_mu, y_spend, reduction="none")
        return (mse_per_step * mask).sum() / mask.sum().clamp(min=1.0)

    def _compute_loss(self, batch: dict):
        """
        Masked sequence-to-sequence loss.

        Returns:
            total_loss: scalar tensor
            metrics:    dict with float values for logging (includes task weights if joint)
        """
        mask = batch["mask"].to(self.device)
        y_freq = batch["y_freq"].to(self.device)

        freq_logits, spend_mu, spend_log_var = self._forward(batch)
        B, T, n_classes = freq_logits.shape

        ce_per_step = F.cross_entropy(
            freq_logits.reshape(-1, n_classes),
            y_freq.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        freq_loss = (ce_per_step * mask).sum() / mask.sum()

        if self.joint:
            y_spend = batch["y_spend"].to(self.device)
            active_mask = batch.get("active_mask")
            if active_mask is not None:
                active_mask = active_mask.to(self.device)
            else:
                active_mask = (y_freq > 0).float() * mask
            spend_loss = self._compute_spend_loss(
                spend_mu, y_spend, mask, active_mask, spend_log_var
            )
            total_loss = self.multi_task_loss([freq_loss, spend_loss])

            # Log Kendall task weights (exp(-log_var) per task)
            weights = self.multi_task_loss.task_weights.cpu().tolist()
            return total_loss, {
                "total_loss": total_loss.item(),
                "freq_loss": freq_loss.item(),
                "spend_loss": spend_loss.item(),
                "task_weight_freq": weights[0],
                "task_weight_spend": weights[1],
            }

        return freq_loss, {"freq_loss": freq_loss.item()}

    def train_epoch(self, dataloader) -> dict:
        """One training epoch. Returns average metrics dict."""
        self.model.train()
        totals = {}
        n_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss, metrics = self._compute_loss(batch)
            self._assert_finite_loss(loss, "training")
            loss.backward()
            self._assert_finite_gradients()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self._optimized_parameters(), self.max_grad_norm
                )
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError(
                        f"Non-finite gradient norm before clipping: {grad_norm.item()}"
                    )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    def validate(self, dataloader) -> dict:
        """Validation epoch (no gradients). Returns average metrics dict."""
        self.model.eval()
        totals = {}
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, metrics = self._compute_loss(batch)
                self._assert_finite_loss(loss, "validation")
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
                n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        early_stopping=None,
    ) -> dict:
        """
        Full training loop with optional early stopping.

        Returns:
            history: dict with per-epoch lists:
                train_loss, val_loss, and (if joint) task_weight_freq, task_weight_spend
        """
        history: dict = {"train_loss": [], "val_loss": []}
        if self.joint:
            history["task_weight_freq"] = []
            history["task_weight_spend"] = []

        # Freeze Kendall log_vars during warm-up so a large initial spend MSE
        # doesn't drive log_var_spend to +10 and collapse the spend loss weight.
        if self.joint and self.multi_task_loss is not None and self.kendall_warmup_epochs > 0:
            self.multi_task_loss.freeze_log_vars()
            print(f"[Kendall] log_vars frozen for {self.kendall_warmup_epochs} warm-up epochs.")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Unfreeze Kendall log_vars after warm-up
            if (
                self.joint
                and self.multi_task_loss is not None
                and epoch == self.kendall_warmup_epochs
            ):
                self.multi_task_loss.unfreeze_log_vars()
                print(f"[Kendall] log_vars unfrozen at epoch {epoch + 1}.")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            train_loss = train_metrics.get("total_loss", train_metrics["freq_loss"])
            val_loss = val_metrics.get("total_loss", val_metrics["freq_loss"])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.joint:
                history["task_weight_freq"].append(
                    train_metrics.get("task_weight_freq", float("nan"))
                )
                history["task_weight_spend"].append(
                    train_metrics.get("task_weight_spend", float("nan"))
                )

            print(
                f"Epoch {epoch + 1:3d}/{epochs} — "
                f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
                + (
                    f"  freq: {val_metrics['freq_loss']:.4f}"
                    f"  spend: {val_metrics.get('spend_loss', float('nan')):.4f}"
                    f"  w_freq: {train_metrics.get('task_weight_freq', float('nan')):.3f}"
                    f"  w_spend: {train_metrics.get('task_weight_spend', float('nan')):.3f}"
                    if self.joint else ""
                )
            )

            if early_stopping is not None:
                if early_stopping(val_loss, self.model):
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        if early_stopping is not None and self.restore_best_checkpoint:
            early_stopping.load_best(self.model)

        return history
