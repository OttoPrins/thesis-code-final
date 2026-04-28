"""
Training loop for LSTM and Transformer CLV models.

Handles sequence-to-sequence training with masked cross-entropy loss.
Supports both base (frequency only) and joint (frequency + spend) models.
Supports optional covariate features (Extension 3 — Dunnhumby only).

Architecture:
    - LSTMModel.forward(week, trans, hidden=None, covariates=None) → (freq_logits, hidden)
      or (freq_logits, log_spend, hidden) when joint=True
    - TransformerModel.forward(week, trans, padding_mask=None, covariates=None) → freq_logits
      or (freq_logits, log_spend) when joint=True

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.joint = joint
        self.multi_task_loss = multi_task_loss
        self.max_grad_norm = max_grad_norm

    def _forward(self, batch: dict):
        """
        Run forward pass, normalising different return signatures.

        Returns:
            freq_logits: (B, T, n_classes)
            log_spend:   (B, T) or None
        """
        week = batch["week"].to(self.device)
        trans = batch["trans"].to(self.device)
        mask = batch["mask"].to(self.device)

        # Covariates are optional (Extension 3 only)
        covariates = None
        if "covariates" in batch and batch["covariates"] is not None:
            covariates = batch["covariates"].to(self.device)  # (B, T, C)

        if isinstance(self.model, LSTMModel):
            if self.joint:
                freq_logits, log_spend, _ = self.model(week, trans, covariates=covariates)
            else:
                freq_logits, _ = self.model(week, trans, covariates=covariates)
                log_spend = None
        elif isinstance(self.model, TransformerModel):
            out = self.model(week, trans, padding_mask=mask, covariates=covariates)
            if self.joint:
                freq_logits, log_spend = out
            else:
                freq_logits = out
                log_spend = None
        else:
            raise TypeError(f"Unknown model type: {type(self.model)}")

        return freq_logits, log_spend

    def _compute_loss(self, batch: dict):
        """
        Masked sequence-to-sequence loss.

        Returns:
            total_loss: scalar tensor
            metrics:    dict with float values for logging (includes task weights if joint)
        """
        mask = batch["mask"].to(self.device)
        y_freq = batch["y_freq"].to(self.device)

        freq_logits, log_spend = self._forward(batch)
        B, T, n_classes = freq_logits.shape

        ce_per_step = F.cross_entropy(
            freq_logits.reshape(-1, n_classes),
            y_freq.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        freq_loss = (ce_per_step * mask).sum() / mask.sum()

        if self.joint:
            y_spend = batch["y_spend"].to(self.device)
            mse_per_step = F.mse_loss(log_spend, y_spend, reduction="none")
            # Mask zero-purchase weeks: spend regression target is 0 on inactive
            # weeks, which would teach the model to predict ~0 everywhere. Train
            # the spend head only on weeks where a purchase actually happened.
            activity_mask = (y_freq > 0).float()
            spend_mask = mask * activity_mask
            spend_denom = spend_mask.sum().clamp(min=1.0)
            spend_loss = (mse_per_step * spend_mask).sum() / spend_denom
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
            loss.backward()
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

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
                _, metrics = self._compute_loss(batch)
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

        for epoch in range(epochs):
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
                    early_stopping.load_best(self.model)
                    break

        return history
