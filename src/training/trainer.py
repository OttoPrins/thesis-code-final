"""
Training loop for LSTM and Transformer CLV models.

Handles sequence-to-sequence training with masked cross-entropy loss.
Supports both base (frequency only) and joint (frequency + spend) models.

Architecture:
    - LSTMModel.forward(week, trans, hidden=None) → (freq_logits, hidden)
      or (freq_logits, log_spend, hidden) when joint=True
    - TransformerModel.forward(week, trans, padding_mask=None) → freq_logits
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
            log_spend:   (B, T) or None if not joint
        """
        week = batch["week"].to(self.device)    # (B, T-1)
        trans = batch["trans"].to(self.device)  # (B, T-1)
        mask = batch["mask"].to(self.device)    # (B, T-1)

        if isinstance(self.model, LSTMModel):
            if self.joint:
                freq_logits, log_spend, _ = self.model(week, trans)
            else:
                freq_logits, _ = self.model(week, trans)
                log_spend = None
        elif isinstance(self.model, TransformerModel):
            out = self.model(week, trans, padding_mask=mask)
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

        Cross-entropy and MSE are computed at every time step, then
        masked and averaged over valid (non-padded) positions.

        Returns:
            total_loss: scalar tensor
            metrics:    dict with float values for logging
        """
        mask = batch["mask"].to(self.device)     # (B, T-1), float32
        y_freq = batch["y_freq"].to(self.device)  # (B, T-1), long

        freq_logits, log_spend = self._forward(batch)
        B, T, n_classes = freq_logits.shape

        # Masked cross-entropy
        ce_per_step = F.cross_entropy(
            freq_logits.reshape(-1, n_classes),
            y_freq.reshape(-1),
            reduction="none",
        ).reshape(B, T)  # (B, T-1)

        freq_loss = (ce_per_step * mask).sum() / mask.sum()

        if self.joint:
            y_spend = batch["y_spend"].to(self.device)  # (B, T-1)
            mse_per_step = F.mse_loss(log_spend, y_spend, reduction="none")  # (B, T-1)
            spend_loss = (mse_per_step * mask).sum() / mask.sum()
            total_loss = self.multi_task_loss([freq_loss, spend_loss])
            return total_loss, {
                "total_loss": total_loss.item(),
                "freq_loss": freq_loss.item(),
                "spend_loss": spend_loss.item(),
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

        Args:
            train_loader:    DataLoader for training set
            val_loader:      DataLoader for validation set
            epochs:          Maximum number of epochs
            early_stopping:  EarlyStopping instance or None

        Returns:
            history: dict with lists of train/val losses per epoch
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Use total_loss for joint models, freq_loss for base
            train_loss = train_metrics.get("total_loss", train_metrics["freq_loss"])
            val_loss = val_metrics.get("total_loss", val_metrics["freq_loss"])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch + 1:3d}/{epochs} — "
                f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
                + (f"  freq: {val_metrics['freq_loss']:.4f}  spend: {val_metrics.get('spend_loss', float('nan')):.4f}" if self.joint else "")
            )

            if early_stopping is not None:
                if early_stopping(val_loss, self.model):
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    early_stopping.load_best(self.model)
                    break

        return history
