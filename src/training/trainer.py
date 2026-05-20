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

from contextlib import nullcontext
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.training.inference import _zero_inactive_spend_feedback


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
        spend_loss_normalize: bool = False,
        scheduled_sampling: dict | None = None,
        amp_enabled: bool = True,
        dataset_name: str = "dataset",
        run_id: str = "run",
        checkpoint_dir: str | Path | None = None,
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
        # Optional spend-loss scale normalization (paired with Kendall warm-up):
        # divide spend_loss by its mean magnitude observed during warm-up so the
        # adaptive Kendall weighting operates on comparably-scaled tasks. Frozen
        # after warm-up. NOTE: shifts the absolute val-loss magnitude — early
        # stopping is relative + best-checkpoint restore, so this is safe.
        self.spend_loss_normalize = spend_loss_normalize
        self._spend_norm: float = 1.0
        self._spend_norm_sum: float = 0.0
        self._spend_norm_count: int = 0
        self._spend_norm_frozen: bool = False
        # Scheduled sampling (Bengio et al. 2015) — exposure-bias mitigation so
        # the autoregressive rollout stops collapsing on sparse data. LSTM-only,
        # config-gated, default OFF (preserves paper-faithful teacher forcing).
        self.scheduled_sampling = scheduled_sampling
        self._total_epochs: int = 1
        # Automatic mixed precision: uses FP16 Tensor Cores on T4/A100 for ~1.5x
        # speedup. Config-gated so numerical repair runs can fall back to FP32.
        # Disabled on CPU/MPS where AMP has no benefit.
        self.amp_enabled = bool(amp_enabled and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self._nan_count: int = 0
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("results/checkpoints")

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

    def _finalize_spend_norm(self) -> None:
        """Freeze the spend-loss normalizer at the mean warm-up magnitude."""
        if not self.spend_loss_normalize or self._spend_norm_frozen:
            return
        if self._spend_norm_count > 0:
            self._spend_norm = max(self._spend_norm_sum / self._spend_norm_count, 1e-6)
        self._spend_norm_frozen = True
        print(
            f"[spend-norm] frozen at {self._spend_norm:.4f} "
            f"(absolute val-loss magnitude shifts accordingly; "
            f"early stopping is relative so this is safe)"
        )

    @staticmethod
    def _assert_finite_loss(loss: torch.Tensor, stage: str) -> None:
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite {stage} loss: {loss.item()}")

    @staticmethod
    def _safe_token(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "run"

    def _save_last_valid_checkpoint(self) -> Path:
        dataset = self._safe_token(self.dataset_name)
        run = self._safe_token(self.run_id)
        path = self.checkpoint_dir / f"last_valid_{dataset}_{run}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

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

        # Restrict loss to active (purchase) weeks only.  Training on inactive weeks
        # (log1p(0) = 0 target) teaches the head to predict ≈0 everywhere → -93% bias.
        mse_per_step = F.mse_loss(spend_mu, y_spend, reduction="none")
        active = (active_mask * mask).to(dtype=mse_per_step.dtype)
        denom = active.sum().clamp(min=1.0)
        return (mse_per_step * active).sum() / denom

    def _ss_prob(self) -> float:
        """Scheduled-sampling probability ε for the current epoch (0 disables it).

        ε is the per-step probability of feeding the model's own previous
        prediction instead of the ground-truth token (Bengio et al. 2015).
        Ramps from 0 at start_epoch up to max_prob, linear or inverse-sigmoid.
        Only ever active for the LSTM during training.
        """
        cfg = self.scheduled_sampling
        if not cfg or not cfg.get("enabled", False):
            return 0.0
        if not isinstance(self.model, LSTMModel) or not self.model.training:
            return 0.0
        start = int(cfg.get("start_epoch", self.kendall_warmup_epochs))
        if self.current_epoch < start:
            return 0.0
        max_prob = float(cfg.get("max_prob", 0.25))
        total = max(int(self._total_epochs), start + 1)
        denom = max(total - start, 1)
        progress = (self.current_epoch - start) / denom  # 0 → ~1
        schedule = str(cfg.get("schedule", "linear")).lower()
        if schedule == "inv_sigmoid":
            # Inverse-sigmoid decay of teacher forcing → ε grows S-shaped.
            import math
            frac = 1.0 / (1.0 + math.exp(-6.0 * (progress - 0.5)))
        else:
            frac = progress
        return float(max(0.0, min(max_prob, max_prob * frac)))

    def _scheduled_sampling_forward_lstm(self, batch: dict, ss_prob: float):
        """Step-by-step LSTM unroll with token-level scheduled sampling.

        At each step t>0, with per-sample probability ss_prob, feed the model's
        own sampled previous frequency (and the matching predicted spend / delta_t
        state) instead of the ground-truth token — mirroring autoregressive
        inference so the model is trained on its own error distribution and the
        rollout stops collapsing on sparse data. Loss is still computed against
        the ground-truth targets in _reduce_losses.

        Returns (freq_logits (B,T,C), spend_mu (B,T)|None, spend_log_var|None).
        """
        week = batch["week"].to(self.device)
        trans = batch["trans"].to(self.device)
        B, T = week.shape
        joint = self.joint
        spend = batch.get("spend")
        spend = spend.to(self.device) if (spend is not None and joint) else None
        if joint and spend is None:
            spend = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        state_features = batch.get("state_features")
        has_state = (
            state_features is not None
            and getattr(self.model, "state_feature_dim", 0) > 0
        )
        if has_state:
            state_features = state_features.to(self.device)
        static_cov = batch.get("static_covariates")
        if static_cov is not None:
            static_cov = static_cov.to(self.device)
        dynamic_cov = batch.get("dynamic_covariates")
        if dynamic_cov is not None:
            dynamic_cov = dynamic_cov.to(self.device)

        hidden = None
        logits_steps: list = []
        mu_steps: list = []
        logvar_steps: list = []

        prev_class: torch.Tensor | None = None
        prev_spend: torch.Tensor | None = None
        run_delta: torch.Tensor | None = None
        if has_state:
            run_delta = state_features[:, 0, 0].clone()

        for t in range(T):
            if t == 0 or ss_prob <= 0.0:
                use_pred = torch.zeros(B, dtype=torch.bool, device=self.device)
            else:
                use_pred = torch.rand(B, device=self.device) < ss_prob

            trans_t = trans[:, t]
            if prev_class is not None:
                trans_t = torch.where(use_pred, prev_class, trans_t)

            spend_in = None
            if joint:
                spend_t = spend[:, t]
                if prev_spend is not None:
                    spend_t = torch.where(use_pred, prev_spend, spend_t)
                spend_in = spend_t.unsqueeze(1)

            state_in = None
            if has_state:
                true_state_t = state_features[:, t, :]
                if run_delta is not None:
                    pred_state_t = true_state_t.clone()
                    pred_state_t[:, 0] = run_delta
                    state_t = torch.where(
                        use_pred.unsqueeze(-1), pred_state_t, true_state_t
                    )
                else:
                    state_t = true_state_t
                state_in = state_t.unsqueeze(1)

            dyn_in = None
            if dynamic_cov is not None:
                dyn_in = dynamic_cov[:, t : t + 1, :]

            out = self.model(
                week[:, t : t + 1],
                trans_t.unsqueeze(1),
                hidden,
                spend=spend_in,
                state_features=state_in,
                static_covariates=static_cov,
                dynamic_covariates=dyn_in,
            )
            if joint:
                if len(out) == 4:
                    logits_t, mu_t, logvar_t, hidden = out
                    logvar_steps.append(logvar_t)
                else:
                    logits_t, mu_t, hidden = out
                mu_steps.append(mu_t)
            else:
                logits_t, hidden = out
            logits_steps.append(logits_t)

            # Build the model's own feedback for the next step (detached: the
            # discrete/continuous feedback is a constant input, as in inference).
            with torch.no_grad():
                probs = torch.softmax(logits_t[:, 0, :], dim=-1)
                prev_class = torch.multinomial(probs, 1).squeeze(-1)
                if joint:
                    prev_spend = _zero_inactive_spend_feedback(
                        mu_t[:, 0].detach(), prev_class
                    )
                if run_delta is not None:
                    purchased = (trans_t > 0).to(run_delta.dtype)
                    run_delta = (1.0 - purchased) * (run_delta + 1.0)

        freq_logits = torch.cat(logits_steps, dim=1)
        spend_mu = torch.cat(mu_steps, dim=1) if mu_steps else None
        spend_log_var = torch.cat(logvar_steps, dim=1) if logvar_steps else None
        return freq_logits, spend_mu, spend_log_var

    def _compute_loss(self, batch: dict):
        """
        Masked sequence-to-sequence loss.

        Returns:
            total_loss: scalar tensor
            metrics:    dict with float values for logging (includes task weights if joint)
        """
        mask = batch["mask"].to(self.device)
        y_freq = batch["y_freq"].to(self.device)

        ss_prob = self._ss_prob()
        if ss_prob > 0.0:
            freq_logits, spend_mu, spend_log_var = self._scheduled_sampling_forward_lstm(
                batch, ss_prob
            )
        else:
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
            if self.spend_loss_normalize:
                # Collect the raw spend-loss scale during the warm-up window
                # (training batches only); divide by the frozen scale thereafter.
                if (
                    not self._spend_norm_frozen
                    and self.model.training
                    and self.current_epoch < self.kendall_warmup_epochs
                ):
                    self._spend_norm_sum += float(spend_loss.detach())
                    self._spend_norm_count += 1
                spend_loss = spend_loss / self._spend_norm
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
        self._nan_count = 0
        totals = {}
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            amp_context = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if self.amp_enabled
                else nullcontext()
            )
            with amp_context:
                loss, metrics = self._compute_loss(batch)
            if not torch.isfinite(loss).item():
                self.optimizer.zero_grad()
                self._nan_count += 1
                print(
                    "WARNING: non-finite training loss "
                    f"at epoch {self.current_epoch + 1}, batch {batch_idx}; "
                    f"strike {self._nan_count}/5. Skipping batch."
                )
                if self._nan_count > 5:
                    path = self._save_last_valid_checkpoint()
                    raise RuntimeError(
                        f"Training aborted: >5 NaN losses in epoch {self.current_epoch + 1}. "
                        f"Last valid checkpoint saved to {path}."
                    )
                continue
            self.scaler.scale(loss).backward()
            # Unscale before gradient checks and clipping so they operate on the
            # true gradient magnitudes, not the AMP-scaled values.
            self.scaler.unscale_(self.optimizer)
            self._assert_finite_gradients()
            if self.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self._optimized_parameters(), self.max_grad_norm
                )
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError(
                        f"Non-finite gradient norm before clipping: {grad_norm.item()}"
                    )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

        if n_batches == 0:
            path = self._save_last_valid_checkpoint()
            raise RuntimeError(
                f"Training aborted: no finite training batches in epoch {self.current_epoch + 1}. "
                f"Last valid checkpoint saved to {path}."
            )
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
        self._total_epochs = int(epochs)  # used by the scheduled-sampling ε schedule
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
                # Freeze the spend-loss normalizer using the warm-up statistics
                # so it is a constant when Kendall starts adapting.
                self._finalize_spend_norm()

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
