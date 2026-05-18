"""
Validation-only calibration helpers for deep CLV models.

These utilities deliberately operate on validation data only. They do not touch
the final temporal holdout, which keeps thesis evaluation leakage-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel


@dataclass(frozen=True)
class AggregateCalibration:
    """Multiplicative calibration factors estimated on validation data."""

    freq_factor: float = 1.0
    spend_factor: float = 1.0


def conservative_ratio_factor(
    true_total: float,
    pred_total: float,
    *,
    shrinkage: float = 0.5,
    min_factor: float = 0.25,
    max_factor: float = 4.0,
) -> float:
    """
    Conservative multiplicative correction from aggregate validation totals.

    A raw true/pred ratio can overreact on sparse validation windows. We shrink
    the ratio halfway back to 1.0 by default and clip the final factor.
    """
    if true_total <= 0 or pred_total <= 0 or not np.isfinite(pred_total):
        return 1.0
    raw = true_total / pred_total
    shrunk = 1.0 + float(shrinkage) * (raw - 1.0)
    return float(np.clip(shrunk, min_factor, max_factor))


def _unpack_model_output(model, out):
    if isinstance(model, LSTMModel):
        if len(out) == 4:
            freq_logits, spend_mu, spend_log_var, _ = out
            return freq_logits, spend_mu, spend_log_var
        if len(out) == 3:
            freq_logits, spend_mu, _ = out
            return freq_logits, spend_mu, None
        freq_logits, _ = out
        return freq_logits, None, None
    if isinstance(model, TransformerModel):
        if model.joint:
            if len(out) == 3:
                return out
            freq_logits, spend_mu = out
            return freq_logits, spend_mu, None
        return out, None, None
    raise TypeError(f"Unsupported model type: {type(model)}")


@torch.no_grad()
def collect_teacher_forced_validation(
    model,
    val_loader,
    *,
    device: torch.device,
    scaler=None,
    temperature: float = 1.0,
) -> dict:
    """
    Collect teacher-forced validation predictions for calibration fitting.

    Frequency predictions use expected counts under the model softmax.
    Spend predictions are converted to raw spend if a scaler is supplied.
    """
    model.eval()
    temp = max(float(temperature), 1e-6)
    freq_true_total = 0.0
    freq_pred_total = 0.0
    spend_true_total = 0.0
    spend_pred_total = 0.0

    for batch in val_loader:
        week = batch["week"].to(device)
        trans = batch["trans"].to(device)
        spend = batch.get("spend")
        if spend is not None:
            spend = spend.to(device)
        position = batch.get("position")
        if position is not None:
            position = position.to(device)
        delta_t = batch.get("delta_t")
        if delta_t is not None:
            delta_t = delta_t.to(device)
        state_features = batch.get("state_features")
        if state_features is not None:
            state_features = state_features.to(device)
        mask = batch["mask"].to(device)
        y_freq = batch["y_freq"].to(device)
        y_spend = batch.get("y_spend")
        if y_spend is not None:
            y_spend = y_spend.to(device)

        static_cov = batch.get("static_covariates")
        if static_cov is not None:
            static_cov = static_cov.to(device)
        dynamic_cov = batch.get("dynamic_covariates")
        if dynamic_cov is not None:
            dynamic_cov = dynamic_cov.to(device)

        if isinstance(model, LSTMModel):
            out = model(
                week,
                trans,
                spend=spend,
                state_features=state_features,
                static_covariates=static_cov,
                dynamic_covariates=dynamic_cov,
            )
        else:
            out = model(
                week,
                trans,
                spend=spend,
                state_features=state_features,
                position=position,
                delta_t=delta_t,
                static_covariates=static_cov,
                dynamic_covariates=dynamic_cov,
            )

        freq_logits, spend_mu, _ = _unpack_model_output(model, out)
        probs = torch.softmax(freq_logits / temp, dim=-1)
        class_vals = torch.arange(probs.size(-1), dtype=probs.dtype, device=device)
        pred_freq = (probs * class_vals).sum(dim=-1)
        freq_true_total += float((y_freq.float() * mask).sum().cpu())
        freq_pred_total += float((pred_freq * mask).sum().cpu())

        if scaler is not None and spend_mu is not None and y_spend is not None:
            active = ((y_freq > 0).float() * mask).cpu().numpy().astype(np.float32)
            pred_activity = ((1.0 - probs[..., 0]) * mask).cpu().numpy().astype(np.float32)
            true_raw = scaler.inverse_transform_spend(y_spend.cpu().numpy()) * active
            pred_raw = scaler.inverse_transform_spend(spend_mu.cpu().numpy()) * pred_activity
            spend_true_total += float(np.sum(true_raw))
            spend_pred_total += float(np.sum(pred_raw))

    return {
        "freq_true_total": freq_true_total,
        "freq_pred_total": freq_pred_total,
        "spend_true_total": spend_true_total,
        "spend_pred_total": spend_pred_total,
    }


def fit_aggregate_calibration(
    validation_totals: dict,
    *,
    shrinkage: float = 0.5,
    min_factor: float = 0.25,
    max_factor: float = 4.0,
) -> AggregateCalibration:
    """Fit conservative frequency and spend aggregate factors."""
    return AggregateCalibration(
        freq_factor=conservative_ratio_factor(
            validation_totals.get("freq_true_total", 0.0),
            validation_totals.get("freq_pred_total", 0.0),
            shrinkage=shrinkage,
            min_factor=min_factor,
            max_factor=max_factor,
        ),
        spend_factor=conservative_ratio_factor(
            validation_totals.get("spend_true_total", 0.0),
            validation_totals.get("spend_pred_total", 0.0),
            shrinkage=shrinkage,
            min_factor=min_factor,
            max_factor=max_factor,
        ),
    )


def fit_temperature_from_loader(
    model,
    val_loader,
    *,
    device: torch.device,
    max_iter: int = 50,
) -> float:
    """Fit a scalar softmax temperature by minimizing validation CE."""
    model.eval()
    logits_all = []
    targets_all = []
    masks_all = []

    with torch.no_grad():
        for batch in val_loader:
            week = batch["week"].to(device)
            trans = batch["trans"].to(device)
            spend = batch.get("spend")
            if spend is not None:
                spend = spend.to(device)
            position = batch.get("position")
            if position is not None:
                position = position.to(device)
            delta_t = batch.get("delta_t")
            if delta_t is not None:
                delta_t = delta_t.to(device)
            state_features = batch.get("state_features")
            if state_features is not None:
                state_features = state_features.to(device)
            static_cov = batch.get("static_covariates")
            if static_cov is not None:
                static_cov = static_cov.to(device)
            dynamic_cov = batch.get("dynamic_covariates")
            if dynamic_cov is not None:
                dynamic_cov = dynamic_cov.to(device)

            if isinstance(model, LSTMModel):
                out = model(
                    week,
                    trans,
                    spend=spend,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                )
            else:
                out = model(
                    week,
                    trans,
                    spend=spend,
                    state_features=state_features,
                    position=position,
                    delta_t=delta_t,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                )
            freq_logits, _, _ = _unpack_model_output(model, out)
            logits_all.append(freq_logits.detach())
            targets_all.append(batch["y_freq"].to(device))
            masks_all.append(batch["mask"].to(device))

    if not logits_all:
        return 1.0
    logits = torch.cat([x.reshape(-1, x.size(-1)) for x in logits_all], dim=0)
    targets = torch.cat([x.reshape(-1) for x in targets_all], dim=0)
    mask = torch.cat([x.reshape(-1) for x in masks_all], dim=0).bool()
    logits = logits[mask]
    targets = targets[mask]
    if logits.numel() == 0:
        return 1.0

    log_temp = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(0.25, 4.0)
        loss = F.cross_entropy(logits / temp, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).clamp(0.25, 4.0).detach().cpu())
