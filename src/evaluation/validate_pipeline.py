"""Pre-holdout validation gates for trained deep CLV models."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel


def _to_device(batch: dict, key: str, device: torch.device):
    value = batch.get(key)
    return value.to(device) if value is not None else None


def _forward_once(model, batch: dict, device: torch.device):
    week = batch["week"].to(device)
    trans = batch["trans"].to(device)
    spend = _to_device(batch, "spend", device)
    state_features = _to_device(batch, "state_features", device)
    static_cov = _to_device(batch, "static_covariates", device)
    dynamic_cov = _to_device(batch, "dynamic_covariates", device)
    if dynamic_cov is not None and dynamic_cov.shape[1] != week.shape[1]:
        dynamic_cov = dynamic_cov[:, : week.shape[1], :]

    if isinstance(model, LSTMModel):
        out = model(
            week,
            trans,
            spend=spend,
            state_features=state_features,
            static_covariates=static_cov,
            dynamic_covariates=dynamic_cov,
        )
        if getattr(model, "joint", False):
            if len(out) == 4:
                freq_logits, spend_mu, spend_log_var, _ = out
                return freq_logits, (spend_mu, spend_log_var)
            freq_logits, spend_mu, _ = out
            return freq_logits, (spend_mu,)
        freq_logits, _ = out
        return freq_logits, ()

    if isinstance(model, TransformerModel):
        out = model(
            week,
            trans,
            spend=spend,
            state_features=state_features,
            position=_to_device(batch, "position", device),
            padding_mask=_to_device(batch, "padding_mask", device),
            static_covariates=static_cov,
            dynamic_covariates=dynamic_cov,
            delta_t=_to_device(batch, "delta_t", device),
        )
        if getattr(model, "joint", False):
            if len(out) == 3:
                freq_logits, spend_mu, spend_log_var = out
                return freq_logits, (spend_mu, spend_log_var)
            freq_logits, spend_mu = out
            return freq_logits, (spend_mu,)
        return out, ()

    raise TypeError(f"Unsupported model type for validation gate: {type(model)}")


def validate_pipeline_inputs(model, dataloader, config, checkpoint_path, device):
    """Validate core inputs and model outputs before holdout evaluation or plotting."""
    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError("Validation gate received an empty dataloader") from exc

    y_spend = batch.get("y_spend")
    if y_spend is None:
        raise ValueError("Validation gate requires batch['y_spend']")
    if not torch.all(torch.isfinite(y_spend)).item():
        raise ValueError("Non-finite spend targets in validation gate batch")

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise ValueError(f"Checkpoint does not exist: {ckpt}")
    try:
        torch.load(ckpt, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Checkpoint is not loadable: {ckpt}") from exc

    model.eval()
    with torch.no_grad():
        freq_logits, spend_outputs = _forward_once(model, batch, device)
        probs = torch.softmax(freq_logits, dim=-1)
        row_sums = probs.sum(dim=-1)
        if not torch.all((row_sums >= 0.999) & (row_sums <= 1.001)).item():
            raise ValueError("Softmax sanity check failed: probability row sums out of range")

        if bool(config.get("model", {}).get("joint", getattr(model, "joint", False))):
            if not spend_outputs:
                raise ValueError("Joint model validation expected spend outputs")
            for spend_pred in spend_outputs:
                if not torch.all(torch.isfinite(spend_pred)).item():
                    raise ValueError("Non-finite regression head output in validation gate")

    return True
