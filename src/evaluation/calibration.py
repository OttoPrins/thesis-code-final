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
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import aggregate_spend_to_raw_total, compute_smearing_factor
from src.evaluation.rolling_origin import origins_for_dataset
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.models.covariates import align_dynamic_covariates
from src.training.inference import (
    autoregressive_inference_lstm,
    autoregressive_inference_transformer,
)


def _rolling_collate(batch):
    """Collate minimal inference batches for rolling-origin calibration."""
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = torch.as_tensor(values)
    return result


@dataclass(frozen=True)
class AggregateCalibration:
    """Multiplicative calibration factors estimated on validation data."""

    freq_factor: float = 1.0
    spend_factor: float = 1.0


class RollingOriginInferenceDataset(Dataset):
    """
    Inference-style view over validation customers for one rolling-origin window.

    The base dataset must be a CustomerDataset built with include_seed=True. We
    use the first `train_weeks` calibration periods as the autoregressive seed
    and score the next `validation_weeks` periods, all before the final holdout.
    """

    def __init__(self, base_dataset, train_weeks: int, validation_weeks: int):
        self.base = base_dataset
        self.train_weeks = int(train_weeks)
        self.validation_weeks = int(validation_weeks)
        seed_week = getattr(base_dataset, "seed_week", None)
        if seed_week is None:
            raise ValueError("Rolling-origin calibration requires include_seed=True validation data")
        total_weeks = int(seed_week.shape[1])
        if self.train_weeks <= 0 or self.validation_weeks <= 0:
            raise ValueError("rolling-origin train_weeks and validation_weeks must be positive")
        if self.train_weeks + self.validation_weeks > total_weeks:
            raise ValueError(
                "rolling-origin window extends beyond the available calibration history: "
                f"{self.train_weeks}+{self.validation_weeks}>{total_weeks}"
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        start = 0
        stop = self.train_weeks
        item = {
            "customer_id": self.base.customer_ids[idx],
            "seed_week": self.base.seed_week[idx, start:stop],
            "seed_trans": self.base.seed_trans[idx, start:stop],
            "seed_spend": self.base.seed_spend[idx, start:stop],
        }
        if getattr(self.base, "seed_position", None) is not None:
            item["seed_position"] = self.base.seed_position[idx, start:stop]
        if getattr(self.base, "seed_delta_t", None) is not None:
            item["seed_delta_t"] = self.base.seed_delta_t[idx, start:stop]
        if getattr(self.base, "seed_state_features", None) is not None:
            item["seed_state_features"] = self.base.seed_state_features[idx, start:stop]
        if getattr(self.base, "seed_raw_trans", None) is not None:
            item["seed_raw_trans"] = self.base.seed_raw_trans[idx, start:stop]
        if getattr(self.base, "static_covariates", None) is not None:
            item["static_covariates"] = self.base.static_covariates[idx]
        if getattr(self.base, "dynamic_covariates", None) is not None:
            item["dynamic_covariates"] = self.base.dynamic_covariates[idx]
        return item

    def true_freq(self) -> np.ndarray:
        start = self.train_weeks
        stop = self.train_weeks + self.validation_weeks
        raw = getattr(self.base, "seed_raw_trans", None)
        if raw is None:
            raw = self.base.seed_trans
        return raw[:, start:stop].detach().cpu().numpy().astype(np.float32)

    def true_spend(self) -> np.ndarray:
        start = self.train_weeks
        stop = self.train_weeks + self.validation_weeks
        return self.base.seed_spend[:, start:stop].detach().cpu().numpy().astype(np.float32)

    def customer_ids(self) -> np.ndarray:
        return self.base.customer_ids.detach().cpu().numpy()


def validate_rolling_origins(dataset_cfg: dict, origins: list[tuple[int, int]]) -> None:
    """Ensure rolling-origin windows stay strictly inside the calibration period."""
    calibration_weeks = int(dataset_cfg["calibration_weeks"])
    for train_weeks, validation_weeks in origins:
        train_weeks = int(train_weeks)
        validation_weeks = int(validation_weeks)
        if train_weeks <= 0 or validation_weeks <= 0:
            raise ValueError(
                f"Invalid rolling origin ({train_weeks}, {validation_weeks}): "
                "both values must be positive."
            )
        if train_weeks + validation_weeks > calibration_weeks:
            raise ValueError(
                f"Rolling origin ({train_weeks}, {validation_weeks}) reaches week "
                f"{train_weeks + validation_weeks}, beyond calibration_weeks={calibration_weeks}. "
                "Rolling-origin calibration must not touch the final holdout."
            )


def _configured_rolling_origins(dataset_cfg: dict, validation_cfg: dict) -> list[tuple[int, int]]:
    origins = origins_for_dataset(dataset_cfg.get("name", ""), validation_cfg.get("origins"))
    validate_rolling_origins(dataset_cfg, origins)
    return origins


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
    class_values: torch.Tensor | np.ndarray | None = None,
    smearing_factor: float | None = None,
) -> dict:
    """
    Collect teacher-forced validation predictions for calibration fitting.

    Frequency predictions use expected counts under the model softmax.
    Spend predictions are converted to raw spend if a scaler is supplied.
    """
    model.eval()
    temp = max(float(temperature), 1e-6)
    class_vals = None
    if class_values is not None:
        class_vals = torch.as_tensor(class_values, dtype=torch.float32, device=device)
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
            dynamic_cov = align_dynamic_covariates(dynamic_cov, week.shape[1])

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
        if class_vals is None:
            class_vals_batch = torch.arange(probs.size(-1), dtype=probs.dtype, device=device)
        else:
            if class_vals.numel() != probs.size(-1):
                raise ValueError(
                    f"class_values length {class_vals.numel()} does not match "
                    f"model classes {probs.size(-1)}"
                )
            class_vals_batch = class_vals.to(dtype=probs.dtype)
        pred_freq = (probs * class_vals_batch).sum(dim=-1)
        freq_true_total += float((y_freq.float() * mask).sum().cpu())
        freq_pred_total += float((pred_freq * mask).sum().cpu())

        if scaler is not None and spend_mu is not None and y_spend is not None:
            active = ((y_freq > 0).float() * mask).cpu().numpy().astype(np.float32)
            pred_activity = ((1.0 - probs[..., 0]) * mask).cpu().numpy().astype(np.float32)
            true_raw = scaler.inverse_transform_spend(y_spend.cpu().numpy()) * active
            pred_raw = scaler.inverse_transform_spend(spend_mu.cpu().numpy()) * pred_activity
            if smearing_factor is not None:
                pred_raw = pred_raw * float(smearing_factor)
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


@torch.no_grad()
def collect_autoregressive_rolling_validation(
    model,
    val_dataset,
    *,
    dataset_cfg: dict,
    validation_cfg: dict,
    batch_size: int,
    device: torch.device,
    scaler=None,
    temperature: float = 1.0,
    class_values: torch.Tensor | np.ndarray | None = None,
    smearing_factor: float | None = None,
    mode: str = "expected",
    n_scenarios: int = 1,
    use_kv_cache: bool = True,
) -> dict:
    """
    Score held-out validation customers with autoregressive rolling origins.

    Unlike teacher-forced calibration, this matches the final holdout mechanism:
    a prefix is used as a seed and future weeks are generated step by step. All
    origins must lie within the calibration period, so the final holdout remains
    untouched.
    """
    origins = _configured_rolling_origins(dataset_cfg, validation_cfg)
    if not origins or len(val_dataset) == 0:
        return {
            "freq_true_total": 0.0,
            "freq_pred_total": 0.0,
            "spend_true_total": 0.0,
            "spend_pred_total": 0.0,
            "n_origins": 0,
            "n_customers": int(len(val_dataset)),
        }

    class_vals = None
    if class_values is not None:
        class_vals = torch.as_tensor(class_values, dtype=torch.float32)

    freq_true_total = 0.0
    freq_pred_total = 0.0
    spend_true_total = 0.0
    spend_pred_total = 0.0
    true_freq_by_origin = []
    pred_freq_by_origin = []
    true_log_active = []
    pred_log_active = []

    for train_weeks, validation_weeks in origins:
        ro_ds = RollingOriginInferenceDataset(val_dataset, train_weeks, validation_weeks)
        loader = DataLoader(
            ro_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_rolling_collate,
            num_workers=0,
        )
        if isinstance(model, LSTMModel):
            results = autoregressive_inference_lstm(
                model=model,
                inference_loader=loader,
                holdout_weeks=validation_weeks,
                calibration_weeks=train_weeks,
                n_scenarios=n_scenarios,
                device=device,
                mode=mode,
                temperature=temperature,
                class_values=class_vals,
            )
        elif isinstance(model, TransformerModel):
            results = autoregressive_inference_transformer(
                model=model,
                inference_loader=loader,
                holdout_weeks=validation_weeks,
                calibration_weeks=train_weeks,
                n_scenarios=n_scenarios,
                device=device,
                use_kv_cache=use_kv_cache,
                mode=mode,
                temperature=temperature,
                class_values=class_vals,
            )
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        true_ids = ro_ds.customer_ids()
        if not np.array_equal(results["customer_ids"], true_ids):
            raise AssertionError("Rolling-origin customer ordering changed during inference")

        true_freq = ro_ds.true_freq()
        pred_freq = np.asarray(results["pred_freq"], dtype=np.float32)
        freq_true_total += float(true_freq.sum())
        freq_pred_total += float(pred_freq.sum())
        true_freq_by_origin.append(true_freq.sum(axis=1))
        pred_freq_by_origin.append(pred_freq.sum(axis=1))

        if scaler is not None and "pred_spend" in results:
            true_spend = ro_ds.true_spend()
            pred_spend = np.asarray(results["pred_spend"], dtype=np.float32)
            pred_activity = np.asarray(results["pred_activity"], dtype=np.float32)
            true_activity = (true_freq > 0).astype(np.float32)
            true_total = aggregate_spend_to_raw_total(
                true_spend,
                scaler,
                activity_weights=true_activity,
            )
            pred_total = aggregate_spend_to_raw_total(
                pred_spend,
                scaler,
                activity_weights=pred_activity,
                smearing_factor=smearing_factor,
            )
            spend_true_total += float(true_total.sum())
            spend_pred_total += float(pred_total.sum())

            active = true_activity.astype(bool)
            if np.any(active):
                true_log = scaler.inverse_transform_log1p(true_spend)
                pred_log = scaler.inverse_transform_log1p(pred_spend)
                true_log_active.append(true_log[active])
                pred_log_active.append(pred_log[active])

    out = {
        "freq_true_total": freq_true_total,
        "freq_pred_total": freq_pred_total,
        "spend_true_total": spend_true_total,
        "spend_pred_total": spend_pred_total,
        "n_origins": len(origins),
        "n_customers": int(len(val_dataset)),
    }
    if true_freq_by_origin:
        true_freq_pc = np.concatenate(true_freq_by_origin)
        pred_freq_pc = np.concatenate(pred_freq_by_origin)
        out["freq_mae"] = float(np.mean(np.abs(true_freq_pc - pred_freq_pc)))
        out["freq_rmse"] = float(np.sqrt(np.mean((true_freq_pc - pred_freq_pc) ** 2)))
    if true_log_active and pred_log_active:
        true_log = np.concatenate(true_log_active)
        pred_log = np.concatenate(pred_log_active)
        out["smearing_factor"] = compute_smearing_factor(true_log, pred_log)
    return out


def fit_temperature_from_rolling_origin(
    model,
    val_dataset,
    *,
    dataset_cfg: dict,
    validation_cfg: dict,
    batch_size: int,
    device: torch.device,
    class_values: torch.Tensor | np.ndarray | None = None,
    candidates: list[float] | None = None,
    mode: str = "expected",
    n_scenarios: int = 1,
    use_kv_cache: bool = True,
) -> float:
    """Pick a scalar temperature by autoregressive rolling-origin validation."""
    if candidates is None:
        candidates = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    best_temp = 1.0
    best_score = float("inf")
    for temp in candidates:
        totals = collect_autoregressive_rolling_validation(
            model,
            val_dataset,
            dataset_cfg=dataset_cfg,
            validation_cfg=validation_cfg,
            batch_size=batch_size,
            device=device,
            scaler=None,
            temperature=float(temp),
            class_values=class_values,
            mode=mode,
            n_scenarios=n_scenarios,
            use_kv_cache=use_kv_cache,
        )
        score = float(totals.get("freq_mae", float("inf")))
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_temp = float(temp)
    return best_temp


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
                dynamic_cov = align_dynamic_covariates(dynamic_cov, week.shape[1])

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
