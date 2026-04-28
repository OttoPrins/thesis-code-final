"""
Autoregressive inference for holdout period evaluation.

Two inference modes are supported:

    mode='sample'  — Monte Carlo (Valendin et al. 2022):
        At each step, sample next-period transactions from the softmax via
        torch.multinomial. Average n_scenarios stochastic paths to reduce noise.

    mode='expected' — Deterministic expected value:
        At each step, take the softmax expectation E[k] = Σ k·p(k). Round to the
        nearest valid class for the next-step embedding lookup but track the
        unrounded expectation as the prediction. Lower variance than MC; helps
        in sparse regimes (e.g. CDNOW) where multinomial sampling can drift
        positive once a non-zero class is drawn. n_scenarios is forced to 1 in
        expected mode (the result is deterministic).

LSTM inference procedure:
    1. Feed full calibration seed through LSTM to warm up hidden state (h, c).
    2. Take the first prediction from the last calibration-step softmax.
    3. For each subsequent holdout week: feed (prev_trans, week_idx) as a single
       time step, carry (h, c) forward, take new prediction.

Transformer inference procedure (with KV caching):
    1. Full forward pass over calibration seed → collect per-layer KV cache.
    2. Take prediction from the last position of the warmup output.
    3. For each holdout week: forward single new token using cached K,V (O(1) per step).

Holdout week indices cycle through 0..(calibration_weeks-1) using modulo so the model
sees the same week-of-year embeddings it was trained on (Valendin et al. week-of-year 0-51).

Spend prediction:
    The model also outputs per-step log-spend, which the caller post-processes
    with aggregate_spend_to_raw_total() in evaluation.metrics. To match the
    masked-spend training (loss only on weeks where freq>0), the inference loop
    also returns a per-week activity weight:
        sample mode:   1 if sampled freq>0, else 0
        expected mode: P(freq>0) = 1 - softmax(logits)[:, 0]
    The caller multiplies raw per-week spend by this weight before summing, so
    inactive weeks contribute zero raw revenue.

Covariate support (Extension 3):
    When the inference batch contains a "covariates" key of shape (B, T_total, C),
    the calibration slice [:,  :calibration_weeks, :] is fed during warm-up, and
    covariate[:, calibration_weeks + h, :] is fed at each holdout step h.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel


def _step_from_logits(
    logits: torch.Tensor,
    mode: str,
    max_trans: int,
) -> tuple:
    """
    Convert per-step softmax logits to (next_input_class, prediction_value, P(active)).

    Args:
        logits:    (B, n_classes) raw logits at one time step
        mode:      'sample' (multinomial) or 'expected' (deterministic E[k])
        max_trans: top class index (for clamping the expected-mode embedding lookup)

    Returns:
        next_class: (B,) long — what to feed back into the next step's embedding
        pred:       (B,) float — prediction added to the holdout running total
        p_active:   (B,) float — P(freq>0) for spend gating
    """
    probs = torch.softmax(logits, dim=-1)
    p_active = 1.0 - probs[:, 0]
    if mode == "sample":
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        return sampled, sampled.float(), p_active
    if mode == "expected":
        n_classes = probs.size(-1)
        class_vals = torch.arange(n_classes, dtype=probs.dtype, device=probs.device)
        ev = (probs * class_vals).sum(dim=-1)  # (B,)
        next_class = ev.round().long().clamp(0, max_trans)
        return next_class, ev, p_active
    raise ValueError(f"Unknown inference mode: {mode!r} — expected 'sample' or 'expected'.")


@torch.no_grad()
def autoregressive_inference_lstm(
    model: LSTMModel,
    inference_loader: DataLoader,
    holdout_weeks: int,
    calibration_weeks: int,
    n_scenarios: int = 30,
    device: Optional[torch.device] = None,
    mode: str = "sample",
) -> dict:
    """
    Run autoregressive LSTM inference over the holdout period.

    The inference_loader must come from a CustomerDataset built with
    include_seed=True. DataLoader must use shuffle=False to preserve ordering.

    Args:
        model:             Trained LSTMModel (joint or base)
        inference_loader:  DataLoader with seed sequences, shuffle=False
        holdout_weeks:     Number of holdout steps to generate (H)
        calibration_weeks: Number of calibration steps (T)
        n_scenarios:       Monte Carlo paths to average (>= 20 per Valendin et al.).
                           Forced to 1 when mode='expected' (deterministic).
        device:            torch.device; defaults to model's parameter device
        mode:              'sample' (default) or 'expected' — see module docstring.

    Returns:
        dict with:
            customer_ids:    (N,) np.ndarray int64
            pred_freq:       (N, H) float32 — per-week mean predicted freq
            pred_total_freq: (N,) float32 — total predicted transactions
            pred_activity:   (N, H) float32 — per-week activity weight:
                              sample mode    → 1 if sampled freq>0, else 0
                              expected mode  → P(freq>0) = 1 - softmax_0
                              Caller multiplies raw spend by this before summing.
            pred_spend:      (N, H) float32 — per-week mean scaled-log spend (joint only)

        Spend totals in raw currency must be computed by the caller via
        aggregate_spend_to_raw_total(pred_spend, scaler, activity_weights=pred_activity).
    """
    if device is None:
        device = next(model.parameters()).device
    if mode == "expected":
        n_scenarios = 1  # deterministic; averaging adds nothing

    model.eval()
    H = holdout_weeks
    joint = model.joint
    max_trans = model.max_trans

    all_customer_ids = []
    all_pred_freq = []
    all_pred_activity = []
    all_pred_spend: list = [] if joint else None  # type: ignore[assignment]

    for batch in inference_loader:
        seed_week = batch["seed_week"].to(device)    # (B, T_calib)
        seed_trans = batch["seed_trans"].to(device)  # (B, T_calib)
        customer_ids = batch["customer_id"].numpy()
        B = seed_week.size(0)

        # Covariates: (B, T_total, C) or None
        covariates = None
        if "covariates" in batch and batch["covariates"] is not None:
            covariates = batch["covariates"].to(device)

        scenario_freq = np.zeros((B, H), dtype=np.float64)
        scenario_activity = np.zeros((B, H), dtype=np.float64)
        scenario_spend = np.zeros((B, H), dtype=np.float64) if joint else None

        for _ in range(n_scenarios):
            # Warm up: feed full calibration seed
            cov_warmup = covariates[:, :calibration_weeks, :] if covariates is not None else None
            if joint:
                freq_logits, log_spend, hidden = model(
                    seed_week, seed_trans, covariates=cov_warmup
                )
            else:
                freq_logits, hidden = model(seed_week, seed_trans, covariates=cov_warmup)

            # Step 0 of holdout from last calibration output
            last_logits = freq_logits[:, -1, :]
            prev_trans, pred_val, p_active = _step_from_logits(last_logits, mode, max_trans)

            preds = np.zeros((B, H), dtype=np.float32)
            activity = np.zeros((B, H), dtype=np.float32)
            preds[:, 0] = pred_val.cpu().numpy()
            activity[:, 0] = p_active.cpu().numpy()

            spend_preds = np.zeros((B, H), dtype=np.float32) if joint else None
            if joint:
                spend_preds[:, 0] = log_spend[:, -1].cpu().numpy()

            for h in range(1, H):
                # Feed the week-of-year for the *previous* holdout step (h-1), since
                # that is the "current" context from which we predict step h.
                week_idx = (h - 1) % calibration_weeks
                week_input = torch.full((B, 1), week_idx, dtype=torch.long, device=device)
                trans_input = prev_trans.unsqueeze(1)

                # Per-step covariate slice: (B, 1, C)
                cov_step = None
                if covariates is not None:
                    step_idx = calibration_weeks + h
                    if step_idx < covariates.size(1):
                        cov_step = covariates[:, step_idx: step_idx + 1, :]

                if joint:
                    freq_logits_step, log_spend_step, hidden = model(
                        week_input, trans_input, hidden, covariates=cov_step
                    )
                    spend_preds[:, h] = log_spend_step[:, 0].cpu().numpy()
                else:
                    freq_logits_step, hidden = model(
                        week_input, trans_input, hidden, covariates=cov_step
                    )

                prev_trans, pred_val, p_active = _step_from_logits(
                    freq_logits_step[:, 0, :], mode, max_trans
                )
                preds[:, h] = pred_val.cpu().numpy()
                activity[:, h] = p_active.cpu().numpy()

            scenario_freq += preds
            scenario_activity += activity
            if joint:
                scenario_spend += spend_preds

        avg_freq = (scenario_freq / n_scenarios).astype(np.float32)
        avg_activity = (scenario_activity / n_scenarios).astype(np.float32)
        all_customer_ids.append(customer_ids)
        all_pred_freq.append(avg_freq)
        all_pred_activity.append(avg_activity)

        if joint:
            avg_spend = (scenario_spend / n_scenarios).astype(np.float32)
            all_pred_spend.append(avg_spend)

    all_customer_ids = np.concatenate(all_customer_ids)
    all_pred_freq = np.concatenate(all_pred_freq, axis=0)
    all_pred_activity = np.concatenate(all_pred_activity, axis=0)

    result = {
        "customer_ids": all_customer_ids,
        "pred_freq": all_pred_freq,
        "pred_total_freq": all_pred_freq.sum(axis=1),
        "pred_activity": all_pred_activity,
    }

    if joint:
        all_pred_spend_arr = np.concatenate(all_pred_spend, axis=0)
        result["pred_spend"] = all_pred_spend_arr

    return result


@torch.no_grad()
def autoregressive_inference_transformer(
    model: TransformerModel,
    inference_loader: DataLoader,
    holdout_weeks: int,
    calibration_weeks: int,
    n_scenarios: int = 30,
    device: Optional[torch.device] = None,
    use_kv_cache: bool = True,
    mode: str = "sample",
) -> dict:
    """
    Run autoregressive Transformer inference over the holdout period.

    When use_kv_cache=True (default), the calibration seed is encoded once and its
    per-layer K,V matrices are cached; each holdout step only encodes the single new
    token, giving O(H) total complexity instead of O(H*T²).

    Args:
        model:             Trained TransformerModel (joint or base)
        inference_loader:  DataLoader with seed sequences, shuffle=False
        holdout_weeks:     Number of holdout steps H
        calibration_weeks: Number of calibration steps T
        n_scenarios:       Monte Carlo paths to average. Forced to 1 in 'expected' mode.
        device:            torch.device
        use_kv_cache:      If True (default), use KV caching; False for correctness checks
        mode:              'sample' (default) or 'expected' — see module docstring.

    Returns:
        dict with same keys as autoregressive_inference_lstm
        (customer_ids, pred_freq, pred_total_freq, pred_activity, [pred_spend]).
    """
    if device is None:
        device = next(model.parameters()).device
    if mode == "expected":
        n_scenarios = 1

    model.eval()
    H = holdout_weeks
    joint = model.joint
    max_trans = getattr(model, "max_trans", None)
    if max_trans is None:
        # TransformerModel may store n_classes - 1 elsewhere; fall back to embedding size
        max_trans = model.embed_trans.num_embeddings - 1

    all_customer_ids = []
    all_pred_freq = []
    all_pred_activity = []
    all_pred_spend: list = [] if joint else None  # type: ignore[assignment]

    for batch in inference_loader:
        seed_week = batch["seed_week"].to(device)    # (B, T_calib)
        seed_trans = batch["seed_trans"].to(device)  # (B, T_calib)
        customer_ids = batch["customer_id"].numpy()
        B = seed_week.size(0)

        covariates = None
        if "covariates" in batch and batch["covariates"] is not None:
            covariates = batch["covariates"].to(device)  # (B, T_total, C)

        scenario_freq = np.zeros((B, H), dtype=np.float64)
        scenario_activity = np.zeros((B, H), dtype=np.float64)
        scenario_spend = np.zeros((B, H), dtype=np.float64) if joint else None

        for _ in range(n_scenarios):
            preds = np.zeros((B, H), dtype=np.float32)
            activity = np.zeros((B, H), dtype=np.float32)
            spend_preds = np.zeros((B, H), dtype=np.float32) if joint else None

            if use_kv_cache:
                # --- KV-cached path ---
                cov_warmup = (
                    covariates[:, :calibration_weeks, :] if covariates is not None else None
                )
                warmup_out = model(
                    seed_week, seed_trans, kv_cache=[], covariates=cov_warmup
                )
                if joint:
                    freq_logits, log_spend, kv_cache = warmup_out
                else:
                    freq_logits, kv_cache = warmup_out

                last_logits = freq_logits[:, -1, :]
                prev_trans, pred_val, p_active = _step_from_logits(
                    last_logits, mode, max_trans
                )
                preds[:, 0] = pred_val.cpu().numpy()
                activity[:, 0] = p_active.cpu().numpy()
                if joint:
                    spend_preds[:, 0] = log_spend[:, -1].cpu().numpy()

                for h in range(1, H):
                    week_idx = (h - 1) % calibration_weeks
                    week_input = torch.full((B, 1), week_idx, dtype=torch.long, device=device)
                    trans_input = prev_trans.unsqueeze(1)

                    cov_step = None
                    if covariates is not None:
                        step_idx = calibration_weeks + h
                        if step_idx < covariates.size(1):
                            cov_step = covariates[:, step_idx: step_idx + 1, :]

                    step_out = model(
                        week_input, trans_input, kv_cache=kv_cache, covariates=cov_step
                    )
                    if joint:
                        freq_logits_step, log_spend_step, kv_cache = step_out
                        spend_preds[:, h] = log_spend_step[:, 0].cpu().numpy()
                    else:
                        freq_logits_step, kv_cache = step_out

                    prev_trans, pred_val, p_active = _step_from_logits(
                        freq_logits_step[:, 0, :], mode, max_trans
                    )
                    preds[:, h] = pred_val.cpu().numpy()
                    activity[:, h] = p_active.cpu().numpy()

            else:
                # --- Non-cached path (for correctness verification) ---
                ctx_week = seed_week.clone()
                ctx_trans = seed_trans.clone()

                for h in range(H):
                    cov_ctx = (
                        covariates[:, :calibration_weeks + h, :] if covariates is not None else None
                    )
                    out = model(ctx_week, ctx_trans, covariates=cov_ctx)
                    if joint:
                        freq_logits, log_spend = out
                        spend_preds[:, h] = log_spend[:, -1].cpu().numpy()
                    else:
                        freq_logits = out

                    last_logits = freq_logits[:, -1, :]
                    next_class, pred_val, p_active = _step_from_logits(
                        last_logits, mode, max_trans
                    )
                    preds[:, h] = pred_val.cpu().numpy()
                    activity[:, h] = p_active.cpu().numpy()

                    next_week = torch.full(
                        (B, 1), h % calibration_weeks, dtype=torch.long, device=device
                    )
                    ctx_week = torch.cat([ctx_week, next_week], dim=1)
                    ctx_trans = torch.cat([ctx_trans, next_class.unsqueeze(1)], dim=1)

            scenario_freq += preds
            scenario_activity += activity
            if joint:
                scenario_spend += spend_preds

        avg_freq = (scenario_freq / n_scenarios).astype(np.float32)
        avg_activity = (scenario_activity / n_scenarios).astype(np.float32)
        all_customer_ids.append(customer_ids)
        all_pred_freq.append(avg_freq)
        all_pred_activity.append(avg_activity)

        if joint:
            avg_spend = (scenario_spend / n_scenarios).astype(np.float32)
            all_pred_spend.append(avg_spend)

    all_customer_ids = np.concatenate(all_customer_ids)
    all_pred_freq = np.concatenate(all_pred_freq, axis=0)
    all_pred_activity = np.concatenate(all_pred_activity, axis=0)

    result = {
        "customer_ids": all_customer_ids,
        "pred_freq": all_pred_freq,
        "pred_total_freq": all_pred_freq.sum(axis=1),
        "pred_activity": all_pred_activity,
    }

    if joint:
        all_pred_spend_arr = np.concatenate(all_pred_spend, axis=0)
        result["pred_spend"] = all_pred_spend_arr

    return result
