"""
Autoregressive Monte Carlo inference for holdout period evaluation.

LSTM inference procedure (Valendin et al. 2022):
    1. Feed full calibration seed through LSTM to warm up hidden state (h, c).
    2. Sample the first holdout prediction from the softmax at the last calibration step.
    3. For each subsequent holdout week: feed (sampled_trans, week_idx) as a single
       time step, carry (h, c) forward, sample from new softmax.
    4. Repeat for n_scenarios stochastic paths (Monte Carlo) and average predictions.

Transformer inference procedure (with KV caching):
    1. Full forward pass over calibration seed → collect per-layer KV cache.
    2. Sample from last position of the warmup output.
    3. For each holdout week: forward single new token using cached K,V (O(1) per step).
    4. Repeat for n_scenarios Monte Carlo paths and average.

Holdout week indices cycle through 0..(calibration_weeks-1) using modulo so the model
sees the same week-of-year embeddings it was trained on (Valendin et al. week-of-year 0-51).

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


@torch.no_grad()
def autoregressive_inference_lstm(
    model: LSTMModel,
    inference_loader: DataLoader,
    holdout_weeks: int,
    calibration_weeks: int,
    n_scenarios: int = 30,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Run autoregressive LSTM Monte Carlo inference over the holdout period.

    The inference_loader must come from a CustomerDataset built with
    include_seed=True. DataLoader must use shuffle=False to preserve ordering.

    Args:
        model:             Trained LSTMModel (joint or base)
        inference_loader:  DataLoader with seed sequences, shuffle=False
        holdout_weeks:     Number of holdout steps to generate (H)
        calibration_weeks: Number of calibration steps (T)
        n_scenarios:       Monte Carlo paths to average (>= 20 per Valendin et al.)
        device:            torch.device; defaults to model's parameter device

    Returns:
        dict with:
            customer_ids:    (N,) np.ndarray int64
            pred_freq:       (N, H) float32 — per-week mean predicted freq
            pred_total_freq: (N,) float32 — total predicted transactions
            pred_spend:      (N, H) float32 — per-week mean log-spend (joint only)
            pred_total_spend:(N,) float32 — total predicted log-spend (joint only)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    H = holdout_weeks
    joint = model.joint

    all_customer_ids = []
    all_pred_freq = []
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

            # Sample from last calibration output (step 0 of holdout)
            last_logits = freq_logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            prev_trans = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

            preds = np.zeros((B, H), dtype=np.float32)
            preds[:, 0] = prev_trans.cpu().numpy()

            spend_preds = np.zeros((B, H), dtype=np.float32) if joint else None
            if joint:
                spend_preds[:, 0] = log_spend[:, -1].cpu().numpy()

            for h in range(1, H):
                week_idx = h % calibration_weeks
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

                step_probs = torch.softmax(freq_logits_step[:, 0, :], dim=-1)
                prev_trans = torch.multinomial(step_probs, 1).squeeze(-1)
                preds[:, h] = prev_trans.cpu().numpy()

            scenario_freq += preds
            if joint:
                scenario_spend += spend_preds

        avg_freq = (scenario_freq / n_scenarios).astype(np.float32)
        all_customer_ids.append(customer_ids)
        all_pred_freq.append(avg_freq)

        if joint:
            avg_spend = (scenario_spend / n_scenarios).astype(np.float32)
            all_pred_spend.append(avg_spend)

    all_customer_ids = np.concatenate(all_customer_ids)
    all_pred_freq = np.concatenate(all_pred_freq, axis=0)

    result = {
        "customer_ids": all_customer_ids,
        "pred_freq": all_pred_freq,
        "pred_total_freq": all_pred_freq.sum(axis=1),
    }

    if joint:
        all_pred_spend_arr = np.concatenate(all_pred_spend, axis=0)
        result["pred_spend"] = all_pred_spend_arr
        result["pred_total_spend"] = all_pred_spend_arr.sum(axis=1)

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
) -> dict:
    """
    Run autoregressive Transformer Monte Carlo inference over the holdout period.

    When use_kv_cache=True (default), the calibration seed is encoded once and its
    per-layer K,V matrices are cached; each holdout step only encodes the single new
    token, giving O(H) total complexity instead of O(H*T²).

    Args:
        model:             Trained TransformerModel (joint or base)
        inference_loader:  DataLoader with seed sequences, shuffle=False
        holdout_weeks:     Number of holdout steps H
        calibration_weeks: Number of calibration steps T
        n_scenarios:       Monte Carlo paths to average
        device:            torch.device
        use_kv_cache:      If True (default), use KV caching; False for correctness checks

    Returns:
        dict with same keys as autoregressive_inference_lstm.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    H = holdout_weeks
    joint = model.joint

    all_customer_ids = []
    all_pred_freq = []
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
        scenario_spend = np.zeros((B, H), dtype=np.float64) if joint else None

        for _ in range(n_scenarios):
            preds = np.zeros((B, H), dtype=np.float32)
            spend_preds = np.zeros((B, H), dtype=np.float32) if joint else None

            if use_kv_cache:
                # --- KV-cached path ---
                # Step 1: Warm up over full calibration seed, collect KV cache
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

                # Step 2: Sample from last calibration position
                last_logits = freq_logits[:, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                prev_trans = torch.multinomial(probs, 1).squeeze(-1)
                preds[:, 0] = prev_trans.cpu().numpy()
                if joint:
                    spend_preds[:, 0] = log_spend[:, -1].cpu().numpy()

                # Step 3: Autoregressive generation with cached K,V
                for h in range(1, H):
                    week_idx = h % calibration_weeks
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

                    step_probs = torch.softmax(freq_logits_step[:, 0, :], dim=-1)
                    prev_trans = torch.multinomial(step_probs, 1).squeeze(-1)
                    preds[:, h] = prev_trans.cpu().numpy()

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
                    probs = torch.softmax(last_logits, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    preds[:, h] = sampled.cpu().numpy()

                    next_week = torch.full(
                        (B, 1), h % calibration_weeks, dtype=torch.long, device=device
                    )
                    ctx_week = torch.cat([ctx_week, next_week], dim=1)
                    ctx_trans = torch.cat([ctx_trans, sampled.unsqueeze(1)], dim=1)

            scenario_freq += preds
            if joint:
                scenario_spend += spend_preds

        avg_freq = (scenario_freq / n_scenarios).astype(np.float32)
        all_customer_ids.append(customer_ids)
        all_pred_freq.append(avg_freq)

        if joint:
            avg_spend = (scenario_spend / n_scenarios).astype(np.float32)
            all_pred_spend.append(avg_spend)

    all_customer_ids = np.concatenate(all_customer_ids)
    all_pred_freq = np.concatenate(all_pred_freq, axis=0)

    result = {
        "customer_ids": all_customer_ids,
        "pred_freq": all_pred_freq,
        "pred_total_freq": all_pred_freq.sum(axis=1),
    }

    if joint:
        all_pred_spend_arr = np.concatenate(all_pred_spend, axis=0)
        result["pred_spend"] = all_pred_spend_arr
        result["pred_total_spend"] = all_pred_spend_arr.sum(axis=1)

    return result
