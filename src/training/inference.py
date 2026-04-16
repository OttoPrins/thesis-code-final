"""
Autoregressive inference for holdout period evaluation.

LSTM inference procedure (Valendin et al. 2022):
    1. Feed full calibration seed through LSTM to warm up hidden state (h, c).
    2. Sample the first holdout prediction from the softmax at the last calibration step.
    3. For each subsequent holdout week: feed (sampled_trans, week_idx) as a single
       time step, carry (h, c) forward, sample from new softmax.
    4. Repeat for n_scenarios stochastic scenarios and average predictions.

Transformer inference procedure:
    1. Start with full calibration seed sequence (length T_calib).
    2. Forward pass through Transformer (causal mask) → sample from last position.
    3. For each subsequent holdout week: append new (week, trans) to the growing
       sequence and do another full forward pass, sampling from the last position.
    4. Repeat for n_scenarios stochastic scenarios and average predictions.
    Note: This is O(H * T²) per scenario due to full re-computation each step.
    For typical holdout sizes (H ≤ 52) this is acceptable.

Week indices during holdout (calibration_weeks, calibration_weeks+1, ...) are clamped
by the model's embedding layer to max_week — this is intentional, matching the reference
implementation where holdout week embeddings saturate and the model relies on hidden state.

Returns per-customer predicted weekly frequencies averaged over scenarios.
"""

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
    n_scenarios: int = 50,
    device: torch.device = None,
) -> dict:
    """
    Run autoregressive LSTM inference over the holdout period.

    The inference_loader must come from a CustomerDataset built with
    include_seed=True (so seed_week, seed_trans are available in batches).
    The DataLoader must use shuffle=False to preserve customer ordering.

    Args:
        model:              Trained LSTMModel (joint or base)
        inference_loader:   DataLoader with seed sequences, shuffle=False
        holdout_weeks:      Number of holdout steps to generate (H)
        calibration_weeks:  Number of calibration steps (T)
        n_scenarios:        Stochastic scenarios to average over (>= 20)
        device:             torch.device; defaults to model's parameter device

    Returns:
        dict with:
            customer_ids:    (N,) np.ndarray int64
            pred_freq:       (N, H) np.ndarray float32 — per-week predicted freq
            pred_total_freq: (N,) np.ndarray float32 — total predicted freq
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    H = holdout_weeks

    all_customer_ids = []
    all_pred_freq = []

    for batch in inference_loader:
        seed_week = batch["seed_week"].to(device)    # (B, T)
        seed_trans = batch["seed_trans"].to(device)  # (B, T)
        customer_ids = batch["customer_id"].numpy()
        B = seed_week.size(0)

        # Accumulate scenario predictions before averaging
        scenario_preds = np.zeros((B, H), dtype=np.float64)

        for _ in range(n_scenarios):
            # Step 1: Warm up hidden state with full calibration seed
            freq_logits, hidden = model(seed_week, seed_trans)
            # freq_logits: (B, T, n_classes)  hidden: (h, c)

            # Step 2: Sample first holdout prediction from last calibration output
            last_logits = freq_logits[:, -1, :]             # (B, n_classes)
            probs = torch.softmax(last_logits, dim=-1)
            prev_trans = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

            preds = np.zeros((B, H), dtype=np.float32)
            preds[:, 0] = prev_trans.cpu().numpy()

            # Step 3: Autoregressively generate remaining holdout weeks
            for h in range(1, H):
                # Week index: calibration_weeks + h (clamped inside model embedding)
                week_idx = calibration_weeks + h
                week_input = torch.full(
                    (B, 1), week_idx, dtype=torch.long, device=device
                )
                trans_input = prev_trans.unsqueeze(1)  # (B, 1)

                freq_logits_step, hidden = model(week_input, trans_input, hidden)
                # freq_logits_step: (B, 1, n_classes)

                step_probs = torch.softmax(freq_logits_step[:, 0, :], dim=-1)
                prev_trans = torch.multinomial(step_probs, 1).squeeze(-1)
                preds[:, h] = prev_trans.cpu().numpy()

            scenario_preds += preds

        # Average across scenarios
        avg_preds = (scenario_preds / n_scenarios).astype(np.float32)
        all_customer_ids.append(customer_ids)
        all_pred_freq.append(avg_preds)

    all_customer_ids = np.concatenate(all_customer_ids)
    all_pred_freq = np.concatenate(all_pred_freq, axis=0)

    return {
        "customer_ids": all_customer_ids,
        "pred_freq": all_pred_freq,
        "pred_total_freq": all_pred_freq.sum(axis=1),
    }


@torch.no_grad()
def autoregressive_inference_transformer(
    model: TransformerModel,
    inference_loader: DataLoader,
    holdout_weeks: int,
    calibration_weeks: int,
    n_scenarios: int = 50,
    device: torch.device = None,
) -> dict:
    """
    Run autoregressive Transformer inference over the holdout period.

    The Transformer has no hidden state, so each generation step requires a full
    forward pass over the growing context window. The calibration seed warms up
    attention over historical context; each holdout step appends the latest sampled
    transaction count and the next week index to the sequence.

    The inference_loader must come from a CustomerDataset built with
    include_seed=True (so seed_week, seed_trans are available in batches).
    The DataLoader must use shuffle=False to preserve customer ordering.

    Args:
        model:              Trained TransformerModel (joint or base)
        inference_loader:   DataLoader with seed sequences, shuffle=False
        holdout_weeks:      Number of holdout steps to generate (H)
        calibration_weeks:  Number of calibration steps (T)
        n_scenarios:        Stochastic scenarios to average over (>= 20)
        device:             torch.device; defaults to model's parameter device

    Returns:
        dict with:
            customer_ids:    (N,) np.ndarray int64
            pred_freq:       (N, H) np.ndarray float32 — per-week predicted freq
            pred_total_freq: (N,) np.ndarray float32 — total predicted freq
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    H = holdout_weeks

    all_customer_ids = []
    all_pred_freq = []

    for batch in inference_loader:
        seed_week = batch["seed_week"].to(device)    # (B, T_calib)
        seed_trans = batch["seed_trans"].to(device)  # (B, T_calib)
        customer_ids = batch["customer_id"].numpy()
        B = seed_week.size(0)

        scenario_preds = np.zeros((B, H), dtype=np.float64)

        for _ in range(n_scenarios):
            # Growing context window: start with full calibration seed
            ctx_week = seed_week.clone()    # (B, T_calib)
            ctx_trans = seed_trans.clone()  # (B, T_calib)

            preds = np.zeros((B, H), dtype=np.float32)

            for h in range(H):
                # Full forward pass over current context (causal mask applied inside model)
                out = model(ctx_week, ctx_trans)
                if isinstance(out, tuple):
                    freq_logits = out[0]  # (B, T_ctx, n_classes)
                else:
                    freq_logits = out

                # Sample from the last position
                last_logits = freq_logits[:, -1, :]       # (B, n_classes)
                probs = torch.softmax(last_logits, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

                preds[:, h] = sampled.cpu().numpy()

                # Append next step to context: week_idx = calibration_weeks + h,
                # trans = sampled prediction (clamped by embedding inside model)
                next_week = torch.full(
                    (B, 1), calibration_weeks + h, dtype=torch.long, device=device
                )
                next_trans = sampled.unsqueeze(1)  # (B, 1)
                ctx_week = torch.cat([ctx_week, next_week], dim=1)
                ctx_trans = torch.cat([ctx_trans, next_trans], dim=1)

            scenario_preds += preds

        avg_preds = (scenario_preds / n_scenarios).astype(np.float32)
        all_customer_ids.append(customer_ids)
        all_pred_freq.append(avg_preds)

    all_customer_ids = np.concatenate(all_customer_ids)
    all_pred_freq = np.concatenate(all_pred_freq, axis=0)

    return {
        "customer_ids": all_customer_ids,
        "pred_freq": all_pred_freq,
        "pred_total_freq": all_pred_freq.sum(axis=1),
    }
