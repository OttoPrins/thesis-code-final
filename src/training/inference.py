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
       time step, carry (h, c) forward, take new prediction. Joint models also
       feed the previous predicted log-spend as the continuous spend input.

Transformer inference procedure (with KV caching):
    1. Full forward pass over calibration seed → collect per-layer KV cache.
    2. Take prediction from the last position of the warmup output.
    3. For each holdout week: forward single new token using cached K,V (O(1) per step).
       Joint models append the previous predicted log-spend to the autoregressive
       context exactly as they append the previous frequency prediction.

Holdout week indices cycle through 0..51 so the model sees calendar
week-of-year embeddings. Transformer sinusoidal PE receives a separate absolute
position tensor so sequence order is not confused with week-of-year.

Spend prediction:
    The model also outputs per-step log-spend, which the caller post-processes
    with aggregate_spend_to_raw_total() in evaluation.metrics. The inference loop
    also returns a per-week activity weight:
        sample mode:   1 if sampled freq>0, else 0
        expected mode: P(freq>0) = 1 - softmax(logits)[:, 0]
    The caller multiplies raw per-week spend by this weight before summing, so
    inactive weeks contribute zero raw revenue.

Covariate support (Extension 3):
    When the inference batch contains "static_covariates" (B, S) and/or
    "dynamic_covariates" (B, T_total, D) keys:
    - static_covariates are passed unchanged at every step (they don't vary over time).
    - dynamic_covariates[:,  :calibration_weeks, :] is fed during warm-up, and
      dynamic_covariates[:, calibration_weeks + h, :] is fed at each holdout step h.
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
    temperature: float = 1.0,
    class_values: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Convert per-step softmax logits to feedback class, prediction, and activity.

    Args:
        logits:       (B, n_classes) raw logits at one time step
        mode:         'sample' (multinomial) or 'expected' (deterministic E[k])
        max_trans:    top class index (for clamping the expected-mode embedding lookup)
        class_values: (n_classes,) float — value to assign to each class when
                      accumulating predictions.  The last entry should be
                      E[count | count >= max_trans], not just max_trans, so the
                      censored top bin is decoded to its calibration conditional mean
                      rather than the clipping threshold.  If None, falls back to
                      arange(n_classes) (original behaviour — top bin = max_trans).

    Returns:
        next_class: (B,) long — what to feed back into the next step's embedding
        pred:       (B,) float — prediction added to the holdout running total
        p_active:   (B,) float — sample mode: binary sampled activity;
                    expected mode: P(freq>0) for spend gating
    """
    temp = max(float(temperature), 1e-6)
    probs = torch.softmax(logits / temp, dim=-1)
    p_active = 1.0 - probs[:, 0]
    n_classes = probs.size(-1)

    if class_values is not None:
        cv = class_values.to(device=probs.device, dtype=probs.dtype)
    else:
        cv = torch.arange(n_classes, dtype=probs.dtype, device=probs.device)

    if mode == "sample":
        sampled = torch.multinomial(probs, 1).squeeze(-1)  # (B,) integer class index
        sampled_active = (sampled > 0).to(probs.dtype)
        # Map each sampled class to its representative value via class_values.
        pred_val = cv[sampled]
        return sampled, pred_val, sampled_active
    if mode == "expected":
        ev = (probs * cv).sum(dim=-1)  # (B,) — weighted mean using representative values
        next_class = ev.round().long().clamp(0, max_trans)
        return next_class, ev, p_active
    raise ValueError(f"Unknown inference mode: {mode!r} — expected 'sample' or 'expected'.")


def _zero_inactive_spend_feedback(
    log_spend: torch.Tensor,
    next_class: torch.Tensor,
) -> torch.Tensor:
    """
    Feed zero spend into the next step whenever the generated frequency is zero.

    Joint models condition the next token on the previous generated frequency and
    spend. A no-purchase week must therefore carry zero spend, otherwise sample
    paths can keep a positive spend state alive after activity has stopped.
    """
    return torch.where(
        next_class > 0,
        log_spend.clamp_min(0.0),
        torch.zeros_like(log_spend),
    )


@torch.no_grad()
def autoregressive_inference_lstm(
    model: LSTMModel,
    inference_loader: DataLoader,
    holdout_weeks: int,
    calibration_weeks: int,
    n_scenarios: int = 30,
    device: Optional[torch.device] = None,
    mode: str = "sample",
    temperature: float = 1.0,
    class_values: Optional[torch.Tensor] = None,
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
        class_values:      (n_classes,) float tensor — representative value per class.
                           Last entry = E[count | count >= max_trans] for top-bin
                           decode correction.  None → arange(n_classes) (original).

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
        seed_position = batch.get("seed_position")
        if seed_position is not None:
            seed_position = seed_position.to(device)
        seed_trans = batch["seed_trans"].to(device)  # (B, T_calib)
        seed_spend = batch.get("seed_spend")
        if seed_spend is not None:
            seed_spend = seed_spend.to(device)
        elif joint:
            seed_spend = torch.zeros_like(seed_week, dtype=torch.float32, device=device)
        seed_state_features = batch.get("seed_state_features")
        if seed_state_features is not None:
            seed_state_features = seed_state_features.to(device)
        customer_ids = batch["customer_id"].numpy()
        B = seed_week.size(0)

        # Static covariates: (B, S) — same at every step
        static_cov = None
        if "static_covariates" in batch and batch["static_covariates"] is not None:
            static_cov = batch["static_covariates"].to(device)

        # Dynamic covariates: (B, T_total, D) — slice per step
        dynamic_cov = None
        if "dynamic_covariates" in batch and batch["dynamic_covariates"] is not None:
            dynamic_cov = batch["dynamic_covariates"].to(device)

        scenario_freq = np.zeros((B, H), dtype=np.float64)
        scenario_activity = np.zeros((B, H), dtype=np.float64)
        scenario_spend = np.zeros((B, H), dtype=np.float64) if joint else None

        for _ in range(n_scenarios):
            cur_state_delta = None
            if seed_state_features is not None and seed_state_features.size(-1) > 0:
                cur_state_delta = seed_state_features[:, -1, 0].clone()

            # Warm up: feed full calibration seed
            dyn_warmup = dynamic_cov[:, :calibration_weeks, :] if dynamic_cov is not None else None
            if joint:
                warmup_out = model(
                    seed_week, seed_trans,
                    spend=seed_spend,
                    state_features=seed_state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dyn_warmup,
                )
                if len(warmup_out) == 4:
                    freq_logits, log_spend, _, hidden = warmup_out
                else:
                    freq_logits, log_spend, hidden = warmup_out
            else:
                freq_logits, hidden = model(
                    seed_week, seed_trans,
                    state_features=seed_state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dyn_warmup,
                )

            # Step 0 of holdout from last calibration output
            last_logits = freq_logits[:, -1, :]
            prev_trans, pred_val, p_active = _step_from_logits(
                last_logits, mode, max_trans, temperature=temperature,
                class_values=class_values,
            )

            preds = np.zeros((B, H), dtype=np.float32)
            activity = np.zeros((B, H), dtype=np.float32)
            preds[:, 0] = pred_val.cpu().numpy()
            activity[:, 0] = p_active.cpu().numpy()

            spend_preds = np.zeros((B, H), dtype=np.float32) if joint else None
            prev_spend = None
            if joint:
                spend_preds[:, 0] = log_spend[:, -1].cpu().numpy()
                prev_spend = _zero_inactive_spend_feedback(log_spend[:, -1], prev_trans)

            for h in range(1, H):
                # Feed the week-of-year for the *previous* holdout step (h-1), since
                # that is the "current" context from which we predict step h.
                week_idx = (calibration_weeks + h - 1) % 52
                week_input = torch.full((B, 1), week_idx, dtype=torch.long, device=device)
                trans_input = prev_trans.unsqueeze(1)
                spend_input = prev_spend.unsqueeze(1) if prev_spend is not None else None
                state_step = None
                if cur_state_delta is not None:
                    purchased = (prev_trans > 0).to(cur_state_delta.dtype)
                    cur_state_delta = (1.0 - purchased) * (cur_state_delta + 1.0)
                    state_step = cur_state_delta.unsqueeze(1).unsqueeze(-1)

                # Dynamic covariate slice for this holdout step: (B, 1, D)
                dyn_step = None
                if dynamic_cov is not None:
                    step_idx = calibration_weeks + h - 1
                    if step_idx < dynamic_cov.size(1):
                        dyn_step = dynamic_cov[:, step_idx: step_idx + 1, :]

                if joint:
                    step_out = model(
                        week_input, trans_input, hidden,
                        spend=spend_input,
                        state_features=state_step,
                        static_covariates=static_cov,
                        dynamic_covariates=dyn_step,
                    )
                    if len(step_out) == 4:
                        freq_logits_step, log_spend_step, _, hidden = step_out
                    else:
                        freq_logits_step, log_spend_step, hidden = step_out
                    spend_preds[:, h] = log_spend_step[:, 0].cpu().numpy()
                else:
                    freq_logits_step, hidden = model(
                        week_input, trans_input, hidden,
                        state_features=state_step,
                        static_covariates=static_cov,
                        dynamic_covariates=dyn_step,
                    )

                prev_trans, pred_val, p_active = _step_from_logits(
                    freq_logits_step[:, 0, :], mode, max_trans, temperature=temperature,
                    class_values=class_values,
                )
                preds[:, h] = pred_val.cpu().numpy()
                activity[:, h] = p_active.cpu().numpy()
                if joint:
                    prev_spend = _zero_inactive_spend_feedback(
                        log_spend_step[:, 0], prev_trans
                    )

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
    temperature: float = 1.0,
    class_values: Optional[torch.Tensor] = None,
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
        seed_position = batch.get("seed_position")
        if seed_position is not None:
            seed_position = seed_position.to(device)
        seed_trans = batch["seed_trans"].to(device)  # (B, T_calib)
        seed_spend = batch.get("seed_spend")
        if seed_spend is not None:
            seed_spend = seed_spend.to(device)
        elif joint:
            seed_spend = torch.zeros_like(seed_week, dtype=torch.float32, device=device)
        seed_state_features = batch.get("seed_state_features")
        if seed_state_features is not None:
            seed_state_features = seed_state_features.to(device)
        customer_ids = batch["customer_id"].numpy()
        B = seed_week.size(0)

        # Elapsed-time seed (precomputed in SequenceBuilder); fall back to zeros
        # for legacy datasets that don't expose it.
        seed_delta_t = batch.get("seed_delta_t")
        if seed_delta_t is not None:
            seed_delta_t = seed_delta_t.to(device)

        # Separate static / dynamic covariates (Extension 3 — Dunnhumby only)
        static_cov = None
        dynamic_cov = None
        if "static_covariates" in batch and batch["static_covariates"] is not None:
            static_cov = batch["static_covariates"].to(device)   # (B, S)
        if "dynamic_covariates" in batch and batch["dynamic_covariates"] is not None:
            dynamic_cov = batch["dynamic_covariates"].to(device)  # (B, T_total, D)

        scenario_freq = np.zeros((B, H), dtype=np.float64)
        scenario_activity = np.zeros((B, H), dtype=np.float64)
        scenario_spend = np.zeros((B, H), dtype=np.float64) if joint else None

        for _ in range(n_scenarios):
            preds = np.zeros((B, H), dtype=np.float32)
            activity = np.zeros((B, H), dtype=np.float32)
            spend_preds = np.zeros((B, H), dtype=np.float32) if joint else None

            # delta_t state at the end of warm-up: weeks since last purchase as
            # of step T_calib - 1. Updated each holdout step from the freshly
            # sampled (or expected) prev_trans.
            cur_delta_t: Optional[torch.Tensor] = None
            if seed_delta_t is not None:
                cur_delta_t = seed_delta_t[:, -1].clone()  # (B,)
            prev_spend: Optional[torch.Tensor] = None

            if use_kv_cache:
                # --- KV-cached path ---
                dyn_warmup = (
                    dynamic_cov[:, :calibration_weeks, :] if dynamic_cov is not None else None
                )
                warmup_out = model(
                    seed_week, seed_trans, spend=seed_spend, position=seed_position, kv_cache=[],
                    state_features=seed_state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dyn_warmup,
                    delta_t=seed_delta_t,
                )
                if joint:
                    if len(warmup_out) == 4:
                        freq_logits, log_spend, _, kv_cache = warmup_out
                    else:
                        freq_logits, log_spend, kv_cache = warmup_out
                else:
                    freq_logits, kv_cache = warmup_out

                last_logits = freq_logits[:, -1, :]
                prev_trans, pred_val, p_active = _step_from_logits(
                    last_logits, mode, max_trans, temperature=temperature,
                    class_values=class_values,
                )
                preds[:, 0] = pred_val.cpu().numpy()
                activity[:, 0] = p_active.cpu().numpy()
                if joint:
                    spend_preds[:, 0] = log_spend[:, -1].cpu().numpy()
                    prev_spend = _zero_inactive_spend_feedback(log_spend[:, -1], prev_trans)

                for h in range(1, H):
                    week_idx = (calibration_weeks + h - 1) % 52
                    week_input = torch.full((B, 1), week_idx, dtype=torch.long, device=device)
                    position_input = torch.full(
                        (B, 1), calibration_weeks + h - 1,
                        dtype=torch.long, device=device,
                    )
                    trans_input = prev_trans.unsqueeze(1)
                    spend_input = prev_spend.unsqueeze(1) if prev_spend is not None else None

                    # Update elapsed-time state from the just-sampled prev_trans.
                    delta_t_step: Optional[torch.Tensor] = None
                    if cur_delta_t is not None:
                        purchased = (prev_trans > 0).to(cur_delta_t.dtype)
                        cur_delta_t = (1.0 - purchased) * (cur_delta_t + 1.0)
                        delta_t_step = cur_delta_t.unsqueeze(1)  # (B, 1)
                    state_step = (
                        delta_t_step.unsqueeze(-1) if delta_t_step is not None else None
                    )

                    # Dynamic covariate slice for this step: (B, 1, D)
                    dyn_step = None
                    if dynamic_cov is not None:
                        step_idx = calibration_weeks + h - 1
                        if step_idx < dynamic_cov.size(1):
                            dyn_step = dynamic_cov[:, step_idx: step_idx + 1, :]

                    step_out = model(
                        week_input, trans_input, spend=spend_input,
                        position=position_input, kv_cache=kv_cache,
                        state_features=state_step,
                        static_covariates=static_cov,
                        dynamic_covariates=dyn_step,
                        delta_t=delta_t_step,
                    )
                    if joint:
                        if len(step_out) == 4:
                            freq_logits_step, log_spend_step, _, kv_cache = step_out
                        else:
                            freq_logits_step, log_spend_step, kv_cache = step_out
                        spend_preds[:, h] = log_spend_step[:, 0].cpu().numpy()
                    else:
                        freq_logits_step, kv_cache = step_out

                    prev_trans, pred_val, p_active = _step_from_logits(
                        freq_logits_step[:, 0, :], mode, max_trans, temperature=temperature,
                        class_values=class_values,
                    )
                    preds[:, h] = pred_val.cpu().numpy()
                    activity[:, h] = p_active.cpu().numpy()
                    if joint:
                        prev_spend = _zero_inactive_spend_feedback(
                            log_spend_step[:, 0], prev_trans
                        )

            else:
                # --- Non-cached path (for correctness verification) ---
                ctx_week = seed_week.clone()
                ctx_position = (
                    seed_position.clone()
                    if seed_position is not None
                    else torch.arange(seed_week.shape[1], device=device).unsqueeze(0).expand(B, -1)
                )
                ctx_trans = seed_trans.clone()
                ctx_spend = seed_spend.clone() if seed_spend is not None else None
                ctx_delta_t = (
                    seed_delta_t.clone() if seed_delta_t is not None else None
                )
                ctx_state_features = (
                    seed_state_features.clone() if seed_state_features is not None else None
                )

                for h in range(H):
                    dyn_ctx = (
                        dynamic_cov[:, :calibration_weeks + h, :] if dynamic_cov is not None else None
                    )
                    out = model(
                        ctx_week, ctx_trans,
                        spend=ctx_spend,
                        state_features=ctx_state_features,
                        position=ctx_position,
                        static_covariates=static_cov,
                        dynamic_covariates=dyn_ctx,
                        delta_t=ctx_delta_t,
                    )
                    if joint:
                        if len(out) == 3:
                            freq_logits, log_spend, _ = out
                        else:
                            freq_logits, log_spend = out
                        spend_preds[:, h] = log_spend[:, -1].cpu().numpy()
                    else:
                        freq_logits = out
                    next_spend = None

                    last_logits = freq_logits[:, -1, :]
                    next_class, pred_val, p_active = _step_from_logits(
                        last_logits, mode, max_trans, temperature=temperature,
                        class_values=class_values,
                    )
                    preds[:, h] = pred_val.cpu().numpy()
                    activity[:, h] = p_active.cpu().numpy()
                    if joint:
                        next_spend = _zero_inactive_spend_feedback(
                            log_spend[:, -1], next_class
                        ).unsqueeze(1)

                    next_week = torch.full(
                        (B, 1), (calibration_weeks + h) % 52, dtype=torch.long, device=device
                    )
                    next_position = torch.full(
                        (B, 1), calibration_weeks + h, dtype=torch.long, device=device
                    )
                    ctx_week = torch.cat([ctx_week, next_week], dim=1)
                    ctx_position = torch.cat([ctx_position, next_position], dim=1)
                    ctx_trans = torch.cat([ctx_trans, next_class.unsqueeze(1)], dim=1)
                    if ctx_spend is not None and next_spend is not None:
                        ctx_spend = torch.cat([ctx_spend, next_spend], dim=1)
                    if ctx_delta_t is not None:
                        purchased = (next_class > 0).to(ctx_delta_t.dtype)
                        next_delta_t = (1.0 - purchased) * (ctx_delta_t[:, -1] + 1.0)
                        ctx_delta_t = torch.cat(
                            [ctx_delta_t, next_delta_t.unsqueeze(1)], dim=1
                        )
                        if ctx_state_features is not None:
                            next_state = next_delta_t.unsqueeze(1).unsqueeze(-1)
                            ctx_state_features = torch.cat(
                                [ctx_state_features, next_state], dim=1
                            )

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
