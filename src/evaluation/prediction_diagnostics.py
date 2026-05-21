"""Prediction-failure diagnostics for the v4 repair track.

This module deliberately does not train or tune models. It loads an existing
checkpoint and scores the same model under several regimes so forecast failure
can be attributed to the right mechanism: teacher-forced one-step accuracy,
autoregressive exposure bias, sampling variance, or calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.collate import collate_fn
from src.evaluation.calibration import (
    collect_autoregressive_rolling_validation,
    collect_teacher_forced_validation,
    fit_aggregate_calibration,
    fit_temperature_from_loader,
    fit_temperature_from_rolling_origin,
)
from src.evaluation.metrics import compute_all_metrics, mase_scale
from src.training.inference import (
    _step_from_logits,
    _zero_inactive_spend_feedback,
    autoregressive_inference_lstm,
    autoregressive_inference_transformer,
)
from src.utils.config import apply_kaggle_overrides, load_config
from train import PIPELINES, build_model


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_values(model_cfg: dict, inference_ds) -> torch.Tensor:
    max_trans = int(model_cfg["max_trans"])
    top_bin = float(getattr(inference_ds, "top_bin_value", max_trans))
    return torch.tensor(list(range(max_trans)) + [top_bin], dtype=torch.float32)


def _score_holdout(
    results: dict,
    holdout_gt: dict,
    inference_ds,
    scaler,
    config: dict,
    *,
    freq_factor: float = 1.0,
    spend_factor: float = 1.0,
    smearing_factor: float | None = None,
) -> dict:
    true_ids = holdout_gt["customer_ids"]
    assert np.array_equal(results["customer_ids"], true_ids)
    true_per_week = np.asarray(holdout_gt["raw_freq"], dtype=np.float32)
    freq_kwargs = {
        "y_freq_true_per_week": true_per_week,
        "y_freq_pred_per_week": np.asarray(results["pred_freq"], dtype=np.float32),
    }
    if getattr(inference_ds, "seed_trans", None) is not None:
        freq_kwargs["freq_mase_scale"] = mase_scale(inference_ds.seed_trans.cpu().numpy())

    joint = bool(config.get("model", {}).get("joint", False))
    spend_kwargs = {}
    if joint and "pred_spend" in results:
        spend_kwargs = {
            "y_spend_true_per_week": holdout_gt["spend"].astype(np.float32),
            "y_spend_pred_per_week": results["pred_spend"].astype(np.float32),
            "pred_activity_per_week": results["pred_activity"].astype(np.float32),
            "true_activity_per_week": (holdout_gt["raw_freq"] > 0).astype(np.float32),
        }
        if smearing_factor is not None:
            spend_kwargs["smearing_factor"] = smearing_factor
        if getattr(inference_ds, "seed_spend", None) is not None:
            calib_spend_raw = scaler.inverse_transform_spend(
                inference_ds.seed_spend.cpu().numpy()
            )
            spend_kwargs["spend_mase_scale"] = mase_scale(calib_spend_raw)

    return compute_all_metrics(
        y_freq_true=holdout_gt["total_freq"].astype(np.float32),
        y_freq_pred=results["pred_total_freq"].astype(np.float32),
        customer_ids=true_ids,
        scaler=scaler if joint else None,
        weekly_discount_rate=config.get("evaluation", {}).get("weekly_discount_rate", 0.0),
        freq_calibration_factor=freq_factor,
        spend_calibration_factor=spend_factor,
        **freq_kwargs,
        **spend_kwargs,
    )


def _subset_holdout_gt(holdout_gt: dict, indices: np.ndarray) -> dict:
    """Return a holdout_gt dict subset to the same customer rows as a Subset."""
    subset = {}
    n = len(holdout_gt["customer_ids"])
    for key, value in holdout_gt.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == n:
            subset[key] = value[indices]
        else:
            subset[key] = value
    return subset


@torch.no_grad()
def oracle_holdout_inference_lstm(
    model,
    inference_loader,
    holdout_gt: dict,
    config: dict,
    device: torch.device,
    class_values: torch.Tensor,
    *,
    mode: str = "expected",
    temperature: float = 1.0,
) -> dict:
    """Autoregressive LSTM scoring with true holdout feedback after each step."""
    model.eval()
    dataset_cfg = config["dataset"]
    H = int(dataset_cfg["holdout_weeks"])
    T_cal = int(dataset_cfg["calibration_weeks"])
    max_trans = int(config["model"]["max_trans"])
    joint = bool(config["model"].get("joint", False))
    id_to_row = {int(cid): i for i, cid in enumerate(holdout_gt["customer_ids"])}

    all_ids, all_freq, all_activity = [], [], []
    all_spend = [] if joint else None
    cv = class_values.to(device)

    for batch in inference_loader:
        ids = batch["customer_id"].numpy()
        rows = np.array([id_to_row[int(cid)] for cid in ids])
        true_freq = torch.as_tensor(
            holdout_gt["raw_freq"][rows],
            device=device,
            dtype=torch.long,
        ).clamp(0, max_trans)
        true_spend = torch.as_tensor(
            holdout_gt["spend"][rows],
            device=device,
            dtype=torch.float32,
        )

        seed_week = batch["seed_week"].to(device)
        seed_trans = batch["seed_trans"].to(device)
        seed_spend = batch.get("seed_spend")
        seed_spend = seed_spend.to(device) if seed_spend is not None else None
        state_features = batch.get("seed_state_features")
        state_features = state_features.to(device) if state_features is not None else None
        B = seed_week.size(0)

        if joint:
            out = model(
                seed_week,
                seed_trans,
                spend=seed_spend,
                state_features=state_features,
            )
            if len(out) == 4:
                freq_logits, log_spend, _, hidden = out
            else:
                freq_logits, log_spend, hidden = out
        else:
            freq_logits, hidden = model(
                seed_week,
                seed_trans,
                state_features=state_features,
            )
            log_spend = None

        preds = torch.zeros((B, H), device=device)
        activity = torch.zeros((B, H), device=device)
        spend_preds = torch.zeros((B, H), device=device) if joint else None

        step_logits = freq_logits[:, -1, :]
        step_spend = log_spend[:, -1] if joint else None
        for h in range(H):
            _, pred_val, p_active = _step_from_logits(
                step_logits,
                mode,
                max_trans,
                temperature=temperature,
                class_values=cv,
            )
            preds[:, h] = pred_val
            activity[:, h] = p_active
            if joint and step_spend is not None:
                spend_preds[:, h] = step_spend

            if h == H - 1:
                break

            week_input = torch.full(
                (B, 1),
                (T_cal + h) % 52,
                dtype=torch.long,
                device=device,
            )
            trans_input = true_freq[:, h].unsqueeze(1)
            if joint:
                spend_input = _zero_inactive_spend_feedback(
                    true_spend[:, h],
                    true_freq[:, h],
                ).unsqueeze(1)
                out = model(
                    week_input,
                    trans_input,
                    hidden,
                    spend=spend_input,
                )
                if len(out) == 4:
                    freq_step, spend_step, _, hidden = out
                else:
                    freq_step, spend_step, hidden = out
                step_spend = spend_step[:, 0]
            else:
                freq_step, hidden = model(week_input, trans_input, hidden)
            step_logits = freq_step[:, 0, :]

        all_ids.append(ids)
        all_freq.append(preds.cpu().numpy())
        all_activity.append(activity.cpu().numpy())
        if joint:
            all_spend.append(spend_preds.cpu().numpy())

    pred_freq = np.concatenate(all_freq, axis=0).astype(np.float32)
    result = {
        "customer_ids": np.concatenate(all_ids),
        "pred_freq": pred_freq,
        "pred_total_freq": pred_freq.sum(axis=1),
        "pred_activity": np.concatenate(all_activity, axis=0).astype(np.float32),
    }
    if joint:
        result["pred_spend"] = np.concatenate(all_spend, axis=0).astype(np.float32)
    return result


@torch.no_grad()
def oracle_holdout_inference_transformer(
    model,
    inference_loader,
    holdout_gt: dict,
    config: dict,
    device: torch.device,
    class_values: torch.Tensor,
    *,
    mode: str = "expected",
    temperature: float = 1.0,
) -> dict:
    """Transformer holdout oracle scoring by appending true previous tokens."""
    model.eval()
    dataset_cfg = config["dataset"]
    H = int(dataset_cfg["holdout_weeks"])
    T_cal = int(dataset_cfg["calibration_weeks"])
    max_trans = int(config["model"]["max_trans"])
    joint = bool(config["model"].get("joint", False))
    id_to_row = {int(cid): i for i, cid in enumerate(holdout_gt["customer_ids"])}

    all_ids, all_freq, all_activity = [], [], []
    all_spend = [] if joint else None
    cv = class_values.to(device)

    for batch in inference_loader:
        ids = batch["customer_id"].numpy()
        rows = np.array([id_to_row[int(cid)] for cid in ids])
        true_freq = torch.as_tensor(
            holdout_gt["raw_freq"][rows],
            device=device,
            dtype=torch.long,
        ).clamp(0, max_trans)
        true_spend = torch.as_tensor(
            holdout_gt["spend"][rows],
            device=device,
            dtype=torch.float32,
        )
        B = batch["seed_week"].shape[0]
        ctx_week = batch["seed_week"].to(device)
        ctx_position = batch.get("seed_position")
        if ctx_position is not None:
            ctx_position = ctx_position.to(device)
        else:
            ctx_position = torch.arange(ctx_week.shape[1], device=device).unsqueeze(0).expand(B, -1)
        ctx_trans = batch["seed_trans"].to(device)
        ctx_spend = batch.get("seed_spend")
        ctx_spend = ctx_spend.to(device) if ctx_spend is not None else None
        ctx_delta = batch.get("seed_delta_t")
        ctx_delta = ctx_delta.to(device) if ctx_delta is not None else None
        ctx_state = batch.get("seed_state_features")
        ctx_state = ctx_state.to(device) if ctx_state is not None else None

        preds = torch.zeros((B, H), device=device)
        activity = torch.zeros((B, H), device=device)
        spend_preds = torch.zeros((B, H), device=device) if joint else None

        for h in range(H):
            out = model(
                ctx_week,
                ctx_trans,
                spend=ctx_spend,
                position=ctx_position,
                state_features=ctx_state,
                delta_t=ctx_delta,
            )
            if joint:
                if len(out) == 3:
                    freq_logits, spend_mu, _ = out
                else:
                    freq_logits, spend_mu = out
                spend_preds[:, h] = spend_mu[:, -1]
            else:
                freq_logits = out
            _, pred_val, p_active = _step_from_logits(
                freq_logits[:, -1, :],
                mode,
                max_trans,
                temperature=temperature,
                class_values=cv,
            )
            preds[:, h] = pred_val
            activity[:, h] = p_active

            if h == H - 1:
                break
            ctx_week = torch.cat([
                ctx_week,
                torch.full((B, 1), (T_cal + h) % 52, dtype=torch.long, device=device),
            ], dim=1)
            ctx_position = torch.cat([
                ctx_position,
                torch.full((B, 1), T_cal + h, dtype=torch.long, device=device),
            ], dim=1)
            ctx_trans = torch.cat([ctx_trans, true_freq[:, h].unsqueeze(1)], dim=1)
            if ctx_spend is not None:
                next_spend = _zero_inactive_spend_feedback(
                    true_spend[:, h],
                    true_freq[:, h],
                ).unsqueeze(1)
                ctx_spend = torch.cat([ctx_spend, next_spend], dim=1)
            if ctx_delta is not None:
                purchased = (true_freq[:, h] > 0).to(ctx_delta.dtype)
                next_delta = (1.0 - purchased) * (ctx_delta[:, -1] + 1.0)
                ctx_delta = torch.cat([ctx_delta, next_delta.unsqueeze(1)], dim=1)
                if ctx_state is not None:
                    ctx_state = torch.cat(
                        [ctx_state, next_delta.unsqueeze(1).unsqueeze(-1)],
                        dim=1,
                    )

        all_ids.append(ids)
        all_freq.append(preds.cpu().numpy())
        all_activity.append(activity.cpu().numpy())
        if joint:
            all_spend.append(spend_preds.cpu().numpy())

    pred_freq = np.concatenate(all_freq, axis=0).astype(np.float32)
    result = {
        "customer_ids": np.concatenate(all_ids),
        "pred_freq": pred_freq,
        "pred_total_freq": pred_freq.sum(axis=1),
        "pred_activity": np.concatenate(all_activity, axis=0).astype(np.float32),
    }
    if joint:
        result["pred_spend"] = np.concatenate(all_spend, axis=0).astype(np.float32)
    return result


def run_prediction_diagnostics(
    config: dict,
    checkpoint: Path,
    *,
    max_customers: int | None = None,
) -> dict:
    device = _device()
    pipeline = PIPELINES[config["dataset"]["name"]]()
    train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)
    inference_eval_ds = inference_ds
    if max_customers is not None:
        n = min(max(1, int(max_customers)), len(inference_ds))
        indices = np.arange(n)
        inference_eval_ds = Subset(inference_ds, indices.tolist())
        holdout_gt = _subset_holdout_gt(holdout_gt, indices)
    batch_size = int(config["training"]["batch_size"])
    n_workers = int(config["training"].get("num_workers", 0))
    pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=pin,
    )
    inference_loader = DataLoader(
        inference_eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=pin,
    )

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    class_values = _class_values(config["model"], inference_ds)
    model_type = config["model"]["type"]

    diagnostics: dict[str, dict] = {}
    tf_val = collect_teacher_forced_validation(
        model,
        val_loader,
        device=device,
        scaler=scaler if config["model"].get("joint", False) else None,
        temperature=1.0,
        class_values=class_values.to(device),
        smearing_factor=None,
    )
    diagnostics["teacher_forced_validation"] = {
        "freq_true_total": tf_val["freq_true_total"],
        "freq_pred_total": tf_val["freq_pred_total"],
        "freq_bias_pct": 100.0 * (tf_val["freq_pred_total"] - tf_val["freq_true_total"]) / max(tf_val["freq_true_total"], 1.0),
        "spend_true_total": tf_val.get("spend_true_total", 0.0),
        "spend_pred_total": tf_val.get("spend_pred_total", 0.0),
    }

    oracle_fn = (
        oracle_holdout_inference_lstm
        if model_type == "lstm"
        else oracle_holdout_inference_transformer
    )
    oracle = oracle_fn(
        model,
        inference_loader,
        holdout_gt,
        config,
        device,
        class_values,
        mode="expected",
        temperature=1.0,
    )
    diagnostics["teacher_forced_holdout_oracle"] = _score_holdout(
        oracle,
        holdout_gt,
        inference_ds,
        scaler,
        config,
    )

    infer_fn = (
        autoregressive_inference_lstm
        if model_type == "lstm"
        else autoregressive_inference_transformer
    )
    common_kwargs = {
        "model": model,
        "inference_loader": inference_loader,
        "holdout_weeks": config["dataset"]["holdout_weeks"],
        "calibration_weeks": config["dataset"]["calibration_weeks"],
        "device": device,
        "class_values": class_values,
    }
    if model_type == "transformer":
        common_kwargs["use_kv_cache"] = config.get("inference", {}).get("use_kv_cache", True)

    for mode in ("expected", "sample"):
        raw = infer_fn(
            **common_kwargs,
            n_scenarios=1 if mode == "expected" else int(config.get("inference", {}).get("n_scenarios", 30)),
            mode=mode,
            temperature=1.0,
        )
        diagnostics[f"autoregressive_{mode}_uncalibrated"] = _score_holdout(
            raw,
            holdout_gt,
            inference_ds,
            scaler,
            config,
        )

    calibration_cfg = config.get("calibration", {})
    validation_cfg = config.get("validation", {})
    temperature = 1.0
    if calibration_cfg.get("temperature_scaling", False):
        if validation_cfg.get("mode") == "rolling_origin":
            temperature = fit_temperature_from_rolling_origin(
                model,
                val_ds,
                dataset_cfg=config["dataset"],
                validation_cfg=validation_cfg,
                batch_size=batch_size,
                device=device,
                class_values=class_values,
                candidates=calibration_cfg.get("temperature_candidates"),
                mode=calibration_cfg.get("rolling_origin_mode", "expected"),
                n_scenarios=int(calibration_cfg.get("rolling_origin_n_scenarios", 1)),
                use_kv_cache=config.get("inference", {}).get("use_kv_cache", True),
            )
        else:
            temperature = fit_temperature_from_loader(model, val_loader, device=device)

    freq_factor, spend_factor = 1.0, 1.0
    if calibration_cfg.get("aggregate_scaling", False):
        if validation_cfg.get("mode") == "rolling_origin":
            totals = collect_autoregressive_rolling_validation(
                model,
                val_ds,
                dataset_cfg=config["dataset"],
                validation_cfg=validation_cfg,
                batch_size=batch_size,
                device=device,
                scaler=scaler if config["model"].get("joint", False) else None,
                temperature=temperature,
                class_values=class_values,
                smearing_factor=None,
                mode=calibration_cfg.get("rolling_origin_mode", "expected"),
                n_scenarios=int(calibration_cfg.get("rolling_origin_n_scenarios", 1)),
                use_kv_cache=config.get("inference", {}).get("use_kv_cache", True),
            )
        else:
            totals = collect_teacher_forced_validation(
                model,
                val_loader,
                device=device,
                scaler=scaler if config["model"].get("joint", False) else None,
                temperature=temperature,
                class_values=class_values.to(device),
                smearing_factor=None,
            )
        cal = fit_aggregate_calibration(
            totals,
            shrinkage=float(calibration_cfg.get("aggregate_shrinkage", 0.5)),
            min_factor=float(calibration_cfg.get("min_factor", 0.25)),
            max_factor=float(calibration_cfg.get("max_factor", 4.0)),
        )
        freq_factor, spend_factor = float(cal.freq_factor), float(cal.spend_factor)

    calibrated = infer_fn(
        **common_kwargs,
        n_scenarios=1,
        mode="expected",
        temperature=temperature,
    )
    diagnostics["autoregressive_expected_calibrated"] = _score_holdout(
        calibrated,
        holdout_gt,
        inference_ds,
        scaler,
        config,
        freq_factor=freq_factor,
        spend_factor=spend_factor,
    )
    diagnostics["calibration"] = {
        "temperature": float(temperature),
        "freq_factor": freq_factor,
        "spend_factor": spend_factor,
    }
    return diagnostics


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose prediction failure modes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed_override", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--max-customers",
        type=int,
        default=None,
        help="Bound holdout customers for local smoke diagnostics.",
    )
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--kaggle-data-root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed_override is not None:
        config.setdefault("training", {})["seed"] = args.seed_override
    if args.results_dir is not None:
        config.setdefault("output", {})["results_dir"] = args.results_dir
    if args.kaggle or os.environ.get("KAGGLE_ENV", "0") == "1":
        apply_kaggle_overrides(config, args.kaggle_data_root)

    diagnostics = run_prediction_diagnostics(
        config,
        Path(args.checkpoint),
        max_customers=args.max_customers,
    )
    out_path = Path(args.output) if args.output else (
        Path(config["output"]["results_dir"])
        / "tables"
        / f"{config['output']['run_name']}_prediction_diagnostics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(diagnostics), indent=2, sort_keys=True))
    print(f"Prediction diagnostics saved: {out_path}")
    for name, metrics in diagnostics.items():
        if isinstance(metrics, dict) and "bias_pct" in metrics:
            print(
                f"{name}: bias={metrics.get('bias_pct'):.2f}% "
                f"rmse={metrics.get('freq_rmse'):.4f}"
            )


if __name__ == "__main__":
    main()
