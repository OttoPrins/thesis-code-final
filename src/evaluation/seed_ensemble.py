"""
Seed-ensemble inference for trained deep-learning checkpoints.

Averages the per-week autoregressive predictions of the N trained seeds of one
config (default {42, 7, 123}) at inference time, then scores the ensembled
prediction with the standard metric stack. This reduces residual cross-seed
variance once scheduled sampling has corrected the *mean* autoregressive bias —
it is NOT a substitute for more seeds (the seed count is fixed; this only
combines the seeds that already exist).

Mirrors src/evaluation/rescore.py's building blocks (pipeline, temperature /
aggregate calibration, smearing, autoregressive inference, compute_all_metrics)
so the ensembled result is scored identically to a single-seed run.

Example:
    python -m src.evaluation.seed_ensemble \
      --config experiments/configs/lstm_joint_cdnow_v3.yaml \
      --seeds 42 7 123 --mode sample \
      --checkpoint-dir results/checkpoints \
      --fit-temperature --fit-aggregate-calibration
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.evaluation.calibration import (
    collect_teacher_forced_validation,
    fit_aggregate_calibration,
    fit_temperature_from_loader,
)
from src.evaluation.metrics import compute_all_metrics, mase_scale, save_metrics_with_artifacts
from src.training.inference import (
    autoregressive_inference_lstm,
    autoregressive_inference_transformer,
)
from src.utils.config import apply_kaggle_overrides, load_config
from src.utils.final_manifest import attach_manifest_metadata
from train import PIPELINES, _add_result_validity_checks, _compute_val_smearing, build_model


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_one_seed(config: dict, checkpoint: Path, mode: str, n_scenarios: int,
                  fit_temperature: bool, fit_aggregate: bool, device: torch.device):
    """Reconstruct the pipeline + model for one seed and run autoregressive inference.

    Returns (results, holdout_gt, scaler, inference_ds, smearing_factor,
             temperature, aggregate_cal).
    """
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    pipeline = PIPELINES[dataset_cfg["name"]]()
    train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)
    batch_size = training_cfg["batch_size"]
    _pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=_pin, persistent_workers=True,
    )
    inference_loader = DataLoader(
        inference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=_pin, persistent_workers=True,
    )

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    joint = model_cfg.get("joint", False)

    temperature = 1.0
    if fit_temperature:
        temperature = fit_temperature_from_loader(model, val_loader, device=device)
    validation_totals = collect_teacher_forced_validation(
        model, val_loader, device=device,
        scaler=scaler if joint else None, temperature=temperature,
    )
    aggregate_cal = fit_aggregate_calibration(validation_totals) if fit_aggregate else None
    smearing_factor = _compute_val_smearing(
        model, val_loader, device, joint, scaler if joint else None
    )

    if model_cfg["type"] == "lstm":
        results = autoregressive_inference_lstm(
            model, inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=n_scenarios, device=device, mode=mode, temperature=temperature,
        )
    else:
        results = autoregressive_inference_transformer(
            model, inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=n_scenarios, device=device,
            use_kv_cache=config.get("inference", {}).get("use_kv_cache", True),
            mode=mode, temperature=temperature,
        )
    return (results, holdout_gt, scaler, inference_ds, smearing_factor,
            float(temperature), aggregate_cal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-ensemble a trained CLV config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--mode", choices=["sample", "expected"], default="sample")
    parser.add_argument("--n-scenarios", type=int, default=200)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--run-name", default=None,
                        help="Default: <config run_name>_ensemble_<mode>")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--fit-aggregate-calibration", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--kaggle", action="store_true",
                        help="Apply Kaggle raw_dir/results_dir overrides (or set KAGGLE_ENV=1).")
    parser.add_argument("--kaggle-data-root", default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    on_kaggle = args.kaggle or os.environ.get("KAGGLE_ENV", "0") == "1"
    if on_kaggle:
        apply_kaggle_overrides(base_config, args.kaggle_data_root)
    base_run = base_config.get("output", {}).get("run_name", "model")
    if args.results_dir is not None:
        base_config.setdefault("output", {})["results_dir"] = args.results_dir
    results_dir = base_config.get("output", {}).get("results_dir", "results")
    run_name = args.run_name or f"{base_run}_ensemble_{args.mode}"
    device = _device()
    ckpt_dir = Path(args.checkpoint_dir)

    per_seed = []
    ref = None
    for seed in args.seeds:
        cfg = load_config(args.config)
        if on_kaggle:
            apply_kaggle_overrides(cfg, args.kaggle_data_root)
        cfg.setdefault("training", {})["seed"] = seed
        cfg.setdefault("inference", {})["mode"] = args.mode
        cfg.setdefault("inference", {})["n_scenarios"] = args.n_scenarios
        ckpt = ckpt_dir / f"{base_run}_seed{seed}_{args.mode}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {ckpt}")
        out = _run_one_seed(
            cfg, ckpt, args.mode, args.n_scenarios,
            args.fit_temperature, args.fit_aggregate_calibration, device,
        )
        per_seed.append(out)
        if ref is None:
            ref = out
        print(f"seed {seed}: ran inference from {ckpt.name}")

    # All seeds share the same holdout customers and ordering (shuffle=False,
    # deterministic customer sort) — assert before averaging.
    ref_ids = ref[0]["customer_ids"]
    for results, *_ in per_seed:
        assert np.array_equal(results["customer_ids"], ref_ids), \
            "customer_ids mismatch across seeds — cannot ensemble"

    joint = base_config.get("model", {}).get("joint", False)

    def _stack_mean(key):
        return np.mean([np.asarray(r[0][key], dtype=np.float64) for r in per_seed], axis=0)

    ens_pred_freq = _stack_mean("pred_freq").astype(np.float32)
    ens_pred_total_freq = ens_pred_freq.sum(axis=1)
    ens_pred_activity = _stack_mean("pred_activity").astype(np.float32)
    ens_pred_spend = _stack_mean("pred_spend").astype(np.float32) if joint else None

    smear_vals = [s[4] for s in per_seed if s[4] is not None]
    ens_smearing = float(np.mean(smear_vals)) if smear_vals else None
    ens_temperature = float(np.mean([s[5] for s in per_seed]))
    cal_objs = [s[6] for s in per_seed if s[6] is not None]
    ens_freq_factor = float(np.mean([c.freq_factor for c in cal_objs])) if cal_objs else 1.0
    ens_spend_factor = float(np.mean([c.spend_factor for c in cal_objs])) if cal_objs else 1.0

    _, holdout_gt, scaler, inference_ds, _, _, _ = ref
    true_per_week = np.asarray(holdout_gt["raw_freq"], dtype=np.float32)

    freq_kwargs = {
        "y_freq_true_per_week": true_per_week,
        "y_freq_pred_per_week": ens_pred_freq,
        "freq_mase_scale": mase_scale(inference_ds.seed_trans.cpu().numpy()),
    }
    spend_kwargs = {}
    if joint and ens_pred_spend is not None:
        spend_kwargs = {
            "y_spend_true_per_week": holdout_gt["spend"].astype(np.float32),
            "y_spend_pred_per_week": ens_pred_spend,
            "pred_activity_per_week": ens_pred_activity,
            "true_activity_per_week": (holdout_gt["raw_freq"] > 0).astype(np.float32),
            "spend_mase_scale": mase_scale(
                scaler.inverse_transform_spend(inference_ds.seed_spend.cpu().numpy())
            ),
        }
        if ens_smearing is not None:
            spend_kwargs["smearing_factor"] = ens_smearing

    metrics = compute_all_metrics(
        y_freq_true=holdout_gt["total_freq"].astype(np.float32),
        y_freq_pred=ens_pred_total_freq,
        customer_ids=ref_ids,
        scaler=scaler if joint else None,
        weekly_discount_rate=base_config.get("evaluation", {}).get("weekly_discount_rate", 0.0),
        freq_calibration_factor=ens_freq_factor,
        spend_calibration_factor=ens_spend_factor,
        **freq_kwargs,
        **spend_kwargs,
    )
    metrics["ensemble_seeds"] = list(args.seeds)
    metrics["ensemble_temperature"] = ens_temperature
    # Distinct model tag so compare.aggregate_all_results lists it as its own row.
    metrics["model"] = f"{base_config.get('model', {}).get('type', 'model')}_ensemble"

    attach_manifest_metadata(
        metrics, config_path=args.config, config=base_config, run_name=run_name
    )
    _add_result_validity_checks(
        metrics,
        results={
            "pred_freq": ens_pred_freq,
            "pred_total_freq": ens_pred_total_freq,
            "pred_activity": ens_pred_activity,
            **({"pred_spend": ens_pred_spend} if joint else {}),
        },
        true_per_week=true_per_week,
        evaluation_cfg=base_config.get("evaluation", {}),
        joint=joint,
    )

    tables_dir = Path(results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"{run_name}_metrics.json"
    save_metrics_with_artifacts(metrics, metrics_path)
    print(f"Ensemble metrics saved: {metrics_path}")
    print(f"seeds={args.seeds} temperature={ens_temperature:.4f} "
          f"smearing={ens_smearing} bias_pct={metrics.get('bias_pct')}")


if __name__ == "__main__":
    main()
