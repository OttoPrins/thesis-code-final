"""
Rescore trained deep-learning checkpoints without retraining.

Examples:
    python -m src.evaluation.rescore \
      --config experiments/configs/lstm_joint_cdnow.yaml \
      --checkpoint results_kaggle_notebook/checkpoints/lstm_joint_cdnow_final_seed42_sample.pt \
      --run-name lstm_joint_cdnow_final_seed42_expected_rescore \
      --mode expected
"""

from __future__ import annotations

import argparse
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
from src.utils.config import load_config
from src.utils.final_manifest import attach_manifest_metadata
from train import PIPELINES, _add_result_validity_checks, _compute_val_smearing, build_model


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore a trained CLV checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["sample", "expected"], default="expected")
    parser.add_argument("--n-scenarios", type=int, default=200)
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--fit-aggregate-calibration", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--kaggle-data-root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.kaggle:
        from src.utils.config import apply_kaggle_overrides
        apply_kaggle_overrides(config, args.kaggle_data_root)
    if args.results_dir is not None:
        config.setdefault("output", {})["results_dir"] = args.results_dir
    config.setdefault("output", {})["run_name"] = args.run_name
    config.setdefault("inference", {})["mode"] = args.mode
    config.setdefault("inference", {})["n_scenarios"] = args.n_scenarios

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    output_cfg = config["output"]
    device = _device()

    pipeline = PIPELINES[dataset_cfg["name"]]()
    train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)
    batch_size = training_cfg["batch_size"]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    inference_loader = DataLoader(
        inference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    joint = model_cfg.get("joint", False)

    temperature = 1.0
    if args.fit_temperature:
        temperature = fit_temperature_from_loader(model, val_loader, device=device)
    validation_totals = collect_teacher_forced_validation(
        model,
        val_loader,
        device=device,
        scaler=scaler if joint else None,
        temperature=temperature,
    )
    aggregate_cal = fit_aggregate_calibration(validation_totals) if args.fit_aggregate_calibration else None

    smearing_factor = _compute_val_smearing(model, val_loader, device, joint, scaler if joint else None)

    if model_cfg["type"] == "lstm":
        results = autoregressive_inference_lstm(
            model,
            inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=args.n_scenarios,
            device=device,
            mode=args.mode,
            temperature=temperature,
        )
    else:
        results = autoregressive_inference_transformer(
            model,
            inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=args.n_scenarios,
            device=device,
            use_kv_cache=config.get("inference", {}).get("use_kv_cache", True),
            mode=args.mode,
            temperature=temperature,
        )

    true_ids = holdout_gt["customer_ids"]
    assert np.array_equal(results["customer_ids"], true_ids)
    true_per_week = np.asarray(holdout_gt["raw_freq"], dtype=np.float32)
    pred_per_week = np.asarray(results["pred_freq"], dtype=np.float32)

    freq_kwargs = {
        "y_freq_true_per_week": true_per_week,
        "y_freq_pred_per_week": pred_per_week,
        "freq_mase_scale": mase_scale(inference_ds.seed_trans.cpu().numpy()),
    }
    spend_kwargs = {}
    if joint and "pred_spend" in results:
        spend_kwargs = {
            "y_spend_true_per_week": holdout_gt["spend"].astype(np.float32),
            "y_spend_pred_per_week": results["pred_spend"].astype(np.float32),
            "pred_activity_per_week": results["pred_activity"].astype(np.float32),
            "true_activity_per_week": (holdout_gt["raw_freq"] > 0).astype(np.float32),
            "spend_mase_scale": mase_scale(
                scaler.inverse_transform_spend(inference_ds.seed_spend.cpu().numpy())
            ),
        }
        if smearing_factor is not None:
            spend_kwargs["smearing_factor"] = smearing_factor

    metrics = compute_all_metrics(
        y_freq_true=holdout_gt["total_freq"].astype(np.float32),
        y_freq_pred=results["pred_total_freq"],
        customer_ids=true_ids,
        scaler=scaler if joint else None,
        weekly_discount_rate=config.get("evaluation", {}).get("weekly_discount_rate", 0.0),
        freq_calibration_factor=aggregate_cal.freq_factor if aggregate_cal else 1.0,
        spend_calibration_factor=aggregate_cal.spend_factor if aggregate_cal else 1.0,
        **freq_kwargs,
        **spend_kwargs,
    )
    metrics["rescore_checkpoint"] = str(args.checkpoint)
    metrics["rescore_temperature"] = float(temperature)
    metrics.update({f"validation_{k}": float(v) for k, v in validation_totals.items()})
    if aggregate_cal:
        metrics["validation_freq_calibration_factor"] = aggregate_cal.freq_factor
        metrics["validation_spend_calibration_factor"] = aggregate_cal.spend_factor

    attach_manifest_metadata(metrics, config_path=args.config, config=config, run_name=args.run_name)
    _add_result_validity_checks(
        metrics,
        results=results,
        true_per_week=true_per_week,
        evaluation_cfg=config.get("evaluation", {}),
        joint=joint,
    )

    tables_dir = Path(output_cfg["results_dir"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"{args.run_name}_metrics.json"
    save_metrics_with_artifacts(metrics, metrics_path)
    print(f"Metrics saved: {metrics_path}")
    print(f"temperature={temperature:.4f}")
    if aggregate_cal:
        print(f"freq_factor={aggregate_cal.freq_factor:.4f} spend_factor={aggregate_cal.spend_factor:.4f}")


if __name__ == "__main__":
    main()
