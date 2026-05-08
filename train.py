"""
Main training script for deep learning CLV models.

Runs a full experiment: load data → build model → train → evaluate on holdout.

Usage:
    python train.py --config experiments/configs/lstm_base_cdnow.yaml
    python train.py --config experiments/configs/lstm_base_uci.yaml
    python train.py --config experiments/configs/lstm_base_tafeng.yaml
    python train.py --config experiments/configs/lstm_base_dunnhumby.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.data.datasets import (
    CDNOWPipeline,
    DunnhumbyPipeline,
    TaFengPipeline,
    UCIRetailPipeline,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_smearing_factor,
    save_metrics_with_artifacts,
)
from src.models import KendallMultiTaskLoss, LSTMModel, TransformerModel
from src.training.callbacks import EarlyStopping
from src.training.inference import (
    autoregressive_inference_lstm,
    autoregressive_inference_transformer,
)
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.final_manifest import attach_manifest_metadata
from src.utils.seed import set_seed

PIPELINES = {
    "cdnow": CDNOWPipeline,
    "uci": UCIRetailPipeline,
    "tafeng": TaFengPipeline,
    "dunnhumby": DunnhumbyPipeline,
}


@torch.no_grad()
def _compute_val_smearing(model, val_loader, device, joint, scaler):
    """
    Forward pass on validation set in teacher-forcing mode to estimate
    Duan's smearing factor from held-out residuals.

    Only meaningful for joint models with a spend head. Returns None otherwise.
    Smearing corrects the Jensen's-Inequality bias that arises from exponentiating
    predicted log-spend: E[exp(Z)] > exp(E[Z]).
    """
    if not joint or scaler is None:
        return None

    model.eval()
    from src.models.lstm import LSTMModel
    from src.models.transformer import TransformerModel

    true_log_all = []
    pred_log_all = []
    mask_all = []

    for batch in val_loader:
        week = batch["week"].to(device)
        position = batch.get("position")
        if position is not None:
            position = position.to(device)
        trans = batch["trans"].to(device)
        spend = batch.get("spend")
        if spend is not None:
            spend = spend.to(device)
        mask = batch["mask"].to(device)
        y_spend = batch.get("y_spend")
        y_freq = batch.get("y_freq")
        if y_spend is None:
            continue
        y_spend = y_spend.to(device)
        y_freq = y_freq.to(device) if y_freq is not None else None

        static_cov = None
        dynamic_cov = None
        if "static_covariates" in batch and batch["static_covariates"] is not None:
            static_cov = batch["static_covariates"].to(device)
        if "dynamic_covariates" in batch and batch["dynamic_covariates"] is not None:
            dynamic_cov = batch["dynamic_covariates"].to(device)

        if isinstance(model, LSTMModel):
            _, log_spend, _ = model(week, trans,
                                    spend=spend,
                                    static_covariates=static_cov,
                                    dynamic_covariates=dynamic_cov)
        elif isinstance(model, TransformerModel):
            delta_t = batch.get("delta_t")
            if delta_t is not None:
                delta_t = delta_t.to(device)
            _, log_spend = model(week, trans, position=position, padding_mask=mask,
                                 spend=spend,
                                 static_covariates=static_cov,
                                 dynamic_covariates=dynamic_cov,
                                 delta_t=delta_t)
        else:
            return None

        # Activity mask: only active weeks contribute to smearing estimate
        if y_freq is not None:
            activity = (y_freq > 0).float() * mask
        else:
            activity = mask

        true_log_all.append(y_spend.cpu().numpy())
        pred_log_all.append(log_spend.cpu().numpy())
        mask_all.append(activity.cpu().numpy())

    if not true_log_all:
        return None

    true_log = np.concatenate([x.ravel() for x in true_log_all])
    pred_log = np.concatenate([x.ravel() for x in pred_log_all])
    mask_flat = np.concatenate([x.ravel() for x in mask_all]).astype(bool)
    return compute_smearing_factor(true_log, pred_log, mask=mask_flat)


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    joint = model_cfg.get("joint", False)
    model_type = model_cfg["type"]
    static_cov_dim = model_cfg.get("static_cov_dim", 0)
    dynamic_cov_dim = model_cfg.get("dynamic_cov_dim", 0)
    cov_emb_dim = model_cfg.get("cov_emb_dim", 8)

    if model_type == "lstm":
        return LSTMModel(
            max_week=model_cfg["max_week"],
            max_trans=model_cfg["max_trans"],
            memory_units=model_cfg["hidden_size"],
            dense_units=model_cfg["hidden_size"],
            dropout=model_cfg.get("dropout", 0.0),
            joint=joint,
            static_cov_dim=static_cov_dim,
            dynamic_cov_dim=dynamic_cov_dim,
            cov_emb_dim=cov_emb_dim,
        )
    elif model_type == "transformer":
        return TransformerModel(
            max_week=model_cfg["max_week"],
            max_trans=model_cfg["max_trans"],
            d_model=model_cfg.get("d_model", 64),
            n_heads=model_cfg.get("n_heads", 4),
            n_layers=model_cfg.get("n_layers", 2),
            d_ff=model_cfg.get("d_ff", 256),
            dropout=model_cfg.get("dropout", 0.1),
            time2vec_dim=model_cfg.get("time2vec_dim", 8),
            joint=joint,
            static_cov_dim=static_cov_dim,
            dynamic_cov_dim=dynamic_cov_dim,
            cov_emb_dim=cov_emb_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose 'lstm' or 'transformer'.")


def main():
    parser = argparse.ArgumentParser(description="Train a CLV deep learning model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--seed_override", type=int, default=None,
        help="Override training.seed from YAML. Appends _seed<N> to run_name."
    )
    parser.add_argument(
        "--inference_mode", choices=["sample", "expected"], default=None,
        help="Override inference.mode from YAML. Appends _<mode> to run_name."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Override training.epochs (smoke/CI use)."
    )
    parser.add_argument(
        "--n_scenarios", type=int, default=None,
        help="Override inference.n_scenarios (smoke/CI use)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    if args.seed_override is not None:
        training_cfg["seed"] = args.seed_override
        output_cfg["run_name"] = f"{output_cfg['run_name']}_seed{args.seed_override}"
    if args.inference_mode is not None:
        config.setdefault("inference", {})["mode"] = args.inference_mode
        output_cfg["run_name"] = f"{output_cfg['run_name']}_{args.inference_mode}"
    if args.max_epochs is not None:
        training_cfg["epochs"] = args.max_epochs
    if args.n_scenarios is not None:
        config.setdefault("inference", {})["n_scenarios"] = args.n_scenarios

    print(f"Config: {args.config}")
    print(f"Run: {output_cfg['run_name']}")

    # Reproducibility
    set_seed(training_cfg["seed"])

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data pipeline
    dataset_name = dataset_cfg["name"]
    if dataset_name not in PIPELINES:
        print(f"Unknown dataset: {dataset_name!r}. Choose from: {list(PIPELINES)}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {dataset_name} ...")
    pipeline = PIPELINES[dataset_name]()
    train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)
    print(f"  Train customers: {len(train_ds)}, Val: {len(val_ds)}, Inference: {len(inference_ds)}")

    batch_size = training_cfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    inference_loader = DataLoader(
        inference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_cfg['type']} — {n_params:,} parameters")

    # Optimizer (include KendallMultiTaskLoss params if joint)
    joint = model_cfg.get("joint", False)
    multi_task_loss = None
    opt_params = list(model.parameters())

    if joint:
        multi_task_loss = KendallMultiTaskLoss(n_tasks=2).to(device)
        opt_params += list(multi_task_loss.parameters())

    optimizer = torch.optim.Adam(
        opt_params,
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=training_cfg["early_stopping_patience"])

    # Train
    print(f"\nTraining for up to {training_cfg['epochs']} epochs ...")
    loss_cfg = config.get("loss", {})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        joint=joint,
        multi_task_loss=multi_task_loss,
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        kendall_warmup_epochs=loss_cfg.get("warmup_epochs", 5),
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_cfg["epochs"],
        early_stopping=early_stopping,
    )

    # Save checkpoint
    results_dir = Path(output_cfg["results_dir"])
    if output_cfg.get("save_checkpoint", True):
        ckpt_dir = results_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{output_cfg['run_name']}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"\nCheckpoint saved: {ckpt_path}")

    # Compute smearing factor from validation residuals (joint models only)
    smearing_factor = _compute_val_smearing(model, val_loader, device, joint, scaler if joint else None)
    if smearing_factor is not None:
        print(f"\nDuan's smearing factor (val): {smearing_factor:.4f}")

    # Autoregressive inference on holdout
    print("\nRunning autoregressive inference on holdout period ...")
    inference_cfg = config.get("inference", {})
    n_scenarios = inference_cfg.get("n_scenarios", 30)
    inference_mode = inference_cfg.get("mode", "sample")
    print(f"  inference mode: {inference_mode}  n_scenarios: {n_scenarios}")

    if model_cfg["type"] == "lstm":
        results = autoregressive_inference_lstm(
            model=model,
            inference_loader=inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=n_scenarios,
            device=device,
            mode=inference_mode,
        )
    elif model_cfg["type"] == "transformer":
        results = autoregressive_inference_transformer(
            model=model,
            inference_loader=inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=n_scenarios,
            device=device,
            use_kv_cache=inference_cfg.get("use_kv_cache", True),
            mode=inference_mode,
        )
    else:
        raise ValueError(f"Unknown model type for inference: {model_cfg['type']!r}")

    # Verify customer ordering matches between inference output and holdout ground truth
    pred_ids = results["customer_ids"]
    true_ids = holdout_gt["customer_ids"]
    assert np.array_equal(pred_ids, true_ids), (
        "Customer ID mismatch between inference output and holdout ground truth. "
        "Ensure inference_loader uses shuffle=False."
    )

    # Evaluate: compare total predicted freq vs actual total freq per customer
    pred_total_freq = results["pred_total_freq"]   # (N,)
    true_total_freq = holdout_gt["total_freq"].astype(np.float32)  # (N,)

    # For spend (joint models): pass per-week scaled log arrays; compute_all_metrics
    # inverse-transforms per week and sums to raw currency (the only correct way).
    # Activity weights gate spend so that weeks the model predicts as inactive
    # contribute zero raw revenue (matches the masked spend training regime).
    spend_kwargs = {}
    if joint and "pred_spend" in results:
        spend_kwargs["y_spend_true_per_week"] = holdout_gt["spend"].astype(np.float32)
        spend_kwargs["y_spend_pred_per_week"] = results["pred_spend"].astype(np.float32)
        spend_kwargs["pred_activity_per_week"] = results["pred_activity"].astype(np.float32)
        # Ground-truth spend is multiplied by the binary activity indicator so
        # the inverse log1p(0)≈0 doesn't leak any "active" mass into the totals.
        true_activity = (holdout_gt["raw_freq"] > 0).astype(np.float32)
        spend_kwargs["true_activity_per_week"] = true_activity

    # Pass smearing factor so predictions are corrected for Jensen's Inequality bias.
    if joint and smearing_factor is not None:
        spend_kwargs["smearing_factor"] = smearing_factor

    weekly_discount_rate = config.get("evaluation", {}).get("weekly_discount_rate", 0.0)

    metrics = compute_all_metrics(
        y_freq_true=true_total_freq,
        y_freq_pred=pred_total_freq,
        customer_ids=true_ids,
        scaler=scaler if joint else None,
        weekly_discount_rate=weekly_discount_rate,
        **spend_kwargs,
    )
    attach_manifest_metadata(
        metrics,
        config_path=args.config,
        config=config,
        run_name=output_cfg["run_name"],
    )

    # Prediction diagnostics — helps identify exposure bias / class distribution issues.
    # pred_per_week values are Monte Carlo averages (floats), so we compare means and
    # distributions rather than exact class counts.
    pred_per_week = np.asarray(results["pred_freq"])   # (N, H) float averages over n_scenarios
    true_per_week = holdout_gt.get("raw_freq")  # (N, H) integers or None
    if true_per_week is not None:
        true_per_week = np.asarray(true_per_week)
    print("\n=== Prediction Diagnostics ===")
    print(f"  Mean predicted freq/week:    {pred_per_week.mean():.4f}")
    if true_per_week is not None:
        print(f"  Mean actual freq/week:       {true_per_week.mean():.4f}")
        print(f"  Over-prediction ratio:       {pred_per_week.mean() / max(true_per_week.mean(), 1e-9):.3f}x")
        # % of customers with zero total predicted transactions
        zero_pred = (pred_per_week.sum(axis=1) < 0.5).mean()
        zero_true = (true_per_week.sum(axis=1) == 0).mean()
        print(f"  Customers predicted as zero total:  {zero_pred:.3f}")
        print(f"  Customers actually zero total:      {zero_true:.3f}")
        # Mean non-zero purchase rate per week
        print(f"  Non-zero rate (pred, avg > 0.05):  {(pred_per_week > 0.05).mean():.3f}")
        print(f"  Non-zero rate (true):               {(true_per_week > 0).mean():.3f}")

    # Save metrics first so a crash in the print block doesn't lose results.
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"{output_cfg['run_name']}_metrics.json"
    _, arrays_path = save_metrics_with_artifacts(metrics, metrics_path)
    print(f"\nMetrics saved: {metrics_path}")
    if arrays_path is not None:
        print(f"Array artifacts saved: {arrays_path}")

    print("\n=== Holdout Evaluation ===")
    for k, v in metrics.items():
        if k.startswith("_"):
            continue  # skip internal arrays (per-customer error lists)
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save training history
    history_path = tables_dir / f"{output_cfg['run_name']}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
