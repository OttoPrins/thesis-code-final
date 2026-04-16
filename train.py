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
from src.evaluation.metrics import compute_all_metrics
from src.models import KendallMultiTaskLoss, LSTMModel, TransformerModel
from src.training.callbacks import EarlyStopping
from src.training.inference import (
    autoregressive_inference_lstm,
    autoregressive_inference_transformer,
)
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed

PIPELINES = {
    "cdnow": CDNOWPipeline,
    "uci": UCIRetailPipeline,
    "tafeng": TaFengPipeline,
    "dunnhumby": DunnhumbyPipeline,
}


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    joint = model_cfg.get("joint", False)
    model_type = model_cfg["type"]

    if model_type == "lstm":
        return LSTMModel(
            max_week=model_cfg["max_week"],
            max_trans=model_cfg["max_trans"],
            memory_units=model_cfg["hidden_size"],
            dense_units=model_cfg["hidden_size"],
            dropout=model_cfg.get("dropout", 0.0),
            joint=joint,
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
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose 'lstm' or 'transformer'.")


def main():
    parser = argparse.ArgumentParser(description="Train a CLV deep learning model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    output_cfg = config["output"]

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
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        joint=joint,
        multi_task_loss=multi_task_loss,
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
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

    # Autoregressive inference on holdout
    print("\nRunning autoregressive inference on holdout period ...")
    if model_cfg["type"] == "lstm":
        results = autoregressive_inference_lstm(
            model=model,
            inference_loader=inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=training_cfg.get("n_scenarios", 50),
            device=device,
        )
    elif model_cfg["type"] == "transformer":
        results = autoregressive_inference_transformer(
            model=model,
            inference_loader=inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=training_cfg.get("n_scenarios", 50),
            device=device,
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

    # For spend evaluation (joint models): use log-space totals
    spend_kwargs = {}
    if joint and "pred_total_spend" in results:
        spend_kwargs["y_spend_true_log"] = holdout_gt["total_spend"].astype(np.float32)
        spend_kwargs["y_spend_pred_log"] = results["pred_total_spend"]

    metrics = compute_all_metrics(
        y_freq_true=true_total_freq,
        y_freq_pred=pred_total_freq,
        customer_ids=true_ids,
        **spend_kwargs,
    )

    print("\n=== Holdout Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"{output_cfg['run_name']}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Save training history
    history_path = tables_dir / f"{output_cfg['run_name']}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
