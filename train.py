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
import copy
import json
import os
import sys
import math
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
    mase_scale,
    save_metrics_with_artifacts,
)
from src.evaluation.calibration import (
    collect_teacher_forced_validation,
    fit_aggregate_calibration,
    fit_temperature_from_loader,
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

        state_features = batch.get("state_features")
        if state_features is not None:
            state_features = state_features.to(device)

        if isinstance(model, LSTMModel):
            out = model(week, trans,
                        spend=spend,
                        state_features=state_features,
                        static_covariates=static_cov,
                        dynamic_covariates=dynamic_cov)
            if len(out) == 4:
                _, log_spend, _, _ = out
            else:
                _, log_spend, _ = out
        elif isinstance(model, TransformerModel):
            delta_t = batch.get("delta_t")
            if delta_t is not None:
                delta_t = delta_t.to(device)
            padding_mask = batch.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            out = model(week, trans, position=position, padding_mask=padding_mask,
                        spend=spend,
                        state_features=state_features,
                        static_covariates=static_cov,
                        dynamic_covariates=dynamic_cov,
                        delta_t=delta_t)
            if len(out) == 3:
                _, log_spend, _ = out
            else:
                _, log_spend = out
        else:
            return None

        # Activity mask: only active weeks contribute to smearing estimate
        if y_freq is not None:
            activity = (y_freq > 0).float() * mask
        else:
            activity = mask

        true_log_all.append(scaler.inverse_transform_log1p(y_spend.cpu().numpy()))
        pred_log_all.append(scaler.inverse_transform_log1p(log_spend.cpu().numpy()))
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
    state_feature_dim = model_cfg.get("state_feature_dim", 0)
    spend_head = model_cfg.get("spend_head", "regression")

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
            state_feature_dim=state_feature_dim,
            spend_head=spend_head,
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
            state_feature_dim=state_feature_dim,
            spend_head=spend_head,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}. Choose 'lstm' or 'transformer'.")


def _add_result_validity_checks(
    metrics: dict,
    *,
    results: dict,
    true_per_week: np.ndarray | None,
    evaluation_cfg: dict,
    joint: bool,
) -> None:
    """
    Mark mechanically invalid runs before they enter comparison tables.

    These gates target implementation failures, not ordinary model weakness:
    NaNs, all-zero forecasts on nonzero holdouts, and implausible aggregate
    explosions. A weak but finite model remains valid and should be reported.
    """
    invalid_reasons: list[str] = []
    warnings: list[str] = []

    pred_per_week = (
        np.asarray(results["pred_freq"], dtype=np.float64)
        * float(metrics.get("freq_calibration_factor", 1.0))
    )
    if not np.isfinite(pred_per_week).all():
        invalid_reasons.append("non-finite frequency predictions")

    pred_activity = results.get("pred_activity")
    if pred_activity is not None and not np.isfinite(np.asarray(pred_activity)).all():
        invalid_reasons.append("non-finite activity predictions")

    if joint and "pred_spend" in results:
        if not np.isfinite(np.asarray(results["pred_spend"])).all():
            invalid_reasons.append("non-finite spend predictions")

    if true_per_week is not None:
        true_total = float(np.asarray(true_per_week, dtype=np.float64).sum())
        pred_total = float(pred_per_week.sum())
        if true_total > 0:
            if pred_total <= 0:
                invalid_reasons.append("all-zero frequency forecast for nonzero holdout")
            else:
                freq_ratio = pred_total / true_total
                max_freq_ratio = float(evaluation_cfg.get("validity_max_freq_total_ratio", 10.0))
                if freq_ratio > max_freq_ratio:
                    invalid_reasons.append(
                        f"frequency total ratio {freq_ratio:.2f} exceeds {max_freq_ratio:.2f}"
                    )
                if abs(float(metrics.get("bias_pct", 0.0))) > float(
                    evaluation_cfg.get("warning_abs_bias_pct", 100.0)
                ):
                    warnings.append(f"large frequency bias ({metrics.get('bias_pct'):.1f}%)")

    if joint and "spend_weekly_total_true" in metrics and "spend_weekly_total_pred" in metrics:
        true_spend = float(metrics["spend_weekly_total_true"])
        pred_spend = float(metrics["spend_weekly_total_pred"])
        if true_spend > 0:
            if not np.isfinite(pred_spend):
                invalid_reasons.append("non-finite aggregate spend forecast")
            elif pred_spend <= 0:
                invalid_reasons.append("all-zero spend forecast for nonzero holdout")
            else:
                spend_ratio = pred_spend / true_spend
                max_spend_ratio = float(evaluation_cfg.get("validity_max_spend_total_ratio", 10.0))
                if spend_ratio > max_spend_ratio:
                    invalid_reasons.append(
                        f"spend total ratio {spend_ratio:.2f} exceeds {max_spend_ratio:.2f}"
                    )
                spend_bias = metrics.get("spend_bias_pct")
                if isinstance(spend_bias, (int, float)) and np.isfinite(spend_bias):
                    if abs(float(spend_bias)) > float(
                        evaluation_cfg.get("warning_abs_spend_bias_pct", 100.0)
                    ):
                        warnings.append(f"large spend bias ({spend_bias:.1f}%)")

    primary_keys = ["freq_rmse", "freq_mae", "freq_mape", "bias_pct"]
    if joint:
        primary_keys.extend(["spend_mae_raw", "spend_rmse_raw", "spend_bias_pct"])
    for key in primary_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not np.isfinite(value):
            invalid_reasons.append(f"non-finite metric {key}")

    metrics["run_valid"] = len(invalid_reasons) == 0
    metrics["run_invalid_reason"] = "; ".join(invalid_reasons) if invalid_reasons else ""
    metrics["run_warning"] = "; ".join(warnings)


def _model_label(config: dict) -> str:
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "unknown")
    if model_type == "transformer":
        return "transformer_joint" if model_cfg.get("joint", False) else "transformer_base"
    if model_type == "lstm":
        return "lstm_joint" if model_cfg.get("joint", False) else "lstm_base"
    return str(model_type)


def _looks_like_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cublas" in msg


def _is_repairable_training_failure(exc: BaseException) -> bool:
    """Numerical/resource failures may receive one bounded repair attempt."""
    if isinstance(exc, FloatingPointError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        needles = (
            "non-finite",
            "nan",
            "infinite",
            "out of memory",
            "cuda error",
            "cublas",
            "cudnn",
            "gradscaler",
        )
        return any(needle in msg for needle in needles)
    return False


def _build_repair_config(config: dict, exc: BaseException) -> dict:
    """
    Build the one allowed stability repair variant.

    This is deliberately narrow: disable AMP, lower LR to at most 5e-4,
    tighten gradient clipping to at most 0.5, and halve batch size for OOM-like
    failures. The repaired run receives its own run name so it cannot masquerade
    as the strict replication/extension run in final tables.
    """
    repaired = copy.deepcopy(config)
    training_cfg = repaired.setdefault("training", {})
    output_cfg = repaired.setdefault("output", {})
    strategies = ["amp_disabled"]

    training_cfg["repair_attempt"] = True
    training_cfg["amp_enabled"] = False

    current_lr = float(training_cfg.get("lr", 1e-3))
    fallback_lr = float(training_cfg.get("repair_fallback_lr", 5e-4))
    if current_lr > fallback_lr:
        strategies.append(f"lr_{fallback_lr:g}")
    training_cfg["lr"] = min(current_lr, fallback_lr)

    current_clip = float(training_cfg.get("max_grad_norm", 1.0))
    fallback_clip = float(training_cfg.get("repair_fallback_max_grad_norm", 0.5))
    if current_clip > fallback_clip:
        strategies.append(f"grad_clip_{fallback_clip:g}")
    training_cfg["max_grad_norm"] = min(current_clip, fallback_clip)

    if _looks_like_oom(exc):
        batch_size = int(training_cfg.get("batch_size", 256))
        reduced = max(1, batch_size // 2)
        if reduced < batch_size:
            training_cfg["batch_size"] = reduced
            strategies.append(f"batch_size_{reduced}")

    base_run = output_cfg.get("run_name", "run")
    if not str(base_run).endswith("_repair"):
        output_cfg["run_name"] = f"{base_run}_repair"

    training_cfg["repair_strategy"] = "; ".join(strategies)
    training_cfg["repair_source_failure_type"] = type(exc).__name__
    training_cfg["repair_source_failure_message"] = str(exc)[:500]
    return repaired


def _failed_training_metrics(
    config: dict,
    *,
    config_path: str,
    exc: BaseException,
    repair_attempted: bool = False,
    repair_error: BaseException | None = None,
) -> dict:
    reason = f"training failed: {type(exc).__name__}: {str(exc)[:500]}"
    metrics = {
        "model": _model_label(config),
        "dataset": config.get("dataset", {}).get("name", "unknown"),
        "run_valid": False,
        "run_invalid_reason": reason,
        "run_warning": "",
        "training_failed": True,
        "repair_attempted": bool(repair_attempted),
        "failure_type": type(exc).__name__,
        "failure_message": str(exc)[:1000],
    }
    if repair_error is not None:
        metrics["repair_failed"] = True
        metrics["repair_failure_type"] = type(repair_error).__name__
        metrics["repair_failure_message"] = str(repair_error)[:1000]
        metrics["run_invalid_reason"] = (
            f"{reason}; repair failed: {type(repair_error).__name__}: "
            f"{str(repair_error)[:500]}"
        )
    attach_manifest_metadata(
        metrics,
        config_path=config_path,
        config=config,
        run_name=config.get("output", {}).get("run_name", "run"),
    )
    return metrics


def _save_metrics_artifact(config: dict, metrics: dict) -> Path:
    results_dir = Path(config["output"]["results_dir"])
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"{config['output']['run_name']}_metrics.json"
    save_metrics_with_artifacts(metrics, metrics_path)
    print(f"Metrics saved: {metrics_path}")
    return metrics_path


def run_experiment_with_repair(
    config: dict,
    *,
    config_path: str = "<in-memory>",
    write_outputs: bool = True,
) -> dict:
    """Run one experiment, then one bounded stability repair if training fails."""
    try:
        return run_experiment(config, config_path=config_path, write_outputs=write_outputs)
    except Exception as exc:
        repair_enabled = config.get("training", {}).get("repair_on_failure", True)
        if not repair_enabled or not _is_repairable_training_failure(exc):
            raise

        print(f"\nTraining failed with a repairable error: {type(exc).__name__}: {exc}")
        failed = _failed_training_metrics(
            config,
            config_path=config_path,
            exc=exc,
            repair_attempted=True,
        )
        if write_outputs:
            _save_metrics_artifact(config, failed)

        repaired_config = _build_repair_config(config, exc)
        strategy = repaired_config["training"].get("repair_strategy", "")
        print(f"Retrying once as robustness repair ({strategy}) ...")
        try:
            return run_experiment(
                repaired_config,
                config_path=config_path,
                write_outputs=write_outputs,
            )
        except Exception as repair_exc:
            failed = _failed_training_metrics(
                config,
                config_path=config_path,
                exc=exc,
                repair_attempted=True,
                repair_error=repair_exc,
            )
            if write_outputs:
                _save_metrics_artifact(config, failed)
            return failed


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
    parser.add_argument(
        "--batch_size_override", type=int, default=None,
        help="Override training.batch_size (use to avoid CUDA OOM on large datasets)."
    )
    parser.add_argument(
        "--kaggle", action="store_true",
        help=(
            "Enable Kaggle path overrides: raw_dir → /kaggle/input/<dataset_name>/, "
            "results_dir → /kaggle/working/results. "
            "Equivalent to setting KAGGLE_ENV=1 in the environment. "
            "Use --kaggle-data-root to override the /kaggle/input parent if your "
            "Kaggle dataset slugs differ from the dataset.name values in the YAML configs."
        ),
    )
    parser.add_argument(
        "--kaggle-data-root", dest="kaggle_data_root", default=None,
        metavar="DIR",
        help=(
            "Override the Kaggle input root directory. "
            "Defaults to the KAGGLE_DATA_ROOT env var, or /kaggle/input if unset. "
            "Example: --kaggle-data-root /kaggle/input/my-combined-dataset"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Kaggle environment overrides ──────────────────────────────────────────
    # Activated by --kaggle flag OR KAGGLE_ENV=1 env var (the env var form lets
    # run_seeds.py propagate the override to all its subprocess train.py calls
    # without any changes to run_seeds.py itself).
    if args.kaggle or os.environ.get("KAGGLE_ENV", "0") == "1":
        from src.utils.config import apply_kaggle_overrides
        apply_kaggle_overrides(config, args.kaggle_data_root)
        print(f"[Kaggle] raw_dir  → {config['dataset']['raw_dir']}")
        print(f"[Kaggle] results  → {config['output']['results_dir']}")
    # ─────────────────────────────────────────────────────────────────────────
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
        training_cfg["max_epochs"] = args.max_epochs
    if args.n_scenarios is not None:
        config.setdefault("inference", {})["n_scenarios"] = args.n_scenarios
    if args.batch_size_override is not None:
        training_cfg["batch_size"] = args.batch_size_override

    metrics = run_experiment_with_repair(config, config_path=args.config)
    evaluation_cfg = config.get("evaluation", {})
    if not metrics["run_valid"] and evaluation_cfg.get("fail_on_invalid", True):
        print(f"Invalid run: {metrics['run_invalid_reason']}", file=sys.stderr)
        sys.exit(2)


def run_experiment(
    config: dict,
    *,
    config_path: str = "<in-memory>",
    write_outputs: bool = True,
) -> dict:
    """Programmatic training + holdout inference. Returns the metrics dict.

    `config` must already have any CLI/HPO overrides applied. When
    write_outputs=False (HPO trials) the checkpoint / metrics / history / array
    files are skipped so trials don't pollute results/. Used by CLI main() and
    by tune.py (Optuna HPO).
    """
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    print(f"Config: {config_path}")
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

    # Build class_values for top-bin decode correction (A2).
    # pipeline.run() updates model_cfg["max_trans"] from calibration data, so this
    # reads the correct cap even when freq_cap=auto overrides the YAML placeholder.
    _max_trans = model_cfg["max_trans"]
    _top_bin = getattr(inference_ds, "top_bin_value", float(_max_trans))
    class_values = torch.tensor(
        list(range(_max_trans)) + [_top_bin], dtype=torch.float32
    )
    print(f"  top-bin decode: class_values[-1] = {_top_bin:.3f} (max_trans = {_max_trans})")

    batch_size = training_cfg["batch_size"]
    # Cap at 2 workers per subprocess: with 2 parallel GPU runs (run_seeds.py),
    # 2×2=4 workers stays within Kaggle's 4-core allocation without contention.
    _n_workers = int(training_cfg.get("num_workers", min(2, os.cpu_count() or 1)))
    _n_workers = max(0, _n_workers)
    _pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=_n_workers, pin_memory=_pin, persistent_workers=(_n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=_n_workers, pin_memory=_pin, persistent_workers=(_n_workers > 0),
    )
    inference_loader = DataLoader(
        inference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=_n_workers, pin_memory=_pin, persistent_workers=(_n_workers > 0),
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
        _freq_logvar_max = config.get("loss", {}).get("freq_logvar_max", None)
        _spend_logvar_max = config.get("loss", {}).get("spend_logvar_max", None)
        multi_task_loss = KendallMultiTaskLoss(
            n_tasks=2,
            freq_logvar_max=float(_freq_logvar_max) if _freq_logvar_max is not None else None,
            spend_logvar_max=float(_spend_logvar_max) if _spend_logvar_max is not None else None,
        ).to(device)
        opt_params += list(multi_task_loss.parameters())

    optimizer_name = training_cfg.get("optimizer", "adam").lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        opt_params,
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 100)))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = None
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", training_cfg.get("lr_scheduler", "none"))
    if str(scheduler_type).lower() == "cosine":
        warmup_steps = max(
            1,
            int(total_steps * float(scheduler_cfg.get("warmup_fraction", 0.05))),
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Early stopping
    early_stopping = EarlyStopping(patience=training_cfg["early_stopping_patience"])

    # Train
    print(f"\nTraining for up to {epochs} epochs ...")
    loss_cfg = config.get("loss", {})
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        joint=joint,
        multi_task_loss=multi_task_loss,
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        kendall_warmup_epochs=loss_cfg.get("warmup_epochs", 5),
        spend_loss=loss_cfg.get("spend_loss", "mse"),
        scheduler=scheduler,
        restore_best_checkpoint=training_cfg.get("restore_best_checkpoint", True),
        spend_loss_normalize=bool(loss_cfg.get("spend_loss_normalize", False)),
        scheduled_sampling=training_cfg.get("scheduled_sampling", None),
        amp_enabled=bool(training_cfg.get("amp_enabled", True)),
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
    )

    # Save checkpoint
    results_dir = Path(output_cfg["results_dir"])
    if write_outputs and output_cfg.get("save_checkpoint", True):
        ckpt_dir = results_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{output_cfg['run_name']}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"\nCheckpoint saved: {ckpt_path}")

    inference_cfg = config.setdefault("inference", {})
    calibration_cfg = config.setdefault("calibration", {})
    calibration_metadata: dict[str, float | str] = {}

    # Compute smearing factor from validation residuals (joint models only)
    smearing_factor = _compute_val_smearing(model, val_loader, device, joint, scaler if joint else None)
    if smearing_factor is not None:
        print(f"\nDuan's smearing factor (val): {smearing_factor:.4f}")

    temperature = float(inference_cfg.get("temperature", 1.0))
    if calibration_cfg.get("temperature_scaling", False):
        temperature = fit_temperature_from_loader(model, val_loader, device=device)
        inference_cfg["temperature"] = float(temperature)
        calibration_metadata["calibration_temperature_source"] = "validation_only"
        calibration_metadata["validation_temperature"] = float(temperature)
        print(f"Validation-fitted softmax temperature: {temperature:.4f}")
    else:
        calibration_metadata["calibration_temperature_source"] = (
            "config" if "temperature" in inference_cfg else "default"
        )

    if calibration_cfg.get("aggregate_scaling", False):
        validation_totals = collect_teacher_forced_validation(
            model,
            val_loader,
            device=device,
            scaler=scaler if joint else None,
            temperature=temperature,
            class_values=class_values.to(device),
            smearing_factor=smearing_factor,
        )
        aggregate_calibration = fit_aggregate_calibration(
            validation_totals,
            shrinkage=float(calibration_cfg.get("aggregate_shrinkage", 0.5)),
            min_factor=float(calibration_cfg.get("min_factor", 0.25)),
            max_factor=float(calibration_cfg.get("max_factor", 4.0)),
        )
        calibration_cfg["freq_factor"] = float(aggregate_calibration.freq_factor)
        calibration_cfg["spend_factor"] = float(aggregate_calibration.spend_factor)
        calibration_metadata.update({
            "calibration_source": "validation_only",
            "validation_freq_true_total": float(validation_totals["freq_true_total"]),
            "validation_freq_pred_total": float(validation_totals["freq_pred_total"]),
            "validation_spend_true_total": float(validation_totals["spend_true_total"]),
            "validation_spend_pred_total": float(validation_totals["spend_pred_total"]),
            "validation_freq_calibration_factor": float(aggregate_calibration.freq_factor),
            "validation_spend_calibration_factor": float(aggregate_calibration.spend_factor),
        })
        print(
            "Validation aggregate calibration: "
            f"freq_factor={aggregate_calibration.freq_factor:.4f}, "
            f"spend_factor={aggregate_calibration.spend_factor:.4f}"
        )
    else:
        calibration_metadata["calibration_source"] = (
            "config"
            if ("freq_factor" in calibration_cfg or "spend_factor" in calibration_cfg)
            else "none"
        )

    # Autoregressive inference on holdout
    print("\nRunning autoregressive inference on holdout period ...")
    n_scenarios = inference_cfg.get("n_scenarios", 30)
    inference_mode = inference_cfg.get("mode", "sample")
    temperature = float(inference_cfg.get("temperature", temperature))
    print(
        f"  inference mode: {inference_mode}  n_scenarios: {n_scenarios}  "
        f"temperature: {temperature:.4f}"
    )

    if model_cfg["type"] == "lstm":
        results = autoregressive_inference_lstm(
            model=model,
            inference_loader=inference_loader,
            holdout_weeks=dataset_cfg["holdout_weeks"],
            calibration_weeks=dataset_cfg["calibration_weeks"],
            n_scenarios=n_scenarios,
            device=device,
            mode=inference_mode,
            temperature=temperature,
            class_values=class_values,
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
            temperature=temperature,
            class_values=class_values,
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
    pred_per_week = np.asarray(results["pred_freq"], dtype=np.float32)  # (N, H)
    true_per_week = holdout_gt.get("raw_freq")  # (N, H) integers or None
    if true_per_week is not None:
        true_per_week = np.asarray(true_per_week, dtype=np.float32)

    freq_week_kwargs = {}
    if true_per_week is not None:
        freq_week_kwargs["y_freq_true_per_week"] = true_per_week
        freq_week_kwargs["y_freq_pred_per_week"] = pred_per_week
        if getattr(inference_ds, "seed_trans", None) is not None:
            freq_week_kwargs["freq_mase_scale"] = mase_scale(
                inference_ds.seed_trans.cpu().numpy()
            )

    # For spend (joint models): pass per-week scaled log arrays; compute_all_metrics
    # inverse-transforms per week and sums to raw currency (the only correct way).
    # Activity weights gate spend so that weeks the model predicts as inactive
    # contribute zero raw revenue under sample/expected autoregressive inference.
    spend_kwargs = {}
    if joint and "pred_spend" in results:
        spend_kwargs["y_spend_true_per_week"] = holdout_gt["spend"].astype(np.float32)
        spend_kwargs["y_spend_pred_per_week"] = results["pred_spend"].astype(np.float32)
        spend_kwargs["pred_activity_per_week"] = results["pred_activity"].astype(np.float32)
        # Ground-truth spend is multiplied by the binary activity indicator so
        # the inverse log1p(0)≈0 doesn't leak any "active" mass into the totals.
        true_activity = (holdout_gt["raw_freq"] > 0).astype(np.float32)
        spend_kwargs["true_activity_per_week"] = true_activity
        if getattr(inference_ds, "seed_spend", None) is not None:
            calib_spend_raw = scaler.inverse_transform_spend(
                inference_ds.seed_spend.cpu().numpy()
            )
            spend_kwargs["spend_mase_scale"] = mase_scale(calib_spend_raw)

    # Pass smearing factor so predictions are corrected for Jensen's Inequality bias.
    if joint and smearing_factor is not None:
        spend_kwargs["smearing_factor"] = smearing_factor

    evaluation_cfg = config.get("evaluation", {})
    weekly_discount_rate = evaluation_cfg.get("weekly_discount_rate", 0.0)
    calibration_cfg = config.get("calibration", {})
    freq_calibration_factor = float(calibration_cfg.get("freq_factor", 1.0))
    spend_calibration_factor = float(calibration_cfg.get("spend_factor", 1.0))

    metrics = compute_all_metrics(
        y_freq_true=true_total_freq,
        y_freq_pred=pred_total_freq,
        customer_ids=true_ids,
        scaler=scaler if joint else None,
        weekly_discount_rate=weekly_discount_rate,
        freq_calibration_factor=freq_calibration_factor,
        spend_calibration_factor=spend_calibration_factor,
        **freq_week_kwargs,
        **spend_kwargs,
    )
    metrics.update(calibration_metadata)
    if training_cfg.get("repair_attempt", False):
        metrics["repair_attempted"] = True
        metrics["repair_strategy"] = training_cfg.get("repair_strategy", "")
        metrics["repair_source_failure_type"] = training_cfg.get("repair_source_failure_type", "")
        metrics["training_lr_after_repair"] = float(training_cfg.get("lr", 0.0))
        metrics["training_max_grad_norm_after_repair"] = float(
            training_cfg.get("max_grad_norm", 0.0)
        )
        metrics["training_amp_enabled_after_repair"] = bool(training_cfg.get("amp_enabled", True))
    attach_manifest_metadata(
        metrics,
        config_path=config_path,
        config=config,
        run_name=output_cfg["run_name"],
    )
    _add_result_validity_checks(
        metrics,
        results=results,
        true_per_week=true_per_week,
        evaluation_cfg=evaluation_cfg,
        joint=joint,
    )

    # Prediction diagnostics — helps identify exposure bias / class distribution issues.
    # pred_per_week values are Monte Carlo averages (floats), so we compare means and
    # distributions rather than exact class counts.
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

    if write_outputs:
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

    if write_outputs:
        # Save training history
        history_path = tables_dir / f"{output_cfg['run_name']}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved: {history_path}")

    return metrics


if __name__ == "__main__":
    main()
