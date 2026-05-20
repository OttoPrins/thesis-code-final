"""One-at-a-time sensitivity sweeps for Category C governance parameters."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from train import run_experiment


DEFAULT_CONFIGS = {
    "cdnow": "experiments/configs/lstm_joint_cdnow_v2.yaml",
    "uci": "experiments/configs/lstm_joint_uci_v2.yaml",
    "tafeng": "experiments/configs/lstm_joint_tafeng_v2.yaml",
    "dunnhumby": "experiments/configs/lstm_joint_dunnhumby_v2.yaml",
}

SWEEP_VALUES = {
    "spend_log_transform": ["log1p", "robust_log1p"],
    "regression_head_hidden": [None, 32, 64],
    "kendall_warmup_epochs": [0, 5, 10],
    "early_stopping_patience": [5, 10, 20],
    "max_grad_norm": [0.5, 1.0, 5.0],
    "scheduled_sampling": [None, "linear", "inv_sigmoid"],
}

DEFAULT_VALUES = {
    "spend_log_transform": "log1p",
    "regression_head_hidden": None,
    "kendall_warmup_epochs": 5,
    "early_stopping_patience": 5,
    "max_grad_norm": 1.0,
    "scheduled_sampling": None,
}

METRIC_COLUMNS = ["freq_rmse", "spend_rmse", "freq_mae", "spend_mae"]


def _value_label(value) -> str:
    return json.dumps(value)


def _value_slug(value) -> str:
    return _value_label(value).strip('"').replace(" ", "_").replace("/", "_")


def _apply_sweep_value(config: dict, parameter: str, value) -> None:
    dataset_cfg = config.setdefault("dataset", {})
    model_cfg = config.setdefault("model", {})
    training_cfg = config.setdefault("training", {})
    loss_cfg = config.setdefault("loss", {})

    if parameter == "spend_log_transform":
        dataset_cfg["spend_scaler"] = "robust" if value == "robust_log1p" else "log"
    elif parameter == "regression_head_hidden":
        if value is None:
            model_cfg.pop("regression_head_hidden", None)
        else:
            model_cfg["regression_head_hidden"] = int(value)
    elif parameter == "kendall_warmup_epochs":
        loss_cfg["warmup_epochs"] = int(value)
    elif parameter == "early_stopping_patience":
        training_cfg["early_stopping_patience"] = int(value)
    elif parameter == "max_grad_norm":
        training_cfg["max_grad_norm"] = float(value)
    elif parameter == "scheduled_sampling":
        if value is None:
            training_cfg.pop("scheduled_sampling", None)
        else:
            training_cfg["scheduled_sampling"] = {
                "enabled": True,
                "start_epoch": int(loss_cfg.get("warmup_epochs", 5)),
                "max_prob": 0.25,
                "schedule": str(value),
            }
    else:
        raise KeyError(f"Unknown sensitivity parameter: {parameter}")


def _row_from_metrics(parameter: str, value, metrics: dict) -> dict:
    return {
        "parameter_name": parameter,
        "value_tested": _value_label(value),
        "freq_rmse": metrics.get("freq_rmse", float("nan")),
        "spend_rmse": metrics.get("spend_rmse_log", float("nan")),
        "freq_mae": metrics.get("freq_mae", float("nan")),
        "spend_mae": metrics.get("spend_mae_log", float("nan")),
    }


def _defaults_from_config(config: dict) -> dict:
    training_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    scheduled_sampling = training_cfg.get("scheduled_sampling")
    if scheduled_sampling and scheduled_sampling.get("enabled", False):
        scheduled_default = scheduled_sampling.get("schedule", "linear")
    else:
        scheduled_default = None
    return {
        "spend_log_transform": (
            "robust_log1p"
            if config.get("dataset", {}).get("spend_scaler", "log") == "robust"
            else "log1p"
        ),
        "regression_head_hidden": config.get("model", {}).get("regression_head_hidden", None),
        "kendall_warmup_epochs": loss_cfg.get("warmup_epochs", DEFAULT_VALUES["kendall_warmup_epochs"]),
        "early_stopping_patience": training_cfg.get(
            "early_stopping_patience", DEFAULT_VALUES["early_stopping_patience"]
        ),
        "max_grad_norm": training_cfg.get("max_grad_norm", DEFAULT_VALUES["max_grad_norm"]),
        "scheduled_sampling": scheduled_default,
    }


def _print_ranking(rows: list[dict], defaults: dict) -> None:
    by_param: dict[str, list[dict]] = {}
    for row in rows:
        by_param.setdefault(row["parameter_name"], []).append(row)

    ranking = []
    for param, param_rows in by_param.items():
        default_label = _value_label(defaults.get(param, DEFAULT_VALUES[param]))
        default_row = next(
            (row for row in param_rows if row["value_tested"] == default_label),
            param_rows[0],
        )
        max_delta = 0.0
        for row in param_rows:
            for metric in METRIC_COLUMNS:
                try:
                    delta = abs(float(row[metric]) - float(default_row[metric]))
                except (TypeError, ValueError):
                    delta = 0.0
                max_delta = max(max_delta, delta)
        ranking.append((max_delta, param))

    print("\n=== Sensitivity Ranking (max absolute metric delta) ===")
    for rank, (score, param) in enumerate(sorted(ranking, reverse=True), start=1):
        print(f"{rank:2d}. {param}: {score:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-at-a-time Category C sensitivity sweeps.")
    parser.add_argument("--dataset", default="cdnow", choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--config", default=None, help="Base joint LSTM config. Defaults by dataset.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned runs without training.")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIGS[args.dataset]
    base_config = load_config(config_path)
    defaults = _defaults_from_config(base_config)
    base_run = base_config.get("output", {}).get("run_name", Path(config_path).stem)

    planned = [
        (parameter, value)
        for parameter, values in SWEEP_VALUES.items()
        for value in values
    ]
    print(f"Sensitivity sweep: {len(planned)} runs from {config_path}")

    if args.dry_run:
        for parameter, value in planned:
            print(f"  {parameter} = {_value_label(value)}")
        return

    rows: list[dict] = []
    for parameter, value in planned:
        cfg = copy.deepcopy(base_config)
        _apply_sweep_value(cfg, parameter, value)
        cfg.setdefault("dataset", {})["name"] = args.dataset
        cfg.setdefault("output", {})["run_name"] = (
            f"{base_run}_sensitivity_{parameter}_{_value_slug(value)}"
        )
        print(f"\n[{parameter}] testing {_value_label(value)}")
        metrics = run_experiment(
            cfg,
            config_path=f"{config_path}#sensitivity:{parameter}={_value_label(value)}",
            write_outputs=False,
        )
        rows.append(_row_from_metrics(parameter, value, metrics))

    out_path = ROOT / "results" / f"sensitivity_{args.dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter_name", "value_tested", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved sensitivity results: {out_path}")
    _print_ranking(rows, defaults)


if __name__ == "__main__":
    main()
