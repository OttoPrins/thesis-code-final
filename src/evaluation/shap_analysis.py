"""
Dual-head SHAP analysis for Extension 3 (Dunnhumby covariate ablation).

Attributes SHAP values separately for:
    - Frequency classification head (expected next-week transaction count)
    - Log-spend regression head (predicted log-spend at last calibration step)

Only the 4 covariate features are attributed; the sequential week/trans inputs
are held fixed at each customer's true calibration history.  This isolates the
marginal contribution of demographics and campaign exposure — the research
question of Extension 3.

Usage:
    python -m src.evaluation.shap_analysis \\
        --config experiments/configs/extension3_dunnhumby.yaml \\
        --checkpoint results/checkpoints/<run>.pt \\
        [--n_background 100] [--n_explain 200] [--out_dir experiments/insights]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import DunnhumbyPipeline
from src.data.collate import collate_fn
from src.models import LSTMModel, TransformerModel
from train import build_model


def _load_model(config: dict, checkpoint_path: str, device: torch.device):
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_predictor(model, seed_week, seed_trans, calib_weeks, device, head="freq"):
    """
    Return a function that maps (n_samples, 4) covariate arrays → scalar predictions.

    For head='freq': expected transaction count = sum_k k * softmax(logits)_k at t=-1.
    For head='spend': predicted log-spend at t=-1 (joint models only).
    """
    joint = getattr(model, "joint", False)

    def predict_fn(cov_array: np.ndarray) -> np.ndarray:
        n = cov_array.shape[0]
        # Broadcast each customer's fixed sequence to each covariate row
        with torch.no_grad():
            # Repeat seed for each row in cov_array (batch over covariate perturbations)
            sw = seed_week.expand(n, -1).to(device)   # (n, T_calib)
            st = seed_trans.expand(n, -1).to(device)

            # Build (n, T_calib, 4) covariate: broadcast static values across T
            cov_t = torch.tensor(cov_array, dtype=torch.float32, device=device)
            cov_t = cov_t.unsqueeze(1).expand(-1, sw.size(1), -1)  # (n, T_calib, 4)

            if isinstance(model, LSTMModel):
                if joint:
                    logits, log_spend, _ = model(sw, st, covariates=cov_t)
                else:
                    logits, _ = model(sw, st, covariates=cov_t)
            else:
                if joint:
                    logits, log_spend = model(sw, st, covariates=cov_t)
                else:
                    logits = model(sw, st, covariates=cov_t)

            if head == "freq":
                # Expected count = sum_k k * softmax(logits_k)
                probs = torch.softmax(logits[:, -1, :], dim=-1)  # (n, n_classes)
                n_classes = probs.shape[1]
                class_vals = torch.arange(n_classes, dtype=torch.float32, device=device)
                result = (probs * class_vals).sum(dim=-1)
            elif head == "spend":
                if not joint:
                    raise ValueError("Spend head requires a joint model.")
                result = log_spend[:, -1]
            else:
                raise ValueError(f"Unknown head: {head!r}")

        return result.cpu().numpy().flatten()

    return predict_fn


def run_shap(
    config_path: str,
    checkpoint_path: str,
    n_background: int = 100,
    n_explain: int = 200,
    out_dir: str = "experiments/insights",
    seed: int = 42,
):
    set_seed(seed)
    config = load_config(config_path)
    set_seed(config.get("training", {}).get("seed", seed))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading Dunnhumby pipeline ...")
    pipeline = DunnhumbyPipeline()
    train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)

    model = _load_model(config, checkpoint_path, device)
    joint = getattr(model, "joint", False)

    # Collect covariates and seed sequences from inference dataset
    all_covariates: List[np.ndarray] = []
    all_seed_week: List[torch.Tensor] = []
    all_seed_trans: List[torch.Tensor] = []

    for i in range(len(inference_ds)):
        item = inference_ds[i]
        if "covariates" not in item:
            raise RuntimeError(
                "Inference dataset contains no covariates. "
                "Ensure include_covariates=true in the config."
            )
        all_covariates.append(item["covariates"][:config["dataset"]["calibration_weeks"], :].numpy())
        all_seed_week.append(item["seed_week"])
        all_seed_trans.append(item["seed_trans"])

    # Use only the covariate value at the last calibration step (or mean across T_calib)
    # for the SHAP attribution — we attribute over per-customer static-ish features
    calib_covariates = np.stack([c[-1] for c in all_covariates])  # (N, 4)
    calib_weeks = config["dataset"]["calibration_weeks"]

    rng = np.random.RandomState(seed)
    N = calib_covariates.shape[0]
    idx_background = rng.choice(N, size=min(n_background, N), replace=False)
    idx_explain = rng.choice(N, size=min(n_explain, N), replace=False)

    background = calib_covariates[idx_background]  # (n_background, 4)
    explain_X = calib_covariates[idx_explain]       # (n_explain, 4)

    feature_names = getattr(inference_ds, "covariate_feature_names", None) or [
        "income", "household_size", "coupon_redemptions_week", "campaign_active_flag"
    ]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    heads = ["freq"]
    if joint:
        heads.append("spend")

    for head in heads:
        print(f"Running SHAP KernelExplainer for '{head}' head ...")

        # Use median customer seed for all predictions (focus on covariate attribution)
        median_idx = int(N // 2)
        seed_week = all_seed_week[median_idx].unsqueeze(0)   # (1, T_calib)
        seed_trans = all_seed_trans[median_idx].unsqueeze(0)

        predict_fn = _build_predictor(
            model, seed_week, seed_trans, calib_weeks, device, head=head
        )

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain_X, nsamples=100, l1_reg="aic")

        # Bar chart of mean |SHAP| per feature
        fig, ax = plt.subplots(figsize=(7, 4))
        mean_abs = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]
        ax.bar(range(len(feature_names)), mean_abs[sorted_idx], color="#4C72B0")
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=30, ha="right")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(
            f"Feature importance — {'frequency' if head == 'freq' else 'log-spend'} head"
        )
        fig.tight_layout()

        fig_path = out_path / f"shap_{head}_dunnhumby.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

        # Also save raw SHAP values for reproducibility
        np.save(out_path / f"shap_{head}_values.npy", shap_values)
        with open(out_path / f"shap_{head}_feature_names.json", "w") as f:
            json.dump(feature_names, f)

    print("SHAP analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Dual-head SHAP analysis (Extension 3).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_background", type=int, default=100)
    parser.add_argument("--n_explain", type=int, default=200)
    parser.add_argument("--out_dir", default="experiments/insights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_shap(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        n_background=args.n_background,
        n_explain=args.n_explain,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
