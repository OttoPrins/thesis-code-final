"""
Dual-head SHAP analysis for Extension 3 (Dunnhumby covariate ablation).

Attributes per-feature importance separately for:
    - Frequency classification head (expected next-week transaction count)
    - Log-spend regression head (predicted log-spend at last calibration step)

Attribution method: shap.GradientExplainer (PyTorch-native, gradient-based).
    - Handles 3D dynamic covariate tensors (B, T, D) natively.
    - Uses integration over gradients w.r.t. the input covariates.
    - Dynamic SHAP values (B, T, D) are summed across the time axis T to obtain
      a 2D (B, D) total temporal contribution per feature — comparable to the
      static (B, S) values on the same plot.

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
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import DunnhumbyPipeline
from src.models import LSTMModel, TransformerModel
from train import build_model


class _CovariateShapWrapper(nn.Module):
    """
    Differentiable wrapper that exposes only the covariate tensors as SHAP inputs.

    Fixes week, trans, spend, and delta_t as buffers. Forward takes
    (static_cov, dynamic_cov) and returns a scalar prediction per sample
    (expected frequency or log-spend at the last calibration step).

    This lets GradientExplainer differentiate through the model w.r.t. covariates,
    attributing the prediction to each static and dynamic feature independently.
    """

    def __init__(
        self,
        model: nn.Module,
        seed_week: torch.Tensor,   # (1, T_calib) — representative customer
        seed_trans: torch.Tensor,  # (1, T_calib)
        seed_spend: torch.Tensor,  # (1, T_calib)
        seed_delta_t: Optional[torch.Tensor],  # (1, T_calib) or None
        head: str,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("seed_week", seed_week)
        self.register_buffer("seed_trans", seed_trans)
        self.register_buffer("seed_spend", seed_spend)
        if seed_delta_t is not None:
            self.register_buffer("seed_delta_t", seed_delta_t)
        else:
            self.seed_delta_t = None
        self.head = head

    def forward(
        self,
        static_cov: torch.Tensor,   # (B, S)
        dynamic_cov: torch.Tensor,  # (B, T, D)
    ) -> torch.Tensor:              # (B,) scalar prediction
        B = static_cov.shape[0]
        week = self.seed_week.expand(B, -1)
        trans = self.seed_trans.expand(B, -1)
        spend = self.seed_spend.expand(B, -1)
        delta_t = (
            self.seed_delta_t.expand(B, -1)
            if self.seed_delta_t is not None else None
        )
        state_features = None
        if delta_t is not None and getattr(self.model, "state_feature_dim", 0) > 0:
            state_features = delta_t.unsqueeze(-1)

        joint = getattr(self.model, "joint", False)
        if isinstance(self.model, LSTMModel):
            if joint:
                out = self.model(
                    week, trans,
                    spend=spend,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                )
                if len(out) == 4:
                    logits, log_spend, _, _ = out
                else:
                    logits, log_spend, _ = out
            else:
                logits, _ = self.model(
                    week, trans,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                )
                log_spend = None
        else:
            # Transformer
            if joint:
                out = self.model(
                    week, trans,
                    spend=spend,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                    delta_t=delta_t,
                )
                if len(out) == 3:
                    logits, log_spend, _ = out
                else:
                    logits, log_spend = out
            else:
                logits = self.model(
                    week, trans,
                    state_features=state_features,
                    static_covariates=static_cov,
                    dynamic_covariates=dynamic_cov,
                    delta_t=delta_t,
                )
                log_spend = None

        # Scalar prediction from last calibration time step
        if self.head == "freq":
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            n_classes = probs.shape[1]
            class_vals = torch.arange(
                n_classes, dtype=probs.dtype, device=probs.device
            )
            return (probs * class_vals).sum(dim=-1)  # (B,) expected count
        if self.head == "spend":
            if log_spend is None:
                raise ValueError("Spend head requires a joint model.")
            return log_spend[:, -1]  # (B,)
        raise ValueError(f"Unknown head: {self.head!r}")


def _load_model(config: dict, checkpoint_path: str, device: torch.device):
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _collect_covariate_tensors(
    inference_ds,
    calibration_weeks: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[Optional[torch.Tensor]],
]:
    """
    Pull per-customer static and dynamic calibration covariates and seed tensors.

    Returns:
        static_vals:  (N, S) float32 — static covariate vectors per customer
        dynamic_calib: (N, T_calib, D) float32 — dynamic calibration trajectories
        seed_weeks:   list of (T_calib,) tensors
        seed_trans:   list of (T_calib,) tensors
        seed_spend:   list of (T_calib,) tensors
        seed_delta_t: list of (T_calib,) tensors or None
    """
    all_static: List[np.ndarray] = []
    all_dynamic: List[np.ndarray] = []
    all_seed_week: List[torch.Tensor] = []
    all_seed_trans: List[torch.Tensor] = []
    all_seed_spend: List[torch.Tensor] = []
    all_seed_delta_t: List[Optional[torch.Tensor]] = []

    for i in range(len(inference_ds)):
        item = inference_ds[i]
        if "static_covariates" not in item or "dynamic_covariates" not in item:
            raise RuntimeError(
                "Inference dataset contains no covariates. "
                "Set include_covariates: true in the config."
            )
        all_static.append(item["static_covariates"].numpy())
        all_dynamic.append(item["dynamic_covariates"][:calibration_weeks, :].numpy())
        all_seed_week.append(item["seed_week"])
        all_seed_trans.append(item["seed_trans"])
        all_seed_spend.append(item.get("seed_spend", torch.zeros_like(item["seed_week"], dtype=torch.float32)))
        all_seed_delta_t.append(item.get("seed_delta_t"))

    return (
        np.stack(all_static, axis=0),   # (N, S)
        np.stack(all_dynamic, axis=0),  # (N, T_calib, D)
        all_seed_week,
        all_seed_trans,
        all_seed_spend,
        all_seed_delta_t,
    )


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

    print("Loading Dunnhumby pipeline ...")
    pipeline = DunnhumbyPipeline()
    _, _, inference_ds, _, _ = pipeline.run(config)

    model = _load_model(config, checkpoint_path, device)
    joint = getattr(model, "joint", False)
    calib_weeks = config["dataset"]["calibration_weeks"]

    print("Collecting covariate tensors ...")
    static_vals, dynamic_calib, seed_weeks, seed_trans_list, seed_spend_list, seed_delta_t_list = (
        _collect_covariate_tensors(inference_ds, calib_weeks)
    )
    N = static_vals.shape[0]

    # Feature names
    static_names = getattr(inference_ds, "static_feature_names", None) or [
        f"static_{i}" for i in range(static_vals.shape[1])
    ]
    dynamic_names = getattr(inference_ds, "dynamic_feature_names", None) or [
        f"dynamic_{i}" for i in range(dynamic_calib.shape[2])
    ]

    rng = np.random.RandomState(seed)
    idx_bg = rng.choice(N, size=min(n_background, N), replace=False)
    idx_ex = rng.choice(N, size=min(n_explain, N), replace=False)

    # Use the median customer's seed sequence as the fixed history backbone for the
    # SHAP wrapper. Holding the transaction history constant isolates covariate effects.
    median_idx = int(N // 2)
    seed_week_t = seed_weeks[median_idx].unsqueeze(0).to(device)    # (1, T)
    seed_trans_t = seed_trans_list[median_idx].unsqueeze(0).to(device)  # (1, T)
    seed_spend_t = seed_spend_list[median_idx].unsqueeze(0).to(device)  # (1, T)
    delta_t_seed = (
        seed_delta_t_list[median_idx].unsqueeze(0).to(device)
        if seed_delta_t_list[median_idx] is not None else None
    )

    # Background and explain tensors as float32 (GradientExplainer needs floats)
    bg_static = torch.tensor(static_vals[idx_bg], dtype=torch.float32)
    bg_dynamic = torch.tensor(dynamic_calib[idx_bg], dtype=torch.float32)
    ex_static = torch.tensor(static_vals[idx_ex], dtype=torch.float32)
    ex_dynamic = torch.tensor(dynamic_calib[idx_ex], dtype=torch.float32)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    heads = ["freq"]
    if joint:
        heads.append("spend")

    summary: dict = {}
    summary_rows: list[dict] = []

    for head in heads:
        print(f"\n=== GradientExplainer attribution for '{head}' head ===")

        wrapper = _CovariateShapWrapper(
            model=model,
            seed_week=seed_week_t,
            seed_trans=seed_trans_t,
            seed_spend=seed_spend_t,
            seed_delta_t=delta_t_seed,
            head=head,
        ).to(device)
        wrapper.eval()

        # GradientExplainer: background is a list of [static_bg, dynamic_bg]
        explainer = shap.GradientExplainer(
            wrapper, [bg_static.to(device), bg_dynamic.to(device)]
        )
        # shap_values is a list: [static_shap, dynamic_shap]
        # static_shap:  (n_explain, S)
        # dynamic_shap: (n_explain, T, D)
        shap_vals = explainer.shap_values(
            [ex_static.to(device), ex_dynamic.to(device)]
        )

        static_shap = shap_vals[0]    # (n_explain, S) ndarray
        dynamic_shap = shap_vals[1]   # (n_explain, T, D) ndarray

        # Aggregate dynamic SHAP across time: sum |shap| over T → (n_explain, D)
        dynamic_shap_agg = np.abs(dynamic_shap).sum(axis=1)   # (n_explain, D)

        static_mean_abs = np.abs(static_shap).mean(axis=0)    # (S,)
        dynamic_mean_abs = dynamic_shap_agg.mean(axis=0)      # (D,)

        # Save raw SHAP arrays
        np.save(out_path / f"shap_{head}_static_values.npy", static_shap)
        np.save(out_path / f"shap_{head}_dynamic_values.npy", dynamic_shap)
        with open(out_path / f"shap_{head}_feature_names.json", "w") as f:
            json.dump({"static": static_names, "dynamic": dynamic_names}, f)

        # Combined bar chart: static and dynamic on the same axis
        all_names = static_names + dynamic_names
        all_importance = np.concatenate([static_mean_abs, dynamic_mean_abs])
        method_labels = (
            ["static (SHAP)"] * len(static_names)
            + ["time-varying (SHAP, Σ_T)"] * len(dynamic_names)
        )

        if all_importance.size > 0:
            sort_idx = np.argsort(all_importance)[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            colours = [
                "#4C72B0" if "static" in method_labels[i] else "#DD8452"
                for i in sort_idx
            ]
            ax.bar(range(len(all_names)), all_importance[sort_idx], color=colours)
            ax.set_xticks(range(len(all_names)))
            ax.set_xticklabels(
                [f"{all_names[i]}\n[{method_labels[i]}]" for i in sort_idx],
                rotation=30, ha="right",
            )
            ax.set_ylabel("Mean |SHAP value|")
            ax.set_title(
                f"Feature importance — "
                f"{'frequency' if head == 'freq' else 'log-spend'} head"
            )
            fig.tight_layout()
            fig.savefig(out_path / f"shap_{head}_dunnhumby.png", dpi=300)
            plt.close(fig)
            print(f"  Saved: {out_path / f'shap_{head}_dunnhumby.png'}")

        summary[head] = {
            "static_features": dict(zip(static_names, static_mean_abs.tolist())),
            "dynamic_features": dict(zip(dynamic_names, dynamic_mean_abs.tolist())),
        }
        for name, value in zip(static_names, static_mean_abs.tolist()):
            summary_rows.append({
                "run_name": config["output"]["run_name"],
                "head": head,
                "feature_type": "static",
                "feature": name,
                "mean_abs_shap": float(value),
            })
        for name, value in zip(dynamic_names, dynamic_mean_abs.tolist()):
            summary_rows.append({
                "run_name": config["output"]["run_name"],
                "head": head,
                "feature_type": "dynamic",
                "feature": name,
                "mean_abs_shap": float(value),
            })

    with open(out_path / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if summary_rows:
        import pandas as pd

        csv_path = out_path / f"shap_{config['output']['run_name']}.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    print("\nSHAP analysis complete.")
    return summary


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
