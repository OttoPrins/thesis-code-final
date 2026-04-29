"""
Dual-head SHAP analysis for Extension 3 (Dunnhumby covariate ablation).

Attributes per-feature importance separately for:
    - Frequency classification head (expected next-week transaction count)
    - Log-spend regression head (predicted log-spend at last calibration step)

Two attribution strategies are used, dispatched per feature based on whether
the covariate column varies across time within a customer:

    1. STATIC features (e.g. income, household_size) — values that are
       constant for a given customer across the calibration window. These are
       attributed with KernelExplainer in the standard way: each customer is
       represented by a single (n_static,) feature vector, the explainer
       perturbs it, and the predictor function broadcasts the perturbed
       static values across all T calibration steps before model evaluation.
       Broadcasting is correct here because the original feature really IS
       constant in time.

    2. TIME-VARYING features (e.g. coupon_redemptions_per_week,
       campaign_active_flag) — values that legitimately change from one
       week to the next. KernelExplainer cannot attribute these as scalar
       features without flattening the temporal structure (the previous
       implementation took the last calibration step's value and broadcast
       it across the entire history, fabricating a constant time-series).
       Instead we run a SEQUENTIAL PERTURBATION ablation: for each time-
       varying column we replace the explained customer's full trajectory
       with the per-customer-mean trajectory from the background set, and
       record the resulting change in prediction. The mean absolute change
       is reported alongside the static SHAP values for direct comparison.

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
import shap
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import DunnhumbyPipeline
from src.data.collate import collate_fn  # noqa: F401  (kept for backwards-compat imports)
from src.models import LSTMModel, TransformerModel
from train import build_model


# Threshold below which a feature's per-customer time-variance is considered
# numerically zero (i.e., the feature is static within that customer).
_STATIC_VARIANCE_TOL = 1e-8


def _classify_static_columns(covariate_calib: np.ndarray) -> np.ndarray:
    """
    Identify which covariate columns are static (constant across time) per customer.

    Args:
        covariate_calib: (N, T_calib, C) calibration covariate trajectories.

    Returns:
        is_static: (C,) bool — True if the column is constant in time for every
                   customer. Time-varying columns are False.
    """
    # Variance across the time axis, per (customer, feature)
    time_var = covariate_calib.var(axis=1)  # (N, C)
    # A column is "static" only if it is constant in time for ALL customers;
    # otherwise we treat it as time-varying for SHAP purposes.
    return np.all(time_var <= _STATIC_VARIANCE_TOL, axis=0)


def _load_model(config: dict, checkpoint_path: str, device: torch.device):
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _model_forward(
    model,
    week: torch.Tensor,
    trans: torch.Tensor,
    cov: Optional[torch.Tensor],
    delta_t: Optional[torch.Tensor],
):
    """Run a forward pass that handles the LSTM vs Transformer signature differences."""
    joint = getattr(model, "joint", False)
    if isinstance(model, LSTMModel):
        if joint:
            logits, log_spend, _ = model(week, trans, covariates=cov)
            return logits, log_spend
        logits, _ = model(week, trans, covariates=cov)
        return logits, None
    # Transformer
    if joint:
        logits, log_spend = model(week, trans, covariates=cov, delta_t=delta_t)
        return logits, log_spend
    logits = model(week, trans, covariates=cov, delta_t=delta_t)
    return logits, None


def _scalarize(logits: torch.Tensor, log_spend: Optional[torch.Tensor], head: str):
    """Reduce a forward-pass output to the scalar prediction used for SHAP attribution."""
    if head == "freq":
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        n_classes = probs.shape[1]
        class_vals = torch.arange(n_classes, dtype=torch.float32, device=probs.device)
        return (probs * class_vals).sum(dim=-1)
    if head == "spend":
        if log_spend is None:
            raise ValueError("Spend head requires a joint model.")
        return log_spend[:, -1]
    raise ValueError(f"Unknown head: {head!r}")


def _build_static_predictor(
    model,
    seed_week: torch.Tensor,
    seed_trans: torch.Tensor,
    fixed_time_varying_traj: torch.Tensor,
    seed_delta_t: Optional[torch.Tensor],
    static_idx: np.ndarray,
    time_varying_idx: np.ndarray,
    n_features: int,
    device: torch.device,
    head: str,
):
    """
    Build a SHAP predictor over the STATIC covariate columns only.

    The explained customer's static values are the (n_static,) row passed to
    KernelExplainer. Inside the predictor we reconstruct the full (T, C)
    covariate tensor by broadcasting those perturbed static values across T
    and re-attaching the customer's actual time-varying trajectory.

    Args:
        seed_week:               (1, T_calib) week indices for the explained customer
        seed_trans:              (1, T_calib) trans counts for the explained customer
        fixed_time_varying_traj: (T_calib, n_time_varying) the explained customer's
                                 actual time-varying covariate trajectory
        seed_delta_t:            (1, T_calib) elapsed-time signal (Transformer only); may be None
        static_idx:              column indices of static features in the full covariate tensor
        time_varying_idx:        column indices of time-varying features
    """
    T = seed_week.size(1)

    def predict_fn(static_array: np.ndarray) -> np.ndarray:
        n = static_array.shape[0]
        static_t = torch.tensor(static_array, dtype=torch.float32, device=device)

        # Reconstruct (n, T, C) — static columns broadcast across T (correct,
        # because they are static), time-varying columns held at the explained
        # customer's actual trajectory (no fabrication).
        cov_full = torch.zeros(n, T, n_features, dtype=torch.float32, device=device)
        cov_full[:, :, static_idx] = static_t.unsqueeze(1).expand(-1, T, -1)
        cov_full[:, :, time_varying_idx] = (
            fixed_time_varying_traj.unsqueeze(0).expand(n, -1, -1)
        )

        sw = seed_week.expand(n, -1).to(device)
        st = seed_trans.expand(n, -1).to(device)
        sd = seed_delta_t.expand(n, -1).to(device) if seed_delta_t is not None else None

        with torch.no_grad():
            logits, log_spend = _model_forward(model, sw, st, cov_full, sd)
            return _scalarize(logits, log_spend, head).cpu().numpy().flatten()

    return predict_fn


@torch.no_grad()
def _time_varying_ablation(
    model,
    seed_week: torch.Tensor,
    seed_trans: torch.Tensor,
    seed_delta_t: Optional[torch.Tensor],
    explain_full_traj: torch.Tensor,    # (n_explain, T, C)
    background_full_traj: torch.Tensor, # (n_background, T, C)
    time_varying_idx: np.ndarray,
    device: torch.device,
    head: str,
) -> np.ndarray:
    """
    Sequential perturbation attribution for time-varying covariate columns.

    For each time-varying column j and each explained customer:
        baseline = model prediction with the customer's actual trajectory
        ablated  = model prediction after replacing that column's trajectory
                   with the column-wise mean trajectory across the background set
        attribution[j] = | ablated - baseline |

    The mean-trajectory baseline preserves the temporal pattern observed in the
    cohort, so the perturbation reflects "what if this customer had typical
    campaign exposure?" rather than the broken "constant across time" surrogate
    that scalar SHAP would produce. Returns mean |Δ| across the explain set
    per time-varying feature.
    """
    n_explain, T, C = explain_full_traj.shape
    # Mean trajectory across background customers for each time-varying column.
    # Shape: (T, n_time_varying)
    bg_mean_traj = background_full_traj[:, :, time_varying_idx].mean(dim=0)

    sw = seed_week.expand(n_explain, -1).to(device)
    st = seed_trans.expand(n_explain, -1).to(device)
    sd = seed_delta_t.expand(n_explain, -1).to(device) if seed_delta_t is not None else None

    explain_full_traj = explain_full_traj.to(device)
    bg_mean_traj = bg_mean_traj.to(device)

    logits, log_spend = _model_forward(model, sw, st, explain_full_traj, sd)
    baseline = _scalarize(logits, log_spend, head).cpu().numpy()

    deltas = np.zeros((n_explain, len(time_varying_idx)), dtype=np.float32)
    for j_local, j_global in enumerate(time_varying_idx):
        ablated = explain_full_traj.clone()
        ablated[:, :, j_global] = bg_mean_traj[:, j_local].unsqueeze(0).expand(n_explain, -1)
        logits_a, log_spend_a = _model_forward(model, sw, st, ablated, sd)
        ablated_pred = _scalarize(logits_a, log_spend_a, head).cpu().numpy()
        deltas[:, j_local] = ablated_pred - baseline

    return deltas  # signed; caller takes |.| as needed


def _collect_inference_tensors(
    inference_ds,
    calibration_weeks: int,
) -> Tuple[np.ndarray, List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """
    Pull the per-customer covariate calibration trajectories and seed tensors
    into Python lists, sized (T_calib, C) and (T_calib,) respectively.
    """
    all_calib_traj: List[np.ndarray] = []
    all_seed_week: List[torch.Tensor] = []
    all_seed_trans: List[torch.Tensor] = []
    all_seed_delta_t: List[Optional[torch.Tensor]] = []

    for i in range(len(inference_ds)):
        item = inference_ds[i]
        if "covariates" not in item:
            raise RuntimeError(
                "Inference dataset contains no covariates. "
                "Ensure include_covariates=true in the config."
            )
        cov_calib = item["covariates"][:calibration_weeks, :].numpy()
        all_calib_traj.append(cov_calib)
        all_seed_week.append(item["seed_week"])
        all_seed_trans.append(item["seed_trans"])
        all_seed_delta_t.append(item.get("seed_delta_t"))

    calib_traj_stack = np.stack(all_calib_traj, axis=0)  # (N, T_calib, C)
    return calib_traj_stack, all_seed_week, all_seed_trans, all_seed_delta_t


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
    train_ds, val_ds, inference_ds, holdout_gt, scaler = pipeline.run(config)

    model = _load_model(config, checkpoint_path, device)
    joint = getattr(model, "joint", False)

    calib_weeks = config["dataset"]["calibration_weeks"]
    calib_traj, all_seed_week, all_seed_trans, all_seed_delta_t = _collect_inference_tensors(
        inference_ds, calib_weeks
    )
    N, T, C = calib_traj.shape

    # Identify which columns are static vs time-varying.
    is_static = _classify_static_columns(calib_traj)
    static_idx = np.flatnonzero(is_static)
    time_varying_idx = np.flatnonzero(~is_static)
    print(
        f"Detected {len(static_idx)} static and {len(time_varying_idx)} "
        f"time-varying covariate columns "
        f"(threshold {_STATIC_VARIANCE_TOL})."
    )

    feature_names = getattr(inference_ds, "covariate_feature_names", None) or [
        f"feature_{i}" for i in range(C)
    ]
    static_names = [feature_names[i] for i in static_idx]
    time_varying_names = [feature_names[i] for i in time_varying_idx]

    rng = np.random.RandomState(seed)
    idx_background = rng.choice(N, size=min(n_background, N), replace=False)
    idx_explain = rng.choice(N, size=min(n_explain, N), replace=False)

    # Static SHAP feature matrix — extract the (constant) static value at t=0.
    static_values = calib_traj[:, 0, :][:, static_idx]  # (N, n_static)
    background_static = static_values[idx_background]
    explain_static = static_values[idx_explain]

    # Trajectories used by the time-varying ablation.
    explain_full_traj = torch.tensor(calib_traj[idx_explain], dtype=torch.float32)
    background_full_traj = torch.tensor(calib_traj[idx_background], dtype=torch.float32)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Use the median customer's seed sequence as the "fixed history" backbone
    # for the static SHAP path, matching the previous behaviour.
    median_idx = int(N // 2)
    seed_week = all_seed_week[median_idx].unsqueeze(0)
    seed_trans = all_seed_trans[median_idx].unsqueeze(0)
    seed_delta_t = (
        all_seed_delta_t[median_idx].unsqueeze(0)
        if all_seed_delta_t[median_idx] is not None else None
    )
    fixed_time_varying_traj = torch.tensor(
        calib_traj[median_idx][:, time_varying_idx], dtype=torch.float32, device=device,
    )

    heads = ["freq"]
    if joint:
        heads.append("spend")

    summary: dict = {}

    for head in heads:
        print(f"\n=== SHAP attribution for '{head}' head ===")

        # --- Static features via KernelExplainer ---
        static_mean_abs = np.zeros(len(static_idx), dtype=np.float32)
        if len(static_idx) > 0:
            print(f"Running KernelExplainer over {len(static_idx)} static features ...")
            predict_fn = _build_static_predictor(
                model=model,
                seed_week=seed_week,
                seed_trans=seed_trans,
                fixed_time_varying_traj=fixed_time_varying_traj,
                seed_delta_t=seed_delta_t,
                static_idx=static_idx,
                time_varying_idx=time_varying_idx,
                n_features=C,
                device=device,
                head=head,
            )
            explainer = shap.KernelExplainer(predict_fn, background_static)
            static_shap = explainer.shap_values(explain_static, nsamples=100, l1_reg="aic")
            static_mean_abs = np.abs(static_shap).mean(axis=0)
            np.save(out_path / f"shap_{head}_static_values.npy", static_shap)
            with open(out_path / f"shap_{head}_static_features.json", "w") as f:
                json.dump(static_names, f)
        else:
            print("No static features detected — skipping KernelExplainer.")

        # --- Time-varying features via sequential ablation ---
        time_varying_mean_abs = np.zeros(len(time_varying_idx), dtype=np.float32)
        if len(time_varying_idx) > 0:
            print(
                f"Running sequential ablation over {len(time_varying_idx)} "
                f"time-varying features ..."
            )
            tv_deltas = _time_varying_ablation(
                model=model,
                seed_week=seed_week,
                seed_trans=seed_trans,
                seed_delta_t=seed_delta_t,
                explain_full_traj=explain_full_traj,
                background_full_traj=background_full_traj,
                time_varying_idx=time_varying_idx,
                device=device,
                head=head,
            )
            time_varying_mean_abs = np.abs(tv_deltas).mean(axis=0)
            np.save(out_path / f"shap_{head}_timevarying_deltas.npy", tv_deltas)
            with open(out_path / f"shap_{head}_timevarying_features.json", "w") as f:
                json.dump(time_varying_names, f)
        else:
            print("No time-varying features detected — skipping ablation.")

        # --- Combined bar chart ---
        all_names = static_names + time_varying_names
        all_importance = np.concatenate([static_mean_abs, time_varying_mean_abs])
        method_labels = (
            ["static (SHAP)"] * len(static_idx)
            + ["time-varying (ablation)"] * len(time_varying_idx)
        )
        if all_importance.size > 0:
            sort_idx = np.argsort(all_importance)[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            colours = [
                "#4C72B0" if method_labels[i] == "static (SHAP)" else "#DD8452"
                for i in sort_idx
            ]
            ax.bar(range(len(all_names)), all_importance[sort_idx], color=colours)
            ax.set_xticks(range(len(all_names)))
            ax.set_xticklabels(
                [
                    f"{all_names[i]}\n[{method_labels[i]}]"
                    for i in sort_idx
                ],
                rotation=30, ha="right",
            )
            ax.set_ylabel("Mean |attribution|")
            ax.set_title(
                f"Feature importance — "
                f"{'frequency' if head == 'freq' else 'log-spend'} head"
            )
            fig.tight_layout()
            fig.savefig(out_path / f"shap_{head}_dunnhumby.png", dpi=300)
            plt.close(fig)

        summary[head] = {
            "static_features": dict(zip(static_names, static_mean_abs.tolist())),
            "time_varying_features": dict(
                zip(time_varying_names, time_varying_mean_abs.tolist())
            ),
        }

    with open(out_path / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSHAP analysis complete.")


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
