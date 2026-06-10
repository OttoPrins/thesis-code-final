"""
Customer-specific SHAP attribution for Extension 3 on Dunnhumby.

The analysis explains the first holdout-week predictions from the full-covariate
models while holding each household's own 80-week transaction history fixed.
Only the two static demographic covariates and two dynamic campaign covariates
are varied against a shared empirical background.

The two explained outputs are:
    - expected first-holdout-week transaction frequency;
    - conditional first-holdout-week log1p-spend.

Attributions are associational model explanations conditional on transaction
history. They do not identify causal campaign effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.data.datasets import DunnhumbyPipeline
from src.models import LSTMModel, TransformerModel
from src.utils.config import load_config
from src.utils.seed import set_seed
from train import build_model


HEADS = ("freq", "spend")
STATIC_FEATURES = ("income", "household_size")
DYNAMIC_FEATURES = ("coupon_redemptions_week", "campaign_active_flag")


@dataclass(frozen=True)
class ShapSample:
    background_ids: np.ndarray
    explain_ids: np.ndarray
    cohort: str
    analysis_seed: int


class CovariateShapWrapper(nn.Module):
    """
    Expose only covariates as differentiable inputs for one household history.

    A separate wrapper is used for each explained household. The empirical
    background covariates are therefore evaluated under that same household's
    fixed transaction, spend, week, and elapsed-time history.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        seed_week: torch.Tensor,
        seed_trans: torch.Tensor,
        seed_spend: torch.Tensor,
        seed_delta_t: Optional[torch.Tensor],
        head: str,
        spend_center: float = 0.0,
        spend_scale: float = 1.0,
    ):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"Unknown SHAP head: {head!r}")
        self.model = model
        self.register_buffer("seed_week", seed_week)
        self.register_buffer("seed_trans", seed_trans)
        self.register_buffer("seed_spend", seed_spend)
        if seed_delta_t is not None:
            self.register_buffer("seed_delta_t", seed_delta_t)
        else:
            self.seed_delta_t = None
        self.head = head
        self.spend_center = float(spend_center)
        self.spend_scale = float(spend_scale)
        self._force_lstm_train_for_cudnn = False

    def train(self, mode: bool = True) -> "CovariateShapWrapper":
        super().train(mode)
        if self._force_lstm_train_for_cudnn:
            self.model.lstm.train()
        return self

    def forward(
        self,
        static_cov: torch.Tensor,
        dynamic_cov: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = static_cov.shape[0]
        week = self.seed_week.expand(batch_size, -1)
        trans = self.seed_trans.expand(batch_size, -1)
        spend = self.seed_spend.expand(batch_size, -1)
        delta_t = (
            self.seed_delta_t.expand(batch_size, -1)
            if self.seed_delta_t is not None
            else None
        )
        state_features = None
        if delta_t is not None and getattr(self.model, "state_feature_dim", 0) > 0:
            state_features = delta_t.unsqueeze(-1)

        if isinstance(self.model, LSTMModel):
            output = self.model(
                week,
                trans,
                spend=spend,
                state_features=state_features,
                static_covariates=static_cov,
                dynamic_covariates=dynamic_cov,
            )
            if getattr(self.model, "joint", False):
                if len(output) == 4:
                    logits, spend_mu, _, _ = output
                else:
                    logits, spend_mu, _ = output
            else:
                logits, _ = output
                spend_mu = None
        elif isinstance(self.model, TransformerModel):
            output = self.model(
                week,
                trans,
                spend=spend,
                state_features=state_features,
                static_covariates=static_cov,
                dynamic_covariates=dynamic_cov,
                delta_t=delta_t,
            )
            if getattr(self.model, "joint", False):
                if len(output) == 3:
                    logits, spend_mu, _ = output
                else:
                    logits, spend_mu = output
            else:
                logits = output
                spend_mu = None
        else:
            raise TypeError(f"Unsupported SHAP model type: {type(self.model)!r}")

        if self.head == "freq":
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            class_values = torch.arange(
                probabilities.shape[-1],
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            prediction = (probabilities * class_values).sum(dim=-1)
        else:
            if spend_mu is None:
                raise ValueError("Conditional spend attribution requires a joint model.")
            prediction = spend_mu[:, -1]
            prediction = prediction * self.spend_scale + self.spend_center

        # GradientExplainer indexes outputs[:, output_index].
        return prediction.unsqueeze(-1)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(
    config: dict,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    model = build_model(config).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_wrapper_for_gradients(
    wrapper: CovariateShapWrapper,
    device: torch.device,
) -> None:
    """
    Preserve deterministic inference while enabling cuDNN LSTM backward.

    cuDNN rejects RNN backward in evaluation mode. Only the recurrent module is
    switched when CUDA is used; the full model, including Transformer dropout,
    remains in evaluation mode.
    """
    wrapper._force_lstm_train_for_cudnn = (
        device.type == "cuda" and isinstance(wrapper.model, LSTMModel)
    )
    wrapper.eval()


def normalise_shap_values(
    shap_values: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize current and legacy GradientExplainer multi-input layouts."""
    if isinstance(shap_values, np.ndarray):
        raise RuntimeError(
            "GradientExplainer returned one array for a two-input model."
        )

    if (
        isinstance(shap_values, list)
        and len(shap_values) == 2
        and not isinstance(shap_values[0], list)
    ):
        static_shap = np.asarray(shap_values[0])
        dynamic_shap = np.asarray(shap_values[1])
    elif (
        isinstance(shap_values, list)
        and len(shap_values) == 1
        and isinstance(shap_values[0], list)
        and len(shap_values[0]) == 2
    ):
        static_shap = np.asarray(shap_values[0][0])
        dynamic_shap = np.asarray(shap_values[0][1])
    else:
        raise RuntimeError(
            "Unexpected GradientExplainer output layout for two model inputs."
        )

    if static_shap.ndim == 3 and static_shap.shape[-1] == 1:
        static_shap = static_shap[..., 0]
    if dynamic_shap.ndim == 4 and dynamic_shap.shape[-1] == 1:
        dynamic_shap = dynamic_shap[..., 0]
    if static_shap.ndim != 2 or dynamic_shap.ndim != 3:
        raise RuntimeError(
            "Unexpected SHAP array shapes: "
            f"static={static_shap.shape}, dynamic={dynamic_shap.shape}."
        )
    return static_shap, dynamic_shap


def aggregate_dynamic_signed(dynamic_shap: np.ndarray) -> np.ndarray:
    """
    Group a dynamic feature across time while preserving SHAP additivity.

    Summing signed contributions before taking absolute values prevents positive
    and negative weekly effects from being counted as independent features.
    """
    values = np.asarray(dynamic_shap, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Dynamic SHAP values must have shape (N, T, D).")
    return values.sum(axis=1)


def additivity_diagnostics(
    prediction: np.ndarray,
    baseline: np.ndarray,
    static_shap: np.ndarray,
    dynamic_shap: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    attributed = (
        np.asarray(static_shap, dtype=np.float64).sum(axis=1)
        + np.asarray(dynamic_shap, dtype=np.float64).sum(axis=(1, 2))
    )
    delta = prediction - baseline
    residual = delta - attributed
    denominator = max(float(np.mean(np.abs(delta))), 1e-8)
    normalized_error = float(np.mean(np.abs(residual)) / denominator)
    return {
        "prediction": prediction,
        "baseline": baseline,
        "prediction_delta": delta,
        "attributed_delta": attributed,
        "residual": residual,
        "normalized_error": normalized_error,
        "mean_abs_residual": float(np.mean(np.abs(residual))),
    }


def observed_demographic_customer_ids(raw_dir: str | Path) -> set[int]:
    path = Path(raw_dir) / "hh_demographic.csv"
    frame = pd.read_csv(path, usecols=["household_key"])
    return set(frame["household_key"].astype(int).tolist())


def eligible_customer_ids(
    inference_ds,
    *,
    raw_dir: str | Path,
    cohort: str,
) -> np.ndarray:
    customer_ids = inference_ds.customer_ids.cpu().numpy().astype(np.int64)
    if cohort == "observed_demographics":
        observed = observed_demographic_customer_ids(raw_dir)
        return customer_ids[np.isin(customer_ids, list(observed))]
    if cohort == "all_households":
        return customer_ids
    raise ValueError(
        "cohort must be 'observed_demographics' or 'all_households'"
    )


def make_disjoint_sample(
    customer_ids: Iterable[int],
    *,
    n_background: int,
    n_explain: int,
    analysis_seed: int,
    cohort: str,
) -> ShapSample:
    ids = np.asarray(list(customer_ids), dtype=np.int64)
    required = int(n_background) + int(n_explain)
    if len(ids) < required:
        raise ValueError(
            f"Cohort {cohort!r} has {len(ids)} households; {required} are required."
        )
    rng = np.random.RandomState(analysis_seed)
    selected = rng.choice(ids, size=required, replace=False)
    return ShapSample(
        background_ids=selected[:n_background],
        explain_ids=selected[n_background:],
        cohort=cohort,
        analysis_seed=analysis_seed,
    )


def save_sample_manifest(sample: ShapSample, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cohort": sample.cohort,
        "analysis_seed": int(sample.analysis_seed),
        "background_customer_ids": sample.background_ids.astype(int).tolist(),
        "explain_customer_ids": sample.explain_ids.astype(int).tolist(),
        "disjoint": bool(
            not set(sample.background_ids.tolist())
            & set(sample.explain_ids.tolist())
        ),
    }
    output.write_text(json.dumps(payload, indent=2))
    return output


def load_sample_manifest(path: str | Path) -> ShapSample:
    payload = json.loads(Path(path).read_text())
    sample = ShapSample(
        background_ids=np.asarray(
            payload["background_customer_ids"], dtype=np.int64
        ),
        explain_ids=np.asarray(
            payload["explain_customer_ids"], dtype=np.int64
        ),
        cohort=str(payload["cohort"]),
        analysis_seed=int(payload["analysis_seed"]),
    )
    overlap = set(sample.background_ids.tolist()) & set(sample.explain_ids.tolist())
    if overlap:
        raise ValueError("SHAP background and explanation samples overlap.")
    return sample


def _indices_for_ids(inference_ds, customer_ids: np.ndarray) -> np.ndarray:
    all_ids = inference_ds.customer_ids.cpu().numpy().astype(np.int64)
    index = {int(customer_id): row for row, customer_id in enumerate(all_ids)}
    missing = [int(customer_id) for customer_id in customer_ids if int(customer_id) not in index]
    if missing:
        raise ValueError(f"Sample manifest contains unknown customer IDs: {missing[:5]}")
    return np.asarray([index[int(customer_id)] for customer_id in customer_ids])


def run_output_dir(
    root: str | Path,
    *,
    cohort: str,
    architecture: str,
    seed: int,
    n_integration_samples: int,
) -> Path:
    return (
        Path(root)
        / cohort
        / architecture
        / f"seed{seed}"
        / f"nsamples{n_integration_samples}"
    )


def _feature_names(inference_ds) -> tuple[list[str], list[str]]:
    static_names = getattr(inference_ds, "static_feature_names", None)
    dynamic_names = getattr(inference_ds, "dynamic_feature_names", None)
    return (
        list(static_names or STATIC_FEATURES),
        list(dynamic_names or DYNAMIC_FEATURES),
    )


def _run_cache_payload(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    sample_manifest_path: str | Path,
    head_names: tuple[str, ...],
    n_integration_samples: int,
    analysis_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "sample_manifest_sha256": _sha256(sample_manifest_path),
        "source_sha256": _sha256(__file__),
        "heads": list(head_names),
        "n_integration_samples": int(n_integration_samples),
        "analysis_seed": int(analysis_seed),
        "device_type": device.type,
        "shap_version": shap.__version__,
        "torch_version": torch.__version__,
    }


def _cache_is_valid(output_dir: Path, cache_key: str) -> bool:
    provenance_path = output_dir / "provenance.json"
    values_path = output_dir / "shap_values.npz"
    summary_path = output_dir / "summary.csv"
    if not (provenance_path.exists() and values_path.exists() and summary_path.exists()):
        return False
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return provenance.get("cache_key") == cache_key and provenance.get("complete") is True


def _plot_run_importance(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    architecture: str,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"static": "#4C72B0", "dynamic": "#DD8452"}
    for axis, head in zip(axes, HEADS):
        subset = summary[summary["head"] == head].sort_values("mean_abs_shap")
        axis.barh(
            subset["feature"],
            subset["mean_abs_shap"],
            color=[colors[value] for value in subset["feature_type"]],
        )
        axis.set_title("Frequency" if head == "freq" else "Conditional log1p-spend")
        axis.set_xlabel("Mean absolute grouped SHAP value")
    fig.suptitle(f"{architecture.upper()} covariate attribution, seed {seed}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_shap(
    *,
    config_path: str,
    checkpoint_path: str,
    sample_manifest_path: str,
    output_root: str = "results/final_kaggle/shap",
    n_integration_samples: int = 64,
    device_name: str = "auto",
    model_seed: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    analysis_sample = load_sample_manifest(sample_manifest_path)
    analysis_seed = analysis_sample.analysis_seed
    set_seed(analysis_seed)
    device = _resolve_device(device_name)

    if config.get("dataset", {}).get("name") != "dunnhumby":
        raise ValueError("Extension 3 SHAP is defined only for Dunnhumby.")
    if config.get("dataset", {}).get("covariate_mode") != "full":
        raise ValueError("SHAP requires the full-covariate Extension 3 config.")
    if not config.get("model", {}).get("joint", False):
        raise ValueError("Dual-head SHAP requires a joint model.")

    architecture = str(config["model"]["type"])
    seed = int(
        config["training"]["seed"] if model_seed is None else model_seed
    )
    output_dir = run_output_dir(
        output_root,
        cohort=analysis_sample.cohort,
        architecture=architecture,
        seed=seed,
        n_integration_samples=n_integration_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_payload = _run_cache_payload(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        sample_manifest_path=sample_manifest_path,
        head_names=HEADS,
        n_integration_samples=n_integration_samples,
        analysis_seed=analysis_seed,
        device=device,
    )
    cache_key = _json_hash(cache_payload)
    if not force and _cache_is_valid(output_dir, cache_key):
        print(f"[SHAP] cache hit: {output_dir}")
        return json.loads((output_dir / "provenance.json").read_text())

    print(
        f"[SHAP] {architecture} seed={seed} cohort={analysis_sample.cohort} "
        f"nsamples={n_integration_samples} device={device}"
    )
    pipeline = DunnhumbyPipeline()
    _, _, inference_ds, _, scaler = pipeline.run(config)
    model = _load_model(config, checkpoint_path, device)
    calibration_weeks = int(config["dataset"]["calibration_weeks"])

    background_indices = _indices_for_ids(
        inference_ds, analysis_sample.background_ids
    )
    explain_indices = _indices_for_ids(inference_ds, analysis_sample.explain_ids)
    static_names, dynamic_names = _feature_names(inference_ds)

    background_static = inference_ds.static_covariates[background_indices].to(device)
    background_dynamic = inference_ds.dynamic_covariates[
        background_indices, :calibration_weeks, :
    ].to(device)
    explain_static_values = inference_ds.static_covariates[
        explain_indices
    ].cpu().numpy()
    explain_dynamic_values = inference_ds.dynamic_covariates[
        explain_indices, :calibration_weeks, :
    ].cpu().numpy()

    results: dict[str, dict[str, np.ndarray | float]] = {}
    summary_rows: list[dict[str, Any]] = []
    additivity_rows: list[dict[str, Any]] = []

    for head in HEADS:
        static_values: list[np.ndarray] = []
        dynamic_values: list[np.ndarray] = []
        predictions: list[float] = []
        baselines: list[float] = []

        iterator = tqdm(
            explain_indices,
            desc=f"{architecture} seed {seed} {head}",
            unit="household",
        )
        for local_row, dataset_index in enumerate(iterator):
            item = inference_ds[int(dataset_index)]
            wrapper = CovariateShapWrapper(
                model,
                seed_week=item["seed_week"].unsqueeze(0).to(device),
                seed_trans=item["seed_trans"].unsqueeze(0).to(device),
                seed_spend=item["seed_spend"].unsqueeze(0).to(device),
                seed_delta_t=item.get("seed_delta_t").unsqueeze(0).to(device)
                if item.get("seed_delta_t") is not None
                else None,
                head=head,
                spend_center=scaler.center_,
                spend_scale=scaler.scale_,
            ).to(device)
            _prepare_wrapper_for_gradients(wrapper, device)

            explainer = shap.GradientExplainer(
                wrapper,
                [background_static, background_dynamic],
                batch_size=min(100, len(background_indices)),
            )
            explain_static = torch.as_tensor(
                explain_static_values[local_row : local_row + 1],
                dtype=torch.float32,
                device=device,
            )
            explain_dynamic = torch.as_tensor(
                explain_dynamic_values[local_row : local_row + 1],
                dtype=torch.float32,
                device=device,
            )
            raw_shap = explainer.shap_values(
                [explain_static, explain_dynamic],
                nsamples=n_integration_samples,
                rseed=analysis_seed + local_row,
            )
            static_shap, dynamic_shap = normalise_shap_values(raw_shap)
            static_values.append(static_shap[0])
            dynamic_values.append(dynamic_shap[0])

            with torch.no_grad():
                prediction = wrapper(explain_static, explain_dynamic)
                background_prediction = wrapper(
                    background_static, background_dynamic
                )
            predictions.append(float(prediction[0, 0].detach().cpu()))
            baselines.append(
                float(background_prediction[:, 0].mean().detach().cpu())
            )

        static_array = np.stack(static_values).astype(np.float32)
        dynamic_array = np.stack(dynamic_values).astype(np.float32)
        grouped_dynamic = aggregate_dynamic_signed(dynamic_array).astype(np.float32)
        diagnostics = additivity_diagnostics(
            np.asarray(predictions),
            np.asarray(baselines),
            static_array,
            dynamic_array,
        )
        results[head] = {
            "static_shap": static_array,
            "dynamic_shap": dynamic_array,
            "dynamic_grouped_shap": grouped_dynamic,
            "prediction": diagnostics["prediction"],
            "baseline": diagnostics["baseline"],
            "prediction_delta": diagnostics["prediction_delta"],
            "attributed_delta": diagnostics["attributed_delta"],
            "residual": diagnostics["residual"],
            "normalized_error": diagnostics["normalized_error"],
        }

        for feature_type, names, values in (
            ("static", static_names, static_array),
            ("dynamic", dynamic_names, grouped_dynamic),
        ):
            for feature_index, feature_name in enumerate(names):
                per_customer = values[:, feature_index]
                summary_rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "cohort": analysis_sample.cohort,
                        "head": head,
                        "feature_type": feature_type,
                        "feature": feature_name,
                        "mean_abs_shap": float(np.mean(np.abs(per_customer))),
                        "mean_signed_shap": float(np.mean(per_customer)),
                        "n_households": len(explain_indices),
                        "n_integration_samples": n_integration_samples,
                    }
                )

        for row, customer_id in enumerate(analysis_sample.explain_ids):
            additivity_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "cohort": analysis_sample.cohort,
                    "head": head,
                    "customer_id": int(customer_id),
                    "prediction": float(diagnostics["prediction"][row]),
                    "background_prediction": float(diagnostics["baseline"][row]),
                    "prediction_delta": float(diagnostics["prediction_delta"][row]),
                    "attributed_delta": float(diagnostics["attributed_delta"][row]),
                    "residual": float(diagnostics["residual"][row]),
                }
            )

    arrays: dict[str, np.ndarray] = {
        "customer_ids": analysis_sample.explain_ids.astype(np.int64),
        "background_customer_ids": analysis_sample.background_ids.astype(np.int64),
        "static_feature_values": explain_static_values.astype(np.float32),
        "dynamic_feature_values": explain_dynamic_values.astype(np.float32),
        "dynamic_feature_totals": explain_dynamic_values.sum(axis=1).astype(np.float32),
    }
    for head in HEADS:
        for key, value in results[head].items():
            if isinstance(value, np.ndarray):
                arrays[f"{head}_{key}"] = value
    np.savez_compressed(output_dir / "shap_values.npz", **arrays)

    feature_payload = {
        "static": static_names,
        "dynamic": dynamic_names,
        "dynamic_grouping": "signed_sum_over_calibration_weeks",
        "spend_output": "conditional_unscaled_log1p_spend",
    }
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_payload, indent=2)
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(additivity_rows).to_csv(
        output_dir / "additivity_diagnostics.csv", index=False
    )
    _plot_run_importance(
        summary,
        output_dir / "importance.png",
        architecture=architecture,
        seed=seed,
    )

    normalized_errors = {
        head: float(results[head]["normalized_error"]) for head in HEADS
    }
    provenance = {
        **cache_payload,
        "cache_key": cache_key,
        "complete": True,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "sample_manifest_path": str(sample_manifest_path),
        "output_dir": str(output_dir),
        "run_name": config["output"]["run_name"],
        "architecture": architecture,
        "seed": seed,
        "cohort": analysis_sample.cohort,
        "n_background": len(background_indices),
        "n_explain": len(explain_indices),
        "calibration_weeks": calibration_weeks,
        "heads": list(HEADS),
        "method": "shap.GradientExplainer",
        "explanation_target": {
            "freq": "first-holdout-week expected transaction count",
            "spend": "first-holdout-week conditional log1p-spend",
        },
        "history_policy": "each explained household retains its own calibration history",
        "dynamic_aggregation": "sum_signed_over_time_then_absolute_for_importance",
        "normalized_additivity_error": normalized_errors,
        "causal_warning": (
            "Interventional model attribution conditional on transaction history; "
            "not a causal campaign-effect estimate."
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(
        "[SHAP] complete "
        + ", ".join(
            f"{head} additivity={normalized_errors[head]:.3f}" for head in HEADS
        )
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer-specific dual-head SHAP analysis for Extension 3."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument(
        "--output_root", default="results/final_kaggle/shap"
    )
    parser.add_argument("--n_integration_samples", type=int, default=64)
    parser.add_argument("--model_seed", type=int)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_shap(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        sample_manifest_path=args.sample_manifest,
        output_root=args.output_root,
        n_integration_samples=args.n_integration_samples,
        device_name=args.device,
        model_seed=args.model_seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
