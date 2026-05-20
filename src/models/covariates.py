"""Covariate tensor helpers shared by model training and validation paths."""

from __future__ import annotations

import torch


def align_dynamic_covariates(
    dynamic_covariates: torch.Tensor | None,
    target_len: int,
) -> torch.Tensor | None:
    """Return dynamic covariates aligned to a teacher-forced sequence length."""
    if dynamic_covariates is None:
        return None
    if dynamic_covariates.ndim != 3:
        raise ValueError(
            "dynamic_covariates must have shape (batch, time, features), "
            f"got {tuple(dynamic_covariates.shape)}"
        )
    if dynamic_covariates.size(1) < target_len:
        raise ValueError(
            "dynamic_covariates shorter than model input sequence: "
            f"{dynamic_covariates.size(1)} < {target_len}"
        )
    if dynamic_covariates.size(1) == target_len:
        return dynamic_covariates
    return dynamic_covariates[:, :target_len, :]
