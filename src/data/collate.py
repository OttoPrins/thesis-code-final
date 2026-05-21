"""
Custom collate function for DataLoader.

All sequences within a dataset have the same length (dense weekly grid),
so torch.stack suffices — no dynamic padding needed.

Usage:
    DataLoader(dataset, collate_fn=collate_fn, ...)
"""

import torch


def collate_fn(batch):
    """
    Collate a list of samples into a batched dict.

    Returns dict with tensors:
        week        : (B, T-1)  long
        position    : (B, T-1)  long
        trans       : (B, T-1)  long
        spend       : (B, T-1)  float
        delta_t     : (B, T-1)  float
        state_features: (B, T-1, S) float
        y_freq      : (B, T-1)  long
        y_freq_raw  : (B, T-1)  float
        y_spend     : (B, T-1)  float
        active_mask : (B, T-1)  float
        customer_id : (B,)      long
        mask        : (B, T-1)  float
    Optional (Extension 3 — Dunnhumby only):
        static_covariates  : (B, S)        float — static per-customer features
        dynamic_covariates : (B, T, D)     float — training slice or (B, T_total, D)
                             for inference trajectories
    Optional (inference seed):
        seed_week    : (B, T)    long
        seed_position: (B, T)    long
        seed_trans   : (B, T)    long
        seed_spend   : (B, T)    float
        seed_delta_t : (B, T)    float
        seed_state_features: (B, T, S) float
    """
    result = {
        "week": torch.stack([item["week"] for item in batch]),
        "position": torch.stack([item["position"] for item in batch]),
        "trans": torch.stack([item["trans"] for item in batch]),
        "spend": torch.stack([item["spend"] for item in batch]),
        "state_features": torch.stack([item["state_features"] for item in batch]),
        "y_freq": torch.stack([item["y_freq"] for item in batch]),
        "y_freq_raw": torch.stack([item["y_freq_raw"] for item in batch]),
        "y_spend": torch.stack([item["y_spend"] for item in batch]),
        "active_mask": torch.stack([item["active_mask"] for item in batch]),
        "customer_id": torch.stack([item["customer_id"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }

    if "delta_t" in batch[0]:
        result["delta_t"] = torch.stack([item["delta_t"] for item in batch])

    if "seed_week" in batch[0]:
        result["seed_week"] = torch.stack([item["seed_week"] for item in batch])
        if "seed_position" in batch[0]:
            result["seed_position"] = torch.stack(
                [item["seed_position"] for item in batch]
            )
        result["seed_trans"] = torch.stack([item["seed_trans"] for item in batch])
        if "seed_raw_trans" in batch[0]:
            result["seed_raw_trans"] = torch.stack(
                [item["seed_raw_trans"] for item in batch]
            )
        result["seed_spend"] = torch.stack([item["seed_spend"] for item in batch])
        if "seed_delta_t" in batch[0]:
            result["seed_delta_t"] = torch.stack(
                [item["seed_delta_t"] for item in batch]
            )
        if "seed_state_features" in batch[0]:
            result["seed_state_features"] = torch.stack(
                [item["seed_state_features"] for item in batch]
            )

    if "static_covariates" in batch[0]:
        result["static_covariates"] = torch.stack(
            [item["static_covariates"] for item in batch]
        )  # (B, S)

    if "dynamic_covariates" in batch[0]:
        result["dynamic_covariates"] = torch.stack(
            [item["dynamic_covariates"] for item in batch]
        )  # (B, T, D) or (B, T_total, D)

    return result
