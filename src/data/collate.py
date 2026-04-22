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
        trans       : (B, T-1)  long
        spend       : (B, T-1)  float
        y_freq      : (B, T-1)  long
        y_spend     : (B, T-1)  float
        customer_id : (B,)      long
        mask        : (B, T-1)  float
    Optional:
        seed_week   : (B, T)    long
        seed_trans  : (B, T)    long
        seed_spend  : (B, T)    float
        covariates  : (B, T, C) float  — training slice or (B, T_total, C) inference trajectory
    """
    result = {
        "week": torch.stack([item["week"] for item in batch]),
        "trans": torch.stack([item["trans"] for item in batch]),
        "spend": torch.stack([item["spend"] for item in batch]),
        "y_freq": torch.stack([item["y_freq"] for item in batch]),
        "y_spend": torch.stack([item["y_spend"] for item in batch]),
        "customer_id": torch.stack([item["customer_id"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }

    if "seed_week" in batch[0]:
        result["seed_week"] = torch.stack([item["seed_week"] for item in batch])
        result["seed_trans"] = torch.stack([item["seed_trans"] for item in batch])
        result["seed_spend"] = torch.stack([item["seed_spend"] for item in batch])

    if "covariates" in batch[0]:
        # Handles both (T-1, C) training slices and (T_total, C) inference trajectories
        result["covariates"] = torch.stack([item["covariates"] for item in batch])

    return result
