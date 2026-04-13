"""
Custom collate function for DataLoader.

Pads sequences within a batch to the maximum length in that batch.
Builds a boolean mask tensor (1=real, 0=padding).

Usage:
    DataLoader(dataset, collate_fn=collate_fn, ...)
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate a list of samples into a batched dict.

    If all sequences are the same length (fixed lookback window), padding is a no-op.
    If sequences vary (e.g. customers with short histories), this pads to max length.
    """
    X = torch.stack([item["X"] for item in batch])
    y_freq = torch.stack([item["y_freq"] for item in batch])
    y_spend = torch.stack([item["y_spend"] for item in batch])
    customer_ids = torch.stack([item["customer_id"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])

    result = {
        "X": X,                     # (B, T, F)
        "y_freq": y_freq,           # (B,)
        "y_spend": y_spend,         # (B,)
        "customer_id": customer_ids,# (B,)
        "mask": mask,               # (B, T)
    }

    if "covariate" in batch[0]:
        result["covariate"] = torch.stack([item["covariate"] for item in batch])  # (B, C)

    return result
