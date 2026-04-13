"""
PyTorch Dataset and DataLoader utilities for customer sequence data.

CustomerDataset wraps the tensor outputs from SequenceBuilder.
collate_fn in collate.py handles variable-length padding within a batch.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomerDataset(Dataset):
    """
    PyTorch Dataset for customer transaction sequences.

    Tensor shapes (all are torch tensors):
        X           : (T, F)   — input sequence; T=lookback, F=features
        y_freq      : ()        — scalar class label {0,1,2,3}
        y_spend     : ()        — scalar log-spend label
        mask        : (T,)      — 1=real data, 0=padding
        customer_id : ()        — scalar int customer identifier
        covariate   : (C,) or None — static covariate vector (Extension 3 only)
    """

    def __init__(
        self,
        X: np.ndarray,
        y_freq: np.ndarray,
        y_spend: np.ndarray,
        customer_ids: np.ndarray,
        mask: np.ndarray,
        covariates: np.ndarray = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_freq = torch.tensor(y_freq, dtype=torch.long)
        self.y_spend = torch.tensor(y_spend, dtype=torch.float32)
        self.customer_ids = torch.tensor(customer_ids, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.covariates = (
            torch.tensor(covariates, dtype=torch.float32)
            if covariates is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "X": self.X[idx],
            "y_freq": self.y_freq[idx],
            "y_spend": self.y_spend[idx],
            "customer_id": self.customer_ids[idx],
            "mask": self.mask[idx],
        }
        if self.covariates is not None:
            item["covariate"] = self.covariates[idx]
        return item
