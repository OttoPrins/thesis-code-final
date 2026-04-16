"""
PyTorch Dataset for customer transaction sequences (seq-to-seq format).

Stores separate integer tensors for week and transaction count (for embedding layers),
plus continuous spend, and sequence-level targets at every time step.

Matches the input format expected by LSTMModel and TransformerModel:
    - week:  (B, T) integer week indices → nn.Embedding
    - trans: (B, T) integer transaction counts → nn.Embedding
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomerDataset(Dataset):
    """
    PyTorch Dataset for seq-to-seq customer transaction sequences.

    Tensor shapes per sample:
        week        : (T-1,)  int   — week indices for input steps
        trans       : (T-1,)  int   — transaction counts (teacher forcing input)
        spend       : (T-1,)  float — log-spend at each input step
        y_freq      : (T-1,)  int   — target frequency class at each step
        y_spend     : (T-1,)  float — target log-spend at each step
        mask        : (T-1,)  float — 1=real data, 0=padding
        customer_id : ()      int   — customer identifier

    Optional (for autoregressive inference):
        seed_week   : (T,)    int   — full calibration week sequence
        seed_trans  : (T,)    int   — full calibration transaction sequence
        seed_spend  : (T,)    float — full calibration spend sequence
    """

    def __init__(self, data: dict, include_seed: bool = False):
        """
        Args:
            data: dict from SequenceBuilder.build() with keys:
                  week_input, trans_input, spend_input,
                  y_freq, y_spend, customer_ids, mask
                  Optionally: seed_week, seed_trans, seed_spend
            include_seed: if True, also store seed sequences (for inference)
        """
        self.week = torch.tensor(data["week_input"], dtype=torch.long)
        self.trans = torch.tensor(data["trans_input"], dtype=torch.long)
        self.spend = torch.tensor(data["spend_input"], dtype=torch.float32)
        self.y_freq = torch.tensor(data["y_freq"], dtype=torch.long)
        self.y_spend = torch.tensor(data["y_spend"], dtype=torch.float32)
        self.customer_ids = torch.tensor(data["customer_ids"], dtype=torch.long)
        self.mask = torch.tensor(data["mask"], dtype=torch.float32)
        self.max_trans = data["max_trans"]

        if include_seed and "seed_week" in data:
            self.seed_week = torch.tensor(data["seed_week"], dtype=torch.long)
            self.seed_trans = torch.tensor(data["seed_trans"], dtype=torch.long)
            self.seed_spend = torch.tensor(data["seed_spend"], dtype=torch.float32)
        else:
            self.seed_week = None

    def __len__(self):
        return len(self.week)

    def __getitem__(self, idx):
        item = {
            "week": self.week[idx],
            "trans": self.trans[idx],
            "spend": self.spend[idx],
            "y_freq": self.y_freq[idx],
            "y_spend": self.y_spend[idx],
            "customer_id": self.customer_ids[idx],
            "mask": self.mask[idx],
        }
        if self.seed_week is not None:
            item["seed_week"] = self.seed_week[idx]
            item["seed_trans"] = self.seed_trans[idx]
            item["seed_spend"] = self.seed_spend[idx]
        return item
