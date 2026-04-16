"""
Abstract data pipeline. Each dataset implements this interface.

Usage:
    from src.data.datasets.cdnow import CDNOWPipeline
    train_ds, val_ds, test_ds = CDNOWPipeline().run(config)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import yaml


class BasePipeline(ABC):
    """
    Base class for all dataset pipelines.

    Subclasses override load_raw(), clean(), and optionally build_covariates().
    The run() method orchestrates the full pipeline using shared transforms.
    """

    def run(self, config: dict) -> Tuple:
        """
        Full pipeline: raw data → PyTorch datasets + holdout ground truth.

        Args:
            config: Loaded YAML config dict (see experiments/configs/).

        Returns:
            (train_ds, val_ds, inference_ds, holdout_ground_truth, scaler)

            train_ds:             CustomerDataset for training (no seed sequences)
            val_ds:               CustomerDataset for validation (with seed for inference)
            inference_ds:         CustomerDataset with all customers + seeds (for holdout eval)
            holdout_ground_truth: dict with customer_ids, freq, spend arrays for holdout period
            scaler:               fitted SpendScaler (for inverse-transforming predictions)
        """
        raise NotImplementedError

    @abstractmethod
    def load_raw(self, raw_dir: str):
        """Read original files from data/raw/<dataset>/. Return a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def clean(self, df):
        """Remove bad rows, enforce dtypes, standardise column names.

        Output columns must be: [customer_id, date, transaction_amount]
        """
        raise NotImplementedError

    def build_covariates(self, df):
        """Optional. Build static per-customer covariate matrix.

        Only implemented by DunnhumbyPipeline for Extension 3.
        Returns None by default.
        """
        return None


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)
