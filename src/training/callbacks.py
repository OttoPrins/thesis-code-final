"""
Training callbacks.

EarlyStopping monitors validation loss and stops training when performance
stops improving. Best model weights are stored in memory and restored at the end.
"""

import copy
import torch.nn as nn


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Stores best model weights in memory (deepcopy of state_dict).
    Call load_best() after training to restore the best weights.

    Args:
        patience:   Number of epochs with no improvement before stopping.
        min_delta:  Minimum change to count as an improvement (default 0).
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss.
            model:    Current model (saved if this is the best so far).

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience

    def load_best(self, model: nn.Module):
        """Restore model to the best saved weights."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
