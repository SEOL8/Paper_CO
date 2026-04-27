import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience: int = 20, delta: float = 0.01, verbose: bool = False):
        self.patience     = patience
        self.delta        = delta
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, val_loss: float, model: nn.Module, path: str):
        if self.verbose:
            print(f'Val loss improved ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model.')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
