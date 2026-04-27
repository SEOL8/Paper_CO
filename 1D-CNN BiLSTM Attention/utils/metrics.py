import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, loss: float) -> dict:
    return {
        'loss': loss,
        'mse':  mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae':  mean_absolute_error(targets, predictions),
        'r2':   r2_score(targets, predictions),
        'mape': mean_absolute_percentage_error(targets, predictions),
    }


def print_metrics(metrics: dict):
    print(f"Loss: {metrics['loss']:.4f} | "
          f"MSE: {metrics['mse']:.4f} | "
          f"RMSE: {metrics['rmse']:.4f} | "
          f"MAE: {metrics['mae']:.4f} | "
          f"R2: {metrics['r2']:.4f} | "
          f"MAPE: {metrics['mape']:.4f}")
