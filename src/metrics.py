"""Métriques communes (incluant le R² pondéré à moyenne nulle de Jane Street)."""
from __future__ import annotations
import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if not (y.ndim == pred.ndim == w.ndim == 1 and len(y) == len(pred) == len(w)):
        raise ValueError('y_true, y_pred et weights doivent être des vecteurs de même longueur.')
    if len(y) == 0 or not (np.isfinite(y).all() and np.isfinite(pred).all() and np.isfinite(w).all()):
        raise ValueError('Entrées vides ou non finies pour les métriques.')
    if np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError('Poids négatifs ou poids totaux nuls.')
    mse = float(np.average(np.square(y - pred), weights=w))
    mae = float(np.average(np.abs(y - pred), weights=w))
    denom = float(np.sum(w * np.square(y)))
    if denom <= 0:
        raise ValueError('R² pondéré non défini lorsque somme(weight * responder_6²) = 0.')
    return {'weighted_r2': float(1.0 - (np.sum(w * np.square(y - pred)) / denom)),
            'weighted_rmse': float(np.sqrt(mse)), 'weighted_mae': mae}
