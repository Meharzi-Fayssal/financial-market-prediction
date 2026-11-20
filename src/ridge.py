"""Ridge de scikit-learn : entrée standardisée + indicateurs d'absence."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge

from config import Settings
from metrics import calculate_metrics
from preprocessing import PreprocessedData


def train_ridge(data: PreprocessedData, cfg: Settings, model_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    x_train = data.numerical_with_indicators(data.train)
    x_val = data.numerical_with_indicators(data.val)
    x_test = data.numerical_with_indicators(data.test)
    best, best_score, best_alpha = None, -np.inf, None
    for alpha in cfg.ridge_alphas:
        model = Ridge(alpha=alpha, solver='lsqr')
        model.fit(x_train, data.train.y, sample_weight=data.train.weights)
        score_train = calculate_metrics(data.train.y, model.predict(x_train), data.train.weights)['weighted_r2']
        score_val = calculate_metrics(data.val.y, model.predict(x_val), data.val.weights)['weighted_r2']
        print(f'[Ridge] alpha={alpha:g} | R² pondéré train={score_train:.6f} | R² pondéré val={score_val:.6f}')
        if score_val > best_score:
            best, best_score, best_alpha = model, score_val, alpha
    if best is None:
        raise RuntimeError('Aucun modèle Ridge sélectionné.')
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_dir / 'ridge.joblib')
    return (best.predict(x_val), best.predict(x_test),
            {'alpha': float(best_alpha), 'best_validation_r2': float(best_score),
             'input_columns': data.ridge_lstm_columns})
