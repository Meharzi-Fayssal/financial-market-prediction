"""Gradient Boosting LightGBM : conserve naturellement les NaN des features."""
from __future__ import annotations

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Settings
from metrics import calculate_metrics
from preprocessing import PreprocessedData


def train_gradient_boosting(data: PreprocessedData, cfg: Settings, model_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    model = lgb.LGBMRegressor(
        n_estimators=cfg.gbm_n_estimators,
        learning_rate=cfg.gbm_learning_rate,
        num_leaves=cfg.gbm_num_leaves,
        min_child_samples=cfg.gbm_min_child_samples,
        random_state=cfg.seed,
        verbosity=-1,
        n_jobs=-1,
    )
    # Noms de colonnes conservés jusque dans le modèle et ses prédictions.
    x_train = pd.DataFrame(data.train.x, columns=data.columns)
    x_val = pd.DataFrame(data.val.x, columns=data.columns)
    x_test = pd.DataFrame(data.test.x, columns=data.columns)
    history = {}

    model.fit(
        x_train, data.train.y,
        sample_weight=data.train.weights,

        # Suivre les performances sur train et validation
        eval_set=[
            (x_train, data.train.y),
            (x_val, data.val.y)
        ],
        eval_names=['train', 'validation'],

        # Appliquer les poids sur les deux ensembles
        eval_sample_weight=[
            data.train.weights,
            data.val.weights
        ],

        # Fonction de coût : erreur quadratique moyenne pondérée
        eval_metric='l2',

        callbacks=[
            lgb.early_stopping(
                cfg.gbm_early_stopping_rounds,
                verbose=False
            ),
            lgb.record_evaluation(history)
        ],
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_dir / 'gradient_boosting.txt'))
    best_iteration = int(model.best_iteration_ or cfg.gbm_n_estimators)
    print(f'[LightGBM] meilleure itération : {best_iteration}')

    # Courbes d'apprentissage LightGBM
    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(history['train']['l2']) + 1),
        history['train']['l2'],
        label='Train'
    )

    plt.plot(
        range(1, len(history['validation']['l2']) + 1),
        history['validation']['l2'],
        label='Validation'
    )

    plt.axvline(
        best_iteration,
        color='red',
        linestyle='--',
        label=f'Meilleure itération : {best_iteration}'
    )

    plt.xlabel("Nombre d'itérations")
    plt.ylabel("MSE pondérée (L2)")
    plt.title("Courbes d'apprentissage LightGBM")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        model_dir.parent / 'figures' / 'lightgbm_learning_curve.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"[LightGBM] R² pondéré train = {calculate_metrics(data.train.y, model.predict(x_train, num_iteration=best_iteration), data.train.weights)['weighted_r2']:.6f}")
    return (np.asarray(model.predict(x_val, num_iteration=best_iteration)),
            np.asarray(model.predict(x_test, num_iteration=best_iteration)),
            {'best_iteration': best_iteration, 'feature_importances': model.feature_importances_.tolist(),
             'input_columns': data.columns, 'missing_values': 'NaN conservés (LightGBM)'})
