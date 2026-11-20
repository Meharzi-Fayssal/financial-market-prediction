"""Point d'entrée : mêmes données, mêmes poids, trois modèles et comparaison finale."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd

from compare import compare_models, plot_predictions
from config import Settings
from metrics import calculate_metrics
from preprocessing import load_and_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description='Jane Street : Ridge, Gradient Boosting, LSTM.')
    parser.add_argument('--models', nargs='+', choices=['ridge', 'gradient_boosting', 'lstm'],
                        help='Optionnel : remplacer MODELS dans .env pour cette exécution.')
    args = parser.parse_args()
    cfg = Settings.from_env()
    selected = tuple(args.models) if args.models else cfg.models
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    for subdir in ('models', 'figures', 'predictions'):
        (cfg.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Modèles : {", ".join(selected)}')
    print(f'[INFO] Cible : {cfg.target} | poids : {cfg.weight} | {len(cfg.model_features)} features retenues')
    print('[INFO] NaN préservés pour LightGBM ; Ridge/LSTM : zéro après standardisation + masque.')
    data = load_and_preprocess(cfg)
    data.save_transformers(cfg.output_dir / 'models')
    rows = []
    for name in selected:
        print('\n' + '=' * 24 + f' {name.upper()} ' + '=' * 24)
        if name == 'ridge':
            from ridge import train_ridge
            pred_val, pred_test, meta = train_ridge(data, cfg, cfg.output_dir / 'models')
        elif name == 'gradient_boosting':
            from gradient_boosting import train_gradient_boosting
            pred_val, pred_test, meta = train_gradient_boosting(data, cfg, cfg.output_dir / 'models')
        else:
            from lstm import train_lstm
            pred_val, pred_test, meta = train_lstm(data, cfg, cfg.output_dir / 'models')

        for split, preds in ((data.val, pred_val), (data.test, pred_test)):
            if len(preds) != len(split):
                raise RuntimeError(f'{name}: nombre de prédictions incohérent pour {split.name}.')
            values = calculate_metrics(split.y, preds, split.weights)
            rows.append({'model': name, 'split': split.name, **values})
            print(f"[{name}] {split.name} | R² pondéré={values['weighted_r2']:.6f} "
                  f"| RMSE pondéré={values['weighted_rmse']:.6f} "
                  f"| MAE pondéré={values['weighted_mae']:.6f}")
        # Le test n'est jamais utilisé pour choisir les hyperparamètres / epochs.
        plot_predictions(data.test.y, pred_test, name, cfg)
        if cfg.save_predictions:
            pd.DataFrame({cfg.date: data.test.dates, cfg.time: data.test.times,
                          cfg.symbol: data.test.symbols, cfg.target: data.test.y,
                          cfg.weight: data.test.weights, 'prediction': pred_test}).to_csv(
                              cfg.output_dir / 'predictions' / f'{name}_test.csv', index=False)
        # Importances éventuelles conservées dans un fichier dédié (évite les longs logs).
        importances = meta.pop('feature_importances', None)
        if importances is not None:
            pd.DataFrame({'feature': data.columns, 'importance_split_count': importances}).to_csv(
                cfg.output_dir / 'gradient_boosting_importances.csv', index=False)
        (cfg.output_dir / 'models' / f'{name}_metadata.json').write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    test_ranked = compare_models(rows, cfg.output_dir)
    print('\n=== Comparaison finale (test, score décroissant) ===')
    print(test_ranked.to_string(index=False))
    print(f'\nRésultats enregistrés dans : {cfg.output_dir.resolve()}')


if __name__ == '__main__':
    main()
