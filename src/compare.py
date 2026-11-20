"""Comparaison des modèles sur mêmes jeux de validation/test et mêmes métriques."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Settings


def compare_models(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    """Sauvegarde le tableau et le graphique des scores ; renvoie le classement test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    needed = {'model', 'split', 'weighted_r2', 'weighted_rmse', 'weighted_mae'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Colonnes manquantes pour la comparaison : {needed - set(df.columns)}')
    if df.duplicated(subset=['model', 'split']).any():
        raise ValueError('Métriques dupliquées pour un couple (model, split).')
    df.to_csv(output_dir / 'model_metrics.csv', index=False)
    test = df.loc[df['split'] == 'test'].sort_values('weighted_r2', ascending=False).copy()
    test.to_csv(output_dir / 'comparison_test.csv', index=False)
    if len(test):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(test['model'], test['weighted_r2'])
        ax.axhline(0, linewidth=0.8, linestyle='--', color='gray')
        ax.set_ylabel('R² pondéré à moyenne nulle (test)')
        ax.set_title('Jane Street — comparaison des modèles')
        fig.tight_layout()
        fig.savefig(output_dir / 'figures' / 'comparison_r2.png', dpi=150)
        plt.close(fig)
    return test


def plot_predictions(y_true: np.ndarray, predictions: np.ndarray, name: str,
                     cfg: Settings) -> None:
    """Figure indicative sur échantillon ; métriques calculées sur 100 % du split."""
    n = len(y_true)
    count = min(n, cfg.plot_sample_size)
    sample = np.random.default_rng(cfg.seed).choice(n, size=count, replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true[sample], predictions[sample], alpha=.35, s=7)
    ax.set_xlabel(f'{cfg.target} observé')
    ax.set_ylabel(f'Prédiction {cfg.target}')
    ax.set_title(f'{name} — test ({count:,} observations)')
    fig.tight_layout()
    folder = cfg.output_dir / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}_predictions.png', dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Regénérer la comparaison depuis les métriques enregistrées.')
    parser.add_argument('--metrics', type=Path, default=None,
                        help='CSV des métriques ; par défaut outputs/model_metrics.csv')
    args = parser.parse_args()
    cfg = Settings.from_env()
    path = args.metrics or cfg.output_dir / 'model_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} absent ; exécutez python main.py avant compare.py.')
    (cfg.output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    ranked = compare_models(pd.read_csv(path).to_dict('records'), cfg.output_dir)
    print(ranked.to_string(index=False))


if __name__ == '__main__':
    main()
