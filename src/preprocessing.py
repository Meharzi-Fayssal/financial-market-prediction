"""Preprocessing partagé pour Ridge, LightGBM et LSTM.

Les neuf features fixées par l’EDA sont exclues des entrées des trois jeux.
Aucune imputation statistique : les valeurs manquantes restent des NaN
pour LightGBM ; pour Ridge et LSTM, elles deviennent 0 *après*
standardisation et leur absence est signalée par un indicateur binaire.
Les statistiques sont calculées exclusivement sur train.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from config import Settings


@dataclass
class SplitData:
    name: str
    x: np.ndarray  # Features retenues ; NaN conservés pour LightGBM
    y: np.ndarray
    weights: np.ndarray
    dates: np.ndarray
    times: np.ndarray
    symbols: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class PreprocessedData:
    train: SplitData
    val: SplitData
    test: SplitData
    columns: list[str]
    excluded_columns: list[str]  # Liste fixée par l’EDA, identique sur les trois jeux
    centers: np.ndarray  # Moyennes de train (uniquement les valeurs observées)
    scales: np.ndarray   # Écarts-types de train (uniquement les valeurs observées)
    all_missing_in_train: np.ndarray

    @property
    def ridge_lstm_columns(self) -> list[str]:
        """Masques fixés sur toutes les features, même sans NaN pendant train.

        Cela garantit les mêmes colonnes de train à test, y compris lorsqu'une
        absence apparaît pour la première fois en validation ou test.
        """
        return self.columns + [f'{name}_missing' for name in self.columns]

    def numerical_with_indicators(self, split: SplitData) -> np.ndarray:
        """Features standardisées et masques binaires, sans fuite de données.

        Pour une feature entièrement manquante sur le train : centre=0,
        échelle=1 (valeurs de secours TECHNIQUES, pas une estimation du marché).
        L'indicateur reste 1 pour chaque NaN, même si sa partie numérique vaut 0.
        """
        mask = np.isnan(split.x)
        scaled = (split.x - self.centers) / self.scales
        scaled[mask] = 0.0
        return np.concatenate([scaled.astype(np.float32, copy=False),
                               mask.astype(np.float32)], axis=1)

    def save_transformers(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'feature_columns': self.columns,
            'excluded_feature_columns': self.excluded_columns,
            'ridge_lstm_columns': self.ridge_lstm_columns,
            'centers': self.centers,
            'scales': self.scales,
            'all_missing_in_train': self.all_missing_in_train,
            'missing_strategy': 'zero_after_standardization_plus_binary_masks',
            'fit_split': 'train',
        }, folder / 'preprocessing.joblib')


def _read_split(name: str, cfg: Settings) -> SplitData:
    path = cfg.data_dir / f'{name}.csv'
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f'{path} est absent ou vide. Les CSV fournis dans le projet initial sont vides ; '
            'utilise les CSV issus de ton prepare.py avant de lancer les modèles.'
        )
    # Même liste d'exclusions prédéfinie pour train, validation et test.
    # Les CSV originaux et prepare.py ne sont pas modifiés.
    cols = cfg.model_features + [cfg.target, cfg.weight, cfg.date, cfg.time, cfg.symbol]
    headers = pd.read_csv(path, nrows=0).columns.to_list()
    missing_cols = sorted(set(cols).difference(headers))
    if missing_cols:
        raise ValueError(f'Colonnes attendues absentes de {path.name} : {missing_cols}')

    limits = {'train': cfg.max_train_rows, 'val': cfg.max_val_rows, 'test': cfg.max_test_rows}
    limit = limits[name]
    if limit:
        warnings.warn(f'{name} : mode DEBUG limité à {limit} lignes ; métriques non représentatives.',
                      stacklevel=2)
    dtype = {c: 'float32' for c in cfg.model_features + [cfg.target, cfg.weight]}
    dtype.update({c: 'int64' for c in [cfg.date, cfg.time, cfg.symbol]})
    df = pd.read_csv(path, usecols=cols, dtype=dtype, nrows=limit or None)
    if df.empty:
        raise ValueError(f'{path} ne contient aucune observation.')

    ranges = {'train': (cfg.train_start, cfg.train_end),
              'val': (cfg.val_start, cfg.val_end),
              'test': (cfg.test_start, cfg.test_end)}
    low, high = ranges[name]
    if not df[cfg.date].between(low, high).all():
        raise ValueError(f'{name} : date_id hors de la plage existante [{low}, {high}].')
    keys = [cfg.date, cfg.time, cfg.symbol]
    if df.duplicated(subset=keys).any():
        raise ValueError(f'{name} : doublons détectés sur {keys}.')

    # On ne retire aucune observation silencieusement, même en cas de cible/poids invalides.
    y = df[cfg.target].to_numpy(dtype=np.float32)
    w = df[cfg.weight].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError(f'{name} : cible {cfg.target} non finie ; contrôler les données.')
    if not np.isfinite(w).all() or (w < 0).any() or w.sum(dtype=np.float64) <= 0:
        raise ValueError(f'{name} : poids invalides (NaN, inf, négatifs ou somme nulle).')

    x = df[cfg.model_features].to_numpy(dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan  # +/-inf ne sont pas des valeurs utilisables
    print(f'[DATA] {name}: {len(y):,} lignes | NaN features : {np.isnan(x).sum():,}')
    return SplitData(name, x, y, w, df[cfg.date].to_numpy(),
                     df[cfg.time].to_numpy(), df[cfg.symbol].to_numpy())


def _train_standardization(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Statistiques des valeurs observées du train, sans estimation des NaN."""
    # Les colonnes vides sont traitées explicitement ; pas de warning 'empty slice'.
    observed = np.isfinite(train_x)
    count = observed.sum(axis=0, dtype=np.int64)
    total = np.nansum(train_x, axis=0, dtype=np.float64)
    centers = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    diffs = np.where(observed, train_x.astype(np.float64) - centers, 0.0)
    variance = np.divide((diffs * diffs).sum(axis=0, dtype=np.float64), count,
                         out=np.zeros_like(total), where=count > 0)
    scales = np.sqrt(variance)
    scales[scales <= 1e-12] = 1.0  # feature constante / entièrement absente
    return centers.astype(np.float32), scales.astype(np.float32), (count == 0)


def load_and_preprocess(cfg: Settings) -> PreprocessedData:
    """Lit les trois CSV EXISTANTS et ajuste les paramètres seulement sur train."""
    train = _read_split('train', cfg)
    val = _read_split('val', cfg)
    test = _read_split('test', cfg)
    centers, scales, all_missing = _train_standardization(train.x)
    print(f'[PREPROCESSING] {len(cfg.excluded_features)} features fixées par l’EDA exclues des 3 jeux : '
          f'{list(cfg.excluded_features)}')
    print(f'[PREPROCESSING] {len(cfg.model_features)} features utilisées, '
          f'{all_missing.sum()} entièrement vides sur train parmi celles-ci (conservées).')
    return PreprocessedData(train, val, test, cfg.model_features,
                            list(cfg.excluded_features), centers, scales, all_missing)
