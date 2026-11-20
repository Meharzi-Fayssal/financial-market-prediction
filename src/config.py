"""Configuration centrale du projet. Les valeurs se modifient dans .env."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw not in {'true', 'false', '1', '0', 'yes', 'no'}:
        raise ValueError(f'{name} doit valoir true/false (ou 1/0).')
    return raw in {'true', '1', 'yes'}


def _env_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    partition_0_last_date: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    target: str
    weight: str
    date: str
    time: str
    symbol: str
    feature_prefix: str
    n_features: int
    excluded_features: tuple[str, ...]
    seed: int
    max_train_rows: int
    max_val_rows: int
    max_test_rows: int
    models: tuple[str, ...]
    ridge_alphas: tuple[float, ...]
    gbm_n_estimators: int
    gbm_learning_rate: float
    gbm_num_leaves: int
    gbm_min_child_samples: int
    gbm_early_stopping_rounds: int
    lstm_sequence_length: int
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_dropout: float
    lstm_epochs: int
    lstm_batch_size: int
    lstm_learning_rate: float
    lstm_patience: int
    lstm_device: str
    plot_sample_size: int
    save_predictions: bool

    @property
    def features(self) -> list[str]:
        return [f'{self.feature_prefix}{idx:02d}' for idx in range(self.n_features)]

    @property
    def model_features(self) -> list[str]:
        """Features utilisées par les modèles, après exclusion fixe issue de l'EDA."""
        excluded = set(self.excluded_features)
        return [feature for feature in self.features if feature not in excluded]

    @classmethod
    def from_env(cls) -> 'Settings':
        configured_models = tuple(
            v.strip().lower() for v in os.getenv('MODELS', 'ridge,gradient_boosting,lstm').split(',')
            if v.strip()
        )
        alphas = tuple(float(v.strip()) for v in os.getenv('RIDGE_ALPHAS', '0.1,1,10,100').split(','))
        config = cls(
            data_dir=_env_path('DATA_DIR', 'data'),
            output_dir=_env_path('OUTPUT_DIR', 'outputs'),
            partition_0_last_date=_env_int('PARTITION_0_LAST_DATE', 169),
            train_start=_env_int('TRAIN_START_DATE', 0),
            train_end=_env_int('TRAIN_END_DATE', 109),
            val_start=_env_int('VAL_START_DATE', 110),
            val_end=_env_int('VAL_END_DATE', 153),
            test_start=_env_int('TEST_START_DATE', 154),
            test_end=_env_int('TEST_END_DATE', 219),
            target=os.getenv('TARGET_COLUMN', 'responder_6'),
            weight=os.getenv('WEIGHT_COLUMN', 'weight'),
            date=os.getenv('DATE_COLUMN', 'date_id'),
            time=os.getenv('TIME_COLUMN', 'time_id'),
            symbol=os.getenv('SYMBOL_COLUMN', 'symbol_id'),
            feature_prefix=os.getenv('FEATURE_PREFIX', 'feature_'),
            n_features=_env_int('N_FEATURES', 79),
            excluded_features=tuple(v.strip() for v in os.getenv(
                'EXCLUDED_FEATURES', 'feature_00,feature_01,feature_02,feature_03,feature_04,feature_21,feature_26,feature_27,feature_31').split(',') if v.strip()),
            seed=_env_int('RANDOM_SEED', 42),
            max_train_rows=_env_int('MAX_TRAIN_ROWS', 0),
            max_val_rows=_env_int('MAX_VAL_ROWS', 0),
            max_test_rows=_env_int('MAX_TEST_ROWS', 0),
            models=configured_models,
            ridge_alphas=alphas,
            gbm_n_estimators=_env_int('GBM_N_ESTIMATORS', 500),
            gbm_learning_rate=_env_float('GBM_LEARNING_RATE', 0.05),
            gbm_num_leaves=_env_int('GBM_NUM_LEAVES', 31),
            gbm_min_child_samples=_env_int('GBM_MIN_CHILD_SAMPLES', 20),
            gbm_early_stopping_rounds=_env_int('GBM_EARLY_STOPPING_ROUNDS', 30),
            lstm_sequence_length=_env_int('LSTM_SEQUENCE_LENGTH', 20),
            lstm_hidden_size=_env_int('LSTM_HIDDEN_SIZE', 64),
            lstm_num_layers=_env_int('LSTM_NUM_LAYERS', 2),
            lstm_dropout=_env_float('LSTM_DROPOUT', 0.2),
            lstm_epochs=_env_int('LSTM_EPOCHS', 20),
            lstm_batch_size=_env_int('LSTM_BATCH_SIZE', 256),
            lstm_learning_rate=_env_float('LSTM_LEARNING_RATE', 0.001),
            lstm_patience=_env_int('LSTM_PATIENCE', 4),
            lstm_device=os.getenv('LSTM_DEVICE', 'auto').lower(),
            plot_sample_size=_env_int('PLOT_SAMPLE_SIZE', 5000),
            save_predictions=_env_bool('SAVE_TEST_PREDICTIONS', False),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not (self.train_start <= self.train_end < self.val_start <= self.val_end
                < self.test_start <= self.test_end):
            raise ValueError('Plages de dates incorrectes ou chevauchantes dans .env.')
        if not (self.val_end <= self.partition_0_last_date < self.test_end):
            raise ValueError('Le découpage doit conserver train/val en partition 0 et test sur deux partitions.')
        if self.test_start > self.partition_0_last_date:
            raise ValueError('Le test doit commencer dans la partition 0.')
        if self.n_features <= 0 or not self.feature_prefix:
            raise ValueError('N_FEATURES doit être positif et FEATURE_PREFIX non vide.')
        if (len(set(self.excluded_features)) != len(self.excluded_features)
                or not set(self.excluded_features).issubset(self.features)
                or len(self.model_features) == 0):
            raise ValueError('EXCLUDED_FEATURES : doublons, nom inconnu ou aucune feature conservée.')
        if any(v < 0 for v in [self.max_train_rows, self.max_val_rows, self.max_test_rows]):
            raise ValueError('MAX_*_ROWS doit être >= 0 (0 signifie toutes les lignes).')
        allowed = {'ridge', 'gradient_boosting', 'lstm'}
        if not self.models or any(m not in allowed for m in self.models) or len(set(self.models)) != len(self.models):
            raise ValueError('MODELS doit contenir les noms uniques ridge,gradient_boosting,lstm.')
        if not self.ridge_alphas or any(a <= 0 for a in self.ridge_alphas):
            raise ValueError('RIDGE_ALPHAS doit contenir des nombres positifs.')
        if (self.gbm_n_estimators < 1 or self.gbm_num_leaves < 2 or self.gbm_learning_rate <= 0
                or self.gbm_early_stopping_rounds < 1 or self.gbm_min_child_samples < 1):
            raise ValueError('Paramètres Gradient Boosting incorrects.')
        if (self.lstm_sequence_length < 1 or self.lstm_hidden_size < 1 or self.lstm_num_layers < 1
                or self.lstm_epochs < 1 or self.lstm_batch_size < 1 or self.lstm_patience < 1
                or self.lstm_learning_rate <= 0 or not 0 <= self.lstm_dropout < 1):
            raise ValueError('Paramètres LSTM incorrects.')
        if self.lstm_device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('LSTM_DEVICE doit être auto, cpu ou cuda.')
        if self.plot_sample_size < 1:
            raise ValueError('PLOT_SAMPLE_SIZE doit être positif.')
