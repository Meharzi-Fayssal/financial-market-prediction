"""LSTM : historique causal par symbol_id ; normalisation apprise sur train uniquement.

Pour un exemple à l'instant t, seule la fenêtre [t-L+1, ..., t] est utilisée.
Les features des instants précédents peuvent provenir du split chronologique
antérieur ; ni cible passée, ni variable future ne sont injectées en entrée.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import Settings
from metrics import calculate_metrics
from preprocessing import PreprocessedData


class SequenceContext:
    """Index global ordonné par symbole et temps, sans fenêtres matérialisées en RAM."""

    def __init__(self, data: PreprocessedData, cfg: Settings):
        self.length = cfg.lstm_sequence_length
        splits = [data.train, data.val, data.test]
        x = np.concatenate([data.numerical_with_indicators(part) for part in splits], axis=0)
        dates = np.concatenate([part.dates for part in splits])
        times = np.concatenate([part.times for part in splits])
        symbols = np.concatenate([part.symbols for part in splits])
        ordering = np.lexsort((times, dates, symbols))
        self.features = np.ascontiguousarray(x[ordering], dtype=np.float32)
        ordered_symbols = symbols[ordering]
        first_of_symbol = np.concatenate(([True], ordered_symbols[1:] != ordered_symbols[:-1]))
        self.group_starts = np.maximum.accumulate(
            np.where(first_of_symbol, np.arange(len(ordering)), 0)
        )
        self.positions = np.empty(len(ordering), dtype=np.int64)
        self.positions[ordering] = np.arange(len(ordering))
        self.offsets = {'train': 0, 'val': len(data.train),
                        'test': len(data.train) + len(data.val)}


class SequenceDataset(Dataset):
    def __init__(self, context: SequenceContext, name: str, labels: np.ndarray, weights: np.ndarray):
        self.context = context
        self.positions = context.positions[context.offsets[name]:context.offsets[name] + len(labels)]
        self.labels = labels.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        pos = int(self.positions[idx])
        first_pos = int(self.context.group_starts[pos])
        start = max(first_pos, pos - self.context.length + 1)
        past = self.context.features[start:pos + 1]
        window = np.zeros((self.context.length, self.context.features.shape[1]), dtype=np.float32)
        window[-len(past):] = past
        return (torch.from_numpy(window),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.tensor(self.weights[idx], dtype=torch.float32))


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, cfg: Settings):
        super().__init__()
        self.network = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(cfg.lstm_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.network(x)
        return self.output(hidden[-1]).squeeze(-1)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    for xb, _, _ in loader:
        batches.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(batches).astype(np.float32, copy=False)


def train_lstm(data: PreprocessedData, cfg: Settings, model_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if cfg.lstm_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('LSTM_DEVICE=cuda demandé, mais aucun GPU CUDA détecté.')
    device = torch.device('cuda' if cfg.lstm_device == 'cuda' or
                          (cfg.lstm_device == 'auto' and torch.cuda.is_available()) else 'cpu')
    print(f'[LSTM] device={device}, fenêtre={cfg.lstm_sequence_length}')

    context = SequenceContext(data, cfg)
    datasets = {part.name: SequenceDataset(context, part.name, part.y, part.weights)
                for part in (data.train, data.val, data.test)}
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {
        key: DataLoader(ds, batch_size=cfg.lstm_batch_size, shuffle=(key == 'train'),
                        num_workers=0, generator=generator if key == 'train' else None)
        for key, ds in datasets.items()
    }
    model = LSTMRegressor(len(data.ridge_lstm_columns), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lstm_learning_rate)
    best_score = -np.inf
    best_epoch = 0
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / 'lstm.pt'

    for epoch in range(1, cfg.lstm_epochs + 1):
        model.train()
        weighted_loss_sum = 0.0
        total_weight = 0.0
        for xb, yb, wb in loaders['train']:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            prediction = model(xb)
            loss = torch.sum(wb * (prediction - yb).square()) / wb.sum().clamp(min=1e-12)
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float(torch.sum(wb * (prediction.detach() - yb).square()).item())
            total_weight += float(wb.sum().item())

        predictions = _predict(model, loaders['val'], device)
        score = calculate_metrics(data.val.y, predictions, data.val.weights)['weighted_r2']
        print(f'[LSTM] epoch {epoch}/{cfg.lstm_epochs} | loss train='
              f'{weighted_loss_sum / max(total_weight, 1e-12):.6f} | R² val={score:.6f}')
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), weights_path)
        elif epoch - best_epoch >= cfg.lstm_patience:
            print(f'[LSTM] arrêt anticipé après {cfg.lstm_patience} époques sans progrès.')
            break

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    val_pred = _predict(model, loaders['val'], device)
    test_pred = _predict(model, loaders['test'], device)
    metadata = {'best_epoch': best_epoch, 'sequence_length': cfg.lstm_sequence_length,
                'n_features': len(data.ridge_lstm_columns), 'hidden_size': cfg.lstm_hidden_size,
                'num_layers': cfg.lstm_num_layers, 'dropout': cfg.lstm_dropout,
                'best_validation_r2': float(best_score), 'device_used': str(device)}
    (model_dir / 'lstm_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return val_pred, test_pred, metadata
