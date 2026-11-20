"""Tests unitaires autonomes, sans les données réelles Jane Street."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings
from lstm import SequenceContext, SequenceDataset
from metrics import calculate_metrics
from preprocessing import load_and_preprocess


class TestPipeline(unittest.TestCase):
    def test_weighted_r2_baseline_zero(self):
        y = np.array([1.0, -2.0])
        w = np.array([1.0, 3.0])
        self.assertAlmostEqual(calculate_metrics(y, np.zeros_like(y), w)['weighted_r2'], 0.0)
        self.assertAlmostEqual(calculate_metrics(y, y, w)['weighted_r2'], 1.0)

    def _files(self, path: Path, cfg: Settings):
        for name, dates in [('train', [0, 1, 2, 3]),
                            ('val', [110, 111]), ('test', [154, 155])]:
            df = pd.DataFrame({c: np.ones(len(dates), dtype=np.float32) for c in cfg.features})
            df[cfg.date] = dates
            df[cfg.time] = 0
            df[cfg.symbol] = 8
            df[cfg.target] = np.arange(1, len(dates) + 1)
            df[cfg.weight] = 1.0
            # Feature exclue : même si elle devient renseignée en validation/test,
            # elle ne doit jamais se retrouver dans les entrées des modèles.
            df['feature_00'] = np.nan if name == 'train' else 5.0
            # Feature retenue, partiellement manquante (indicateur obligatoire).
            if name == 'train':
                df['feature_05'] = [1.0, np.nan, 3.0, 4.0]
            elif name == 'val':
                df['feature_05'] = [np.nan, 100.0]
            else:
                df['feature_05'] = [200.0, np.nan]
            # Feature retenue : premier NaN apparaît seulement en validation.
            if name == 'val':
                df.loc[0, 'feature_06'] = np.nan
            df.to_csv(path / f'{name}.csv', index=False)

    def test_eda_columns_excluded_and_partial_nan_masks_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = replace(Settings.from_env(), data_dir=root, output_dir=root / 'outputs',
                          max_train_rows=0, max_val_rows=0, max_test_rows=0,
                          lstm_sequence_length=3)
            self._files(root, cfg)
            data = load_and_preprocess(cfg)
            excluded = {'feature_00', 'feature_01', 'feature_02', 'feature_03',
                        'feature_04', 'feature_21', 'feature_26', 'feature_27',
                        'feature_31'}
            self.assertEqual(set(data.excluded_columns), excluded)
            self.assertEqual(len(data.columns), 70)
            self.assertTrue(excluded.isdisjoint(data.columns))
            self.assertEqual(len(data.ridge_lstm_columns), 140)
            self.assertEqual(data.columns[0], 'feature_05')
            self.assertEqual(data.columns[1], 'feature_06')
            self.assertTrue(np.isnan(data.train.x[1, 0]))
            self.assertTrue(np.isnan(data.val.x[0, 0]))  # NaN natif conservé pour LightGBM
            self.assertTrue(np.isnan(data.val.x[0, 1]))  # manque uniquement en val
            self.assertEqual(data.train.x.shape, (4, 70))
            self.assertEqual(data.val.x.shape, (2, 70))
            self.assertEqual(data.test.x.shape, (2, 70))

            tr = data.numerical_with_indicators(data.train)
            val = data.numerical_with_indicators(data.val)
            tst = data.numerical_with_indicators(data.test)
            self.assertEqual(tr.shape, (4, 140))
            self.assertEqual(val.shape, (2, 140))
            self.assertEqual(tst.shape, (2, 140))
            self.assertEqual(tr[1, 0], 0)     # 0 technique après standardisation
            self.assertEqual(tr[1, 70], 1)    # feature_05 manquante en train
            self.assertEqual(tr[0, 70], 0)    # feature_05 observée en train
            self.assertEqual(val[0, 70], 1)   # feature_05 manquante en validation
            self.assertEqual(tst[1, 70], 1)   # feature_05 manquante en test
            self.assertEqual(val[0, 71], 1)   # feature_06 manque uniquement en val
            self.assertAlmostEqual(data.centers[0], 8/3)
            self.assertAlmostEqual(float(val[1, 0]), float((100 - 8/3)/data.scales[0]), places=4)
            self.assertTrue(np.isfinite(tr).all())
            self.assertTrue(np.isfinite(val).all())
            self.assertTrue(np.isfinite(tst).all())

            context = SequenceContext(data, cfg)
            val_set = SequenceDataset(context, 'val', data.val.y, data.val.weights)
            first, _, _ = val_set[0]
            np.testing.assert_allclose(first.numpy()[-1], val[0], atol=1e-6)
            np.testing.assert_allclose(first.numpy()[-2], tr[-1], atol=1e-6)

            data.save_transformers(root / 'outputs' / 'models')
            path = root / 'outputs' / 'models' / 'preprocessing.joblib'
            self.assertTrue(path.is_file())
            import joblib
            meta = joblib.load(path)
            self.assertEqual(set(meta['excluded_feature_columns']), excluded)
            self.assertEqual(len(meta['feature_columns']), 70)

    def test_invalid_date_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = replace(Settings.from_env(), data_dir=root)
            frame = pd.DataFrame({feature: [0.0] for feature in cfg.features})
            frame[cfg.target] = 1.0
            frame[cfg.weight] = 1.0
            frame[cfg.date] = 999
            frame[cfg.time] = 0
            frame[cfg.symbol] = 0
            frame.to_csv(root / 'train.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'date_id'):
                load_and_preprocess(cfg)

    def test_missing_real_csv_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = replace(Settings.from_env(), data_dir=Path(folder))
            with self.assertRaisesRegex(FileNotFoundError, 'CSV'):
                load_and_preprocess(cfg)


if __name__ == '__main__':
    unittest.main()
