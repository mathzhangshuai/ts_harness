import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fixtures import DUMMY

from tszoo.data import write_store
from tszoo.evaluation.metrics import Scores
from tszoo.evaluation.predict import backtest_windows, run
from tszoo.models.chronos2 import load_model


class EvaluationTests(unittest.TestCase):
    def test_global_metrics_and_undefined_denominator(self):
        scores = Scores()
        scores.update([[2, 4]], [[1, 3]])
        scores.update([[10, 0]], [[8, 1]])
        self.assertEqual(scores.result()["mae"], 5 / 4)
        self.assertEqual(scores.result()["1-wape"], 1 - 5 / 16)
        zero = Scores()
        zero.update([[0]], [[1]])
        self.assertIsNone(zero.result()["1-wape"])
        with self.assertRaises(ValueError):
            zero.update([[0]], [[float("nan")]])

    def test_both_checkpoint_formats_share_prediction_and_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_store(
                [
                    {"item_id": str(i), "start": "2020-01-01", "target": np.arange(14)}
                    for i in range(3)
                ],
                root / "store",
            )
            load_model(DUMMY).save_local(root / "saved")
            config = {
                "store": str(root / "store"),
                "output": str(root / "out"),
                "origin": 10,
                "horizon": 3,
                "context": 7,
                "device": "cpu",
                "batch_size": 2,
                "models": {"original": str(DUMMY), "saved": str(root / "saved")},
                "model_formats": {"original": "pretrained", "saved": "finetuned"},
            }
            result = run(config)
            original = np.load(root / "out/original.npy")
            saved = np.load(root / "out/saved.npy")
            np.testing.assert_array_equal(original, saved)
            truth = np.load(root / "out/targets.npy")
            np.testing.assert_array_equal(truth, [[10, 11, 12]] * 3)
            self.assertEqual(original.shape, (3, 1, 3))
            self.assertEqual(result["series"], 3)
            error = np.abs(original[:, 0].astype(np.float64) - truth)
            for metrics in result["models"].values():
                self.assertAlmostEqual(metrics["mae"], error.mean())
                self.assertAlmostEqual(metrics["1-wape"], 1 - error.sum() / truth.sum())
            self.assertFalse((root / "out/seasonal_naive.npy").exists())
            metadata = json.loads((root / "out/metrics.json").read_text())
            self.assertEqual(metadata["quantiles"], [0.5])
            with self.assertRaises(FileExistsError):
                run(config)
            with self.assertRaises(ValueError):
                backtest_windows(dict(config, origin=12))
