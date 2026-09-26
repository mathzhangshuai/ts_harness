import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from fixtures import DUMMY, PROJECT

from tszoo.data import write_store
from tszoo.utils.evaluate_m5 import Scores, backtest_windows, load_config, run


class M5EvaluationTests(unittest.TestCase):
    def test_scores_global_weighting_and_zero_denominator(self):
        scores = Scores([0.25, 0.5, 0.75])
        scores.update([[2, 4]], [[[0, 2], [1, 3], [2, 4]]])
        result = scores.result()
        self.assertAlmostEqual(result["mae"], 1)
        self.assertAlmostEqual(result["wape"], 1 / 3)
        self.assertAlmostEqual(result["mean_pinball_loss"], 1 / 3)
        zero = Scores([0.5])
        zero.update([[0]], [[[1]]])
        self.assertIsNone(zero.result()["wape"])
        with self.assertRaises(ValueError):
            zero.update([[0]], [[[float("nan")]]])

    def test_boundary_no_future_leakage_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "store"
            write_store(
                [{"item_id": "a", "start": "2020-01-01", "target": np.arange(14)}],
                store,
                "D",
            )
            config = load_config(PROJECT / "config/m5-zero-shot.yaml")
            config.update(
                store=str(store),
                output=str(root / "out"),
                context=7,
                horizon=3,
                origin=10,
                batch_size=1,
                device="cpu",
                quantiles=[0.1, 0.5, 0.9],
                models={"dummy": str(DUMMY)},
            )
            inputs, labels = backtest_windows(config)
            np.testing.assert_array_equal(inputs[0]["target"], [np.arange(3, 10)])
            self.assertNotIn("future_target", inputs[0])
            np.testing.assert_array_equal(labels[0]["future_target"], [[10, 11, 12]])
            del inputs, labels
            result = run(config)
            self.assertEqual(result["models"]["dummy"]["observations"], 3)
            np.testing.assert_array_equal(
                np.load(root / "out/seasonal_naive.npy"), [[3, 4, 5]]
            )
            self.assertTrue((root / "out/metrics.json").exists())
            with self.assertRaises(FileExistsError):
                run(config)
            config.update(output=str(root / "point"), quantiles=[0.5])
            point = run(config, point_only=True)
            self.assertNotIn("seasonal_naive_7", point)
            self.assertFalse((root / "point/seasonal_naive.npy").exists())
            prediction = np.load(root / "point/dummy.npy")[:, 0]
            truth = np.load(root / "point/targets.npy")
            error = np.abs(prediction.astype(np.float64) - truth)
            metrics = point["models"]["dummy"]
            self.assertEqual(
                set(metrics), {"mae", "1-wape", "elapsed_seconds", "checkpoint"}
            )
            self.assertAlmostEqual(metrics["mae"], error.mean())
            self.assertAlmostEqual(metrics["1-wape"], 1 - error.sum() / truth.sum())
            config["origin"] = 12
            with self.assertRaises(ValueError):
                backtest_windows(config)

    def test_yaml_paths_and_invalid_quantiles(self):
        config = load_config(PROJECT / "config/m5-zero-shot.yaml")
        config.pop("model_sources")
        self.assertTrue(Path(config["store"]).is_absolute())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            config["quantiles"] = [0.5, 0.5]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
            config["quantiles"] = [0.5]
            config["models"] = {"targets": "."}
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
