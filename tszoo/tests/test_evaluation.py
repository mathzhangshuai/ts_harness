import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fixtures import DUMMY

from tszoo.data import write_store
from tszoo.evaluation import evaluate, wrmsse
from tszoo.evaluation.metrics import Scores
from tszoo.evaluation.predict import backtest_windows, run
from tszoo.models.chronos2 import load_model


class EvaluationTests(unittest.TestCase):
    def test_retired_horizon_is_rejected_before_loading_resources(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(evaluate, "ensure_resources") as resources,
        ):
            with self.assertRaisesRegex(ValueError, "horizon=7"):
                evaluate.run(
                    {
                        "output": str(Path(temporary) / "out"),
                        "origin": 1913,
                        "horizon": 28,
                    }
                )
            resources.assert_not_called()

    def test_seven_day_evaluation_reports_correct_interval_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "out"
            config = {
                "output": str(root),
                "origin": 1913,
                "horizon": 7,
                "context": 49,
                "models": {"small": "unused"},
                "dataset": {"path": "unused"},
            }

            def forecast(settings):
                self.assertEqual((settings["context"], settings["horizon"]), (49, 7))
                Path(settings["output"]).mkdir(parents=True)
                return {"models": {"small": {"1-wape": 0.5, "mae": 2.0}}}

            with (
                patch.object(evaluate, "ensure_resources"),
                patch.object(evaluate, "predict", side_effect=forecast),
                patch.object(
                    evaluate,
                    "score_wrmsse",
                    return_value={"models": {"small": {"wrmsse": 1.2}}},
                ),
            ):
                result = evaluate.run(config)
            self.assertEqual(
                result["small"], {"1-wape": 0.5, "mae": 2.0, "wrmsse": 1.2}
            )
            report = (root / "README.md").read_text()
            self.assertIn("d_1914-d_1920", report)
            self.assertIn("custom backtest", report)

    def test_wrmsse_accepts_seven_day_forecast_metadata(self):
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics.json").write_text(
                json.dumps(
                    {"series": 30490, "quantiles": [0.5], "postprocessing": "none"}
                )
            )
            (root / "resolved_config.yaml").write_text(
                yaml.safe_dump({"origin": 1913, "horizon": 7, "quantiles": [0.5]})
            )
            with (
                patch.object(
                    wrmsse.pd, "read_csv", side_effect=RuntimeError("metadata accepted")
                ) as read,
                self.assertRaisesRegex(RuntimeError, "metadata accepted"),
            ):
                wrmsse.run(root, root, root / "out")
            columns = read.call_args.kwargs["usecols"]
            self.assertIn("d_1920", columns)
            self.assertNotIn("d_1921", columns)

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
            attention = {
                "variate_attention_policy": "target_aware",
                "variate_grouping": "series",
            }
            load_model(DUMMY, attention=attention).save_local(root / "saved")
            config = {
                "store": str(root / "store"),
                "output": str(root / "out"),
                "origin": 10,
                "horizon": 3,
                "context": 7,
                "device": "cpu",
                "batch_size": 2,
                "attention": attention,
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
            with self.assertRaisesRegex(ValueError, "attention settings differ"):
                run(
                    dict(
                        config,
                        output=str(root / "mismatch"),
                        attention={"variate_attention_policy": "bidirectional"},
                    )
                )
            with self.assertRaises(ValueError):
                backtest_windows(dict(config, origin=12))
