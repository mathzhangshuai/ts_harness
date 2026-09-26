"""Data/configuration checks only: never execute a training forward or update."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tszoo.data import MemmapWindows
from tszoo.data.prepare import main as prepare
from tszoo.models.chronos2 import Chronos2Backbone, CoreConfig, SplitChronos2
from tszoo.utils.train import parse_args, run

PROJECT = Path(__file__).resolve().parents[1]


class M5TargetTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.store = self.root / "store"
        days = [f"d_{i}" for i in range(1, 1914)]
        with (self.source / "calendar.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["d", "date"])
            writer.writerows(
                zip(
                    days, pd.date_range("2011-01-29", periods=1913).strftime("%Y-%m-%d")
                )
            )
        with (self.source / "sales_train_validation.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", *days])
            writer.writerow(["a", *range(1, 1914)])
            writer.writerow(["b", *range(10001, 11914)])
        # No evaluation file or static metadata exists in this fixture.

    def tearDown(self):
        self.temporary.cleanup()

    def prepare_store(self):
        argv = [
            "prepare",
            "--dataset",
            "m5",
            "--source",
            str(self.source),
            "--output",
            str(self.store),
            "--m5-split",
            "validation",
            "--target-only",
        ]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            prepare()

    def training_args(self, *extra):
        return parse_args(
            [
                "--config",
                str(PROJECT / "config/m5-small-target-train.yaml"),
                "--store",
                str(self.store),
                "--schema",
                str(self.store / "schema.json"),
                "--output",
                str(self.root / "out"),
                "--dry-run",
                *extra,
            ]
        )

    def test_training_store_excludes_covariates_and_holdout(self):
        self.prepare_store()
        manifest = json.loads((self.store / "manifest.json").read_text())
        self.assertEqual(set(manifest["fields"]), {"target"})
        self.assertEqual([x["length"] for x in manifest["series"]], [1913, 1913])
        windows = MemmapWindows(self.store, 512, 28, fields=["target"], end=1913)
        self.assertEqual(len(windows), 2 * 1374)
        first, last = windows[0], windows[len(windows) - 1]
        np.testing.assert_array_equal(first["target"], [np.arange(1, 513)])
        np.testing.assert_array_equal(first["future_target"], [np.arange(513, 541)])
        np.testing.assert_array_equal(last["future_target"], [np.arange(11886, 11914)])
        self.assertEqual(
            set(first), {"item_id", "forecast_start", "target", "future_target"}
        )
        self.assertEqual(str(last["forecast_start"] + 27), "2016-04-24")
        windows._maps.clear()

    def test_cloud_config_selects_small_target_and_unfrozen_backbone(self):
        args = self.training_args()
        self.assertEqual(args.fields, ["target"])
        self.assertFalse(args.freeze_backbone)
        self.assertIn("models--autogluon--chronos-2-small", args.pretrained)
        self.assertEqual((args.context, args.horizon, args.train_end), (512, 28, 1913))
        self.assertEqual(args.device, "cuda")
        override = self.training_args("--steps", "2000", "--batch-size", "4")
        self.assertEqual((override.steps, override.batch_size), (2000, 4))

    def test_dry_run_never_forwards_updates_or_saves(self):
        self.prepare_store()
        backbone = Chronos2Backbone(
            CoreConfig(
                d_model=8,
                d_kv=4,
                d_ff=16,
                num_layers=1,
                num_heads=2,
            )
        )
        with (
            patch.object(Chronos2Backbone, "from_local", return_value=backbone),
            patch.object(
                SplitChronos2, "forward", side_effect=AssertionError("No forward")
            ),
            patch.object(
                SplitChronos2, "save_local", side_effect=AssertionError("No save")
            ),
            patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            report = run(self.training_args())
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["fields"], ["target"])
        self.assertEqual(report["training_windows"], 2748)
        self.assertGreater(report["trainable_parameters"], 0)
        self.assertEqual(report["example_label_shapes"], [[1, 28], [1, 28]])
        self.assertFalse((self.root / "out").exists())
        self.assertEqual(next(backbone.parameters()).device.type, "cpu")

    def test_target_only_rejects_feature_mapping(self):
        with (
            patch(
                "sys.argv",
                [
                    "prepare",
                    "--dataset",
                    "m5",
                    "--source",
                    str(self.source),
                    "--output",
                    str(self.store),
                    "--target-only",
                    "--mapping",
                    "features.json",
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            prepare()
        self.assertFalse(self.store.exists())


if __name__ == "__main__":
    unittest.main()
