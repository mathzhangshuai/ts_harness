import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fixtures import DUMMY, PROJECT

from tszoo.data import write_store
from tszoo.models.chronos2 import SplitChronos2
from tszoo.training.trainer import parse_args, run, training_loader


class TrainingChecks(unittest.TestCase):
    def test_dry_run_does_not_forward_optimize_or_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_store(
                [{"item_id": "a", "start": "2011-01-29", "target": np.arange(1913)}],
                root / "store",
            )
            args = parse_args(
                [
                    "--config",
                    str(PROJECT / "configs/train_small.yaml"),
                    "--store",
                    str(root / "store"),
                    "--pretrained",
                    str(DUMMY),
                    "--output",
                    str(root / "out"),
                    "--dry-run",
                ]
            )
            with (
                patch.object(
                    SplitChronos2, "forward", side_effect=AssertionError("No forward")
                ),
                patch.object(
                    SplitChronos2, "save_local", side_effect=AssertionError("No save")
                ),
                patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = run(args)
            self.assertEqual(report["fields"], ["target"])
            self.assertEqual(report["training_windows"], 1374)
            self.assertEqual(report["epochs"], 1)
            self.assertEqual(report["batches_per_epoch"], 172)
            self.assertEqual(report["total_steps"], 172)
            self.assertGreater(report["trainable_parameters"], 0)
            self.assertFalse((root / "out").exists())

    def test_cli_override_and_holdout_guard(self):
        args = parse_args(["--epochs", "10", "--batch-size", "4", "--dry-run"])
        self.assertEqual((args.epochs, args.batch_size), (10, 4))
        self.assertIn("models--autogluon--chronos-2-small", args.pretrained)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--train-end", "1914"])

    def test_epoch_visits_each_window_once_and_keeps_partial_batch(self):
        args = parse_args(["--epochs", "2", "--batch-size", "4", "--dry-run"])
        dataset = list(range(19))
        loader = training_loader(dataset, args)
        orders = []
        for _ in range(args.epochs):
            batches = list(loader)
            self.assertEqual([len(batch) for batch in batches], [4, 4, 4, 4, 3])
            order = [index for batch in batches for index in batch]
            self.assertEqual(sorted(order), dataset)
            orders.append(order)
        self.assertNotEqual(orders[0], orders[1])
        replay = training_loader(dataset, args)
        for order in orders:
            self.assertEqual(order, [index for batch in replay for index in batch])

    def test_nonpositive_epochs_and_old_steps_are_rejected(self):
        for option in (["--epochs", "0"], ["--epochs", "-1"], ["--steps", "1000"]):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(option)
