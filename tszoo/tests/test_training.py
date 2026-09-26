import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from fixtures import DUMMY, PROJECT

from tszoo.data import write_store
from tszoo.models.chronos2 import SplitChronos2, load_model
from tszoo.training.trainer import (
    ForecastTraining,
    make_trainer,
    parse_args,
    run,
    training_loader,
)


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
            self.assertEqual(report["single_device_batches_per_epoch"], 171)
            self.assertEqual(report["single_device_total_steps"], 171)
            self.assertTrue(report["drop_last"])
            self.assertTrue(report["model_settings"]["use_arcsinh"])
            self.assertEqual(
                report["model_settings"]["variate_attention_policy"], "target_aware"
            )
            self.assertGreater(report["trainable_parameters"], 0)
            self.assertFalse((root / "out").exists())

    def test_cli_override_and_holdout_guard(self):
        args = parse_args(["--epochs", "10", "--batch-size", "4", "--dry-run"])
        self.assertEqual((args.epochs, args.batch_size), (10, 4))
        self.assertIn("models--autogluon--chronos-2-small", args.pretrained)
        self.assertEqual(args.devices, "auto")
        self.assertEqual(parse_args(["--devices", "auto", "--dry-run"]).devices, "auto")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--train-end", "1914"])

    def test_epoch_samples_without_replacement_and_drops_partial_batch(self):
        args = parse_args(["--epochs", "2", "--batch-size", "4", "--dry-run"])
        dataset = list(range(19))
        loader = training_loader(dataset, args)
        orders = []
        for _ in range(args.epochs):
            batches = list(loader)
            self.assertEqual([len(batch) for batch in batches], [4, 4, 4, 4])
            order = [index for batch in batches for index in batch]
            self.assertEqual(len(set(order)), 16)
            self.assertTrue(set(order).issubset(dataset))
            orders.append(order)
        self.assertNotEqual(orders[0], orders[1])
        replay = training_loader(dataset, args)
        for order in orders:
            self.assertEqual(order, [index for batch in replay for index in batch])

    def test_distributed_epochs_have_disjoint_samples_without_padding(self):
        args = parse_args(["--batch-size", "4", "--dry-run"])
        loaders = [
            training_loader(list(range(19)), args, world_size=2, rank=rank)
            for rank in range(2)
        ]
        previous = None
        for epoch in range(2):
            for loader in loaders:
                loader.sampler.set_epoch(epoch)
            ranks = [
                [index for batch in loader for index in batch] for loader in loaders
            ]
            self.assertEqual([len(indices) for indices in ranks], [8, 8])
            self.assertEqual(len(set(ranks[0] + ranks[1])), 16)
            self.assertFalse(set(ranks[0]) & set(ranks[1]))
            if previous is not None:
                self.assertNotEqual(previous, ranks)
            previous = ranks

    def test_lightning_settings_and_loss_delegation_without_training(self):
        args = parse_args(["--epochs", "2", "--devices", "2", "--dry-run"])
        with patch("tszoo.training.trainer.L.Trainer") as trainer:
            make_trainer(args)
        settings = trainer.call_args.kwargs
        self.assertEqual(settings["accelerator"], "gpu")
        self.assertEqual(settings["devices"], 2)
        self.assertEqual(settings["max_epochs"], 2)
        self.assertTrue(settings["enable_progress_bar"])
        self.assertTrue(settings["use_distributed_sampler"])
        model = load_model(DUMMY).eval()
        module = ForecastTraining(model, args)
        batch = [
            {
                "target": torch.arange(32).float()[None],
                "future_target": torch.ones(1, args.horizon),
            }
        ]
        with torch.no_grad(), patch.object(module, "log") as log:
            expected = model(batch, args.horizon)["loss"]
            actual = module.training_step(batch, 0)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(log.call_args.kwargs["sync_dist"])
        self.assertTrue(log.call_args.kwargs["prog_bar"])

    def test_interrupted_lightning_run_does_not_save_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_store(
                [{"item_id": "a", "start": "2020-01-01", "target": np.arange(40)}],
                root / "store",
            )
            args = parse_args(
                [
                    "--store",
                    str(root / "store"),
                    "--pretrained",
                    str(DUMMY),
                    "--output",
                    str(root / "out"),
                    "--context",
                    "16",
                    "--horizon",
                    "7",
                    "--train-end",
                    "40",
                ]
            )
            trainer = Mock(interrupted=True)
            with (
                patch("tszoo.training.trainer.make_trainer", return_value=trainer),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                run(args)
            trainer.fit.assert_called_once()
            self.assertFalse((root / "out").exists())

    def test_nonpositive_epochs_and_old_steps_are_rejected(self):
        for option in (["--epochs", "0"], ["--epochs", "-1"], ["--steps", "1000"]):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(option)
