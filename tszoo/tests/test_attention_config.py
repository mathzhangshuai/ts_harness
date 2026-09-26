import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from test_chronos2 import DUMMY, entry, schema
from tszoo.config import load_training_config, training_config
from tszoo.data import MemmapWindows, write_store
from tszoo.models.chronos2 import Chronos2Backbone, CoreConfig, SplitChronos2
from tszoo.utils.train import parse_args


def encoders():
    config = CoreConfig(
        d_model=12, d_kv=4, d_ff=24, num_layers=2, num_heads=2, dropout_rate=0.0
    )
    global_model = Chronos2Backbone(config)
    grouped_model = Chronos2Backbone(replace(config, variate_attention="grouped"))
    grouped_model.load_state_dict(global_model.state_dict(), strict=True)
    return global_model.encoder, grouped_model.encoder


class GroupedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20)
        torch.set_num_threads(1)

    def test_output_and_gradients_match_for_unequal_interleaved_groups(self):
        dense, grouped = encoders()
        groups = torch.tensor([-5, 7, -5, 99, 7, 7])
        observed = torch.rand(6, 5) > 0.3
        observed[[0, 1, 3]] = True
        left = torch.randn(6, 5, 12, requires_grad=True)
        right = left.detach().clone().requires_grad_(True)
        a, b = dense(left, observed, groups), grouped(right, observed, groups)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        weights = torch.randn_like(a)
        (a * weights).sum().backward()
        (b * weights).sum().backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-5)
        for (name, p), (_, q) in zip(
            dense.named_parameters(), grouped.named_parameters()
        ):
            with self.subTest(parameter=name):
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5)

    def test_attention_operator_uses_only_within_group_pairs(self):
        _, grouped = encoders()
        groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        observed, tokens = torch.ones(8, 5, dtype=torch.bool), torch.randn(8, 5, 12)
        calls = []

        def capture(module, args):
            calls.append((tuple(args[0].shape), tuple(args[1].shape)))

        handle = (
            grouped.block[0].layer[1].self_attention.register_forward_pre_hook(capture)
        )
        grouped(tokens, observed, groups)
        handle.remove()
        self.assertEqual(calls, [((20, 2, 12), (20, 1, 1, 2))])
        grouped_pairs = sum(shape[0] * shape[1] ** 2 for shape, _ in calls)
        dense_pairs = 5 * 8**2
        self.assertEqual(dense_pairs // grouped_pairs, 4)
        # Unequal groups are bucketed without padding every group to the maximum size.
        calls.clear()
        handle = (
            grouped.block[0].layer[1].self_attention.register_forward_pre_hook(capture)
        )
        grouped(tokens, observed, torch.tensor([0, 1, 1, 2, 2, 2, 2, 2]))
        handle.remove()
        self.assertEqual(sum(s[0] * s[1] ** 2 for s, _ in calls), 5 * (1 + 4 + 25))

    def test_missing_groups_remain_isolated_and_finite(self):
        _, grouped = encoders()
        grouped.eval()
        groups = torch.tensor([3, 3, 10, 10])
        observed = torch.ones(4, 5, dtype=torch.bool)
        observed[:2] = False
        tokens = torch.randn(4, 5, 12, requires_grad=True)
        before = grouped(tokens, observed, groups)
        modified = tokens.detach().clone()
        modified[2:] *= -1000
        after = grouped(modified, observed, groups)
        torch.testing.assert_close(before[:2], after[:2])
        self.assertTrue(torch.isfinite(before).all())
        before.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_single_group_and_permutations(self):
        dense, grouped = encoders()
        tokens, mask = torch.randn(4, 3, 12), torch.ones(4, 3, dtype=torch.bool)
        groups = torch.zeros(4, dtype=torch.long)
        torch.testing.assert_close(
            dense(tokens, mask, groups), grouped(tokens, mask, groups)
        )
        groups = torch.tensor([2, 9, 9, 2])
        order = torch.tensor([2, 0, 3, 1])
        expected = grouped(tokens, mask, groups)[order]
        actual = grouped(tokens[order], mask[order], groups[order])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_pretrained_mode_and_checkpoint_roundtrip(self):
        if not DUMMY.exists():
            self.skipTest("Reference weights absent")
        a = Chronos2Backbone.from_local(DUMMY).eval()
        b = Chronos2Backbone.from_local(DUMMY, variate_attention="grouped").eval()
        past, groups = torch.randn(4, 32), torch.tensor([0, 0, 1, 1])
        torch.testing.assert_close(
            a(past, groups=groups, prediction_length=7),
            b(past, groups=groups, prediction_length=7),
            rtol=2e-5,
            atol=2e-5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_store([entry(), entry(offset=30)], root / "data", "D")
            windows = MemmapWindows(root / "data", 8, 3, end=16)
            model = SplitChronos2(b, schema())
            batch = [windows[0], windows[len(windows) // 2]]
            loss = model(batch, 3)["loss"]
            loss.backward()
            self.assertGreater(
                model.embeddings["feat_static_cat"][0].weight.grad.abs().sum().item(), 0
            )
            self.assertTrue(torch.isfinite(loss))
            model.save_local(root / "checkpoint")
            restored = SplitChronos2.from_local(root / "checkpoint").eval()
            self.assertEqual(restored.backbone.config.variate_attention, "grouped")
            model.eval()
            torch.testing.assert_close(
                model(batch, 3)["quantile_preds"], restored(batch, 3)["quantile_preds"]
            )
            override = SplitChronos2.from_local(
                root / "checkpoint", variate_attention="global_masked"
            )
            self.assertEqual(
                override.backbone.config.variate_attention, "global_masked"
            )
            config_path = root / "checkpoint/config.json"
            config = json.loads(config_path.read_text())
            config["core"].pop("variate_attention")
            config_path.write_text(json.dumps(config))
            self.assertEqual(
                SplitChronos2.from_local(
                    root / "checkpoint"
                ).backbone.config.variate_attention,
                "global_masked",
            )
            windows._maps.clear()

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            CoreConfig(variate_attention="typo")


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "train.yaml"
        self.values = {
            "data": {
                "store": "data",
                "schema": "data/schema.json",
                "context": 16,
                "horizon": 7,
                "train_end": 60,
            },
            "model": {
                "pretrained": "weights",
                "variate_attention": "grouped",
                "freeze_backbone": True,
            },
            "training": {"output": "run", "steps": 3},
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self, values):
        self.path.write_text(yaml.safe_dump(values), encoding="utf-8")

    def test_yaml_paths_defaults_cli_overrides_and_roundtrip(self):
        self.write(self.values)
        args = parse_args(
            [
                "--config",
                str(self.path),
                "--steps",
                "2",
                "--no-freeze-backbone",
                "--variate-attention",
                "global_masked",
            ]
        )
        self.assertEqual(args.store, str(self.path.parent / "data"))
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.steps, 2)
        self.assertFalse(args.freeze_backbone)
        self.assertEqual(args.variate_attention, "global_masked")
        self.write(training_config(args))
        again = parse_args(["--config", str(self.path)])
        self.assertEqual(training_config(again), training_config(args))

    def test_existing_cli_is_compatible(self):
        args = parse_args(
            [
                "--store",
                "data",
                "--schema",
                "schema.json",
                "--pretrained",
                "weights",
                "--output",
                "run",
                "--context",
                "32",
                "--horizon",
                "7",
                "--train-end",
                "60",
                "--freeze-backbone",
            ]
        )
        self.assertEqual(args.variate_attention, "global_masked")
        self.assertTrue(args.freeze_backbone)
        self.assertEqual(args.store, "data")

    def test_bad_config_is_rejected(self):
        cases = [
            "model:\n  variate_attention: typo\n",
            "model:\n  variate_attention: grouped\n  variate_attention: global_masked\n",
            "model:\n  freeze_backbone: 'false'\n",
            "training:\n  steps: true\n",
            "training:\n  lr: .nan\n",
            "data:\n  fields: [target, target]\n",
            "data:\n  fields: [feat_dynamic_real]\n",
            "model:\n  typo: grouped\n",
            "training: [1, 2]\n",
            "[]\n",
            "1: grouped\n",
            "!!python/object:os.PathLike {}\n",
        ]
        for text in cases:
            with self.subTest(config=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises((TypeError, ValueError, yaml.YAMLError)):
                    load_training_config(self.path)

    def test_missing_required_options_rejected(self):
        self.write({"model": {"variate_attention": "grouped"}})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--config", str(self.path)])


if __name__ == "__main__":
    unittest.main()
