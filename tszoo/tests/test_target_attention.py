import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from fixtures import PROJECT
from test_attention_config import encoders
from test_chronos2 import core, entry, schema
from tszoo.config import load_training_config
from tszoo.data import MemmapWindows, write_store
from tszoo.models.chronos2 import Chronos2Backbone, CoreConfig, SplitChronos2
from tszoo.utils.train import parse_args


class TargetAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(70)
        torch.set_num_threads(1)

    def test_target_cannot_influence_covariate_states_or_gradients(self):
        for encoder in encoders():
            encoder.config.variate_attention_policy = "target_aware"
            groups = torch.tensor([2, 2, 2, 9, 9])
            targets = torch.tensor([True, False, True, False, True])
            observed = torch.ones(5, 4, dtype=torch.bool)
            # No covariate key is visible in the second group at the first token.
            observed[3, 0] = False
            values = torch.randn(5, 4, 12, requires_grad=True)
            output = encoder(values, observed, groups, targets)
            changed = values.detach().clone()
            changed[targets] = torch.randn_like(changed[targets]) * 100
            actual = encoder(changed, observed, groups, targets)
            torch.testing.assert_close(
                output[~targets], actual[~targets], rtol=0, atol=0
            )
            loss = (output[~targets] * torch.randn_like(output[~targets])).sum()
            gradient = torch.autograd.grad(loss, values, retain_graph=True)[0]
            self.assertEqual(gradient[targets].abs().sum().item(), 0.0)
            loss = (output[targets] * torch.randn_like(output[targets])).sum()
            gradient = torch.autograd.grad(loss, values)[0]
            self.assertGreater(gradient[~targets].abs().sum().item(), 0.0)
            self.assertTrue(torch.isfinite(output).all())

    def test_layouts_match_with_roles_missing_keys_and_ragged_groups(self):
        dense, grouped = encoders()
        for encoder in (dense, grouped):
            encoder.config.variate_attention_policy = "target_aware"
        groups = torch.tensor([9, 3, 9, 4, 3, 9])
        targets = torch.tensor([True, True, False, True, False, False])
        observed = torch.rand(6, 4) > 0.5
        observed[:, 1] = False
        left = torch.randn(6, 4, 12, requires_grad=True)
        right = left.detach().clone().requires_grad_(True)
        a, b = (
            dense(left, observed, groups, targets),
            grouped(right, observed, groups, targets),
        )
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        weight = torch.randn_like(a)
        (a * weight).sum().backward()
        (b * weight).sum().backward()
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-5)
        for p, q in zip(dense.parameters(), grouped.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5)

    def test_grouped_mask_direction_and_no_cross_group_axis(self):
        _, encoder = encoders()
        encoder.config.variate_attention_policy = "target_aware"
        targets = torch.tensor([True, False, False, True, False, False])
        calls = []

        def capture(module, args):
            calls.append((args[0].shape, args[1].clone()))

        handle = (
            encoder.block[0].layer[1].self_attention.register_forward_pre_hook(capture)
        )
        encoder(
            torch.randn(6, 4, 12),
            torch.ones(6, 4, dtype=torch.bool),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            targets,
        )
        handle.remove()
        shape, mask = calls[0]
        self.assertEqual(tuple(shape), (8, 3, 12))
        self.assertEqual(tuple(mask.shape), (8, 1, 3, 3))
        expected = torch.tensor(
            [[True, True, True], [False, True, True], [False, True, True]]
        )
        torch.testing.assert_close(mask, expected[None, None].expand_as(mask))

    def test_explicit_roles_and_policy_validation(self):
        with self.assertRaises(ValueError):
            CoreConfig(variate_attention_policy="typo")
        model = Chronos2Backbone(
            replace(core().config, variate_attention_policy="target_aware")
        )
        with self.assertRaises(ValueError):
            model(torch.randn(2, 8), prediction_length=3)
        with self.assertRaises(ValueError):
            model(
                torch.randn(2, 8), prediction_length=3, target_mask=torch.tensor([1, 0])
            )
        for roles in (torch.tensor([True, True]), torch.tensor([False, False])):
            self.assertTrue(
                torch.isfinite(
                    model(torch.randn(2, 8), prediction_length=3, target_mask=roles)
                ).all()
            )

    def test_seven_field_role_assignment_training_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_store([entry(), entry(offset=30)], root / "data", "D")
            data = MemmapWindows(root / "data", 8, 3, end=16)
            backbone = Chronos2Backbone(
                replace(
                    core().config,
                    variate_attention="grouped",
                    variate_attention_policy="target_aware",
                )
            )
            model = SplitChronos2(backbone, schema())
            roles = []

            def capture(module, args):
                roles.append(args[3].clone())

            handle = backbone.encoder.register_forward_pre_hook(capture)
            batch = [data[0], data[len(data) // 2]]
            result = model(batch, 3)
            result["loss"].backward()
            handle.remove()
            self.assertEqual(
                roles[0].tolist(), [True, False, False, False, False, False, False] * 2
            )
            self.assertGreater(
                model.embeddings["feat_static_cat"][0].weight.grad.abs().sum().item(),
                0.0,
            )
            model.save_local(root / "checkpoint")
            restored = SplitChronos2.from_local(root / "checkpoint").eval()
            self.assertEqual(
                restored.backbone.config.variate_attention_policy, "target_aware"
            )
            torch.testing.assert_close(
                model.eval()(batch, 3)["quantile_preds"],
                restored(batch, 3)["quantile_preds"],
            )
            changed = SplitChronos2.from_local(
                root / "checkpoint", variate_attention_policy="bidirectional"
            )
            self.assertEqual(
                changed.backbone.config.variate_attention_policy, "bidirectional"
            )
            path = root / "checkpoint/config.json"
            config = json.loads(path.read_text())
            config["core"].pop("variate_attention_policy")
            path.write_text(json.dumps(config))
            self.assertEqual(
                SplitChronos2.from_local(
                    root / "checkpoint"
                ).backbone.config.variate_attention_policy,
                "bidirectional",
            )
            data._maps.clear()

    def test_yaml_cli_policy_override_and_rejection(self):
        args = parse_args(["--config", str(PROJECT / "config/chronos2.yaml")])
        self.assertEqual(args.variate_attention_policy, "target_aware")
        args = parse_args(
            [
                "--config",
                str(PROJECT / "config/chronos2.yaml"),
                "--variate-attention-policy",
                "bidirectional",
            ]
        )
        self.assertEqual(args.variate_attention_policy, "bidirectional")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("model:\n  variate_attention_policy: wrong\n")
            with self.assertRaises(ValueError):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
