import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fixtures import DUMMY, REFERENCE

from tszoo.models.chronos2 import Chronos2Backbone, SplitChronos2, load_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = load_model(DUMMY).eval()
        self.windows = [
            {
                "item_id": str(i),
                "forecast_start": pd.Period("2020-02-01", freq="D"),
                "target": torch.arange(32).float()[None] + i,
            }
            for i in range(2)
        ]

    def test_checkpoint_roundtrip_and_legacy_target_only_format(self):
        before = self.model.predict(self.windows, 7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            self.model.save_local(root)
            config = json.loads((root / "config.json").read_text())
            config["schema"] = {"target_dim": 1, "feat_static_cat": []}
            config["embedding_dim"] = 8
            (root / "config.json").write_text(json.dumps(config))
            after = load_model(root, "finetuned").predict(self.windows, 7)
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a.forecast_array, b.forecast_array)
            config["fields"] = ["target", "feat_static_cat"]
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                SplitChronos2.from_local(root)

    def test_predictions_ignore_labels_and_other_series(self):
        before = self.model.predict(self.windows, 7)
        modified = [
            dict(w, future_target=torch.full((1, 7), 9999.0)) for w in self.windows
        ]
        modified[1]["target"] = torch.full((1, 32), 12345.0)
        after = self.model.predict(modified, 7)
        np.testing.assert_array_equal(before[0].forecast_array, after[0].forecast_array)
        self.assertTrue(np.isfinite(after[1].forecast_array).all())

    def test_upstream_target_only_numerical_parity(self):
        if not REFERENCE.exists():
            self.skipTest("Optional upstream source absent")
        from upstream_reference import load_upstream

        original = load_upstream(REFERENCE, DUMMY)
        context = torch.cat([w["target"] for w in self.windows])
        with torch.no_grad():
            expected = original(context=context, num_output_patches=1).quantile_preds[
                ..., :7
            ]
            actual = self.model(self.windows, 7)["quantile_preds"][:, 0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_upstream_normalization_and_valid_point_loss_rescaling(self):
        if not REFERENCE.exists():
            self.skipTest("Optional upstream source absent")
        from upstream_reference import load_upstream

        original = load_upstream(REFERENCE, DUMMY)
        context = torch.cat([w["target"] for w in self.windows])
        context[0, 2] = float("nan")
        with torch.no_grad():
            normalized, stats = self.model.backbone.normalize(context)
            expected, expected_stats = original.instance_norm(context)
            torch.testing.assert_close(normalized, expected, equal_nan=True)
            for actual_stat, expected_stat in zip(stats, expected_stats):
                torch.testing.assert_close(actual_stat, expected_stat)
            for horizon in (7, 16, 28):
                for missing in (False, True):
                    labels = torch.arange(2 * horizon).float().reshape(2, horizon)
                    if missing:
                        labels[0] = float("nan")
                        labels[1, ::3] = float("nan")
                    windows = [
                        dict(
                            w,
                            target=context[i : i + 1],
                            future_target=labels[i : i + 1],
                        )
                        for i, w in enumerate(self.windows)
                    ]
                    expected_output = original(
                        context=context,
                        future_target=labels,
                        num_output_patches=(horizon + 15) // 16,
                    )
                    actual_output = self.model(windows, horizon)
                    padded_points = (
                        labels.shape[0] * expected_output.quantile_preds.shape[-1]
                    )
                    expected_loss = (
                        expected_output.loss
                        * padded_points
                        / torch.isfinite(labels).sum()
                    )
                    torch.testing.assert_close(
                        actual_output["loss"],
                        expected_loss,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                    torch.testing.assert_close(
                        actual_output["quantile_preds"][:, 0],
                        expected_output.quantile_preds[..., :horizon],
                        rtol=1e-5,
                        atol=1e-5,
                    )

    def test_padding_and_missing_only_series_do_not_dilute_loss(self):
        labels = torch.arange(28).float()[None]
        labels[:, ::5] = float("nan")
        window = dict(self.windows[0], future_target=labels)
        padded = dict(
            window,
            future_target=torch.cat([labels, torch.full((1, 4), float("nan"))], -1),
        )
        missing = dict(self.windows[1], future_target=torch.full((1, 28), float("nan")))
        with torch.no_grad():
            expected = self.model([window], 28)["loss"]
            torch.testing.assert_close(self.model([padded], 32)["loss"], expected)
            torch.testing.assert_close(
                self.model([window, missing], 28)["loss"], expected
            )
            with self.assertRaisesRegex(ValueError, "No observed target labels"):
                self.model([missing], 28)

    def test_attention_policy_and_grouped_engine(self):
        groups = torch.tensor([0, 0, 0, 1])
        targets = torch.tensor([True, False, False, True])
        tokens = torch.randn(4, 5, self.model.backbone.config.d_model)
        observed = torch.ones(4, 5, dtype=torch.bool)
        observed[1, 0] = False
        with torch.no_grad():
            for policy in ("bidirectional", "target_aware"):
                outputs = []
                for engine in ("global_masked", "grouped"):
                    backbone = Chronos2Backbone(
                        replace(
                            self.model.backbone.config,
                            variate_attention=engine,
                            variate_attention_policy=policy,
                        )
                    ).eval()
                    backbone.load_state_dict(self.model.backbone.state_dict())
                    before = backbone.encoder(tokens, observed, groups, targets)
                    changed = tokens.clone()
                    changed[0] += torch.randn_like(changed[0]) * 100
                    after = backbone.encoder(changed, observed, groups, targets)
                    torch.testing.assert_close(before[3], after[3], rtol=0, atol=0)
                    if policy == "target_aware":
                        torch.testing.assert_close(
                            before[1:3], after[1:3], rtol=0, atol=0
                        )
                    else:
                        self.assertFalse(torch.equal(before[1:3], after[1:3]))
                    outputs.append(before)
                torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)

    def test_attention_settings_survive_checkpoint_and_grouping_is_configurable(self):
        attention = {
            "variate_attention": "global_masked",
            "variate_attention_policy": "target_aware",
            "variate_grouping": "batch",
        }
        model = load_model(DUMMY, attention=attention).eval()
        before = model.predict(self.windows, 7)
        changed = [dict(w) for w in self.windows]
        changed[1]["target"] = torch.randn_like(changed[1]["target"]) * 100
        after = model.predict(changed, 7)
        self.assertFalse(
            np.array_equal(before[0].forecast_array, after[0].forecast_array)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            model.save_local(root)
            restored = load_model(root, "finetuned", attention=attention)
            for name, value in attention.items():
                self.assertEqual(getattr(restored.backbone.config, name), value)
            np.testing.assert_array_equal(
                before[0].forecast_array,
                restored.predict(self.windows, 7)[0].forecast_array,
            )
            with self.assertRaises(ValueError):
                load_model(
                    root,
                    "finetuned",
                    attention={"variate_attention_policy": "bidirectional"},
                )
