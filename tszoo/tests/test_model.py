import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fixtures import DUMMY, REFERENCE

from tszoo.models.chronos2 import SplitChronos2, load_model


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
