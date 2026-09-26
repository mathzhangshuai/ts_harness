import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from fixtures import DUMMY, PROJECT

from tszoo.config import load_training_config
from tszoo.data import MemmapWindows, write_store
from tszoo.data.features import FIELDS, selection
from tszoo.models.chronos2 import load_model


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = selection(
            {
                "target": ["sales"],
                "feat_static_cat": ["store_id", "item_id"],
                "feat_static_real": ["capacity"],
                "feat_dynamic_cat": ["event_name_1"],
                "feat_dynamic_real": ["snap_CA"],
                "past_feat_dynamic_cat": ["wday"],
                "past_feat_dynamic_real": ["sell_price"],
            }
        )
        pd.DataFrame(
            {
                "d": [f"d_{i}" for i in range(1, 41)],
                "date": pd.date_range("2020-01-01", periods=40),
                "wm_yr_wk": np.arange(40),
                "snap_CA": np.arange(40) % 2,
                "wday": np.arange(40) % 7 + 1,
                "event_name_1": ["event" if i % 2 else "" for i in range(40)],
            }
        ).to_csv(self.root / "calendar.csv", index=False)
        pd.DataFrame(
            [{"id": "a_validation", "item_id": "item", "store_id": "store"}]
        ).to_csv(self.root / "sales_train_validation.csv", index=False)
        pd.DataFrame([{"id": "a", "capacity": 12.5}]).to_csv(
            self.root / "static_features.csv", index=False
        )
        pd.DataFrame(
            {
                "store_id": ["store"] * 40,
                "item_id": ["item"] * 40,
                "wm_yr_wk": np.arange(40),
                "sell_price": np.arange(40) + 1,
            }
        ).to_csv(self.root / "sell_prices.csv", index=False)
        write_store(
            [
                {
                    "item_id": "a_evaluation",
                    "start": "2020-01-01",
                    "target": np.arange(40),
                }
            ],
            self.root / "store",
        )

    def dataset(self, **kwargs):
        data = MemmapWindows(
            self.root / "store",
            16,
            7,
            features=self.features,
            source=self.root,
            **kwargs,
        )
        self.addCleanup(data.close)
        return data

    def test_seven_fields_names_order_and_time_boundaries(self):
        data = self.dataset(end=23)
        window = data[0]
        self.assertEqual(
            set(window) - {"item_id", "forecast_start", "future_target"}, set(FIELDS)
        )
        self.assertEqual(data.features["feat_static_cat"], ["store_id", "item_id"])
        self.assertEqual(window["feat_static_real"].shape, (1,))
        self.assertEqual(window["feat_dynamic_cat"].shape, (1, 23))
        self.assertEqual(window["past_feat_dynamic_cat"].shape, (1, 16))
        np.testing.assert_array_equal(
            window["past_feat_dynamic_real"], [np.arange(1, 17)]
        )
        np.testing.assert_array_equal(window["future_target"], [np.arange(16, 23)])
        predicted = self.dataset(end=16, mode="predict")[0]
        self.assertNotIn("future_target", predicted)
        for field in FIELDS:
            torch.testing.assert_close(window[field], predicted[field])

    def test_all_fields_forward_checkpoint_and_covariate_effect(self):
        data = self.dataset(end=16, mode="predict")
        window = data[0]
        model = load_model(DUMMY, feature_schema=data.feature_schema).eval()
        before = model.predict([window], 7)[0].forecast_array
        modified = dict(window, feat_dynamic_real=1 - window["feat_dynamic_real"])
        after = model.predict([modified], 7)[0].forecast_array
        self.assertFalse(np.array_equal(before, after))
        with torch.no_grad():
            labeled = dict(window, future_target=torch.ones(1, 7))
            self.assertTrue(torch.isfinite(model([labeled], 7)["loss"]))
        checkpoint = self.root / "checkpoint"
        model.save_local(checkpoint)
        restored = load_model(checkpoint, "finetuned", data.feature_schema)
        np.testing.assert_array_equal(
            before, restored.predict([window], 7)[0].forecast_array
        )
        mismatch = {
            **data.feature_schema,
            "features": {**data.features, "feat_static_cat": ["item_id", "store_id"]},
        }
        with self.assertRaises(ValueError):
            load_model(checkpoint, "finetuned", mismatch)

    def test_past_price_cannot_observe_future(self):
        before = self.dataset(end=16, mode="predict")[0]
        prices = pd.read_csv(self.root / "sell_prices.csv")
        prices.loc[16:, "sell_price"] = 999999
        prices.to_csv(self.root / "sell_prices.csv", index=False)
        after = self.dataset(end=16, mode="predict")[0]
        for field in FIELDS:
            torch.testing.assert_close(before[field], after[field])

    def test_yaml_rejects_indices_unknown_and_duplicate_columns(self):
        for invalid in (
            {"feat_dynamic_real": [0]},
            {"feat_dynamic_real": ["0"]},
            {"feat_dynamic_real": ["typo"]},
            {"feat_dynamic_cat": ["wday"], "past_feat_dynamic_cat": ["wday"]},
        ):
            with self.assertRaises(ValueError):
                selection({"target": ["sales"], **invalid})
        raw = yaml.safe_load((PROJECT / "configs/train_small.yaml").read_text())
        raw["data"]["features"] = self.features
        raw["data"]["source"] = "."
        path = self.root / "train.yaml"
        path.write_text(yaml.safe_dump(raw))
        config = load_training_config(path)
        self.assertEqual(config["features"], self.features)
        self.assertEqual(config["source"], str(self.root))
