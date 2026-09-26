import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fixtures import DUMMY, PROJECT
from tszoo.data.prepare_m5_covariates import calendar_features, completed_weeks, prepare
from tszoo.utils.evaluate_m5 import backtest_windows, load_config, run


class M5CovariateTests(unittest.TestCase):
    def calendar(self):
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=14).astype(str),
                "d": [f"d_{i}" for i in range(1, 15)],
                "wm_yr_wk": [1] * 7 + [2] * 7,
                "event_name_1": [""] * 10 + ["event"] + [""] * 3,
                "event_name_2": [""] * 14,
                "snap_CA": [0, 1] * 7,
                "snap_TX": [1, 0] * 7,
            }
        )

    def test_calendar_state_alignment_and_week_cutoff(self):
        calendar = self.calendar()
        ca = calendar_features(calendar, "CA")
        tx = calendar_features(calendar, "TX")
        self.assertEqual(ca.shape, (6, 14))
        np.testing.assert_array_equal(ca[4], calendar["snap_CA"])
        np.testing.assert_array_equal(tx[4], calendar["snap_TX"])
        np.testing.assert_allclose(ca[:2, 0], ca[:2, 7], atol=1e-6)
        self.assertEqual(ca[5, 10], 1)
        self.assertEqual(completed_weeks(calendar, 10), {1})

    def test_prepare_covariate_windows_and_frozen_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.calendar().to_csv(root / "calendar.csv", index=False)
            pd.DataFrame(
                [
                    dict(
                        id="a",
                        item_id="item",
                        store_id="CA_1",
                        state_id="CA",
                        cat_id="FOODS",
                        **{f"d_{i}": i for i in range(1, 15)},
                    )
                ]
            ).to_csv(root / "sales_train_evaluation.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "item_id": "item",
                        "store_id": "CA_1",
                        "wm_yr_wk": 1,
                        "sell_price": 2,
                    },
                    {
                        "item_id": "item",
                        "store_id": "CA_1",
                        "wm_yr_wk": 2,
                        "sell_price": 999,
                    },
                ]
            ).to_csv(root / "sell_prices.csv", index=False)
            prepare(root, root / "store", per_stratum=1, origin=10)
            config = load_config(PROJECT / "config/m5-calendar-price.yaml")
            config.update(
                store=str(root / "store"),
                output=str(root / "out"),
                origin=10,
                context=7,
                horizon=3,
                device="cpu",
                quantiles=[0.1, 0.5, 0.9],
                models={"dummy": str(DUMMY)},
            )
            inputs, labels = backtest_windows(config)
            window = inputs[0]
            self.assertEqual(tuple(window["feat_dynamic_real"].shape), (6, 10))
            self.assertEqual(tuple(window["past_feat_dynamic_real"].shape), (1, 7))
            np.testing.assert_array_equal(
                window["past_feat_dynamic_real"][0, :4], [2] * 4
            )
            self.assertTrue(
                np.isnan(window["past_feat_dynamic_real"][0, 4:].numpy()).all()
            )
            self.assertEqual(window["feat_dynamic_real"][5, 7], 1)
            self.assertNotIn("future_target", window)
            del inputs, labels
            result = run(config)
            self.assertEqual(result["models"]["dummy"]["observations"], 3)
            selection = json.loads((root / "store/selection.json").read_text())
            self.assertEqual(selection["baseline_rows"], [0])
            config["origin"] = 9
            with self.assertRaises(ValueError):
                backtest_windows(config)
            prepare(root, root / "known", per_stratum=1, origin=10, future_price=True)
            config.update(
                store=str(root / "known"),
                output=str(root / "out-known"),
                origin=10,
                fields=["target", "feat_dynamic_real"],
                feature_columns={"feat_dynamic_real": [6, 0]},
            )
            inputs, labels = backtest_windows(config)
            window = inputs[0]
            self.assertEqual(tuple(window["feat_dynamic_real"].shape), (2, 10))
            np.testing.assert_array_equal(
                window["feat_dynamic_real"][0, -3:], [999] * 3
            )
            self.assertNotIn("future_target", window)
            self.assertNotIn("past_feat_dynamic_real", window)
            window["feat_dynamic_real"][0, 0] = -123
            self.assertEqual(inputs[0]["feat_dynamic_real"][0, 0], 2)
            del inputs, labels
            self.assertEqual(run(config)["models"]["dummy"]["observations"], 3)
            config["feature_columns"] = {"feat_dynamic_real": [6, 6]}
            with self.assertRaises(ValueError):
                backtest_windows(config)
            config["feature_columns"] = {"feat_dynamic_real": [7]}
            with self.assertRaises(ValueError):
                backtest_windows(config)
            config["feature_columns"] = []
            with self.assertRaises(TypeError):
                backtest_windows(config)


if __name__ == "__main__":
    unittest.main()
