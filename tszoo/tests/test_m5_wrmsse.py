import unittest

import numpy as np
import pandas as pd
from tszoo.utils.score_m5_wrmsse import rmsse_scale, score_hierarchy, trailing_revenue


class M5WRMSSETests(unittest.TestCase):
    def test_scale_excludes_launch_jump_but_retains_later_zeros(self):
        np.testing.assert_allclose(
            rmsse_scale([[0, 0, 2, 0, 4], [1, 2, 3, 4, 5]]), [10, 1]
        )
        for history in ([[0, 0, 0]], [[0, 0, 2]], [[2, 2, 2]]):
            with self.assertRaises(ValueError):
                rmsse_scale(history)

    def test_hierarchy_weights_and_aggregation_before_error(self):
        metadata = pd.DataFrame(
            {
                "state_id": ["CA", "CA"],
                "store_id": ["CA_1", "CA_1"],
                "cat_id": ["FOODS", "FOODS"],
                "dept_id": ["FOODS_1", "FOODS_1"],
                "item_id": ["a", "b"],
            }
        )
        history = np.array([[1, 2, 3], [2, 4, 6]])
        target = np.array([[4, 4], [8, 8]])
        prediction = np.array([[5, 5], [7, 7]])
        result, details = score_hierarchy(
            metadata,
            history,
            target,
            {"opposite": prediction, "perfect": target},
            np.array([1, 3]),
        )
        # First nine levels cancel errors. Three item levels: 1/4*1 + 3/4*1/2.
        self.assertAlmostEqual(result["opposite"]["wrmsse"], 3 / 12 * 0.625)
        self.assertEqual(result["perfect"]["wrmsse"], 0)
        self.assertAlmostEqual(details.weight.sum(), 1)
        for row in result["opposite"]["levels"][:9]:
            self.assertEqual(row["wrmsse_within_level"], 0)
        with self.assertRaises(ValueError):
            score_hierarchy(
                metadata,
                history,
                target,
                {"x": prediction},
                np.array([1, 3]),
                require_full=True,
            )
        with self.assertRaises(ValueError):
            score_hierarchy(
                metadata, history, target, {"x": prediction[:, :1]}, np.array([1, 3])
            )

    def test_revenue_uses_historical_28_days_and_matching_week(self):
        metadata = pd.DataFrame({"store_id": ["CA_1"], "item_id": ["a"]})
        calendar = pd.DataFrame(
            {
                "d": [f"d_{i}" for i in range(1, 31)],
                "wm_yr_wk": [1] * 15 + [2] * 14 + [3],
            }
        )
        prices = pd.DataFrame(
            {
                "store_id": ["CA_1"] * 3,
                "item_id": ["a"] * 3,
                "wm_yr_wk": [1, 2, 3],
                "sell_price": [2, 3, 9999],
            }
        )
        history = np.ones((1, 29))
        history[0, 0] = 10000  # Before the weight window.
        dollars, missing = trailing_revenue(metadata, history, calendar, prices, 29)
        np.testing.assert_array_equal(dollars, [14 * 2 + 14 * 3])
        self.assertEqual(missing, 0)
        with self.assertRaises(ValueError):
            trailing_revenue(metadata, history, calendar, prices.iloc[1:], 29)
        with self.assertRaises(ValueError):
            trailing_revenue(
                metadata, history, calendar, pd.concat([prices, prices.iloc[:1]]), 29
            )
        history[0, :15] = 0
        dollars, missing = trailing_revenue(
            metadata, history, calendar, prices.iloc[1:], 29
        )
        np.testing.assert_array_equal(dollars, [42])
        self.assertEqual(missing, 14)


if __name__ == "__main__":
    unittest.main()
