import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gluonts.dataset.field_names import FieldName as FN
from tszoo.data.prepare import Categories, fresh_entries, m5_entries


class PrepareTests(unittest.TestCase):
    def test_category_reuse_and_oov(self):
        categories = Categories()
        np.testing.assert_array_equal(
            categories.encode("id", ["a", None, "b"]), [1, 0, 2]
        )
        frozen = Categories(categories.values)
        np.testing.assert_array_equal(frozen.encode("id", ["b", "new", "a"]), [2, 0, 1])
        self.assertEqual(len(frozen.values["id"]), 2)
        with self.assertRaises(ValueError):
            frozen.encode("another", ["a"])

    def test_parquet_series_and_static_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.parquet"
            rows = [
                {
                    "store_id": 0,
                    "product_id": 5,
                    "dt": "2024-01-01",
                    "sale_amount": 2.0,
                },
                {
                    "store_id": 0,
                    "product_id": 5,
                    "dt": "2024-01-02",
                    "sale_amount": 3.0,
                },
                {
                    "store_id": 0,
                    "product_id": 7,
                    "dt": "2024-01-01",
                    "sale_amount": 4.0,
                },
            ]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
            entries = list(
                fresh_entries(path, {FN.FEAT_STATIC_CAT: ["product_id"]}, Categories())
            )
            self.assertEqual(len(entries), 2)
            np.testing.assert_array_equal(entries[0][FN.TARGET], [2, 3])
            with self.assertRaises(ValueError):
                list(
                    fresh_entries(
                        path, {FN.FEAT_STATIC_REAL: ["sale_amount"]}, Categories()
                    )
                )
            rows[1]["dt"] = "2024-01-03"
            pq.write_table(pa.Table.from_pylist(rows), path)
            with self.assertRaises(ValueError):
                list(fresh_entries(path, {}, Categories()))

    def test_real_m5_one_series(self):
        source = Path(__file__).resolve().parents[1] / "datasets/m5"
        if not source.exists():
            self.skipTest("Local M5 absent")
        entries = list(m5_entries(source, Categories(), max_series=1))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][FN.TARGET].shape, (1941,))
        self.assertEqual(entries[0][FN.FEAT_STATIC_CAT].shape, (5,))


if __name__ == "__main__":
    unittest.main()
