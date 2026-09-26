import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tszoo.data import MemmapWindows, write_store
from tszoo.data.prepare import m5_entries, prepare_store


class DatasetTests(unittest.TestCase):
    def test_recent_training_span_bounds_history_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            write_store(
                [{"item_id": "a", "start": "2011-01-29", "target": np.arange(1, 1942)}],
                root,
            )
            data = MemmapWindows(root, 49, 7, start=1913 - 180 + 49, end=1913)
            try:
                self.assertEqual(len(data), 125)
                for index in range(len(data)):
                    np.testing.assert_array_equal(
                        data[index]["target"], [np.arange(1734 + index, 1783 + index)]
                    )
                    np.testing.assert_array_equal(
                        data[index]["future_target"],
                        [np.arange(1783 + index, 1790 + index)],
                    )
                self.assertEqual(data[len(data) - 1]["future_target"][0, -1], 1913)
            finally:
                data.close()

    def test_training_boundary_and_prediction_label_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            write_store(
                [{"item_id": "a", "start": "2011-01-29", "target": np.arange(1, 1942)}],
                root,
            )
            train = MemmapWindows(root, 49, 7, end=1913)
            try:
                self.assertEqual(len(train), 1858)
                np.testing.assert_array_equal(train[0]["target"], [np.arange(1, 50)])
                np.testing.assert_array_equal(
                    train[len(train) - 1]["future_target"], [np.arange(1907, 1914)]
                )
                self.assertEqual(
                    str(train[len(train) - 1]["forecast_start"] + 6), "2016-04-24"
                )
            finally:
                train.close()
            predict = MemmapWindows(root, 49, 7, mode="predict", end=1913)
            try:
                self.assertNotIn("future_target", predict[0])
                np.testing.assert_array_equal(
                    predict[0]["target"], [np.arange(1865, 1914)]
                )
            finally:
                predict.close()
            with self.assertRaises(ValueError):
                MemmapWindows(root, 49, 7, end=1942)

    def test_prepare_reads_validation_only_and_reuses_existing_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            days = [f"d_{i}" for i in range(1, 15)]
            pd.DataFrame(
                {"d": days, "date": pd.date_range("2020-01-01", periods=14)}
            ).to_csv(root / "calendar.csv", index=False)
            pd.DataFrame([dict(id="a", **dict(zip(days, range(14))))]).to_csv(
                root / "sales_train_validation.csv", index=False
            )
            store = root / "store"
            self.assertTrue(prepare_store(root, store, split="validation"))
            self.assertFalse(prepare_store(root, store, split="validation", reuse=True))
            manifest = json.loads((store / "manifest.json").read_text())
            self.assertEqual(set(manifest["fields"]), {"target"})
            self.assertFalse((store / "schema.json").exists())
            self.assertFalse((store / "vocabulary.json").exists())
            self.assertEqual(len(list(m5_entries(root, split="validation"))), 1)
            with self.assertRaises(FileExistsError):
                prepare_store(root, store, split="validation")
            with (store / "target.bin").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises(ValueError):
                prepare_store(root, store, split="validation", reuse=True)

    def test_existing_extra_fields_are_never_mapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            manifest = write_store(
                [{"item_id": "a", "start": "2020-01-01", "target": np.arange(14)}], root
            )
            manifest["fields"]["feat_static_cat"] = {"file": "absent.bin", "width": 5}
            (root / "manifest.json").write_text(json.dumps(manifest))
            data = MemmapWindows(root, 7, 3, end=10)
            try:
                self.assertEqual(
                    set(data[0]),
                    {"item_id", "forecast_start", "target", "future_target"},
                )
            finally:
                data.close()

    def test_duplicate_ids_and_covariates_are_rejected(self):
        entry = {"item_id": "a", "start": "2020-01-01", "target": np.arange(14)}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_store([entry, entry], Path(temporary) / "duplicates")
            with self.assertRaises(ValueError):
                write_store(
                    [dict(entry, feat_dynamic_real=np.ones((1, 14)))],
                    Path(temporary) / "covariates",
                )
