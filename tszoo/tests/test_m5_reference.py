import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tszoo.utils.evaluate_m5_reference import raw_batches


class RawM5Tests(unittest.TestCase):
    def test_raw_boundary_and_partial_final_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sales.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["id"] + [f"d_{i}" for i in range(1, 15)]
                )
                writer.writeheader()
                for index in range(3):
                    writer.writerow(
                        dict(
                            id=str(index),
                            **{f"d_{i}": i + index * 100 for i in range(1, 15)},
                        )
                    )
            batches = list(raw_batches(source, 7, 10, 3, 3, 2))
            self.assertEqual([len(batch) for batch in batches], [2, 1])
            np.testing.assert_array_equal(batches[0][0][1], np.arange(4, 11))
            np.testing.assert_array_equal(batches[0][0][2], np.arange(11, 14))
            self.assertEqual(batches[1][0][0], "2")
            with self.assertRaises(ValueError):
                list(raw_batches(source, 7, 13, 3, 3, 2))


if __name__ == "__main__":
    unittest.main()
