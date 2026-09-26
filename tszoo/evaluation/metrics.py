"""Global point errors, without averaging per-series percentages."""

import numpy as np


class Scores:
    def __init__(self):
        self.count = 0
        self.absolute_target = 0.0
        self.absolute_error = 0.0

    def update(self, target, prediction):
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if target.ndim != 2 or target.shape != prediction.shape or not target.size:
            raise ValueError("Expected matching nonempty (series, horizon) arrays")
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            raise ValueError("Scores require finite values")
        self.count += target.size
        self.absolute_target += np.abs(target).sum()
        self.absolute_error += np.abs(target - prediction).sum()

    def result(self):
        if not self.count:
            raise ValueError("No observations")
        return {
            "mae": self.absolute_error / self.count,
            "1-wape": 1 - self.absolute_error / self.absolute_target
            if self.absolute_target
            else None,
        }
