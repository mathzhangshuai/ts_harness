"""Streaming NumPy point and quantile metrics."""

import numpy as np


class Scores:
    """Global observation-weighted scores; undefined ratios are JSON null."""

    def __init__(self, quantiles):
        self.quantiles = np.asarray(quantiles, dtype=np.float64)
        self.median = list(quantiles).index(0.5)
        self.count = 0
        self.absolute_target = 0.0
        self.absolute_error = 0.0
        self.pinball = np.zeros(len(quantiles), dtype=np.float64)

    def update(self, target, prediction):
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != (len(target), len(self.quantiles), target.shape[-1]):
            raise ValueError("Invalid forecast shape")
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            raise ValueError("M5 scores require finite targets and forecasts")
        error = target[:, None] - prediction
        q = self.quantiles[None, :, None]
        self.pinball += np.maximum(q * error, (q - 1) * error).sum(axis=(0, 2))
        self.absolute_error += np.abs(error[:, self.median]).sum()
        self.absolute_target += np.abs(target).sum()
        self.count += target.size

    def result(self):
        if not self.count:
            raise ValueError("Cannot score empty predictions")
        return {
            "observations": self.count,
            "mae": self.absolute_error / self.count,
            "wape": self.absolute_error / self.absolute_target
            if self.absolute_target
            else None,
            "mean_pinball_loss": float(self.pinball.mean() / self.count),
            "mean_weighted_quantile_loss": float(
                2 * self.pinball.mean() / self.absolute_target
            )
            if self.absolute_target
            else None,
            "pinball_by_quantile": {
                str(q): float(x / self.count)
                for q, x in zip(self.quantiles, self.pinball)
            },
        }
