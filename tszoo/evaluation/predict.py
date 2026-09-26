"""Shared median prediction for original and fine-tuned checkpoints."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data import MemmapWindows
from ..models.chronos2 import load_model
from .metrics import Scores


def backtest_windows(config):
    origin, horizon = config["origin"], config["horizon"]
    inputs = MemmapWindows(
        config["store"],
        config["context"],
        horizon,
        mode="predict",
        end=origin,
        features=config.get("features"),
        source=config.get("dataset", {}).get("path"),
    )
    records = inputs.manifest["series"]
    if (
        any(r["length"] < origin + horizon for r in records)
        or len({r["start"] for r in records}) != 1
    ):
        raise ValueError("Require aligned series and full holdout")
    labels = MemmapWindows(
        config["store"], 1, horizon, start=origin, end=origin + horizon
    )
    if len(inputs) != len(labels):
        raise ValueError("Input/label mismatch")
    return inputs, labels


def run(config, max_series=None):
    inputs, labels = backtest_windows(config)
    try:
        return _run(config, inputs, labels, max_series)
    finally:
        inputs.close()
        labels.close()


def _run(config, inputs, labels, max_series):
    learned_features = any(
        names
        for field, names in inputs.features.items()
        if field.endswith("_cat") or field == "feat_static_real"
    )
    if learned_features and any(
        fmt == "pretrained" for fmt in config["model_formats"].values()
    ):
        raise ValueError(
            "Categorical/static-real encoders require a finetuned checkpoint for evaluation"
        )
    count = len(inputs) if max_series is None else min(max_series, len(inputs))
    if count < 1:
        raise ValueError("At least one series is required")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    # Retain the saved-array contract used by existing M5 scoring artifacts.
    resolved = dict(
        config,
        device=device,
        fields=[f for f, names in inputs.features.items() if names],
        feature_schema=inputs.feature_schema,
        quantiles=[0.5],
    )
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    horizon = config["horizon"]
    target = np.lib.format.open_memmap(
        root / "targets.npy", mode="w+", dtype="float32", shape=(count, horizon)
    )
    with (root / "series.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "item_id", "forecast_start"])
        for i in range(count):
            target[i] = labels[i]["future_target"][0].numpy()
            window = inputs[i]
            writer.writerow([i, window["item_id"], str(window["forecast_start"])])
    target.flush()
    summary = {
        "series": count,
        "available_series": len(inputs),
        "quantiles": [0.5],
        "postprocessing": "none",
        "models": {},
        "torch_version": torch.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name(torch.device(device))
        if device.startswith("cuda")
        else "cpu",
    }
    for name, checkpoint in config["models"].items():
        started = time.perf_counter()
        model = (
            load_model(
                checkpoint,
                config["model_formats"][name],
                feature_schema=inputs.feature_schema,
                attention=config.get("attention"),
            )
            .to(device)
            .eval()
        )
        model.requires_grad_(False)
        median = model.backbone.config.quantiles.index(0.5)
        predictions = np.lib.format.open_memmap(
            root / f"{name}.npy", mode="w+", dtype="float32", shape=(count, 1, horizon)
        )
        scores = Scores()
        for left in range(0, count, config["batch_size"]):
            right = min(count, left + config["batch_size"])
            forecasts = model.predict([inputs[i] for i in range(left, right)], horizon)
            values = np.stack([f.forecast_array[median] for f in forecasts])
            predictions[left:right, 0] = values
            scores.update(target[left:right], values)
            if left == 0 or right == count or right % (config["batch_size"] * 50) == 0:
                print(
                    f"{name}: {right}/{count}, {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        predictions.flush()
        result = dict(
            scores.result(),
            elapsed_seconds=time.perf_counter() - started,
            checkpoint=checkpoint,
        )
        summary["models"][name] = result
        (root / f"{name}.metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        del predictions, model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    (root / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
