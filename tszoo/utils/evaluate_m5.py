"""Frozen-checkpoint M5 backtest with optional numeric covariates."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config.m5 import load_config
from ..data import MemmapWindows
from ..data.fields import FeatureSchema
from ..models.chronos2.backbone import Chronos2Backbone
from ..models.chronos2.model import SplitChronos2
from .io import save_json
from .metrics import Scores


def backtest_windows(config):
    origin, horizon = config["origin"], config["horizon"]
    selection_path = Path(config["store"]) / "selection.json"
    if "past_feat_dynamic_real" in config.get("fields", []) and selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("origin") != origin:
            raise ValueError(
                "Rebuild the price store for the requested forecast origin"
            )
    inputs = MemmapWindows(
        config["store"],
        config["context"],
        horizon,
        fields=config.get("fields", ["target"]),
        feature_columns=config.get("feature_columns"),
        mode="predict",
        end=origin,
    )
    manifest = inputs.manifest
    records = manifest["series"]
    if (
        manifest["freq"] != "D"
        or manifest["fields"]["target"]["width"] != 1
        or any(r["length"] < origin + horizon for r in records)
        or len({r["start"] for r in records}) != 1
        or len({r["item_id"] for r in records}) != len(records)
    ):
        raise ValueError(
            "Require aligned daily univariate series, unique IDs, and full holdout"
        )
    # A separate one-origin dataset reads labels; predict() never receives these entries.
    labels = MemmapWindows(
        config["store"],
        1,
        horizon,
        fields=["target"],
        start=origin,
        end=origin + horizon,
        mode="train",
    )
    if len(inputs) != len(labels):
        raise ValueError("Input/label series mismatch")
    return inputs, labels


def run(config, max_series=None, *, point_only=False):
    inputs, labels = backtest_windows(config)
    count = len(inputs) if max_series is None else min(max_series, len(inputs))
    if count < 1:
        raise ValueError("At least one series is required")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict(config, device=device)
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    horizon, quantiles = config["horizon"], config["quantiles"]
    target = np.lib.format.open_memmap(
        root / "targets.npy", mode="w+", dtype="float32", shape=(count, horizon)
    )
    naive = (
        None
        if point_only
        else np.lib.format.open_memmap(
            root / "seasonal_naive.npy",
            mode="w+",
            dtype="float32",
            shape=(count, horizon),
        )
    )
    baseline = Scores([0.5])
    with (root / "series.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "item_id", "forecast_start"])
        for i in range(count):
            window, truth = inputs[i], labels[i]
            target[i] = truth["future_target"][0].numpy()
            if naive is not None:
                naive[i] = np.resize(window["target"][0, -7:].numpy(), horizon)
                baseline.update(target[i : i + 1], naive[i : i + 1, None])
            writer.writerow([i, window["item_id"], str(window["forecast_start"])])
    target.flush()
    if naive is not None:
        naive.flush()
    summary = {
        "series": count,
        "available_series": len(inputs),
        "quantiles": quantiles,
        "postprocessing": "none",
        "models": {},
        "torch_version": torch.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name(torch.device(device))
        if device.startswith("cuda")
        else "cpu",
    }
    if naive is not None:
        summary["seasonal_naive_7"] = baseline.result()
    for name, checkpoint in config["models"].items():
        started = time.perf_counter()
        backbone = Chronos2Backbone.from_local(
            checkpoint,
            variate_attention=config["variate_attention"],
            variate_attention_policy=config["variate_attention_policy"],
        )
        indices = []
        for q in quantiles:
            if q not in backbone.config.quantiles:
                raise ValueError(f"{name} has no quantile {q}")
            indices.append(backbone.config.quantiles.index(q))
        selected = config.get("fields", ["target"])
        schema = FeatureSchema(
            **{
                name: len(inputs.feature_columns[name])
                if name in inputs.feature_columns
                else inputs.manifest["fields"][name]["width"]
                for name in selected
                if name != "target"
            }
        )
        model = SplitChronos2(backbone, schema, fields=selected).to(device).eval()
        model.requires_grad_(False)
        predictions = np.lib.format.open_memmap(
            root / f"{name}.npy",
            mode="w+",
            dtype="float32",
            shape=(count, len(quantiles), horizon),
        )
        scores = Scores(quantiles)
        for left in range(0, count, config["batch_size"]):
            right = min(count, left + config["batch_size"])
            forecasts = model.predict([inputs[i] for i in range(left, right)], horizon)
            values = np.stack([f.forecast_array[indices] for f in forecasts])
            predictions[left:right] = values
            scores.update(target[left:right], values)
            if left == 0 or right == count or right % (config["batch_size"] * 50) == 0:
                print(
                    f"{name}: {right}/{count}, {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        predictions.flush()
        metrics = scores.result()
        if point_only:
            metrics = {
                "mae": metrics["mae"],
                "1-wape": None if metrics["wape"] is None else 1 - metrics["wape"],
            }
        result = dict(
            metrics,
            elapsed_seconds=time.perf_counter() - started,
            checkpoint=checkpoint,
        )
        summary["models"][name] = result
        save_json(root / f"{name}.metrics.json", result)
        del predictions, model, backbone
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    save_json(root / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config/m5-zero-shot.yaml"),
    )
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("device", "batch_size"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.output:
        config["output"] = str(Path(args.output).resolve())
    if args.models:
        if set(args.models) - set(config["models"]):
            parser.error("Unknown model name")
        config["models"] = {name: config["models"][name] for name in args.models}
    if config["batch_size"] < 1 or (
        args.max_series is not None and args.max_series < 1
    ):
        parser.error("batch-size and max-series must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    if Path(config["output"]).exists():
        raise FileExistsError(config["output"])
    from .resources import ensure_resources

    ensure_resources(config)
    run(config, args.max_series)


if __name__ == "__main__":
    main()
