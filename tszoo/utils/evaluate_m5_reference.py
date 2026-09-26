"""Compare the unmodified reference Chronos2Pipeline against a saved M5 run.

Add reference_models/chronos-forecasting/src to PYTHONPATH before execution.
The reference package is imported directly, never installed or monkey-patched.
"""

import argparse
import csv
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config.m5 import load_config
from .io import save_json
from .metrics import Scores


def raw_batches(source, context, origin, horizon, count, batch_size):
    with Path(source).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = [f"d_{d}" for d in range(origin - context + 1, origin + horizon + 1)]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Raw M5 CSV does not cover the requested window")
        batch = []
        for row in islice(reader, count):
            values = np.array([float(row[key]) for key in required], dtype=np.float32)
            batch.append((row["id"], values[:context], values[context:]))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(project / "config/m5-zero-shot.yaml"))
    parser.add_argument(
        "--source", default=str(project / "datasets/m5/sales_train_evaluation.csv")
    )
    parser.add_argument(
        "--reference-source",
        default=str(
            project.parent / "reference_models/chronos-forecasting/src/chronos"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    if args.max_series is not None and args.max_series < 1:
        parser.error("max-series must be positive")
    config = load_config(args.config)
    if config.get("fields", ["target"]) != ["target"]:
        raise ValueError("Reference comparison requires the target-only baseline")
    baseline = Path(config["output"])
    previous = yaml.safe_load(
        (baseline / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    for key in ("context", "origin", "horizon", "batch_size", "quantiles"):
        if config[key] != previous[key]:
            raise ValueError(f"Baseline configuration mismatch: {key}")
    checkpoint = config["models"]["chronos2"]
    previous_checkpoint = Path(previous["models"]["chronos2"])
    same_snapshot = (
        Path(checkpoint).parent.name == "snapshots"
        and previous_checkpoint.parent.name == "snapshots"
        and Path(checkpoint).parts[-3:] == previous_checkpoint.parts[-3:]
    )
    if (checkpoint != str(previous_checkpoint) and not same_snapshot) or previous[
        "variate_attention_policy"
    ] != "bidirectional":
        raise ValueError("Require the same checkpoint and bidirectional baseline")
    saved_metrics = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    count = saved_metrics["series"]
    if args.max_series is not None:
        count = min(count, args.max_series)
    with (baseline / "series.csv").open(newline="", encoding="utf-8") as stream:
        ids = [row["item_id"] for row in csv.DictReader(stream)]
    truth = np.load(baseline / "targets.npy", mmap_mode="r")
    adapted = np.load(baseline / "chronos2.npy", mmap_mode="r")
    import chronos
    import transformers
    from chronos import Chronos2Pipeline

    source_root = Path(args.reference_source).resolve()
    if Path(chronos.__file__).resolve().parent != source_root:
        raise ValueError("chronos must be imported from the local reference checkout")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        config,
        device=previous["device"],
        reference_source=str(source_root),
        raw_csv=str(Path(args.source).resolve()),
        baseline=str(baseline),
        output=str(output.resolve()),
        series=count,
        cross_learning=False,
        torch_dtype="float32",
        transformers_version=transformers.__version__,
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8"
    )
    started = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        checkpoint,
        device_map=previous["device"],
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    quantiles = config["quantiles"]
    indices = [pipeline.quantiles.index(q) for q in quantiles]
    predictions = np.lib.format.open_memmap(
        output / "chronos2_reference.npy",
        mode="w+",
        dtype="float32",
        shape=(count, len(quantiles), config["horizon"]),
    )
    scores = Scores(quantiles)
    offset, max_difference, total_difference, outside_tolerance = 0, 0.0, 0.0, 0
    for batch in raw_batches(
        args.source,
        config["context"],
        config["origin"],
        config["horizon"],
        count,
        config["batch_size"],
    ):
        right = offset + len(batch)
        if [item[0] for item in batch] != ids[offset:right]:
            raise ValueError("Raw CSV and baseline series order differ")
        labels = np.stack([item[2] for item in batch])
        np.testing.assert_array_equal(labels, truth[offset:right])
        forecasts = pipeline.predict(
            [item[1] for item in batch],
            prediction_length=config["horizon"],
            context_length=config["context"],
            batch_size=config["batch_size"],
            cross_learning=False,
        )
        values = np.stack(
            [forecast[0, indices].cpu().numpy() for forecast in forecasts]
        )
        predictions[offset:right] = values
        scores.update(labels, values)
        difference = np.abs(values.astype(np.float64) - adapted[offset:right])
        max_difference = max(max_difference, float(difference.max()))
        total_difference += float(difference.sum())
        outside_tolerance += int(
            (~np.isclose(values, adapted[offset:right], rtol=1e-4, atol=1e-4)).sum()
        )
        if offset == 0 or right == count or right % (config["batch_size"] * 50) == 0:
            print(
                f"reference: {right}/{count}, {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        offset = right
    if offset != count:
        raise ValueError("Raw CSV has fewer series than the completed baseline")
    predictions.flush()
    result = dict(
        scores.result(),
        series=count,
        quantiles=quantiles,
        elapsed_seconds=time.perf_counter() - started,
        max_absolute_difference=max_difference,
        mean_absolute_difference=total_difference / predictions.size,
        values_outside_tolerance=outside_tolerance,
        rtol=1e-4,
        atol=1e-4,
        reference_source=str(source_root),
        checkpoint=checkpoint,
    )
    save_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if outside_tolerance:
        raise AssertionError(
            f"{outside_tolerance} predictions differ beyond floating-point tolerance"
        )


if __name__ == "__main__":
    main()
