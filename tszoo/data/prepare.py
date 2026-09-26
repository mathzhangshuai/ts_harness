"""Prepare only M5 sales; training and evaluation stores remain separate."""

import argparse
import csv
import json
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from .dataset import write_store


def m5_entries(directory, *, split="evaluation", max_series=None):
    if split not in ("validation", "evaluation"):
        raise ValueError("Unknown M5 split")
    root = Path(directory)
    with (root / "calendar.csv").open(encoding="utf-8", newline="") as stream:
        calendar = {row["d"]: row["date"] for row in csv.DictReader(stream)}
    with (root / f"sales_train_{split}.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        days = [name for name in reader.fieldnames if name.startswith("d_")]
        if not days or days != [f"d_{i}" for i in range(1, len(days) + 1)]:
            raise ValueError("M5 day columns must be contiguous")
        dates = pd.DatetimeIndex([calendar[day] for day in days])
        if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D")):
            raise ValueError("M5 calendar must be daily")
        for row in islice(reader, max_series):
            yield {
                "item_id": row["id"],
                "start": str(dates[0].date()),
                "target": np.asarray([float(row[d]) for d in days], dtype=np.float32),
            }


def prepare_store(source, output, *, split="evaluation", reuse=False):
    output = Path(output)
    if output.exists():
        if not reuse:
            raise FileExistsError(output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["version"] != 1
            or manifest["freq"] != "D"
            or manifest["fields"]["target"]["width"] != 1
        ):
            raise ValueError("Invalid existing M5 store")
        first = next(m5_entries(source, split=split, max_series=1))
        records = manifest["series"]
        if not records or any(
            record["length"] != len(first["target"])
            or record["start"] != first["start"]
            for record in records
        ):
            raise ValueError("Existing store does not match the requested M5 split")
        spec = manifest["fields"]["target"]
        if (
            Path(spec["file"]).name != spec["file"]
            or (output / spec["file"]).stat().st_size
            != spec["rows"] * np.dtype(spec["dtype"]).itemsize
        ):
            raise ValueError("Invalid existing target file")
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".m5-build-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "store"
        manifest = write_store(m5_entries(source, split=split), staging)
        (staging / "preparation.json").write_text(
            json.dumps(
                {
                    "source": str(Path(source).resolve()),
                    "split": split,
                    "series": len(manifest["series"]),
                    "fields": ["target"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        staging.rename(output)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "datasets/m5"))
    parser.add_argument(
        "--split", choices=("train", "evaluation", "both"), default="both"
    )
    parser.add_argument("--output", help="Custom store path; requires a single split")
    args = parser.parse_args()
    if args.output and args.split == "both":
        parser.error("--output requires --split train or evaluation")
    splits = ("train", "evaluation") if args.split == "both" else (args.split,)
    for split in splits:
        folder = "m5-train-target" if split == "train" else "m5-zero-shot"
        output = args.output or ROOT / "datasets/processed" / folder
        prepared = prepare_store(
            args.source,
            output,
            split="validation" if split == "train" else "evaluation",
            reuse=True,
        )
        print(
            json.dumps({"store": str(output), "prepared": prepared, "split": split}),
            flush=True,
        )


if __name__ == "__main__":
    main()
