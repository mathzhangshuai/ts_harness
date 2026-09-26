"""M5 sales memmaps with optional named raw-source covariates."""

import json
import os
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import M5Features, selection


def write_store(entries, directory, freq="D"):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    records, seen, offset = [], set(), 0
    with (root / "target.bin").open("wb") as stream:
        for entry in entries:
            if set(entry) - {"item_id", "start", "target"}:
                raise ValueError("Only target and series metadata are supported")
            values = np.asarray(entry["target"], dtype="<f4").reshape(-1)
            if (
                np.asarray(entry["target"]).ndim != 1
                or not len(values)
                or not np.isfinite(values).all()
                or (values < 0).any()
            ):
                raise ValueError("M5 target must be a finite nonnegative 1-D series")
            item_id = str(entry["item_id"])
            if item_id in seen:
                raise ValueError("Duplicate series ID")
            seen.add(item_id)
            records.append(
                {
                    "item_id": item_id,
                    "start": str(pd.Period(entry["start"], freq=freq)),
                    "length": len(values),
                    "fields": {"target": [offset, len(values)]},
                }
            )
            values.tofile(stream)
            offset += len(values)
    if not records:
        raise ValueError("No series")
    manifest = {
        "version": 1,
        "freq": freq,
        "fields": {
            "target": {"width": 1, "dtype": "<f4", "file": "target.bin", "rows": offset}
        },
        "series": records,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class MemmapWindows(Dataset):
    """Training labels stop before end; prediction receives history only."""

    def __init__(
        self,
        directory,
        context_length,
        prediction_length,
        *,
        start=None,
        end=None,
        mode="train",
        features=None,
        source=None,
    ):
        self.root = Path(directory)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = self.manifest
        self.features = selection(features)
        self.feature_source = None
        if any(names for field, names in self.features.items() if field != "target"):
            if source is None:
                raise ValueError("Selected covariates require an M5 source directory")
            self.feature_source = M5Features(source, self.features)
        self.feature_schema = (
            self.feature_source.schema
            if self.feature_source
            else {"features": self.features, "vocabularies": {}}
        )
        if (
            manifest["version"] != 1
            or manifest["freq"] != "D"
            or manifest["fields"]["target"]["width"] != 1
        ):
            raise ValueError("Require daily univariate M5 target data")
        if min(context_length, prediction_length) < 1 or mode not in (
            "train",
            "predict",
        ):
            raise ValueError("Invalid window sizes or mode")
        records = manifest["series"]
        if self.feature_source is not None:
            if any(record["start"] != self.feature_source.start for record in records):
                raise ValueError("Sales store and covariate calendar starts differ")
            ids = [
                r["item_id"].removesuffix("_validation").removesuffix("_evaluation")
                for r in records
            ]
            if any(key not in self.feature_source.metadata.index for key in ids):
                raise ValueError("Sales store series missing from M5 feature metadata")
            if self.feature_source.static is not None and any(
                key not in self.feature_source.static.index for key in ids
            ):
                raise ValueError("Series missing from static_features.csv")
        if not records or len({r["item_id"] for r in records}) != len(records):
            raise ValueError("Require nonempty unique series")
        if end is not None and (end < 1 or any(r["length"] < end for r in records)):
            raise ValueError("Requested boundary exceeds available history")
        self.context_length, self.prediction_length, self.mode = (
            context_length,
            prediction_length,
            mode,
        )
        self._origins, self._counts = [], [0]
        self._map, self._pid = None, None
        for record in records:
            stop = record["length"] if end is None else end
            first = max(context_length, start or 0)
            count = max(0, stop - prediction_length - first + 1)
            if mode == "predict":
                if stop < context_length:
                    raise ValueError("Incomplete prediction context")
                first, count = stop, 1
            self._origins.append(first)
            self._counts.append(self._counts[-1] + count)

    def __len__(self):
        return self._counts[-1]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_map"], state["_pid"] = None, None
        return state

    def close(self):
        if self._map is not None:
            self._map._mmap.close()
            self._map = None

    def _read(self, record, left, right):
        offset, length = record["fields"]["target"]
        if not 0 <= left <= right <= length:
            raise ValueError("Invalid target slice")
        if self._pid != os.getpid() or self._map is None:
            spec = self.manifest["fields"]["target"]
            if Path(spec["file"]).name != spec["file"]:
                raise ValueError("Invalid target filename")
            path = self.root / spec["file"]
            if path.stat().st_size != spec["rows"] * np.dtype(spec["dtype"]).itemsize:
                raise ValueError("Invalid target file size")
            self._map = np.memmap(
                path, mode="r", dtype=spec["dtype"], shape=(spec["rows"],)
            )
            self._pid = os.getpid()
        return torch.from_numpy(
            np.array(self._map[offset + left : offset + right], copy=True)[None]
        )

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        series = bisect_right(self._counts, index) - 1
        record = self.manifest["series"][series]
        origin = self._origins[series] + index - self._counts[series]
        result = {
            "item_id": record["item_id"],
            "forecast_start": pd.Period(record["start"], freq="D") + origin,
            "target": self._read(record, origin - self.context_length, origin),
        }
        if self.mode == "train":
            result["future_target"] = self._read(
                record, origin, origin + self.prediction_length
            )
        if self.feature_source is not None:
            result.update(
                self.feature_source.window(
                    record["item_id"],
                    origin - self.context_length,
                    origin,
                    self.prediction_length,
                )
            )
        return result


def collate_windows(windows):
    return windows
