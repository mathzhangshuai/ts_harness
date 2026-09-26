"""Streaming GluonTS entries -> separate field files -> lazy window tensors."""

import json
import os
from bisect import bisect_right
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName as FN
from torch.utils.data import Dataset

from .fields import CATEGORICAL, FIELDS, KNOWN, STATIC


def write_store(entries, directory, freq: str):
    """Consume one series at a time; never materialize the entire iterable.

    The destination must not exist. A manifest is published only on success.
    Dynamic files are time-major on disk, feature-major at the public boundary.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    records, layout, handles, totals = [], {}, {}, {}
    with ExitStack() as stack:
        for entry in entries:
            names = [name for name in FIELDS if name in entry]
            if FN.TARGET not in names or FN.START not in entry:
                raise ValueError("Each entry requires target and start")
            if layout and set(names) != set(layout):
                raise ValueError("Every series must have the same field schema")
            start = pd.Period(entry[FN.START], freq=freq)
            record = {
                "item_id": str(entry.get(FN.ITEM_ID, len(records))),
                "start": str(start),
                "fields": {},
            }
            target_length = np.asarray(entry[FN.TARGET]).shape[-1]
            for name in names:
                values = np.asarray(entry[name])
                if name == FN.TARGET and values.ndim == 1:
                    values = values[None, :]
                expected_ndim = 1 if name in STATIC else 2
                if values.ndim != expected_ndim or 0 in values.shape:
                    raise ValueError(
                        f"{name}: expected nonempty {expected_ndim}-D array"
                    )
                if name not in STATIC:
                    if values.shape[-1] < target_length:
                        raise ValueError(f"{name}: fewer timestamps than target")
                    if name not in KNOWN and values.shape[-1] != target_length:
                        raise ValueError(
                            f"{name}: only known-future fields may extend beyond target"
                        )
                if name in CATEGORICAL:
                    if (
                        not np.issubdtype(values.dtype, np.integer)
                        or (values < 0).any()
                    ):
                        raise ValueError(
                            f"{name}: categories must be nonnegative integer IDs; 0 = OOV"
                        )
                elif np.isinf(values).any():
                    raise ValueError(f"{name}: use NaN, not infinity")
                width = values.shape[0]
                dtype = "<i8" if name in CATEGORICAL else "<f4"
                rows = values[None, :] if name in STATIC else values.T
                if name not in layout:
                    layout[name] = {
                        "width": width,
                        "dtype": dtype,
                        "file": f"{name}.bin",
                    }
                    handles[name] = stack.enter_context(
                        (root / f"{name}.bin").open("wb")
                    )
                    totals[name] = 0
                if layout[name]["width"] != width:
                    raise ValueError(
                        f"{name}: feature dimension changed between series"
                    )
                record["fields"][name] = [totals[name], len(rows)]
                np.ascontiguousarray(rows, dtype=dtype).tofile(handles[name])
                totals[name] += len(rows)
            record["length"] = target_length
            records.append(record)
    if not records:
        raise ValueError("Cannot write an empty store")
    for name, spec in layout.items():
        spec["rows"] = totals[name]
    manifest = {"version": 1, "freq": freq, "fields": layout, "series": records}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class MemmapWindows(Dataset):
    """Only selected fields are mapped, and only one window is copied to RAM.

    start/end refer to forecast origins/label end in series-relative timestamps.
    Training labels lie entirely before end (exclusive). Predict uses end as the
    forecast origin, returns no labels and may use covariates beyond target.
    """

    def __init__(
        self,
        directory,
        context_length,
        prediction_length,
        *,
        fields=None,
        feature_columns=None,
        start=None,
        end=None,
        stride=1,
        mode="train",
    ):
        self.root = Path(directory)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest["version"] != 1:
            raise ValueError("Unsupported store version")
        self.fields = tuple(self.manifest["fields"] if fields is None else fields)
        if FN.TARGET not in self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("Select target and unique fields")
        if set(self.fields) - set(self.manifest["fields"]) or set(self.fields) - set(
            FIELDS
        ):
            raise ValueError("Unknown or unavailable fields")
        self.feature_columns = {} if feature_columns is None else feature_columns
        if not isinstance(self.feature_columns, dict):
            raise TypeError("feature_columns must be a field-to-columns mapping")
        for name, columns in self.feature_columns.items():
            if (
                name not in self.fields
                or name not in (FN.FEAT_DYNAMIC_REAL, FN.PAST_FEAT_DYNAMIC_REAL)
                or not isinstance(columns, list)
                or not columns
                or any(
                    type(i) is not int
                    or i < 0
                    or i >= self.manifest["fields"][name]["width"]
                    for i in columns
                )
                or len(columns) != len(set(columns))
            ):
                raise ValueError(
                    "Select unique in-range numeric dynamic feature columns"
                )
        if min(context_length, prediction_length, stride) < 1 or mode not in (
            "train",
            "predict",
        ):
            raise ValueError(
                "Positive window sizes and mode=train/predict are required"
            )
        self.context_length, self.prediction_length = context_length, prediction_length
        self.stride, self.mode = stride, mode
        self._maps, self._pid = {}, None
        self._origins, self._counts = [], [0]
        for record in self.manifest["series"]:
            stop = record["length"] if end is None else min(end, record["length"])
            first = max(context_length, start or 0)
            if mode == "predict":
                if stop < 1:
                    raise ValueError(
                        "Prediction requires at least one historical observation"
                    )
                first, count = stop, 1
                for name in self.fields:
                    if (
                        name in KNOWN
                        and record["fields"][name][1] < stop + prediction_length
                    ):
                        raise ValueError(
                            f"{name}: future covariates do not cover prediction horizon"
                        )
            else:
                count = max(0, (stop - prediction_length - first) // stride + 1)
            self._origins.append(first)
            self._counts.append(self._counts[-1] + count)

    def __len__(self):
        return self._counts[-1]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_maps"], state["_pid"] = {}, None
        return state

    def _mapping(self, name):
        if self._pid != os.getpid():
            self._maps, self._pid = {}, os.getpid()
        if name not in self._maps:
            spec = self.manifest["fields"][name]
            filename = spec["file"]
            if Path(filename).name != filename:
                raise ValueError("Store fields must be local filenames")
            path = self.root / filename
            expected = spec["rows"] * spec["width"] * np.dtype(spec["dtype"]).itemsize
            if path.stat().st_size != expected:
                raise ValueError(f"Truncated or invalid field file: {name}")
            self._maps[name] = np.memmap(
                path, mode="r", dtype=spec["dtype"], shape=(spec["rows"], spec["width"])
            )
        return self._maps[name]

    def _read(self, record, name, left, right):
        offset, length = record["fields"][name]
        if not 0 <= left <= right <= length:
            raise ValueError(f"Invalid {name} window")
        # Copy only this slice: torch must never receive a writable view of raw data.
        window = self._mapping(name)[offset + left : offset + right]
        if name in self.feature_columns:
            window = window[:, self.feature_columns[name]]
        values = np.array(window.T, copy=True)
        return torch.from_numpy(values)

    def __getitem__(self, index) -> DataEntry:
        if not 0 <= index < len(self):
            raise IndexError(index)
        series = bisect_right(self._counts, index) - 1
        record = self.manifest["series"][series]
        origin = self._origins[series] + (index - self._counts[series]) * self.stride
        left = max(0, origin - self.context_length)
        result = {
            FN.ITEM_ID: record["item_id"],
            FN.FORECAST_START: pd.Period(record["start"], freq=self.manifest["freq"])
            + origin,
        }
        for name in self.fields:
            if name in STATIC:
                result[name] = self._read(record, name, 0, 1)[:, 0]
                continue
            right = origin + self.prediction_length if name in KNOWN else origin
            value = self._read(record, name, left, right)
            padding = self.context_length - (origin - left)
            fill = 0 if name in CATEGORICAL else float("nan")
            result[name] = torch.nn.functional.pad(value, (padding, 0), value=fill)
        if self.mode == "train":
            result["future_target"] = self._read(
                record, FN.TARGET, origin, origin + self.prediction_length
            )
        return result


def collate_windows(windows):
    """Keep independent GluonTS entries separate until feature encoding."""
    return windows
