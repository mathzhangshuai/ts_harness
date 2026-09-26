"""Prepare local M5/FreshRetailNet as independently mapped GluonTS fields."""

import argparse
import csv
import json
from dataclasses import asdict
from itertools import groupby, islice
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gluonts.dataset.field_names import FieldName as FN

from .fields import CATEGORICAL, FIELDS, STATIC, FeatureSchema
from .storage import write_store


class Categories:
    def __init__(self, vocabulary=None):
        self.frozen = vocabulary is not None
        self.values = vocabulary if vocabulary is not None else {}

    def encode(self, key, values):
        if self.frozen and key not in self.values:
            raise ValueError(f"Training vocabulary does not contain {key}")
        table = self.values.setdefault(key, {})
        result = []
        for value in values:
            if pd.isna(value):
                result.append(0)
                continue
            token = str(value)
            if token not in table and not self.frozen:
                table[token] = len(table) + 1
            result.append(table.get(token, 0))
        return np.asarray(result, dtype=np.int64)


def fresh_entries(path, mapping, categories, max_series=None):
    """Require contiguous series in source; reject gaps/duplicates explicitly."""
    columns = sorted(
        {"store_id", "product_id", "dt", "sale_amount"}
        | {column for names in mapping.values() for column in names}
    )
    parquet = pq.ParquetFile(path)
    if set(columns) - set(parquet.schema_arrow.names):
        raise ValueError("Mapping references missing Parquet columns")

    def rows():
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            yield from batch.to_pylist()

    seen = set()
    groups = groupby(rows(), key=lambda row: (row["store_id"], row["product_id"]))
    for key, group in islice(groups, max_series):
        if key in seen:
            raise ValueError(
                "Parquet series are interleaved; sort/group offline before conversion"
            )
        seen.add(key)
        frame = pd.DataFrame(group).sort_values("dt")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["dt"]))
        if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D")):
            raise ValueError(
                f"Series {key} contains duplicate or missing days; define a policy before conversion"
            )
        entry = {
            FN.ITEM_ID: f"{key[0]}:{key[1]}",
            FN.START: pd.Period(dates[0], freq="D"),
            FN.TARGET: frame["sale_amount"].to_numpy(dtype=np.float32),
        }
        for field, names in mapping.items():
            values = []
            for name in names:
                column = frame[name]
                if field in STATIC:
                    if column.nunique(dropna=False) != 1:
                        raise ValueError(f"{name} is not static within series {key}")
                    column = column.iloc[:1]
                if field in CATEGORICAL:
                    encoded = categories.encode(f"{field}:{name}", column)
                else:
                    encoded = column.to_numpy(dtype=np.float32)
                values.append(encoded)
            if values:
                array = np.stack(values)
                entry[field] = array[:, 0] if field in STATIC else array
        yield entry


def m5_entries(directory, categories, max_series=None):
    """Use only evaluation history, avoiding duplicate validation history."""
    root = Path(directory)
    with (root / "calendar.csv").open(encoding="utf-8", newline="") as stream:
        calendar = {row["d"]: row["date"] for row in csv.DictReader(stream)}
    static_columns = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
    with (root / "sales_train_evaluation.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        days = sorted(
            (name for name in reader.fieldnames if name.startswith("d_")),
            key=lambda name: int(name[2:]),
        )
        if days != [f"d_{index}" for index in range(1, len(days) + 1)]:
            raise ValueError("M5 day columns are not contiguous")
        dates = pd.DatetimeIndex([calendar[day] for day in days])
        if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D")):
            raise ValueError("M5 calendar is not daily/contiguous")
        for row in islice(reader, max_series):
            yield {
                FN.ITEM_ID: row["id"],
                FN.START: pd.Period(dates[0], freq="D"),
                FN.TARGET: np.array(
                    [float(row[day]) for day in days], dtype=np.float32
                ),
                FN.FEAT_STATIC_CAT: np.array(
                    [
                        categories.encode(f"feat_static_cat:{name}", [row[name]])[0]
                        for name in static_columns
                    ]
                ),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("m5", "freshretailnet"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mapping", help="JSON: GluonTS feature field -> ordered source column names"
    )
    parser.add_argument(
        "--vocabulary", help="Reuse training vocabulary; unseen categories become ID 0"
    )
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    if args.max_series is not None and args.max_series < 1:
        parser.error("max-series must be positive")
    mapping = (
        json.loads(Path(args.mapping).read_text(encoding="utf-8"))
        if args.mapping
        else {}
    )
    if set(mapping) - (set(FIELDS) - {FN.TARGET}):
        parser.error("Unknown feature fields in mapping")
    if any(
        not isinstance(names, list)
        or not names
        or len(names) != len(set(names))
        or not all(isinstance(name, str) for name in names)
        for names in mapping.values()
    ):
        parser.error(
            "Each mapping value must be a nonempty list of unique column names"
        )
    if args.dataset == "m5" and mapping:
        parser.error("M5 converter currently supports target and five static IDs only")
    vocabulary = (
        json.loads(Path(args.vocabulary).read_text(encoding="utf-8"))
        if args.vocabulary
        else None
    )
    categories = Categories(vocabulary)
    if args.dataset == "m5":
        entries = m5_entries(args.source, categories, args.max_series)
        mapping = {
            FN.FEAT_STATIC_CAT: ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        }
    else:
        entries = fresh_entries(args.source, mapping, categories, args.max_series)
    manifest = write_store(entries, args.output, "D")
    schema = {"target_dim": 1}
    for name, columns in mapping.items():
        schema[name] = (
            [
                len(categories.values.get(f"{name}:{column}", {})) + 1
                for column in columns
            ]
            if name in CATEGORICAL
            else len(columns)
        )
    root = Path(args.output)
    (root / "schema.json").write_text(
        json.dumps(asdict(FeatureSchema(**schema)), indent=2), encoding="utf-8"
    )
    (root / "vocabulary.json").write_text(
        json.dumps(categories.values, indent=2), encoding="utf-8"
    )
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "mapping": mapping,
                "series": len(manifest["series"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(root),
                "series": len(manifest["series"]),
                "fields": list(manifest["fields"]),
            }
        )
    )


if __name__ == "__main__":
    main()
