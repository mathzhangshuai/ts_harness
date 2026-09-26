"""Named M5 covariates, read independently of sales and sliced at forecast origin."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

FIELDS = (
    "target",
    "feat_static_cat",
    "feat_static_real",
    "feat_dynamic_cat",
    "feat_dynamic_real",
    "past_feat_dynamic_cat",
    "past_feat_dynamic_real",
)
STATIC = {"feat_static_cat", "feat_static_real"}
KNOWN = {"feat_dynamic_cat", "feat_dynamic_real"}
STATIC_CAT = {"item_id", "dept_id", "cat_id", "store_id", "state_id"}
CALENDAR_CAT = {
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
}
CALENDAR_REAL = {"snap_CA", "snap_TX", "snap_WI"}


def selection(value=None):
    if value is None:
        value = {"target": ["sales"]}
    if not isinstance(value, dict) or set(value) - set(FIELDS):
        raise ValueError("features must map the seven field types to column names")
    result = {field: value.get(field, []) for field in FIELDS}
    used = set()
    for field, names in result.items():
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not name or name.isdecimal() for name in names
        ):
            raise ValueError(f"{field}: use explicit column names, never indices")
        if len(set(names)) != len(names) or used.intersection(names):
            raise ValueError("A source column may be selected only once")
        used.update(names)
        allowed = (
            STATIC_CAT
            if field == "feat_static_cat"
            else CALENDAR_CAT
            if field.endswith("_cat")
            else CALENDAR_REAL | {"sell_price"}
        )
        if field not in {"target", "feat_static_real"} and set(names) - allowed:
            raise ValueError(
                f"Unsupported {field} columns: {sorted(set(names) - allowed)}"
            )
    if result["target"] != ["sales"]:
        raise ValueError("M5 target must be [sales]")
    return result


class M5Features:
    def __init__(self, source, features):
        self.features = selection(features)
        self.schema = {"features": self.features, "vocabularies": {}}
        root = Path(source)
        self.calendar = pd.read_csv(root / "calendar.csv", keep_default_na=False)
        dates = pd.DatetimeIndex(self.calendar["date"])
        if not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D")):
            raise ValueError("Calendar dates must be contiguous daily dates")
        self.start = str(dates[0].date())
        if self.calendar["d"].tolist() != [
            f"d_{i}" for i in range(1, len(self.calendar) + 1)
        ]:
            raise ValueError("Calendar days must be contiguous from d_1")
        self.metadata = pd.read_csv(
            root / "sales_train_validation.csv",
            usecols=lambda c: c in STATIC_CAT | {"id"},
            keep_default_na=False,
        )
        self.metadata["id"] = self.metadata["id"].str.removesuffix("_validation")
        self.metadata = self.metadata.set_index("id", verify_integrity=True)
        for field, names in self.features.items():
            table = self.metadata if field == "feat_static_cat" else self.calendar
            if field not in {"target", "feat_static_real"}:
                missing = set(names) - {"sell_price"} - set(table.columns)
                if missing:
                    raise ValueError(
                        f"Missing source columns for {field}: {sorted(missing)}"
                    )
        self.static = None
        if self.features["feat_static_real"]:
            self.static = pd.read_csv(
                root / "static_features.csv", keep_default_na=False
            )
            self.static = self.static.set_index("id", verify_integrity=True)
            missing = set(self.features["feat_static_real"]) - set(self.static.columns)
            if missing:
                raise ValueError(
                    f"Missing static_features.csv columns: {sorted(missing)}"
                )
        self.prices = None
        if any("sell_price" in names for names in self.features.values()):
            self.prices = pd.read_csv(root / "sell_prices.csv").set_index(
                ["store_id", "item_id", "wm_yr_wk"], verify_integrity=True
            )["sell_price"]
        for field, names in self.features.items():
            if field.endswith("_cat"):
                table = self.metadata if field in STATIC else self.calendar
                for name in names:
                    values = sorted(set(table[name].astype(str)) - {""})
                    self.schema["vocabularies"][name] = values
        self.encodings = {
            name: {v: i + 1 for i, v in enumerate(values)}
            for name, values in self.schema["vocabularies"].items()
        }

    def window(self, series_id, left, origin, horizon):
        key = series_id.removesuffix("_validation").removesuffix("_evaluation")
        meta = self.metadata.loc[key]
        result = {}
        for field, names in self.features.items():
            if field == "target" or not names:
                continue
            right = origin + horizon if field in KNOWN else origin
            if right > len(self.calendar):
                raise ValueError("Calendar does not cover requested future features")
            values = []
            for name in names:
                if field == "feat_static_real":
                    raw = np.asarray(float(self.static.loc[key, name]))
                elif field == "feat_static_cat":
                    raw = np.asarray(str(meta[name]))
                elif name == "sell_price":
                    weeks = self.calendar["wm_yr_wk"].iloc[left:right]
                    index = pd.MultiIndex.from_arrays(
                        [
                            [meta["store_id"]] * len(weeks),
                            [meta["item_id"]] * len(weeks),
                            weeks,
                        ]
                    )
                    raw = self.prices.reindex(index).to_numpy()
                else:
                    raw = self.calendar[name].iloc[left:right].to_numpy()
                if field.endswith("_cat"):
                    vocab = self.encodings[name]
                    raw = np.asarray(
                        [vocab.get(str(v), 0) for v in np.asarray(raw).reshape(-1)]
                    )
                    if field in STATIC:
                        raw = raw[0]
                values.append(raw)
            array = np.asarray(
                values, dtype=np.int64 if field.endswith("_cat") else np.float32
            )
            if np.isinf(array).any():
                raise ValueError(f"Infinite values in {field}")
            result[field] = torch.from_numpy(array)
        return result
