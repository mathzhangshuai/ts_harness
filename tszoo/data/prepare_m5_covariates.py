"""Prepare a fixed store/category-stratified M5 covariate experiment."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import save_json
from .fields import FeatureSchema
from .storage import write_store

FEATURES = ["weekday_sin", "weekday_cos", "year_sin", "year_cos", "snap", "event_day"]


def calendar_features(calendar, state):
    dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"]))
    weekly = 2 * np.pi * dates.dayofweek.to_numpy() / 7
    yearly = (
        2
        * np.pi
        * (dates.dayofyear.to_numpy() - 1)
        / np.where(dates.is_leap_year, 366, 365)
    )
    event = (
        calendar[["event_name_1", "event_name_2"]]
        .fillna("")
        .ne("")
        .any(axis=1)
        .to_numpy()
    )
    return np.asarray(
        [
            np.sin(weekly),
            np.cos(weekly),
            np.sin(yearly),
            np.cos(yearly),
            calendar[f"snap_{state}"].to_numpy(),
            event,
        ],
        dtype=np.float32,
    )


def completed_weeks(calendar, origin):
    last_day = (
        calendar.assign(day=calendar["d"].str.removeprefix("d_").astype(int))
        .groupby("wm_yr_wk")["day"]
        .max()
    )
    return set(last_day.index[last_day <= origin])


def prepare(source, output, per_stratum=64, seed=0, origin=1913, future_price=False):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    metadata = pd.read_csv(
        source / "sales_train_evaluation.csv",
        usecols=["id", "store_id", "item_id", "state_id", "cat_id"],
    )
    if metadata["id"].duplicated().any():
        raise ValueError("Duplicate M5 series IDs")
    rng = np.random.default_rng(seed)
    indices = sorted(
        int(i)
        for _, group in metadata.groupby(["store_id", "cat_id"], sort=True)
        for i in rng.choice(
            group.index.to_numpy(), size=min(per_stratum, len(group)), replace=False
        )
    )
    selected = metadata.loc[indices]
    keys = selected[["store_id", "item_id"]]
    calendar = pd.read_csv(source / "calendar.csv")
    calendar["day"] = calendar["d"].str.removeprefix("d_").astype(int)
    calendar = calendar.sort_values("day")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"]))
    if calendar["day"].tolist() != list(
        range(1, len(calendar) + 1)
    ) or not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D")):
        raise ValueError("Invalid daily calendar")
    allowed_weeks = (
        set(calendar["wm_yr_wk"]) if future_price else completed_weeks(calendar, origin)
    )
    prices = {}
    for chunk in pd.read_csv(source / "sell_prices.csv", chunksize=100000):
        subset = chunk[chunk["wm_yr_wk"].isin(allowed_weeks)].merge(
            keys, on=["store_id", "item_id"], validate="many_to_one"
        )
        for row in subset.itertuples(index=False):
            key = (row.store_id, row.item_id, row.wm_yr_wk)
            if key in prices or not np.isfinite(row.sell_price) or row.sell_price <= 0:
                raise ValueError("Invalid or duplicate weekly price")
            prices[key] = row.sell_price
    known = {
        state: calendar_features(calendar, state)
        for state in selected["state_id"].unique()
    }
    wanted = set(selected["id"])
    emitted = []

    def entries():
        with (source / "sales_train_evaluation.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            reader = csv.DictReader(stream)
            days = [name for name in reader.fieldnames if name.startswith("d_")]
            if days != [f"d_{i}" for i in range(1, len(days) + 1)]:
                raise ValueError("Sales day columns must be contiguous")
            if len(days) > len(calendar):
                raise ValueError("Calendar does not cover target")
            weeks = calendar["wm_yr_wk"].to_numpy()[: len(days)]
            for row in reader:
                if row["id"] not in wanted:
                    continue
                past_price = np.asarray(
                    [
                        prices.get((row["store_id"], row["item_id"], int(week)), np.nan)
                        for week in weeks
                    ],
                    dtype=np.float32,
                )
                emitted.append(row["id"])
                entry = {
                    "item_id": row["id"],
                    "start": str(dates[0].date()),
                    "target": np.asarray(
                        [float(row[d]) for d in days], dtype=np.float32
                    ),
                    "feat_dynamic_real": known[row["state_id"]][:, : len(days)],
                    "past_feat_dynamic_real": past_price[None],
                }
                if future_price:
                    entry["feat_dynamic_real"] = np.concatenate(
                        [entry["feat_dynamic_real"], past_price[None]]
                    )
                    del entry["past_feat_dynamic_real"]
                yield entry

    manifest = write_store(entries(), output, freq="D")
    if emitted != selected["id"].tolist():
        raise ValueError("Selected series order mismatch")
    save_json(
        output / "schema.json",
        asdict(
            FeatureSchema(
                feat_dynamic_real=7 if future_price else 6,
                past_feat_dynamic_real=0 if future_price else 1,
            )
        ),
    )
    selection = {
        "seed": seed,
        "per_store_category": per_stratum,
        "origin": origin,
        "baseline_rows": indices,
        "ids": emitted,
        "known_features": FEATURES + (["sell_price"] if future_price else []),
        "past_features": [] if future_price else ["sell_price"],
        "price_policy": "Future prices are known by user-specified experimental assumption; missing values remain NaN"
        if future_price
        else "Only weeks fully completed by origin; missing prices remain NaN; no future prices",
        "strata": selected.groupby(["store_id", "cat_id"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records"),
    }
    save_json(output / "selection.json", selection)
    print(f"Prepared {len(manifest['series'])} series in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="datasets/m5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-stratum", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--origin", type=int, default=1913)
    parser.add_argument("--future-price", action="store_true")
    args = parser.parse_args()
    if args.per_stratum < 1 or args.origin < 1 or args.seed < 0:
        parser.error("per-stratum/origin must be positive and seed nonnegative")
    prepare(
        args.source,
        args.output,
        args.per_stratum,
        args.seed,
        args.origin,
        args.future_price,
    )


if __name__ == "__main__":
    main()
