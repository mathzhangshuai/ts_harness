"""Score saved full-panel M5 point forecasts at all 12 official hierarchy levels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LEVELS = (
    ("total", ()),
    ("state", ("state_id",)),
    ("store", ("store_id",)),
    ("category", ("cat_id",)),
    ("department", ("dept_id",)),
    ("state_category", ("state_id", "cat_id")),
    ("state_department", ("state_id", "dept_id")),
    ("store_category", ("store_id", "cat_id")),
    ("store_department", ("store_id", "dept_id")),
    ("item", ("item_id",)),
    ("state_item", ("state_id", "item_id")),
    ("store_item", ("store_id", "item_id")),
)
EXPECTED_COUNTS = (1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490)


def rmsse_scale(history):
    """Mean squared differences strictly after the first nonzero observation."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] < 2:
        raise ValueError("Need at least two historical observations")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("History must be finite nonnegative sales")
    first = (history != 0).argmax(axis=1)
    differences = np.diff(history, axis=1)
    active = np.arange(differences.shape[1])[None, :] >= first[:, None]
    count = active.sum(axis=1)
    numerator = np.sum(np.square(differences) * active, axis=1)
    scale = np.divide(numerator, count, out=np.zeros_like(numerator), where=count > 0)
    if (scale <= 0).any():
        raise ValueError(
            "Undefined RMSSE scale: constant, all-zero, or too-short history"
        )
    return scale


def aggregate(values, codes):
    return (
        pd.DataFrame(np.asarray(values, dtype=np.float64))
        .groupby(codes, sort=True)
        .sum()
        .to_numpy()
    )


def score_hierarchy(metadata, history, target, forecasts, revenue, require_full=False):
    """Aggregate point forecasts before errors, using equal weight per level."""
    n = len(metadata)
    history = np.asarray(history, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    revenue = np.asarray(revenue, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] != n:
        raise ValueError("History/metadata shape mismatch")
    if target.ndim != 2 or target.shape[0] != n or target.shape[1] == 0:
        raise ValueError("Target/metadata shape mismatch")
    if revenue.shape != (n,) or not np.isfinite(revenue).all() or (revenue < 0).any():
        raise ValueError("Revenue must be finite and nonnegative")
    if revenue.sum() <= 0 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid target or zero total revenue")
    if not forecasts:
        raise ValueError("No forecasts")
    for prediction in forecasts.values():
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError("Forecast shape mismatch or nonfinite values")
    results = {name: {"wrmsse": 0.0, "levels": []} for name in forecasts}
    details = []
    for level, (label, columns) in enumerate(LEVELS, 1):
        if columns:
            keys = pd.MultiIndex.from_frame(metadata[list(columns)])
            codes, unique = pd.factorize(keys, sort=True)
            labels = [json.dumps(list(x)) for x in unique]
        else:
            codes = np.zeros(n, dtype=np.int64)
            labels = ["total"]
        count = len(labels)
        if require_full and count != EXPECTED_COUNTS[level - 1]:
            raise ValueError(f"Incomplete M5 hierarchy at {label}: {count}")
        train_agg = aggregate(history, codes)
        scale = rmsse_scale(train_agg)
        truth = aggregate(target, codes)
        dollars = np.bincount(codes, weights=revenue, minlength=count)
        weight = dollars / revenue.sum()
        np.testing.assert_allclose(weight.sum(), 1, rtol=0, atol=1e-12)
        frame = pd.DataFrame(
            {
                "level": level,
                "name": label,
                "key": labels,
                "scale": scale,
                "revenue": dollars,
                "weight": weight / len(LEVELS),
            }
        )
        for name, prediction in forecasts.items():
            error = aggregate(prediction, codes) - truth
            rmsse = np.sqrt(np.mean(np.square(error), axis=1) / scale)
            score = float(weight @ rmsse)
            results[name]["levels"].append(
                {
                    "level": level,
                    "name": label,
                    "series": count,
                    "wrmsse_within_level": score,
                    "contribution": score / len(LEVELS),
                }
            )
            results[name]["wrmsse"] += score / len(LEVELS)
            frame[f"{name}_rmsse"] = rmsse
            frame[f"{name}_contribution"] = weight * rmsse / len(LEVELS)
        details.append(frame)
        print(f"Scored level {level}: {label}, {count} series", flush=True)
    return results, pd.concat(details, ignore_index=True)


def trailing_revenue(metadata, history, calendar, prices, origin):
    """Use only the 28 historical days ending at the forecast origin."""
    if origin < 28 or history.shape != (len(metadata), origin):
        raise ValueError("Invalid historical revenue window")
    days = [f"d_{d}" for d in range(origin - 27, origin + 1)]
    weeks = calendar.set_index("d").loc[days, "wm_yr_wk"].to_numpy()
    prices = prices[prices.wm_yr_wk.isin(weeks)]
    keys = ["store_id", "item_id", "wm_yr_wk"]
    if prices.duplicated(keys).any():
        raise ValueError("Duplicate weekly price keys")
    indexed = prices.set_index(keys).sell_price
    dollars = np.zeros(len(metadata), dtype=np.float64)
    missing_zero_sales = 0
    for offset, week in enumerate(weeks):
        query = pd.MultiIndex.from_arrays(
            [metadata.store_id, metadata.item_id, np.full(len(metadata), week)],
            names=keys,
        )
        price = indexed.reindex(query).to_numpy(dtype=np.float64)
        units = history[:, origin - 28 + offset]
        missing = np.isnan(price)
        if (missing & (units > 0)).any():
            raise ValueError("Missing price for positive historical sales")
        if ((~missing) & ((price <= 0) | ~np.isfinite(price))).any():
            raise ValueError("Invalid price")
        missing_zero_sales += int(missing.sum())
        dollars += units * np.where(missing, 0.0, price)
    return dollars, missing_zero_sales


def run(source, run_dir, output, *, summary_only=False):
    source, run_dir, output = Path(source), Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    origin, horizon = config["origin"], config["horizon"]
    if summary["series"] != 30490 or horizon != 28:
        raise ValueError("Official M5 scoring requires all 30490 series and 28 days")
    if "target" not in config.get("fields", ["target"]):
        raise ValueError("Scoring requires target predictions")
    if (
        summary["quantiles"] != config["quantiles"]
        or summary["postprocessing"] != "none"
    ):
        raise ValueError("Inconsistent saved forecast metadata")
    metadata_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    days = [f"d_{d}" for d in range(1, origin + horizon + 1)]
    sales = pd.read_csv(
        source / "sales_train_evaluation.csv",
        usecols=metadata_columns + days,
        dtype={d: np.float64 for d in days},
    )
    series = pd.read_csv(run_dir / "series.csv")
    if (
        len(sales) != 30490
        or len(series) != 30490
        or not series.item_id.is_unique
        or not sales.id.is_unique
        or set(series.item_id) != set(sales.id)
        or not np.array_equal(series.row, np.arange(30490))
    ):
        raise ValueError("Saved forecasts do not cover the complete unique M5 panel")
    sales = sales.set_index("id").loc[series.item_id].reset_index()
    metadata = sales[metadata_columns]
    history = sales[days[:origin]].to_numpy()
    target = sales[days[origin:]].to_numpy()
    np.testing.assert_array_equal(
        target, np.load(run_dir / "targets.npy", mmap_mode="r")
    )
    calendar = pd.read_csv(source / "calendar.csv")
    date = calendar.set_index("d").loc[f"d_{origin + 1}", "date"]
    if not series.forecast_start.eq(date).all():
        raise ValueError("Forecast date mismatch")
    prices = pd.read_csv(source / "sell_prices.csv")
    revenue, missing = trailing_revenue(metadata, history, calendar, prices, origin)
    median = summary["quantiles"].index(0.5)
    forecasts = {}
    for name in summary["models"]:
        array = np.load(run_dir / f"{name}.npy", mmap_mode="r")
        if array.shape != (30490, len(summary["quantiles"]), horizon):
            raise ValueError(f"Invalid forecast array: {name}")
        forecasts[name] = array[:, median, :]
    if "seasonal_naive_7" in summary:
        forecasts["seasonal_naive_7"] = np.load(
            run_dir / "seasonal_naive.npy", mmap_mode="r"
        )
        np.testing.assert_array_equal(
            forecasts["seasonal_naive_7"], np.tile(history[:, -7:], (1, 4))
        )
    results, details = score_hierarchy(
        metadata, history, target, forecasts, revenue, require_full=True
    )
    report = {
        "source": str(source.resolve()),
        "forecast_run": str(run_dir.resolve()),
        "origin": origin,
        "horizon": horizon,
        "forecast_start": date,
        "point_forecast": "bottom-level q0.5, summed bottom-up; no clipping or rounding",
        "scale_history": [1, origin],
        "revenue_window": [origin - 27, origin],
        "series": 30490,
        "hierarchical_series": len(details),
        "levels": len(LEVELS),
        "total_trailing_revenue": float(revenue.sum()),
        "missing_price_zero_sales_observations": missing,
        "weight_sum": float(details.weight.sum()),
        "minimum_scale": float(details.scale.min()),
        "models": results,
    }
    output.mkdir(parents=True, exist_ok=False)
    if not summary_only:
        details.to_csv(output / "series_scores.csv", index=False)
    else:
        report["models"] = {
            name: {"wrmsse": values["wrmsse"]} for name, values in results.items()
        }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({name: value["wrmsse"] for name, value in results.items()}, indent=2)
    )
    return report


def main():
    from ..data.download import ROOT, ensure_m5

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "datasets/m5"))
    parser.add_argument("--run", default=str(ROOT / "runs/m5-zero-shot"))
    parser.add_argument("--output", default=str(ROOT / "runs/m5-zero-shot-wrmsse"))
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    ensure_m5(args.source)
    run(args.source, args.run, args.output)


if __name__ == "__main__":
    main()
