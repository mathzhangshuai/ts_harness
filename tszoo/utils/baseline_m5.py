"""Recompute the two-model M5 zero-shot baseline with three point metrics."""

import argparse
from pathlib import Path

import torch

from ..config.m5 import load_config
from .evaluate_m5 import run as evaluate
from .io import save_json
from .resources import ROOT, ensure_resources
from .score_m5_wrmsse import run as score_wrmsse


def run(config):
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError(root)
    if set(config["models"]) != {"chronos2_small", "chronos2"}:
        raise ValueError("Baseline requires exactly Chronos-2-small and Chronos-2")
    if config["quantiles"] != [0.5] or config.get("fields", ["target"]) != ["target"]:
        raise ValueError("Baseline requires target-only median forecasts")
    if (config["context"], config["origin"], config["horizon"]) != (512, 1913, 28):
        raise ValueError("Baseline requires context=512, origin=1913, horizon=28")
    ensure_resources(config)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    forecasts = evaluate(
        dict(config, output=str(root / "predictions")), point_only=True
    )
    hierarchy = score_wrmsse(
        config["dataset"]["path"],
        root / "predictions",
        root / "scoring",
        summary_only=True,
    )
    results = {
        name: {
            "1-wape": forecasts["models"][name]["1-wape"],
            "mae": forecasts["models"][name]["mae"],
            "wrmsse": hierarchy["models"][name]["wrmsse"],
        }
        for name in config["models"]
    }
    save_json(root / "metrics.json", results)
    rows = [
        "# M5 zero-shot baseline",
        "",
        "30,490 series; d_1914-d_1941; 512 historical days; q0.5; no postprocessing.",
        "WRMSSE: 12 levels, full d_1-d_1913 history, trailing 28-day revenue weights.",
        "1-WAPE = 1 - global absolute error / total actual sales.",
        "",
        "| Model | 1-WAPE | MAE | WRMSSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in results.items():
        rows.append(
            f"| {name} | {metrics['1-wape']:.8%} | {metrics['mae']:.8f} | {metrics['wrmsse']:.8f} |"
        )
    (root / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/m5-baseline.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
