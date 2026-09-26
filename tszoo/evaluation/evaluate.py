"""Evaluate original or fine-tuned M5 models using the same three metrics."""

import argparse
import json
from pathlib import Path

import torch

from ..config import ROOT, load_config
from ..data.download import ensure_resources
from .predict import run as predict
from .wrmsse import run as score_wrmsse


def run(config):
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError(root)
    if (config["origin"], config["horizon"]) != (1913, 28):
        raise ValueError("M5 evaluation requires origin=1913 and horizon=28")
    ensure_resources(config)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    forecast = predict(dict(config, output=str(root / "predictions")))
    hierarchy = score_wrmsse(
        config["dataset"]["path"],
        root / "predictions",
        root / "scoring",
        summary_only=True,
    )
    results = {
        name: {
            "1-wape": forecast["models"][name]["1-wape"],
            "mae": forecast["models"][name]["mae"],
            "wrmsse": hierarchy["models"][name]["wrmsse"],
        }
        for name in config["models"]
    }
    (root / "metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = [
        "# M5 evaluation",
        "",
        f"30,490 series; d_1914-d_1941; {config['context']} historical days; q0.5; no postprocessing.",
        "WRMSSE: 12 levels, d_1-d_1913 scale history and trailing 28-day revenue weights.",
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
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    parser.add_argument(
        "--checkpoint", help="Local fine-tuned model.pt/config.json directory"
    )
    parser.add_argument("--name", default="chronos2_small_finetuned")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        import re

        if (
            args.models
            or not re.fullmatch(r"[a-z0-9_-]+", args.name)
            or args.name in {"targets", "seasonal_naive"}
        ):
            parser.error("Use a valid checkpoint name and no --models")
        config["models"] = {args.name: str(Path(args.checkpoint).resolve())}
        config["model_formats"] = {args.name: "finetuned"}
        config["model_sources"] = {}
        config["output"] = str(ROOT / "runs/m5-finetuned")
    elif args.models:
        if set(args.models) - set(config["models"]):
            parser.error("Unknown model")
        config["models"] = {name: config["models"][name] for name in args.models}
    if args.output:
        config["output"] = str(Path(args.output).resolve())
    for key in ("device", "batch_size"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["batch_size"] < 1:
        parser.error("batch-size must be positive")
    run(config)


if __name__ == "__main__":
    main()
