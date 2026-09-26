"""Fine-tune Chronos-2-small using YAML-selected named M5 features."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

from ..config import ROOT, load_training_config, training_config
from ..data import MemmapWindows, collate_windows
from ..models.chronos2 import load_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/train_small.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU checks only; no forward, backward or saving",
    )
    for name in ("store", "pretrained", "output", "device"):
        parser.add_argument("--" + name)
    for name in (
        "context",
        "horizon",
        "train_end",
        "steps",
        "batch_size",
        "workers",
        "seed",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--lr", type=float)
    preliminary, _ = parser.parse_known_args(argv)
    try:
        parser.set_defaults(**load_training_config(preliminary.config))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    if (
        min(args.context, args.horizon, args.train_end, args.steps, args.batch_size) < 1
        or min(args.workers, args.seed) < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error("Invalid training parameters")
    if args.train_end > 1913:
        parser.error("M5 training must not read the d_1914-d_1941 holdout")
    return args


def run(args):
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = MemmapWindows(
        args.store,
        args.context,
        args.horizon,
        end=args.train_end,
        features=args.features,
        source=args.source,
    )
    try:
        model = load_model(args.pretrained, feature_schema=dataset.feature_schema)
        if not len(dataset):
            raise ValueError("No training windows")
        if args.train_end > 1913:
            raise ValueError("Training boundary enters the holdout")
        if (
            args.context > model.backbone.config.context_length
            or args.horizon
            > model.backbone.config.patch_size
            * model.backbone.config.max_output_patches
        ):
            raise ValueError("Window exceeds checkpoint limits")
        if args.dry_run:
            examples = [dataset[0], dataset[len(dataset) - 1]]
            report = {
                "dry_run": True,
                "fields": [f for f, names in args.features.items() if names],
                "features": args.features,
                "series": len(dataset.manifest["series"]),
                "training_windows": len(dataset),
                "train_end": args.train_end,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "requested_device": args.device,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "example_forecast_starts": [str(w["forecast_start"]) for w in examples],
                "example_label_shapes": [
                    list(w["future_target"].shape) for w in examples
                ],
            }
            print(json.dumps(report, indent=2), flush=True)
            return report
        model.to(args.device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=args.steps * args.batch_size,
            generator=torch.Generator().manual_seed(args.seed),
        )
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_windows,
        )
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
        losses = []
        model.train()
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, args.horizon)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            print(json.dumps({"step": step, "loss": losses[-1]}), flush=True)
        model.save_local(root)
        resolved = training_config(args)
        for section, key in (
            ("data", "store"),
            ("model", "pretrained"),
            ("training", "output"),
        ):
            resolved[section][key] = str(Path(resolved[section][key]).resolve())
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        (root / "run.json").write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "features": args.features,
                    "losses": losses,
                    "torch": torch.__version__,
                    "validation": "No validation or test-based selection",
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    finally:
        dataset.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
