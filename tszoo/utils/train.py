"""Local checkpoint fine-tuning over selected memmapped GluonTS fields."""

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

from ..config import load_training_config, training_config
from ..data import MemmapWindows, collate_windows
from ..models.chronos2 import Chronos2Backbone, FeatureSchema, SplitChronos2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config; explicit CLI flags override it")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and checkpoint on CPU without forward/backward or saving",
    )
    parser.add_argument("--store")
    parser.add_argument("--schema")
    parser.add_argument(
        "--pretrained",
        help="Local original Chronos-2 config/safetensors directory",
    )
    parser.add_argument("--output", help="New output directory")
    parser.add_argument("--context", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--train-end",
        type=int,
        help="Exclusive label boundary, relative to each series start",
    )
    parser.add_argument("--fields", nargs="+")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--variate-attention",
        choices=("global_masked", "grouped"),
        default="global_masked",
    )
    parser.add_argument(
        "--variate-attention-policy",
        choices=("bidirectional", "target_aware"),
        default="bidirectional",
    )
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        try:
            parser.set_defaults(**load_training_config(preliminary.config))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
    args = parser.parse_args(argv)
    required = (
        "store",
        "schema",
        "pretrained",
        "output",
        "context",
        "horizon",
        "train_end",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"Provide in YAML or CLI: {', '.join(missing)}")
    if (
        min(args.steps, args.batch_size, args.context, args.horizon, args.train_end) < 1
        or args.workers < 0
        or args.seed < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error("Invalid training parameters")
    return args


def run(args):
    if Path(args.output).exists():
        raise FileExistsError("Output directory already exists")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    schema = FeatureSchema(**json.loads(Path(args.schema).read_text(encoding="utf-8")))
    backbone = Chronos2Backbone.from_local(
        args.pretrained,
        variate_attention=args.variate_attention,
        variate_attention_policy=args.variate_attention_policy,
    )
    model = SplitChronos2(backbone, schema, fields=args.fields)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError(
            "No trainable parameters; enable feature adapters or unfreeze backbone"
        )
    dataset = MemmapWindows(
        args.store, args.context, args.horizon, fields=model.fields, end=args.train_end
    )
    if not len(dataset):
        raise ValueError("No training windows inside train-end")
    if args.context > backbone.config.context_length or args.horizon > (
        backbone.config.patch_size * backbone.config.max_output_patches
    ):
        raise ValueError("Training window exceeds checkpoint limits")
    if args.dry_run:
        examples = [dataset[0], dataset[len(dataset) - 1]]
        report = {
            "dry_run": True,
            "fields": list(model.fields),
            "series": len(dataset.manifest["series"]),
            "training_windows": len(dataset),
            "train_end": args.train_end,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "requested_device": args.device,
            "example_forecast_starts": [str(x["forecast_start"]) for x in examples],
            "example_label_shapes": [list(x["future_target"].shape) for x in examples],
        }
        print(json.dumps(report, indent=2), flush=True)
        return report
    model.to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    # Replacement sampling has bounded memory even for millions of possible windows.
    generator = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=args.steps * args.batch_size,
        generator=generator,
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
    model.save_local(args.output)
    # Record effective options after applying CLI overrides, with portable absolute paths.
    resolved = training_config(args)
    for section, names in (
        ("data", ("store", "schema")),
        ("model", ("pretrained",)),
        ("training", ("output",)),
    ):
        for name in names:
            resolved[section][name] = str(Path(resolved[section][name]).resolve())
    (Path(args.output) / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    run = {
        "arguments": vars(args),
        "schema": asdict(schema),
        "losses": losses,
        "torch": torch.__version__,
        "validation": "training only; no evaluation was run",
    }
    (Path(args.output) / "run.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
