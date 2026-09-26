"""Fine-tune Chronos-2-small using YAML-selected named M5 features."""

import argparse
import json
import math
from pathlib import Path

import lightning as L
import torch
import yaml
from torch.utils.data import DataLoader

from ..config import ROOT, load_training_config, training_config
from ..data import MemmapWindows, collate_windows
from ..data.sampler import BlockShuffleSampler
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
        "train_days",
        "epochs",
        "batch_size",
        "workers",
        "seed",
        "shuffle_block_size",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--devices",
        type=lambda value: value if value == "auto" else int(value),
        help="Positive device count or auto",
    )
    parser.add_argument("--strategy", choices=("auto", "ddp"))
    parser.add_argument("--precision", choices=("bf16-mixed", "16-mixed", "32-true"))
    preliminary, _ = parser.parse_known_args(argv)
    try:
        parser.set_defaults(**load_training_config(preliminary.config))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    if (
        min(
            args.context,
            args.horizon,
            args.train_end,
            args.epochs,
            args.batch_size,
            args.shuffle_block_size,
        )
        < 1
        or min(args.workers, args.seed) < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error("Invalid training parameters")
    if args.train_end > 1913:
        parser.error("M5 training must not read beyond d_1913")
    if not args.context + args.horizon <= args.train_days <= args.train_end:
        parser.error("Require context + horizon <= train_days <= train_end")
    if args.device not in ("cuda", "cpu", "auto"):
        parser.error("Training device must be cuda, cpu or auto")
    if args.devices != "auto" and args.devices < 1:
        parser.error("devices must be positive")
    return args


def training_loader(dataset, args, *, world_size=1, rank=0):
    sampler = BlockShuffleSampler(
        dataset,
        block_size=args.shuffle_block_size,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(args.seed),
    )


class ForecastTraining(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model, self.args = model, args

    def training_step(self, batch, batch_idx):
        loss = self.model(batch, self.args.horizon)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            batch_size=len(batch),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.args.lr
        )


class ForecastData(L.LightningDataModule):
    def __init__(self, dataset, args):
        super().__init__()
        self.dataset, self.args = dataset, args

    def train_dataloader(self):
        loader = training_loader(
            self.dataset,
            self.args,
            world_size=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )
        if not len(loader):
            raise ValueError(
                "No full training batch per device; reduce batch_size or devices"
            )
        return loader


def make_trainer(args):
    torch.set_float32_matmul_precision("medium")
    return L.Trainer(
        accelerator="gpu" if args.device == "cuda" else args.device,
        devices=args.devices,
        strategy=args.strategy,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        precision=args.precision,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        use_distributed_sampler=True,
        num_sanity_val_steps=0,
    )


def run(args):
    if not args.context + args.horizon <= args.train_days <= args.train_end:
        raise ValueError("Require context + horizon <= train_days <= train_end")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    L.seed_everything(args.seed, workers=True)
    dataset = MemmapWindows(
        args.store,
        args.context,
        args.horizon,
        # Dataset start is the first forecast origin, not the first history day.
        start=args.train_end - args.train_days + args.context,
        end=args.train_end,
        features=args.features,
        source=args.source,
    )
    try:
        model = load_model(
            args.pretrained,
            feature_schema=dataset.feature_schema,
            attention=args.attention,
        )
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
        loader = training_loader(dataset, args)
        if not len(loader):
            raise ValueError("No full training batch; reduce batch_size")
        if args.dry_run:
            examples = [dataset[0], dataset[len(dataset) - 1]]
            report = {
                "dry_run": True,
                "fields": [f for f, names in args.features.items() if names],
                "features": args.features,
                "series": len(dataset.manifest["series"]),
                "training_windows": len(dataset),
                "train_end": args.train_end,
                "train_days": args.train_days,
                "train_start": args.train_end - args.train_days + 1,
                "context": args.context,
                "horizon": args.horizon,
                "epochs": args.epochs,
                "single_device_batches_per_epoch": len(loader),
                "single_device_total_steps": args.epochs * len(loader),
                "drop_last": True,
                "shuffle_block_size": args.shuffle_block_size,
                "devices": args.devices,
                "strategy": args.strategy,
                "precision": args.precision,
                "float32_matmul_precision": "medium",
                "model_settings": {
                    "use_arcsinh": model.backbone.config.use_arcsinh,
                    "use_reg_token": model.backbone.config.use_reg_token,
                    "quantiles": list(model.backbone.config.quantiles),
                    "variate_attention": model.backbone.config.variate_attention,
                    "variate_attention_policy": model.backbone.config.variate_attention_policy,
                    "variate_grouping": model.backbone.config.variate_grouping,
                    "loss": "valid_target_mean_quantile_loss",
                },
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
        trainer = make_trainer(args)
        trainer.fit(
            ForecastTraining(model, args), datamodule=ForecastData(dataset, args)
        )
        if trainer.interrupted:
            raise RuntimeError("Training interrupted; final checkpoint was not saved")
        if not trainer.is_global_zero:
            return
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
                    "completed_epochs": args.epochs,
                    "world_size": trainer.world_size,
                    "batches_per_epoch": int(trainer.num_training_batches),
                    "total_steps": trainer.global_step,
                    "metrics": {
                        name: float(value.detach().cpu())
                        for name, value in trainer.callback_metrics.items()
                    },
                    "lightning": L.__version__,
                    "torch": torch.__version__,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
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
