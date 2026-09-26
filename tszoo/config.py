"""Strict M5 configuration, with YAML-relative resource paths."""

import math
import re
from pathlib import Path

import yaml

from .data.features import selection

ROOT = Path(__file__).resolve().parent


class ConfigLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path):
    path = Path(path).resolve()
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
    if not isinstance(data, dict):
        raise TypeError("Expected a YAML mapping")
    return path, data


def _path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty path")
    return str((base.parent / value).resolve())


def _positive(values, names):
    for name in names:
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f"{name} must be a positive integer")


def load_config(path):
    path, config = _read(path)
    required = {
        "store",
        "output",
        "context",
        "horizon",
        "origin",
        "batch_size",
        "device",
        "models",
        "dataset",
    }
    if set(config) - {"features"} != required:
        raise ValueError(f"Evaluation configuration requires {sorted(required)}")
    config["features"] = selection(config.get("features"))
    _positive(config, ("context", "horizon", "origin", "batch_size"))
    if config["context"] > config["origin"]:
        raise ValueError("Context exceeds available history")
    if not isinstance(config["device"], str) or not config["device"]:
        raise ValueError("device must be a nonempty string")
    for name in ("store", "output"):
        config[name] = _path(path, config[name])
    dataset = config["dataset"]
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"name", "path"}
        or dataset["name"] != "m5"
    ):
        raise ValueError("Only the M5 dataset is supported")
    dataset["path"] = _path(path, dataset["path"])
    models = config["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("At least one model is required")
    resolved, sources, formats = {}, {}, {}
    for name, spec in models.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9_-]+", name)
            or name in {"targets", "seasonal_naive"}
        ):
            raise ValueError("Invalid model name")
        if not isinstance(spec, dict):
            raise TypeError(
                "Model requires a path and an explicit checkpoint format or source"
            )
        if set(spec) == {"path", "format"} and spec["format"] == "finetuned":
            formats[name] = "finetuned"
        elif set(spec) == {"path", "repo_id", "revision", "files"}:
            if not isinstance(spec["repo_id"], str) or not re.fullmatch(
                r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", spec["repo_id"]
            ):
                raise ValueError("Invalid model repository")
            if not isinstance(spec["revision"], str) or not re.fullmatch(
                r"[0-9a-f]{40}", spec["revision"]
            ):
                raise ValueError("Pin a full model revision")
            hashes = spec["files"]
            if not isinstance(hashes, dict) or set(hashes) != {
                "config.json",
                "model.safetensors",
            }:
                raise ValueError("Provide config and weight SHA-256")
            if any(
                not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                for h in hashes.values()
            ):
                raise ValueError("Invalid SHA-256")
            sources[name] = {key: spec[key] for key in ("repo_id", "revision", "files")}
            formats[name] = "pretrained"
        else:
            raise ValueError("Expected a pinned pretrained source or format: finetuned")
        resolved[name] = _path(path, spec["path"])
    config.update(models=resolved, model_sources=sources, model_formats=formats)
    return config


SECTIONS = {
    "data": ("store", "context", "horizon", "train_end"),
    "model": ("pretrained",),
    "training": ("output", "epochs", "batch_size", "workers", "lr", "seed", "device"),
}


def load_training_config(path):
    path, config = _read(path)
    if set(config) != set(SECTIONS):
        raise ValueError("Training requires data, model and training sections")
    values = {}
    for section, names in SECTIONS.items():
        content = config[section]
        optional = (
            {"features", "source"}
            if section == "data"
            else {"devices", "strategy"}
            if section == "training"
            else set()
        )
        if not isinstance(content, dict) or set(content) - optional != set(names):
            raise ValueError(f"{section} requires {names}")
        values.update(content)
    values["features"] = selection(values.get("features"))
    values["source"] = _path(path, values.get("source", "../datasets/m5"))
    values.setdefault("devices", "auto")
    values.setdefault("strategy", "auto")
    if values["devices"] != "auto" and (
        type(values["devices"]) is not int or values["devices"] < 1
    ):
        raise ValueError("devices must be auto or a positive device count")
    if values["strategy"] not in ("auto", "ddp"):
        raise ValueError("strategy must be auto or ddp")
    if values["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("Training device must be cpu, cuda or auto")
    _positive(values, ("context", "horizon", "train_end", "epochs", "batch_size"))
    for name in ("workers", "seed"):
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if (
        type(values["lr"]) not in (int, float)
        or not math.isfinite(values["lr"])
        or values["lr"] <= 0
    ):
        raise ValueError("lr must be finite and positive")
    if not isinstance(values["device"], str) or not values["device"]:
        raise ValueError("device must be a nonempty string")
    for name in ("store", "pretrained", "output"):
        values[name] = _path(path, values[name])
    return values


def training_config(args):
    result = {
        section: {name: getattr(args, name) for name in names}
        for section, names in SECTIONS.items()
    }
    result["data"].update(features=args.features, source=args.source)
    result["training"].update(devices=args.devices, strategy=args.strategy)
    return result
