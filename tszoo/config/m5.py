"""Strict M5 evaluation configuration with paths relative to the YAML file."""

import re
from pathlib import Path

import yaml

from .training import ConfigLoader


def load_config(path):
    path = Path(path).resolve()
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
    required = {
        "store",
        "output",
        "models",
        "context",
        "horizon",
        "origin",
        "batch_size",
        "device",
        "quantiles",
        "variate_attention",
        "variate_attention_policy",
    }
    if (
        not isinstance(config, dict)
        or set(config) - {"fields", "feature_columns", "dataset"} != required
    ):
        raise ValueError(f"Configuration requires exactly {sorted(required)}")
    selected = config.setdefault("fields", ["target"])
    if (
        not isinstance(selected, list)
        or "target" not in selected
        or any(not isinstance(x, str) for x in selected)
        or len(selected) != len(set(selected))
        or set(selected) - {"target", "feat_dynamic_real", "past_feat_dynamic_real"}
    ):
        raise ValueError(
            "Zero-shot fields must include target and only numeric dynamic covariates"
        )
    for key in ("context", "horizon", "origin", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["context"] < 7 or config["context"] > config["origin"]:
        raise ValueError("Require 7 <= context <= origin")
    q = config["quantiles"]
    if (
        not isinstance(q, list)
        or not q
        or any(type(x) not in (int, float) or not 0 < x < 1 for x in q)
        or q != sorted(set(q))
        or 0.5 not in q
    ):
        raise ValueError("Quantiles must be sorted, unique, in (0, 1), including 0.5")
    if config["variate_attention"] not in ("grouped", "global_masked"):
        raise ValueError("Invalid variate_attention")
    if config["variate_attention_policy"] not in ("bidirectional", "target_aware"):
        raise ValueError("Invalid variate_attention_policy")
    models = config["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("models must be a nonempty name-to-checkpoint mapping")
    for name in models:
        if (
            name in ("targets", "seasonal_naive")
            or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
        ):
            raise ValueError("Model names must be lowercase filename-safe identifiers")

    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError("Paths must be nonempty strings")
        return str((path.parent / value).resolve())

    for key in ("store", "output"):
        config[key] = resolve(config[key])
    sources = {}
    resolved = {}
    for name, value in models.items():
        if isinstance(value, str):
            resolved[name] = resolve(value)
            continue
        if not isinstance(value, dict) or set(value) != {
            "path",
            "repo_id",
            "revision",
            "files",
        }:
            raise ValueError("Model requires path, repo_id, revision and files")
        if not isinstance(value["repo_id"], str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", value["repo_id"]
        ):
            raise ValueError("Invalid model repository")
        if not isinstance(value["revision"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", value["revision"]
        ):
            raise ValueError("Pin the model revision to a full commit hash")
        checksums = value["files"]
        if not isinstance(checksums, dict) or set(checksums) != {
            "config.json",
            "model.safetensors",
        }:
            raise ValueError("Provide SHA-256 for config.json and model.safetensors")
        if any(
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in checksums.values()
        ):
            raise ValueError("Invalid model SHA-256")
        resolved[name] = resolve(value["path"])
        sources[name] = {key: value[key] for key in ("repo_id", "revision", "files")}
    config["models"] = resolved
    if sources:
        config["model_sources"] = sources
    if "dataset" in config:
        dataset = config["dataset"]
        if (
            not isinstance(dataset, dict)
            or set(dataset) != {"name", "path"}
            or dataset["name"] != "m5"
        ):
            raise ValueError("Automatic dataset download supports only M5")
        config["dataset"] = {"name": "m5", "path": resolve(dataset["path"])}
    if not isinstance(config["device"], str):
        raise TypeError("device must be a string")
    return config
