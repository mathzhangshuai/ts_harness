"""Strict YAML training configuration; CLI values override YAML defaults."""

import math
from pathlib import Path

import yaml

from ..data.fields import FIELDS

SECTIONS = {
    "data": ("store", "schema", "context", "horizon", "train_end", "fields"),
    "model": (
        "pretrained",
        "variate_attention",
        "variate_attention_policy",
        "freeze_backbone",
    ),
    "training": ("output", "steps", "batch_size", "workers", "lr", "seed", "device"),
}
PATHS = {"store", "schema", "pretrained", "output"}
INTEGERS = {"context", "horizon", "train_end", "steps", "batch_size", "workers", "seed"}


class ConfigLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently discarding configuration."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise TypeError("YAML configuration keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_training_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(config, dict) or set(config) - set(SECTIONS):
        raise ValueError("YAML must contain only data, model and training sections")
    values = {}
    for section, content in config.items():
        if not isinstance(content, dict) or set(content) - set(SECTIONS[section]):
            raise ValueError(f"Unknown keys or invalid mapping in {section}")
        for name, value in content.items():
            if name in INTEGERS:
                minimum = 0 if name in ("workers", "seed") else 1
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}")
            elif name == "lr":
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("lr must be a finite positive number")
            elif name == "freeze_backbone":
                if type(value) is not bool:
                    raise ValueError("freeze_backbone must be a YAML boolean")
            elif name == "fields":
                if value is not None and (
                    not isinstance(value, list)
                    or any(not isinstance(field, str) for field in value)
                    or len(value) != len(set(value))
                    or "target" not in value
                    or set(value) - set(FIELDS)
                ):
                    raise ValueError(
                        "fields must be null or unique field names including target"
                    )
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
            if name == "variate_attention" and value not in (
                "global_masked",
                "grouped",
            ):
                raise ValueError("variate_attention must be global_masked or grouped")
            if name in PATHS:
                value = str((path.parent / value).resolve())
            if name == "variate_attention_policy" and value not in (
                "bidirectional",
                "target_aware",
            ):
                raise ValueError(
                    "variate_attention_policy must be bidirectional or target_aware"
                )
            values[name] = value
    return values


def training_config(args):
    return {
        section: {name: getattr(args, name) for name in names}
        for section, names in SECTIONS.items()
    }
