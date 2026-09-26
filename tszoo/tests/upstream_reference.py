"""Test-only execution of local upstream math without installing Chronos.

Only class definitions are extracted from the checked-out reference source.
The small harness replaces Hugging Face's config/output/base-model plumbing;
all patch, normalization, attention and forecast calculations remain upstream.
"""

import ast
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import torch
from einops import rearrange, repeat
from safetensors.torch import load_file
from torch import nn


class Output:
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return tuple(value for value in vars(self).values() if value is not None)[key]


class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def post_init(self):
        pass


def load_upstream(source_root: Path, checkpoint: Path):
    module = ModuleType("_tszoo_upstream_oracle")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(
        torch=torch,
        nn=nn,
        dataclass=dataclass,
        ModelOutput=Output,
        PreTrainedModel=Base,
        Chronos2CoreConfig=SimpleNamespace,
        copy=copy,
        cast=cast,
        List=list,
        rearrange=rearrange,
        repeat=repeat,
        ACT2FN={"relu": nn.ReLU()},
        _TRANSFORMERS_V5=False,
        init=nn.init,
        maybe_autocast=torch.autocast,
    )
    files = [
        ("chronos_bolt.py", {"Patch", "InstanceNorm"}),
        ("chronos2/config.py", {"Chronos2ForecastingConfig"}),
        ("chronos2/layers.py", None),
        ("chronos2/model.py", None),
    ]
    for relative, names in files:
        path = source_root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and (names is None or node.name in names)
        ]
        exec(  # noqa: S102 - execute only class definitions from the trusted local reference checkout
            compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config["_attn_implementation"] = "sdpa"
    model = namespace["Chronos2Model"](SimpleNamespace(**config))
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    return model.eval()
