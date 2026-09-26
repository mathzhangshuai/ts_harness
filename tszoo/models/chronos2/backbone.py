# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Amazon Chronos-2. See NOTICE.md and LICENSE.
# Modifications: standalone PyTorch configuration, token API and local checkpoint IO.
"""Weight-key-compatible Chronos-2 encoder and quantile head."""

import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ...layers import MLP, Attention, ResidualBlock, RMSNorm
from .config import CoreConfig


class AttentionLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.self_attention = Attention(
            config.d_model,
            config.d_kv,
            config.num_heads,
            config.dropout_rate,
            config.rope_theta,
            rotary,
        )
        self.layer_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, mask):
        return x + self.dropout(self.self_attention(self.layer_norm(x), mask))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config.d_model, config.d_ff, config.dropout_rate)
        self.layer_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        return x + self.dropout(self.mlp(self.layer_norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                AttentionLayer(config, True),
                AttentionLayer(config, False),
                FeedForward(config),
            ]
        )

    def forward(self, x, time_mask, group_mask, group_layout=None):
        x = self.layer[0](x, time_mask)
        if group_layout is None:
            x = self.layer[1](x.transpose(0, 1), group_mask).transpose(0, 1)
        else:
            output = torch.empty_like(x)
            for indices, mask in group_layout:
                batch, variables = indices.shape
                time, hidden = x.shape[1:]
                # Equal-size groups share one SDPA call, never one attention axis.
                grouped = (
                    x[indices]
                    .permute(0, 2, 1, 3)
                    .reshape(batch * time, variables, hidden)
                )
                grouped = self.layer[1](grouped, mask)
                grouped = grouped.reshape(batch, time, variables, hidden).permute(
                    0, 2, 1, 3
                )
                output[indices.flatten()] = grouped.reshape(
                    batch * variables, time, hidden
                )
            x = output
        return self.layer[2](x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        self.final_layer_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _group_layout(observed, groups, target_mask=None):
        _, inverse, counts = torch.unique(
            groups, return_inverse=True, return_counts=True
        )
        order = torch.argsort(inverse, stable=True)
        starts = counts.cumsum(0) - counts
        layout = []
        # Bucket by group size: no padded variables and no (total_variables)^2 mask.
        for width in torch.unique(counts).tolist():
            positions = starts[counts == width, None] + torch.arange(
                width, device=groups.device
            )
            indices = order[positions]
            valid = observed[indices].permute(0, 2, 1)
            if target_mask is None:
                mask = valid.reshape(-1, width)[:, None, None, :]
            else:
                roles = target_mask[indices]
                # Rows are queries: covariates may not read target keys/values.
                allowed = roles[:, :, None] | ~roles[:, None, :]
                mask = (valid[:, :, None, :] & allowed[:, None]).reshape(
                    -1, 1, width, width
                )
            layout.append((indices, mask))
        return layout

    def forward(self, tokens, observed, groups, target_mask=None):
        if groups.shape != (tokens.shape[0],) or groups.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("groups must contain one integer group ID per channel")
        if observed.shape != tokens.shape[:2] or observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean (channels, tokens) mask")
        if target_mask is not None and (
            target_mask.shape != groups.shape
            or target_mask.dtype != torch.bool
            or target_mask.device != tokens.device
        ):
            raise ValueError(
                "target_mask must be a boolean role per channel on the input device"
            )
        if self.config.variate_attention_policy == "target_aware":
            if target_mask is None:
                raise ValueError(
                    "target_aware attention requires an explicit target_mask"
                )
        else:
            target_mask = None
        low = torch.finfo(tokens.dtype).min
        time_mask = (~observed[:, None, None, :]).to(tokens.dtype) * low
        group_mask, group_layout = None, None
        if self.config.variate_attention == "grouped":
            group_layout = self._group_layout(observed, groups, target_mask)
        else:
            same = groups[:, None] == groups[None, :]
            valid = same[:, :, None] & observed[None, :, :]
            if target_mask is None:
                group_mask = (~valid.permute(2, 0, 1)[:, None]).to(tokens.dtype) * low
            else:
                allowed = target_mask[:, None] | ~target_mask[None, :]
                # Boolean masks keep forbidden links closed even if no key is valid.
                group_mask = (valid & allowed[:, :, None]).permute(2, 0, 1)[:, None]
        x = self.dropout(tokens)
        for block in self.block:
            x = block(x, time_mask, group_mask, group_layout)
        return self.dropout(self.final_layer_norm(x))


class Chronos2Backbone(nn.Module):
    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(2 if config.use_reg_token else 1, config.d_model)
        self.input_patch_embedding = ResidualBlock(
            config.patch_size * 3, config.d_model, config.d_ff, config.dropout_rate
        )
        self.encoder = Encoder(config)
        self.output_patch_embedding = ResidualBlock(
            config.d_model,
            len(config.quantiles) * config.patch_size,
            config.d_ff,
            config.dropout_rate,
        )
        self.register_buffer(
            "quantiles", torch.tensor(config.quantiles), persistent=False
        )

    @classmethod
    def from_local(
        cls, directory, *, variate_attention=None, variate_attention_policy=None
    ):
        from safetensors import safe_open

        root = Path(directory)
        config = CoreConfig.from_pretrained_config(
            json.loads((root / "config.json").read_text(encoding="utf-8"))
        )
        if variate_attention is not None:
            config = replace(config, variate_attention=variate_attention)
        if variate_attention_policy is not None:
            config = replace(config, variate_attention_policy=variate_attention_policy)
        model = cls(config)
        weights = root / "model.safetensors"
        index = root / "model.safetensors.index.json"
        if weights.exists():
            shards = [weights]
        elif index.exists():
            names = set(
                json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
            )
            if any(Path(name).name != name for name in names):
                raise ValueError("Checkpoint shard names must be local filenames")
            shards = [root / name for name in sorted(names)]
        else:
            raise FileNotFoundError(f"No local safetensors weights in {root}")
        expected, seen = model.state_dict(), set()
        with torch.no_grad():
            for shard in shards:
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for key in handle.keys():  # noqa: SIM118 - safe_open is not a dict/iterable
                        if key not in expected or key in seen:
                            raise ValueError(
                                f"Unexpected or duplicate backbone key: {key}"
                            )
                        tensor = handle.get_tensor(key)
                        if tensor.shape != expected[key].shape:
                            raise ValueError(f"Shape mismatch for {key}")
                        expected[key].copy_(tensor)
                        seen.add(key)
        if seen != set(expected):
            raise ValueError(f"Missing backbone keys: {sorted(set(expected) - seen)}")
        return model

    def normalize(self, x, stats=None):
        x = x.float()
        if stats is None:
            loc = torch.nanmean(x, -1, keepdim=True).nan_to_num(0.0)
            scale = (x - loc).square().nanmean(-1, keepdim=True).sqrt().nan_to_num(1.0)
            scale = torch.where(scale == 0, 1e-5, scale)
            stats = loc, scale
        y = (x - stats[0]) / stats[1]
        return (torch.asinh(y) if self.config.use_arcsinh else y), stats

    def tokenize(self, past, future):
        c = self.config
        if past.ndim != 2 or future.ndim != 2 or past.shape[0] != future.shape[0]:
            raise ValueError(
                "past/future must be (channels, time) with matching channels"
            )
        if not 0 < past.shape[-1] <= c.context_length:
            raise ValueError("History length is outside the model context limit")
        h = future.shape[-1]
        if not 0 < h <= c.max_output_patches * c.patch_size:
            raise ValueError("Prediction horizon is outside the model output limit")
        if torch.isinf(past).any() or torch.isinf(future).any():
            raise ValueError("Use NaN for missing real values, not infinity")
        p = c.patch_size
        x, stats = self.normalize(past)
        y, _ = self.normalize(future, stats)
        x = F.pad(x, ((-x.shape[-1]) % p, 0), value=float("nan"))
        y = F.pad(y, (0, (-h) % p), value=float("nan"))
        parts, masks = [], []
        for values, start in ((x, -x.shape[-1]), (y, 0)):
            valid = ~torch.isnan(values)
            time = (
                torch.arange(start, start + values.shape[-1], device=x.device).float()
                / c.time_encoding_scale
            )
            time = time.expand_as(values).reshape(values.shape[0], -1, p)
            patch = torch.cat(
                [
                    time,
                    values.nan_to_num().reshape(values.shape[0], -1, p),
                    valid.reshape(values.shape[0], -1, p),
                ],
                -1,
            )
            parts.append(self.input_patch_embedding(patch.to(self.shared.weight.dtype)))
            masks.append(valid.reshape(values.shape[0], -1, p).any(-1))
        if c.use_reg_token:
            reg = self.shared.weight[1].reshape(1, 1, -1).expand(past.shape[0], 1, -1)
            parts[0] = torch.cat([parts[0], reg], 1)
            masks[0] = F.pad(masks[0], (0, 1), value=True)
        # Unknown future target tokens must remain visible to the encoder.
        masks[1] = torch.ones_like(masks[1])
        return torch.cat(parts, 1), torch.cat(masks, 1), stats

    def decode(self, hidden, horizon, stats):
        p, nq = self.config.patch_size, len(self.config.quantiles)
        patches = math.ceil(horizon / p)
        z = self.output_patch_embedding(hidden[:, -patches:])
        normalized = (
            z.reshape(z.shape[0], patches, nq, p)
            .permute(0, 2, 1, 3)
            .flatten(2)[..., :horizon]
        )
        values = (
            torch.sinh(normalized.float())
            if self.config.use_arcsinh
            else normalized.float()
        )
        values = values * stats[1][:, None] + stats[0][:, None]
        return normalized, values

    def forward(
        self, past, future=None, groups=None, prediction_length=1, target_mask=None
    ):
        if future is None:
            future = past.new_full((past.shape[0], prediction_length), float("nan"))
        tokens, mask, stats = self.tokenize(past, future)
        if groups is None:
            groups = torch.arange(past.shape[0], device=past.device)
        hidden = self.encoder(tokens, mask, groups, target_mask)
        return self.decode(hidden, future.shape[-1], stats)[1]
