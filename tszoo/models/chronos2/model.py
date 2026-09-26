"""Separate field encoders sharing a pretrained Chronos-2 attention backbone."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from gluonts.dataset.field_names import FieldName as FN
from gluonts.model.forecast import QuantileForecast
from torch import nn
from torch.nn import functional as F

from ...data.fields import CATEGORICAL, FIELDS, KNOWN, STATIC, FeatureSchema
from .backbone import Chronos2Backbone, CoreConfig


class SplitChronos2(nn.Module):
    def __init__(
        self,
        backbone: Chronos2Backbone,
        schema: FeatureSchema,
        *,
        fields=None,
        embedding_dim=8,
    ):
        super().__init__()
        self.backbone, self.schema = backbone, schema
        self.fields = tuple(schema.dimensions() if fields is None else fields)
        if FN.TARGET not in self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("Unique selected fields must include target")
        if set(self.fields) - set(schema.dimensions()):
            raise ValueError("Selected field is not in schema")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        c = backbone.config
        self.embeddings, self.cat_patch = nn.ModuleDict(), nn.ModuleDict()
        for name in self.fields:
            if name in CATEGORICAL:
                size = c.d_model if name in STATIC else embedding_dim
                self.embeddings[name] = nn.ModuleList(
                    [
                        nn.Embedding(cardinality, size, padding_idx=0)
                        for cardinality in getattr(schema, name)
                    ]
                )
                if name not in STATIC:
                    self.cat_patch[name] = nn.Linear(
                        c.patch_size * embedding_dim, c.d_model, bias=False
                    )
        self.static_real = nn.ModuleList(
            [nn.Linear(2, c.d_model) for _ in range(schema.feat_static_real)]
            if FN.FEAT_STATIC_REAL in self.fields
            else []
        )

    def _categorical(self, name, values, history, horizon):
        c = self.backbone.config
        if values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name}: integer category IDs required")
        embedded = []
        for i, layer in enumerate(self.embeddings[name]):
            if ((values[i] < 0) | (values[i] >= layer.num_embeddings)).any():
                raise ValueError(f"{name}[{i}]: category outside schema cardinality")
            embedded.append(layer(values[i]))
        features = torch.stack(embedded)
        if name in STATIC:
            count = (history + c.patch_size - 1) // c.patch_size + int(c.use_reg_token)
            count += (horizon + c.patch_size - 1) // c.patch_size
            tokens = features[:, None].expand(-1, count, -1)
            mask = (values != 0)[:, None].expand(-1, count)
            return tokens, mask

        past = (
            values[:, :history]
            .float()
            .masked_fill(values[:, :history] == 0, float("nan"))
        )
        past = torch.where(torch.isnan(past), past, 0.0)
        future = past.new_full((past.shape[0], horizon), float("nan"))
        if name in KNOWN:
            future = torch.where(values[:, history:] != 0, 0.0, float("nan"))
        base, mask, _ = self.backbone.tokenize(past, future)
        # Embed categories directly; category IDs are never treated as real numbers.
        left = F.pad(features[:, :history], (0, 0, (-history) % c.patch_size, 0))
        left = self.cat_patch[name](
            left.reshape(left.shape[0], -1, c.patch_size * self.embedding_dim)
        )
        if c.use_reg_token:
            left = F.pad(left, (0, 0, 0, 1))
        if name in KNOWN:
            right = F.pad(features[:, history:], (0, 0, 0, (-horizon) % c.patch_size))
            right = self.cat_patch[name](
                right.reshape(right.shape[0], -1, c.patch_size * self.embedding_dim)
            )
        else:
            right = left.new_zeros(
                (left.shape[0], (horizon + c.patch_size - 1) // c.patch_size, c.d_model)
            )
        return base + torch.cat([left, right], 1), mask

    def forward(self, windows, prediction_length: int):
        if not windows:
            raise ValueError("Empty batch")
        c = self.backbone.config
        if not 0 < prediction_length <= c.patch_size * c.max_output_patches:
            raise ValueError("Invalid prediction length")
        device = self.backbone.shared.weight.device
        all_tokens, all_masks, all_groups, target_indices = [], [], [], []
        locations, scales, labels = [], [], []
        cursor, history_length = 0, None
        has_labels = ["future_target" in window for window in windows]
        if any(has_labels) and not all(has_labels):
            raise ValueError("All batch items must agree on label availability")
        for group, window in enumerate(windows):
            history = window[FN.TARGET].shape[-1]
            if history_length is not None and history != history_length:
                raise ValueError("Batch histories must have equal length; pad with NaN")
            if not 0 < history <= c.context_length:
                raise ValueError("Invalid context length")
            history_length = history
            for name in FIELDS:
                if name not in self.fields:
                    continue
                if name not in window:
                    raise ValueError(f"Missing selected field: {name}")
                value = torch.as_tensor(window[name], device=device)
                count = self.schema.dimensions()[name]
                length = history + prediction_length if name in KNOWN else history
                expected = (count,) if name in STATIC else (count, length)
                if value.shape != expected:
                    raise ValueError(
                        f"{name}: expected {expected}, got {tuple(value.shape)}"
                    )
                if name in CATEGORICAL:
                    tokens, mask = self._categorical(
                        name, value, history, prediction_length
                    )
                elif name == FN.FEAT_STATIC_REAL:
                    if torch.isinf(value).any():
                        raise ValueError("Static real features cannot contain infinity")
                    valid = torch.isfinite(value)
                    clean = value.nan_to_num().float()
                    transformed = clean.sign() * torch.log1p(clean.abs())
                    pair = torch.stack([transformed, valid.float()], -1).to(
                        self.backbone.shared.weight.dtype
                    )
                    encoded = torch.stack(
                        [layer(pair[i]) for i, layer in enumerate(self.static_real)]
                    )
                    nt = (history + c.patch_size - 1) // c.patch_size + int(
                        c.use_reg_token
                    )
                    nt += (prediction_length + c.patch_size - 1) // c.patch_size
                    tokens = encoded[:, None].expand(-1, nt, -1)
                    mask = valid[:, None].expand(-1, nt)
                else:
                    past = value[:, :history].float()
                    future = (
                        value[:, history:].float()
                        if name in KNOWN
                        else past.new_full((count, prediction_length), float("nan"))
                    )
                    tokens, mask, stats = self.backbone.tokenize(past, future)
                    if name == FN.TARGET:
                        target_indices.extend(range(cursor, cursor + count))
                        locations.append(stats[0])
                        scales.append(stats[1])
                all_tokens.append(tokens)
                all_masks.append(mask)
                all_groups.append(
                    torch.full((count,), group, dtype=torch.long, device=device)
                )
                cursor += count
            if all(has_labels):
                label = torch.as_tensor(window["future_target"], device=device).float()
                if (
                    label.shape != (self.schema.target_dim, prediction_length)
                    or torch.isinf(label).any()
                ):
                    raise ValueError("Invalid future_target")
                labels.append(label)
        target_mask = torch.zeros(cursor, dtype=torch.bool, device=device)
        target_mask[target_indices] = True
        hidden = self.backbone.encoder(
            torch.cat(all_tokens),
            torch.cat(all_masks),
            torch.cat(all_groups),
            target_mask,
        )
        stats = torch.cat(locations), torch.cat(scales)
        normalized, prediction = self.backbone.decode(
            hidden[target_indices], prediction_length, stats
        )
        result = {
            "quantile_preds": prediction.reshape(
                len(windows),
                self.schema.target_dim,
                len(c.quantiles),
                prediction_length,
            )
        }
        if labels:
            label, _ = self.backbone.normalize(torch.cat(labels), stats)
            valid = torch.isfinite(label)
            if not valid.any():
                raise ValueError("Batch has no observed target labels")
            error = label.nan_to_num()[:, None] - normalized.float()
            q = self.backbone.quantiles[None, :, None]
            loss = 2 * torch.maximum(q * error, (q - 1) * error)
            result["loss"] = (loss * valid[:, None]).sum() / valid.sum()
        return result

    def save_local(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        config = {
            "core": asdict(self.backbone.config),
            "schema": asdict(self.schema),
            "fields": self.fields,
            "embedding_dim": self.embedding_dim,
        }
        torch.save(self.state_dict(), root / "model.pt")
        (root / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_local(
        cls, directory, *, variate_attention=None, variate_attention_policy=None
    ):
        root = Path(directory)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        core = CoreConfig(**config["core"])
        if variate_attention is not None:
            core = replace(core, variate_attention=variate_attention)
        if variate_attention_policy is not None:
            core = replace(core, variate_attention_policy=variate_attention_policy)
        model = cls(
            Chronos2Backbone(core),
            FeatureSchema(**config["schema"]),
            fields=config["fields"],
            embedding_dim=config["embedding_dim"],
        )
        model.load_state_dict(
            torch.load(
                root / "model.pt", map_location="cpu", weights_only=True, mmap=True
            ),
            strict=True,
        )
        return model

    @torch.no_grad()
    def predict(self, windows, prediction_length):
        was_training = self.training
        self.eval()
        try:
            inputs = [
                {key: value for key, value in window.items() if key != "future_target"}
                for window in windows
            ]
            values = self(inputs, prediction_length)["quantile_preds"].cpu().numpy()
            forecasts = []
            for window, value in zip(windows, values):
                array = (
                    value[0]
                    if self.schema.target_dim == 1
                    else np.transpose(value, (1, 2, 0))
                )
                forecasts.append(
                    QuantileForecast(
                        array,
                        start_date=window[FN.FORECAST_START],
                        forecast_keys=[str(q) for q in self.backbone.config.quantiles],
                        item_id=window.get(FN.ITEM_ID),
                    )
                )
            return forecasts
        finally:
            self.train(was_training)
