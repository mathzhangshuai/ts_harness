"""Chronos-2 with named covariates and compatible backbone state names."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from gluonts.model.forecast import QuantileForecast
from torch import nn
from torch.nn import functional as F

from ...data.features import KNOWN, STATIC, selection
from .backbone import Chronos2Backbone
from .config import CoreConfig


class SplitChronos2(nn.Module):
    """Group each sales target with its selected covariate channels."""

    def __init__(self, backbone, feature_schema=None):
        super().__init__()
        self.backbone = backbone
        self.feature_schema = feature_schema or {
            "features": selection(),
            "vocabularies": {},
        }
        self.features = selection(self.feature_schema["features"])
        self.embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        c = backbone.config
        for field, names in self.features.items():
            if field.endswith("_cat"):
                size = c.d_model if field in STATIC else 8
                self.embeddings[field] = nn.ModuleList(
                    [
                        nn.Embedding(
                            len(self.feature_schema["vocabularies"][name]) + 1,
                            size,
                            padding_idx=0,
                        )
                        for name in names
                    ]
                )
                if names and field not in STATIC:
                    self.projections[field] = nn.Linear(
                        c.patch_size * size, c.d_model, bias=False
                    )
            elif field == "feat_static_real" and names:
                self.projections[field] = nn.ModuleList(
                    [nn.Linear(2, c.d_model) for _ in names]
                )

    def _covariate(self, field, value, history, horizon):
        c = self.backbone.config
        count = len(self.features[field])
        length = history + horizon if field in KNOWN else history
        expected = (count,) if field in STATIC else (count, length)
        if tuple(value.shape) != expected:
            raise ValueError(f"{field}: expected {expected}, got {tuple(value.shape)}")
        n_past = (history + c.patch_size - 1) // c.patch_size
        n_future = (horizon + c.patch_size - 1) // c.patch_size
        n_tokens = n_past + int(c.use_reg_token) + n_future
        if field.endswith("_cat"):
            if value.dtype not in (torch.int32, torch.int64):
                raise ValueError("Categorical features require encoded integer IDs")
            embedded = []
            for row, layer in zip(value, self.embeddings[field]):
                if ((row < 0) | (row >= layer.num_embeddings)).any():
                    raise ValueError("Category outside vocabulary")
                embedded.append(layer(row))
            embeddings = torch.stack(embedded)
            if field in STATIC:
                return embeddings[:, None].expand(-1, n_tokens, -1), (value != 0)[
                    :, None
                ].expand(-1, n_tokens)
            past = torch.zeros_like(
                value[:, :history], dtype=torch.float32
            ).masked_fill(value[:, :history] == 0, float("nan"))
            future = past.new_full((count, horizon), float("nan"))
            if field in KNOWN:
                future = torch.zeros_like(
                    value[:, history:], dtype=torch.float32
                ).masked_fill(value[:, history:] == 0, float("nan"))
            base, mask, _ = self.backbone.tokenize(past, future)
            left = F.pad(embeddings[:, :history], (0, 0, (-history) % c.patch_size, 0))
            left = self.projections[field](left.reshape(count, n_past, -1))
            if c.use_reg_token:
                left = F.pad(left, (0, 0, 0, 1))
            right = left.new_zeros((count, n_future, c.d_model))
            if field in KNOWN:
                right = F.pad(
                    embeddings[:, history:], (0, 0, 0, (-horizon) % c.patch_size)
                )
                right = self.projections[field](right.reshape(count, n_future, -1))
            return base + torch.cat([left, right], 1), mask
        if torch.isinf(value).any():
            raise ValueError("Real covariates cannot contain infinity")
        if field in STATIC:
            valid = torch.isfinite(value)
            clean = value.nan_to_num().float()
            pair = torch.stack(
                [clean.sign() * torch.log1p(clean.abs()), valid.float()], -1
            )
            encoded = torch.stack(
                [layer(row) for layer, row in zip(self.projections[field], pair)]
            )
            return encoded[:, None].expand(-1, n_tokens, -1), valid[:, None].expand(
                -1, n_tokens
            )
        past = value[:, :history].float()
        future = (
            value[:, history:].float()
            if field in KNOWN
            else past.new_full((count, horizon), float("nan"))
        )
        tokens, mask, _ = self.backbone.tokenize(past, future)
        return tokens, mask

    def forward(self, windows, prediction_length):
        if not windows:
            raise ValueError("Empty batch")
        device = self.backbone.shared.weight.device
        tokens, masks, locations, scales, labels = [], [], [], [], []
        groups, target_indices, cursor = [], [], 0
        labeled = ["future_target" in w for w in windows]
        if any(labeled) and not all(labeled):
            raise ValueError("Mixed label availability")
        history = windows[0]["target"].shape[-1]
        for group, window in enumerate(windows):
            active = {field for field, names in self.features.items() if names}
            if set(window) - active - {
                "future_target",
                "item_id",
                "forecast_start",
            } or active - set(window):
                raise ValueError("Window features do not match selected named schema")
            past = torch.as_tensor(window["target"], device=device).float()
            if past.shape != (1, history):
                raise ValueError("Require equal-length univariate histories")
            future = past.new_full((1, prediction_length), float("nan"))
            encoded, mask, stats = self.backbone.tokenize(past, future)
            tokens.append(encoded)
            masks.append(mask)
            locations.append(stats[0])
            scales.append(stats[1])
            target_indices.append(cursor)
            groups.append(group)
            cursor += 1
            for field, names in self.features.items():
                if field == "target" or not names:
                    continue
                value = torch.as_tensor(window[field], device=device)
                encoded, mask = self._covariate(
                    field, value, history, prediction_length
                )
                tokens.append(encoded)
                masks.append(mask)
                groups.extend([group] * len(names))
                cursor += len(names)
            if all(labeled):
                label = torch.as_tensor(window["future_target"], device=device).float()
                if label.shape != (1, prediction_length) or torch.isinf(label).any():
                    raise ValueError("Invalid future_target")
                labels.append(label)
        groups = torch.tensor(groups, device=device)
        if self.backbone.config.variate_grouping == "batch":
            groups = torch.zeros_like(groups)
        target_mask = torch.zeros(cursor, dtype=torch.bool, device=device)
        target_mask[target_indices] = True
        hidden = self.backbone.encoder(
            torch.cat(tokens), torch.cat(masks), groups, target_mask
        )
        stats = torch.cat(locations), torch.cat(scales)
        normalized, prediction = self.backbone.decode(
            hidden[target_indices], prediction_length, stats
        )
        result = {"quantile_preds": prediction[:, None]}
        if labels:
            label, _ = self.backbone.normalize(torch.cat(labels), stats)
            label = F.pad(
                label, (0, normalized.shape[-1] - prediction_length), value=float("nan")
            )
            valid = torch.isfinite(label)
            error = label.nan_to_num()[:, None] - normalized.float()
            q = self.backbone.quantiles[None, :, None]
            loss = 2 * torch.abs(
                error * ((label.nan_to_num()[:, None] <= normalized).float() - q)
            )
            # Match upstream: padded horizon mean, quantile sum, then batch mean.
            result["loss"] = (loss * valid[:, None]).mean(-1).sum(-1).mean()
        return result

    def save_local(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        torch.save(self.state_dict(), root / "model.pt")
        (root / "config.json").write_text(
            json.dumps(
                {
                    "core": asdict(self.backbone.config),
                    "fields": [f for f, names in self.features.items() if names],
                    "feature_schema": self.feature_schema,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_local(cls, directory):
        root = Path(directory)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("fields")
            != [
                f
                for f, names in config.get("feature_schema", {"features": selection()})[
                    "features"
                ].items()
                if names
            ]
            or config.get("schema", {}).get("target_dim", 1) != 1
        ):
            raise ValueError("Checkpoint fields do not match its named feature schema")
        core = CoreConfig(**config["core"])
        model = cls(Chronos2Backbone(core), config.get("feature_schema"))
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
                {k: v for k, v in w.items() if k != "future_target"} for w in windows
            ]
            values = self(inputs, prediction_length)["quantile_preds"].cpu().numpy()
            return [
                QuantileForecast(
                    v[0],
                    start_date=w["forecast_start"],
                    forecast_keys=[str(q) for q in self.backbone.config.quantiles],
                    item_id=w["item_id"],
                )
                for w, v in zip(windows, values)
            ]
        finally:
            self.train(was_training)


def load_model(
    path, checkpoint_format="pretrained", feature_schema=None, attention=None
):
    if checkpoint_format == "finetuned":
        model = SplitChronos2.from_local(path)
        if attention and any(
            getattr(model.backbone.config, name) != value
            for name, value in attention.items()
        ):
            raise ValueError(
                "Evaluation attention settings differ from the saved checkpoint"
            )
        if feature_schema is not None and model.feature_schema != feature_schema:
            raise ValueError(
                "Checkpoint feature names, order or vocabularies differ from dataset"
            )
        return model
    if checkpoint_format != "pretrained":
        raise ValueError("Unknown checkpoint format")
    settings = {
        "variate_attention": "grouped",
        "variate_attention_policy": "bidirectional",
        "variate_grouping": "series",
        **(attention or {}),
    }
    backbone = Chronos2Backbone.from_local(
        path,
        variate_attention=settings["variate_attention"],
        variate_attention_policy=settings["variate_attention_policy"],
    )
    backbone.config = replace(
        backbone.config, variate_grouping=settings["variate_grouping"]
    )
    backbone.encoder.config = backbone.config
    return SplitChronos2(backbone, feature_schema)
