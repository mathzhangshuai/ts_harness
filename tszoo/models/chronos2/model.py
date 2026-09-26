"""Target-only Chronos-2 with compatible backbone state names."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from gluonts.model.forecast import QuantileForecast
from torch import nn

from .backbone import Chronos2Backbone
from .config import CoreConfig


class SplitChronos2(nn.Module):
    """One target per series; no feature encoders or covariates."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, windows, prediction_length):
        if not windows:
            raise ValueError("Empty batch")
        device = self.backbone.shared.weight.device
        tokens, masks, locations, scales, labels = [], [], [], [], []
        labeled = ["future_target" in w for w in windows]
        if any(labeled) and not all(labeled):
            raise ValueError("Mixed label availability")
        history = windows[0]["target"].shape[-1]
        for window in windows:
            if set(window) - {"target", "future_target", "item_id", "forecast_start"}:
                raise ValueError("Only target and series metadata are supported")
            past = torch.as_tensor(window["target"], device=device).float()
            if past.shape != (1, history):
                raise ValueError("Require equal-length univariate histories")
            future = past.new_full((1, prediction_length), float("nan"))
            encoded, mask, stats = self.backbone.tokenize(past, future)
            tokens.append(encoded)
            masks.append(mask)
            locations.append(stats[0])
            scales.append(stats[1])
            if all(labeled):
                label = torch.as_tensor(window["future_target"], device=device).float()
                if label.shape != (1, prediction_length) or torch.isinf(label).any():
                    raise ValueError("Invalid future_target")
                labels.append(label)
        groups = torch.arange(len(windows), device=device)
        target_mask = torch.ones(len(windows), dtype=torch.bool, device=device)
        hidden = self.backbone.encoder(
            torch.cat(tokens), torch.cat(masks), groups, target_mask
        )
        stats = torch.cat(locations), torch.cat(scales)
        normalized, prediction = self.backbone.decode(hidden, prediction_length, stats)
        result = {"quantile_preds": prediction[:, None]}
        if labels:
            label, _ = self.backbone.normalize(torch.cat(labels), stats)
            valid = torch.isfinite(label)
            if not valid.any():
                raise ValueError("No observed labels")
            error = label.nan_to_num()[:, None] - normalized.float()
            q = self.backbone.quantiles[None, :, None]
            loss = 2 * torch.maximum(q * error, (q - 1) * error)
            result["loss"] = (loss * valid[:, None]).sum() / valid.sum()
        return result

    def save_local(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        torch.save(self.state_dict(), root / "model.pt")
        (root / "config.json").write_text(
            json.dumps(
                {"core": asdict(self.backbone.config), "fields": ["target"]}, indent=2
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_local(cls, directory):
        root = Path(directory)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("fields") != ["target"]
            or config.get("schema", {}).get("target_dim", 1) != 1
        ):
            raise ValueError("Only target-only checkpoints are supported")
        core = replace(
            CoreConfig(**config["core"]),
            variate_attention="grouped",
            variate_attention_policy="bidirectional",
        )
        model = cls(Chronos2Backbone(core))
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


def load_model(path, checkpoint_format="pretrained"):
    if checkpoint_format == "finetuned":
        return SplitChronos2.from_local(path)
    if checkpoint_format != "pretrained":
        raise ValueError("Unknown checkpoint format")
    return SplitChronos2(
        Chronos2Backbone.from_local(
            path, variate_attention="grouped", variate_attention_policy="bidirectional"
        )
    )
