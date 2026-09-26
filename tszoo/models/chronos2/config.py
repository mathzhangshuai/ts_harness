# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Chronos-2; see NOTICE.md and LICENSE.
"""Chronos-2 architecture and checkpoint configuration."""

from dataclasses import dataclass, fields


@dataclass
class CoreConfig:
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    rope_theta: float = 10000.0
    context_length: int = 8192
    patch_size: int = 16
    max_output_patches: int = 64
    time_encoding_scale: int = 8192
    use_reg_token: bool = True
    use_arcsinh: bool = True
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    variate_attention: str = "global_masked"
    variate_attention_policy: str = "bidirectional"
    variate_grouping: str = "series"

    def __post_init__(self):
        if self.variate_grouping not in ("series", "batch"):
            raise ValueError("variate_grouping must be series or batch")
        if self.variate_attention not in ("global_masked", "grouped"):
            raise ValueError("variate_attention must be global_masked or grouped")
        if self.variate_attention_policy not in ("bidirectional", "target_aware"):
            raise ValueError(
                "variate_attention_policy must be bidirectional or target_aware"
            )
        sizes = (
            self.d_model,
            self.d_kv,
            self.d_ff,
            self.num_layers,
            self.num_heads,
            self.context_length,
            self.patch_size,
            self.max_output_patches,
            self.time_encoding_scale,
        )
        if any(x <= 0 for x in sizes) or self.d_kv % 2:
            raise ValueError("Positive dimensions and even d_kv are required")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.quantiles = tuple(self.quantiles)
        if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be nonempty, unique and sorted")
        if not all(0 < q < 1 for q in self.quantiles):
            raise ValueError("quantiles must be in (0, 1)")

    @classmethod
    def from_pretrained_config(cls, config):
        forecast = config["chronos_config"]
        patch = forecast["input_patch_size"]
        if (
            forecast["input_patch_stride"] != patch
            or forecast["output_patch_size"] != patch
        ):
            raise ValueError(
                "Only equal non-overlapping input/output patches are supported"
            )
        if config.get("feed_forward_proj", "relu") != "relu":
            raise ValueError("Only the Chronos-2 relu backbone is supported")
        keys = {field.name for field in fields(cls)}
        values = {k: v for k, v in config.items() if k in keys}
        values.update({k: v for k, v in forecast.items() if k in keys})
        values["patch_size"] = patch
        values["time_encoding_scale"] = (
            forecast.get("time_encoding_scale") or forecast["context_length"]
        )
        return cls(**values)
