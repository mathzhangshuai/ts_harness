"""Generate tiny local checkpoints; upstream source is an optional parity oracle."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from tszoo.models.chronos2 import Chronos2Backbone, CoreConfig

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT.parent / "reference_models/chronos-forecasting/src/chronos"
TEMPORARY = tempfile.TemporaryDirectory(prefix="tszoo-test-weights-")
DUMMY = Path(TEMPORARY.name)
CONFIG = {
    "d_model": 6,
    "d_kv": 4,
    "d_ff": 8,
    "num_layers": 2,
    "num_heads": 4,
    "dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-6,
    "rope_theta": 10000.0,
    "dense_act_fn": "relu",
    "feed_forward_proj": "relu",
    "initializer_factor": 0.05,
    "is_gated_act": False,
    "pad_token_id": 0,
    "reg_token_id": 1,
    "vocab_size": 2,
    "chronos_config": {
        "context_length": 8192,
        "input_patch_size": 16,
        "input_patch_stride": 16,
        "output_patch_size": 16,
        "max_output_patches": 64,
        "quantiles": [0.1, 0.5, 0.9],
        "time_encoding_scale": 8192,
        "use_arcsinh": True,
        "use_reg_token": True,
    },
}
(DUMMY / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
with torch.random.fork_rng():
    torch.manual_seed(123)
    save_file(
        Chronos2Backbone(CoreConfig.from_pretrained_config(CONFIG)).state_dict(),
        DUMMY / "model.safetensors",
    )
