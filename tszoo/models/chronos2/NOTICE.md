# Source Attribution

The Chronos-2 backbone is adapted from Amazon's Chronos forecasting repository,
commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`:

- `src/chronos/chronos2/model.py`
- `src/chronos/chronos2/layers.py`
- `src/chronos/chronos2/config.py`
- `src/chronos/chronos_bolt.py` (`Patch` and `InstanceNorm`)

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
Original source author attribution: Abdul Fatir Ansari.
Licensed under Apache-2.0; the license is reproduced in `LICENSE`.

Local changes (2026-09-25): standalone PyTorch modules in place of Transformers
and einops runtime dependencies; preserved checkpoint parameter names; an explicit
token interface; separate categorical/static feature encoders; local-only strict
checkpoint loading; target-only observed-value-normalized quantile training loss.
An optional group-size-bucketed variate attention layout avoids cross-group
attention computation while retaining the original projection parameters.
The optional target-aware query/key visibility rule follows TiRex-2's
`src/tirex2/model/component/attention_block.py`, commit
`01e5ca73a4104b2ae2143df2593460d55c21c20e` (Copyright NXAI GmbH, Apache-2.0).
The local adaptation does not claim to implement every upstream pipeline option.

Layout refactor (2026-09-26): common attention, normalization and feed-forward
layers now live in `tszoo/models/chronos2/layers/`; their source attribution and Apache-2.0
license are retained here. Constructors take explicit dimensions rather than
a model configuration object. Checkpoint parameter names and math are unchanged.

M5 simplification (2026-09-26): the outer model accepts only one target per
series; categorical/static adapters were removed. Backbone parameters, patch
math, and target quantile loss are preserved. Existing target-only checkpoints
remain readable.
