# Chronos-2 reference audit

Checked against the local Amazon reference checkout, commit
`10afa9ebe016e514f9d7dc1aa873f66af57e116b`, and installed Lightning 2.4.0 source.
No parameter updates were performed during verification.

## Normalization and loss

- Reference: `src/chronos/chronos_bolt.py`, `InstanceNorm`.
  Standardize using historical mean and population standard deviation, then
  apply `asinh(z)` when the checkpoint enables `use_arcsinh`.
  The inverse applies `sinh`, rescales and adds the historical mean.
- `asinh(z) = sign(z) * log(abs(z) + sqrt(z*z + 1))`; this is not `sign(z) * log1p(abs(z))`.
- Chronos-2-small enables arcsinh and the regression token and predicts 13
  quantiles. These are loaded from its checkpoint, not replaced by YAML defaults.
- Reference: `src/chronos/chronos2/model.py`, `_compute_loss`.
  Quantile loss is masked, averaged over the padded horizon, summed over
  quantiles, and averaged over target channels/batch. Missing targets and
  padded positions contribute zero but remain in the averaging denominator.
- The local loss deliberately divides the sum over all observed target points
  and quantiles by the observed target count in the batch, as requested.
  Padding and missing labels are excluded from both numerator and denominator.
  A completely unobserved batch raises an error. For one complete 28-day target,
  the denominator is 28, not the padded length of 32. This reduction differs
  from upstream; normalization and the pointwise quantile formula are unchanged.
- Only sales channels are prediction targets in this project. Covariate channels
  are model inputs and never receive a forecasting loss or future sales labels.
- The signed-log projection of optional static real attributes is a local
  covariate adapter; it is separate from Chronos-2 time-series normalization.

## Attention

```yaml
model:
  attention:
    variate_attention: grouped
    variate_attention_policy: target_aware
    variate_grouping: series
```

- `grouped` computes attention within groups; `global_masked` computes a full
  matrix with group masks. Both enforce the same group boundaries.
- `target_aware` permits target queries to read covariates but prevents all
  covariate queries, including known-future covariates, from reading targets.
  This also prevents an indirect path through other covariates. `bidirectional`
  allows both directions. This restriction is a local extension, not upstream's default.
- `series` keeps series independent. `batch` puts all series in one group;
  predictions then depend on batch membership, including inference batch size.
- Previous loading code forced bidirectional attention, including when restoring
  a checkpoint. This override has been removed. Saved settings are authoritative.
- Evaluation can supply a top-level `attention` mapping for pretrained models;
  explicit evaluation settings must match a finetuned checkpoint.

## Framework responsibilities

Lightning owns process launch, DDP gradient synchronization, device transfer,
optimization, clipping, progress display, distributed metric reduction and
worker seeding. A LightningDataModule provides the loaders.

M5 CSV joins, named feature selection and leak-free temporal windows remain
dataset-specific code; Lightning does not implement these semantics.

Trainer automatic sampler management is enabled. The DataModule supplies a
`BlockShuffleSampler`, a DistributedSampler subclass with drop_last enabled.
The standard random samplers materialize a full epoch permutation; this local
extension retains only block order and the current block permutation. It shuffles
blocks and their contents, then shards stream positions without duplicate padding.
This is not a uniform global permutation; nearby batches mix fewer series.
Index storage is O(block_size + ceil(window_count / block_size)). Lightning keeps
the existing DistributedSampler and calls set_epoch itself. DataLoader also drops
partial training batches. Process launch and synchronization remain in Lightning.

Evaluation remains a single-process complete traversal, retaining partial batches.

## Verification

Tests compare normalization and predictions directly with the reference and
check loss after converting its denominator to observed target counts at
horizons 7, 16 and 28, including missing labels. Additional tests verify that
padding and entirely missing target series do not dilute loss. Attention tests check blocked
target-to-covariate information flow, group isolation, both computation modes,
cross-series grouping and checkpoint persistence. Distributed sampling is tested
without starting training. Actual multi-GPU training still requires cloud validation.
