# M5 named features

Training reads `data.features` and `data.source` from YAML. Evaluation reads
top-level `features` and `dataset.path`. Both use the same seven field types.
Lists contain source column names, never column positions. List order is preserved.
An empty or omitted list disables that field. `target: [sales]` is required.

The default configs remain target-only. For covariate training, edit the training
YAML and choose a new `training.output`. For example:

```yaml
data:
  source: ../datasets/m5
  store: ../datasets/processed/m5-train-target
  context: 49
  horizon: 7
  train_days: 180
  train_end: 1913
  features:
    target: [sales]
    feat_static_cat: [item_id, dept_id, cat_id, store_id, state_id]
    feat_static_real: []
    feat_dynamic_cat: [wday, month, event_name_1, event_type_1]
    feat_dynamic_real: [snap_CA, snap_TX, snap_WI]
    past_feat_dynamic_cat: []
    past_feat_dynamic_real: [sell_price]
```

Keep the existing `model` and `training` sections. Paths are relative to the YAML.
Sales stores need no rebuilding: sales are memory-mapped as before; selected
covariates are joined from the raw M5 files when constructing windows.

| Field | Available columns | Window shape |
| --- | --- | --- |
| target | sales | (1, context) |
| feat_static_cat | item_id, dept_id, cat_id, store_id, state_id | (features,) |
| feat_static_real | Named numeric columns in optional static_features.csv | (features,) |
| feat_dynamic_cat | weekday, wday, month, year, event_name_1, event_type_1, event_name_2, event_type_2 | (features, context + horizon) |
| feat_dynamic_real | snap_CA, snap_TX, snap_WI, sell_price | (features, context + horizon) |
| past_feat_dynamic_cat | Same calendar categorical columns | (features, context) |
| past_feat_dynamic_real | Same numeric columns | (features, context) |

A source column may belong to only one field. Selecting `sell_price` as a future
known feature explicitly assumes prices are available throughout the forecast
horizon. Use `past_feat_dynamic_real` otherwise. Missing prices remain NaN.
Historical fields never include the forecast horizon; future sales are labels
only, and training labels stop at `train_end <= 1913`.

M5 has no native static numeric attributes. To use them, place
`static_features.csv` in `data.source`, with a unique `id` column holding the
M5 series ID without `_validation`/`_evaluation`, plus named numeric columns,
for example `shelf_capacity`. Select `[shelf_capacity]` in `feat_static_real`.
These attributes must be known before training; do not compute them using
held-out sales. No such attributes are fabricated automatically.

Category vocabularies come from validation-file identifiers and the calendar,
never from future sales. Calendar categories are assumed available in advance.
Empty categories use missing ID 0 internally; users select names, not these IDs.
Categorical embeddings and static numeric projections are newly initialized and
trained with the backbone. They cannot be evaluated as a pretrained zero-shot
baseline. Dynamic numeric covariates use the pretrained tokenizer directly.

Checkpoints record field names, order and vocabularies; evaluation must select
the same features and vocabulary. Different input schemas fail explicitly.
Evaluation still reports only 1-WAPE, MAE and WRMSSE.

Cloud commands, from `tszoo/`:

```sh
python run.py prepare --split train
python run.py train --config configs/train_small.yaml --dry-run
python run.py train --config configs/train_small.yaml
```

For evaluation, copy the same features to your evaluation YAML, prepare the
evaluation sales store, and use `evaluate --config ... --checkpoint ...`.
The dry run constructs windows and encoders but does not run training.
