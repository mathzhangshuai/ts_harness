"""GluonTS field contracts. Category ID 0 is reserved for missing/OOV."""

from dataclasses import dataclass

from gluonts.dataset.field_names import FieldName as FN

STATIC = (FN.FEAT_STATIC_CAT, FN.FEAT_STATIC_REAL)
CATEGORICAL = (FN.FEAT_STATIC_CAT, FN.FEAT_DYNAMIC_CAT, FN.PAST_FEAT_DYNAMIC_CAT)
KNOWN = (FN.FEAT_DYNAMIC_CAT, FN.FEAT_DYNAMIC_REAL)
PAST = (FN.PAST_FEAT_DYNAMIC_CAT, FN.PAST_FEAT_DYNAMIC_REAL)
FIELDS = (FN.TARGET, *STATIC, *KNOWN, *PAST)


@dataclass
class FeatureSchema:
    target_dim: int = 1
    feat_static_cat: tuple[int, ...] = ()
    feat_static_real: int = 0
    feat_dynamic_cat: tuple[int, ...] = ()
    feat_dynamic_real: int = 0
    past_feat_dynamic_cat: tuple[int, ...] = ()
    past_feat_dynamic_real: int = 0

    def __post_init__(self):
        if self.target_dim < 1:
            raise ValueError("target_dim must be positive")
        for name in CATEGORICAL:
            cards = tuple(getattr(self, name))
            if any(c < 1 for c in cards):
                raise ValueError(f"{name}: cardinalities must include reserved ID 0")
            setattr(self, name, cards)
        for name in (
            FN.FEAT_STATIC_REAL,
            FN.FEAT_DYNAMIC_REAL,
            FN.PAST_FEAT_DYNAMIC_REAL,
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}: negative feature count")

    def dimensions(self):
        result = {FN.TARGET: self.target_dim}
        for name in FIELDS[1:]:
            value = getattr(self, name)
            count = len(value) if name in CATEGORICAL else value
            if count:
                result[name] = count
        return result
