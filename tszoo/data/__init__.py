"""Field contracts and lazily imported memory-mapped storage."""

from .fields import FeatureSchema

__all__ = ["FeatureSchema", "MemmapWindows", "collate_windows", "write_store"]


def __getattr__(name):
    if name in ("MemmapWindows", "collate_windows", "write_store"):
        from . import storage

        return getattr(storage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
