"""M5 target data and lazy windows."""

from .dataset import MemmapWindows, collate_windows, write_store

__all__ = ["MemmapWindows", "collate_windows", "write_store"]
