"""Data loading, target definition, and preprocessing utilities."""

from src.data.loader import describe_schema, load_raw
from src.data.targets import make_default_label, make_is_resolved, make_regression_target

__all__ = [
    "load_raw",
    "describe_schema",
    "make_default_label",
    "make_is_resolved",
    "make_regression_target",
]
