"""
Centralized serialization utilities for AION.
Recursively converts dataclasses, enums, paths, sets, and custom objects
into plain JSON-serializable primitives to ensure Flask jsonify never fails.
"""
from __future__ import annotations

import enum
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def jsonable(obj: Any) -> Any:
    """
    Recursively normalize any Python data structure into standard JSON-compliant types:
    None, bool, int, float, str, list, dict.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, enum.Enum):
        return obj.value

    if isinstance(obj, Path):
        return str(obj)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return jsonable(obj.to_dict())
        except Exception:
            pass

    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return {str(k): jsonable(v) for k, v in asdict(obj).items()}
        except Exception:
            pass

    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]

    # Handle custom classes with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {
                str(k): jsonable(v)
                for k, v in obj.__dict__.items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return str(obj)
