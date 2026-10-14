"""
Centralized serialization utilities for AION.
Recursively converts dataclasses, enums, paths, sets, and custom objects
into plain JSON-serializable primitives to ensure Flask jsonify never fails.
"""
import enum
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("aion.serialization")


def jsonable(obj: Any, _strict: Optional[bool] = None) -> Any:
    """
    Recursively normalize any Python data structure into standard JSON-compliant types:
    None, bool, int, float, str, list, dict.

    If _strict is True (or AION_ENV == "development" by default), unknown types will
    raise TypeError rather than silently converting to str/repr.
    """
    if _strict is None:
        _strict = os.environ.get("AION_ENV") == "development"

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, enum.Enum):
        return obj.value

    if isinstance(obj, Path):
        return str(obj)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return jsonable(obj.to_dict(), _strict=_strict)
        except Exception:
            if _strict:
                raise

    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return {str(k): jsonable(v, _strict=_strict) for k, v in asdict(obj).items()}
        except Exception:
            if _strict:
                raise

    if isinstance(obj, dict):
        return {str(k): jsonable(v, _strict=_strict) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v, _strict=_strict) for v in obj]

    # Handle custom classes with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {
                str(k): jsonable(v, _strict=_strict)
                for k, v in obj.__dict__.items()
                if not str(k).startswith("_")
            }
        except Exception:
            if _strict:
                raise

    if _strict:
        raise TypeError(f"jsonable: unhandled type {type(obj).__name__}: {obj!r}")

    logger.warning(f"[serialization] falling back to repr for {type(obj).__name__}")
    return repr(obj)

