"""
AION Core — Module Identity
============================
Single canonical implementation of module_id parsing and construction.

ALL pipeline components MUST import from here.
NEVER write: module_id.split("_")[1]
NEVER write: int(module_id.replace("module_", ""))

These patterns are fragile, inconsistent, and will crash on any
module_id that doesn't follow the exact expected format.
"""

from __future__ import annotations

import re

_MODULE_ID_RE = re.compile(r"^module_(\d+)$")


def parse_module_number(module_id: str) -> int:
    """
    Parse the integer module number from a canonical module_id string.

    Valid inputs  : "module_1", "module_2", ..., "module_12"
    Invalid inputs: "module_", "Module_1", "mod_1", ""

    Raises ValueError for any non-canonical format so callers detect
    bad data immediately rather than silently propagating wrong numbers.
    """
    m = _MODULE_ID_RE.fullmatch(module_id.strip())
    if not m:
        raise ValueError(
            f"Invalid module_id {module_id!r}. "
            f"Expected canonical format 'module_<positive-integer>' "
            f"(e.g. 'module_1', 'module_5')."
        )
    return int(m.group(1))


def make_module_id(number: int) -> str:
    """
    Build a canonical module_id string from a 1-based integer.

    Example: make_module_id(3) -> "module_3"
    """
    if number < 1:
        raise ValueError(f"Module number must be ≥ 1, got {number!r}.")
    return f"module_{number}"


def make_co(module_number: int) -> str:
    """
    Build the Course Outcome label for a module.

    Example: make_co(3) -> "CO3"
    """
    if module_number < 1:
        raise ValueError(f"Module number must be ≥ 1, got {module_number!r}.")
    return f"CO{module_number}"


def parse_co_number(co: str) -> int:
    """
    Parse the integer from a CO label (inverse of make_co).

    Example: parse_co_number("CO3") -> 3
    """
    m = re.fullmatch(r"CO(\d+)", co.strip())
    if not m:
        raise ValueError(
            f"Invalid CO label {co!r}. Expected format 'CO<integer>' (e.g. 'CO1')."
        )
    return int(m.group(1))
