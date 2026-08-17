# runtime/__init__.py

import os
from typing import Optional
from runtime.profiles import (
    RuntimeProfile,
    ProfileName,
    MemoryState,
    get_active_profile as _get_profile_from_registry,
    PROFILE_REGISTRY,
)

_active_profile_override: Optional[RuntimeProfile] = None


def get_active_profile() -> RuntimeProfile:
    """Get the active runtime profile based on environment settings."""
    global _active_profile_override
    if _active_profile_override is not None:
        return _active_profile_override
    return _get_profile_from_registry()


def set_active_profile(profile: RuntimeProfile):
    """Override the active profile explicitly (useful for benchmarks/tests)."""
    global _active_profile_override
    _active_profile_override = profile
