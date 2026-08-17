# tests/integration/test_api_whitelist_guard.py

import os
import pytest
from runtime.profiles import PRODUCTION_PROFILE, LAPTOP_FAST_PROFILE
from runtime import set_active_profile
from v0_1.llm import get_best_llm


def test_api_cannot_bypass_production_model_whitelist():
    """
    Gap 1 Verification:
    Setting AION_PROFILE=PRODUCTION and requesting an unapproved model (e.g. qwen2.5:1.5b)
    MUST cause get_best_llm() / validate_environment() to raise RuntimeError BLOCK
    BEFORE constructing RobustLLMCaller or making any HTTP inference call to Ollama.
    """
    os.environ["AION_PROFILE"] = "PRODUCTION"
    os.environ["AION_MODEL"] = "qwen2.5:1.5b"
    set_active_profile(PRODUCTION_PROFILE)

    # 1. Assert validate_environment raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        PRODUCTION_PROFILE.validate_environment()
    assert "PROFILE INTEGRITY VIOLATION" in str(exc_info.value)

    # 2. Assert get_best_llm raises RuntimeError before constructing LLM caller
    with pytest.raises(RuntimeError) as exc_info:
        get_best_llm()
    assert "PROFILE INTEGRITY VIOLATION" in str(exc_info.value)

    # Cleanup
    os.environ.pop("AION_MODEL", None)
    os.environ.pop("AION_PROFILE", None)
    set_active_profile(None)
