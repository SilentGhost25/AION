"""
AION Emergency Mode Configuration
==================================
Ultra-lightweight local-only configuration.
Production Model: qwen2.5:7b — No silent downgrade.
Even in emergency mode, the production model is used. Resource constraints
are handled via smaller context/chunk settings, NOT model downgrade.

Per AION Development Context:
- No silent model switching
- No automatic downgrade
- Emergency mode reduces workload, not model size
"""

from core.config.production_model import get_production_model

EMERGENCY_CONFIG = {
    # Production model even in emergency — constraints handled via context limits
    "primary_model":   get_production_model(),
    "fallback_model":  None,  # No fallback — one model only (fail loud)

    # Aggressive memory limits (emergency = smaller workload, same model)
    "max_context_words":     200,   # Cut in half
    "max_output_tokens":     128,   # Fast prediction
    "chunk_size_words":      300,   # Small sentence chunks
    "questions_per_module":  1,     # Generate less per pass

    # Fast timeouts — fail fast
    "request_timeout":       20,    # Hard limit
    "token_timeout":         5,     # No token for 5s = abort

    # Sequential processing ONLY
    "parallel_generation":   False,
    "batch_size":            1,

    # Unload model after EVERY generation
    "unload_after_each":     True,

    # Disable heavy features
    "visual_rag":            False,  # Skip figures
    "critic_enabled":        False,  # Skip validation
    "answer_generation":     False,  # Questions only

    # Ollama keep_alive
    "keep_alive":            0,      # Unload immediately
}
