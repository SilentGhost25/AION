"""
AION Emergency Mode Configuration
==================================
Ultra-lightweight local-only configuration.
Designed to work on 16GB RAM systems with zero cloud dependencies.
"""

EMERGENCY_CONFIG = {
    # Use ONLY the smallest model
    "primary_model":   "qwen2.5:1.5b",
    "fallback_model":  None,  # No fallback — one model only

    # Aggressive memory limits
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
