"""
AION Production Server Configuration
====================================
Server deployment settings for Ollama parallel limits, model fallback policies, and device profile.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    device: str = os.environ.get("AION_DEVICE", "server")
    model: str = os.environ.get("AION_MODEL", "qwen2.5:14b")
    allow_model_fallback: bool = os.environ.get("AION_ALLOW_MODEL_FALLBACK", "false").lower() == "true"
    ollama_num_parallel: int = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1"))
    ollama_max_loaded_models: int = int(os.environ.get("OLLAMA_MAX_LOADED_MODELS", "1"))



SERVER_CONFIG = ServerConfig()


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check whether a local TCP port is already in use by another process.

    Returns:
        True if the port is in use (EADDRINUSE / WinError 10048).
        False if the port is available to bind.

    Raises:
        OSError: If a non-EADDRINUSE socket error occurs (e.g. permission denied).
    """
    import errno
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, int(port)))
            return False
        except OSError as e:
            # Differentiate port in use (EADDRINUSE / WSAEADDRINUSE 10048) from permission denied (EACCES 10013)
            addr_in_use_errs = {getattr(errno, "EADDRINUSE", 10048), 10048}
            if e.errno in addr_in_use_errs or getattr(e, "winerror", None) == 10048:
                return True
            # Raise other errors (e.g. permission denied) rather than silently treating as in-use
            raise


def find_available_port(
    start_port: int = 8100, max_attempts: int = 50, host: str = "127.0.0.1"
) -> int:
    """Find the first available TCP port starting from start_port.

    Only increments if the port is genuinely occupied by another process.
    Fails fast if a permission or system configuration error occurs.
    """
    start_port = int(start_port)
    for offset in range(max_attempts):
        port = start_port + offset
        try:
            if not is_port_in_use(port, host):
                return port
        except OSError as err:
            raise RuntimeError(
                f"Cannot bind to port {port} due to system/socket error: {err}"
            ) from err
    raise RuntimeError(
        f"No available port found in range {start_port}..{start_port + max_attempts - 1}"
    )
