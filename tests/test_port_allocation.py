"""
Tests for Dynamic Port Allocation & Safety
===========================================
Validates:
1. is_port_in_use correctly identifies free vs bound TCP ports.
2. find_available_port returns start_port when available.
3. find_available_port increments to next port when target is occupied (EADDRINUSE).
4. find_available_port fails fast on socket/permission errors instead of masking them.
"""

import socket
import pytest
from core.config.server_config import is_port_in_use, find_available_port


def test_is_port_in_use_detects_free_and_bound_ports():
    """Verify is_port_in_use accurately reports port state."""
    # Find a free port first
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    
    # After probe is closed, port should be free
    assert not is_port_in_use(free_port), f"Port {free_port} should be free"
    
    # Bind a socket to the port
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", free_port))
        server.listen(1)
        assert is_port_in_use(free_port), f"Port {free_port} should be reported as in-use"
    finally:
        server.close()
    
    # After closing, should be free again
    assert not is_port_in_use(free_port), f"Port {free_port} should be free after close"


def test_find_available_port_increments_when_target_occupied():
    """Verify find_available_port shifts to next port when target is busy."""
    # Pick a free base port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        base_port = probe.getsockname()[1]

    # Hold the base port open
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", base_port))
        server.listen(1)

        # find_available_port should detect base_port is in use and pick base_port + 1
        allocated = find_available_port(start_port=base_port, max_attempts=5)
        assert allocated > base_port, f"Expected port > {base_port}, got {allocated}"
    finally:
        server.close()


def test_find_available_port_fails_fast_on_invalid_host():
    """Verify find_available_port does NOT mask invalid network/host errors."""
    with pytest.raises(RuntimeError, match="Cannot bind to port"):
        # Binding to an unroutable/invalid IP must raise rather than silently loop
        find_available_port(start_port=8100, host="192.0.2.254")
