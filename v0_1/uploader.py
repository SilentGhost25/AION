"""
AION Module: Uploader
Maturity:    v0.1 — LOCAL FILE VALIDATOR
Upgrades to: Async Multi-Source Ingestion Gateway (S3/HTTP/WebSockets)
Contract:    file_path: str -> str (validated path)
"""

from pathlib import Path

def upload(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path}")
    return str(path.resolve())
