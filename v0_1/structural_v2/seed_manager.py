"""
AION Structural Architecture v2 — Seed Manager & Deterministic Reproducibility
================================================================================
Guarantees 100% paper reproducibility from a single integer seed using HMAC-SHA256
domain-separated seed derivation.
"""

from __future__ import annotations

import hashlib
import hmac
import random
from typing import Dict


class SeedManager:
    """Master and per-slot seed derivation using HMAC-SHA256."""

    @staticmethod
    def hash_seed(master_seed: int, namespace: str) -> int:
        """Derive an integer seed for a given namespace using HMAC-SHA256."""
        key = str(master_seed).encode("utf-8")
        msg = namespace.encode("utf-8")
        raw = hmac.new(key, msg, hashlib.sha256).digest()
        return int.from_bytes(raw[:8], byteorder="big")

    @classmethod
    def derive_seeds(cls, master_seed: int) -> Dict[str, int]:
        """Derive domain-separated sub-seeds from master seed."""
        return {
            "structure": cls.hash_seed(master_seed, ":structure"),
            "bloom": cls.hash_seed(master_seed, ":bloom"),
            "content": cls.hash_seed(master_seed, ":content"),
            "visual": cls.hash_seed(master_seed, ":visual"),
            "template": cls.hash_seed(master_seed, ":template"),
        }

    @classmethod
    def slot_seed(cls, master_seed: int, slot_id: str) -> int:
        """Derive a unique, reproducible integer seed for a specific question slot."""
        return cls.hash_seed(master_seed, f":slot:{slot_id}")

    @classmethod
    def get_rng(cls, seed: int) -> random.Random:
        """Return a seeded Random instance."""
        return random.Random(seed)
