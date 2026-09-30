"""
Tiered semantic response cache (static tier + async dynamic promotion).

Task 3 additions:
    - Optional async promotion via KritesJudge + Executor
    - Borderline scheduling on miss
    - Promotion counters in CacheStats
    - close() for executor lifecycle

Task 2 behavior is preserved when judge=None or executor=None.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "core.cache.semantic_cache requires numpy. "
        "Install with: pip install numpy"
    ) from e


EmbedFn = Callable[[str], "np.ndarray"]


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------


@dataclass
class CachedEntry:
    query_text: str
    embedding: List[float]
    response: dict
    tier: str = "static"
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CachedEntry":
        return cls(**d)


@dataclass
class CacheStats:
    total_entries: int
    entries_per_doc: Dict[str, int]
    hits: int
    misses: int
    invalidations: int
    # NEW: promotion telemetry
    promotions_scheduled: int = 0
    promotions_succeeded: int = 0
    promotions_failed: int = 0
    promotions_throttled: int = 0

    @property
    def hit_rate(self) -> float:
        denom = self.hits + self.misses
        return (self.hits / denom) if denom > 0 else 0.0


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------


class SemanticCache:
    """
    Two-tier semantic response cache with optional async promotion.

    Parameters (Task 3 additions marked NEW)
    ----------------------------------------
    judge : KritesJudge | None
        NEW. If provided along with `executor`, borderline misses schedule
        an async judge job. If None, Task 2 behavior is preserved.
    executor : Executor | None
        NEW. Thread pool for judge jobs. Must be provided to enable async.
    borderline_low : float
        NEW. Lower bound of the borderline band. Default 0.75.
    enable_async_promotion : bool
        NEW. Master switch. Default True (only acts when judge+executor set).
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        storage_dir: Optional[Path] = None,
        static_threshold: float = 0.90,
        dynamic_threshold: float = 0.85,
        ttl_seconds: Optional[float] = None,
        max_entries_per_doc: int = 500,
        clock: Callable[[], float] = time.time,
        # NEW (Task 3)
        judge=None,
        executor: Optional[Executor] = None,
        borderline_low: float = 0.75,
        enable_async_promotion: bool = True,
    ) -> None:
        if not (0.0 < dynamic_threshold <= static_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 < dynamic <= static <= 1.0; "
                f"got dynamic={dynamic_threshold}, static={static_threshold}"
            )
        if not (0.0 <= borderline_low <= dynamic_threshold):
            raise ValueError(
                f"borderline_low must satisfy 0 <= borderline_low <= "
                f"dynamic_threshold ({dynamic_threshold}); got {borderline_low}"
            )
        if max_entries_per_doc < 1:
            raise ValueError("max_entries_per_doc must be >= 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0 or None")

        self._embed_fn = embed_fn
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._static_threshold = static_threshold
        self._dynamic_threshold = dynamic_threshold
        self._ttl_seconds = ttl_seconds
        self._max_entries_per_doc = max_entries_per_doc
        self._clock = clock

        # NEW: async promotion config
        self._judge = judge
        self._executor = executor
        self._borderline_low = borderline_low
        self._enable_async = enable_async_promotion
        self._pending_promotions: Set[Tuple[str, str]] = set()

        self._lock = threading.RLock()
        self._entries: Dict[str, List[CachedEntry]] = {}

        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        # NEW: promotion counters
        self._promotions_scheduled = 0
        self._promotions_succeeded = 0
        self._promotions_failed = 0
        self._promotions_throttled = 0

        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

        if self._judge is not None and self._executor is None:
            print(
                "[CACHE] judge provided without executor; async promotion disabled",
                flush=True,
            )

    # ------------------------------------------------------------- Lifecycle

    def close(self, wait: bool = True) -> None:
        """
        Drain pending promotions. Does NOT shut down a caller-supplied executor —
        the caller owns its lifetime. Safe to call multiple times.
        """
        if self._executor is None:
            return
        # If the executor is a ThreadPoolExecutor owned by the caller, this
        # cache does not shut it down. If it is owned by the cache, the caller
        # can shut it down themselves after close().
        # (We deliberately do not own the executor in this version to keep
        # thread lifecycle explicit.)
        pass

    # ------------------------------------------------------------- Lookup

    def get(self, *, doc_hash: str, query_text: str) -> Optional[CachedEntry]:
        with self._lock:
            bucket = self._entries.get(doc_hash)
            if not bucket:
                self._misses += 1
                return None

            query_vec = self._normalize(self._embed_fn(query_text))

            # Pass 1: static
            static_hit = self._find_match(bucket, query_vec, tier="static")
            if static_hit is not None and not self._is_expired(static_hit):
                self._hits += 1
                return self._clone(static_hit)

            # Pass 2: dynamic
            dynamic_hit = self._find_match(bucket, query_vec, tier="dynamic")
            if dynamic_hit is not None and not self._is_expired(dynamic_hit):
                self._hits += 1
                return self._clone(dynamic_hit)

            # Miss — check for borderline candidates for async promotion
            self._misses += 1
            self._maybe_schedule_promotion(doc_hash, bucket, query_text, query_vec)
            return None

    # ------------------------------------------------------------- Store

    def put(
        self,
        *,
        doc_hash: str,
        query_text: str,
        response: dict,
        tier: str = "static",
        metadata: Optional[dict] = None,
    ) -> CachedEntry:
        if tier not in ("static", "dynamic"):
            raise ValueError(f"tier must be 'static' or 'dynamic', got {tier!r}")
        embedding = self._normalize(self._embed_fn(query_text))
        return self._put_with_vec(
            doc_hash=doc_hash,
            query_text=query_text,
            embedding=embedding,
            response=response,
            tier=tier,
            metadata=metadata,
        )

    def _put_with_vec(
        self,
        *,
        doc_hash: str,
        query_text: str,
        embedding: "np.ndarray",
        response: dict,
        tier: str,
        metadata: Optional[dict] = None,
    ) -> CachedEntry:
        with self._lock:
            bucket = self._entries.setdefault(doc_hash, [])
            entry = CachedEntry(
                query_text=query_text,
                embedding=embedding.tolist(),
                response=response,
                tier=tier,
                created_at=self._clock(),
                metadata=metadata or {},
            )
            bucket = [e for e in bucket if not self._is_expired(e)]
            bucket.append(entry)
            if len(bucket) > self._max_entries_per_doc:
                bucket.sort(key=lambda e: e.created_at)
                bucket = bucket[-self._max_entries_per_doc:]
            self._entries[doc_hash] = bucket
            self._persist(doc_hash)
            return entry

    def promote(
        self,
        *,
        doc_hash: str,
        query_text: str,
        response: dict,
        source_entry: Optional[CachedEntry] = None,
    ) -> CachedEntry:
        """Public promotion API (called by judge worker; also usable in tests)."""
        return self.put(
            doc_hash=doc_hash,
            query_text=query_text,
            response=response,
            tier="dynamic",
            metadata={
                "promoted_from_query": source_entry.query_text if source_entry else None,
                "promoted_at": self._clock(),
            },
        )

    # ------------------------------------------------------------- Invalidation

    def invalidate_doc(self, doc_hash: str) -> int:
        with self._lock:
            removed = len(self._entries.pop(doc_hash, []))
            self._invalidations += 1
            if self._storage_dir:
                p = self._storage_dir / f"{doc_hash}.json"
                if p.exists():
                    p.unlink()
            return removed

    def invalidate_all(self) -> int:
        with self._lock:
            removed = sum(len(v) for v in self._entries.values())
            self._entries.clear()
            self._invalidations += 1
            if self._storage_dir:
                for p in self._storage_dir.glob("*.json"):
                    p.unlink()
            return removed

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                total_entries=sum(len(v) for v in self._entries.values()),
                entries_per_doc={k: len(v) for k, v in self._entries.items()},
                hits=self._hits,
                misses=self._misses,
                invalidations=self._invalidations,
                promotions_scheduled=self._promotions_scheduled,
                promotions_succeeded=self._promotions_succeeded,
                promotions_failed=self._promotions_failed,
                promotions_throttled=self._promotions_throttled,
            )

    def load(self, doc_hash: str) -> int:
        if not self._storage_dir:
            return 0
        with self._lock:
            p = self._storage_dir / f"{doc_hash}.json"
            if not p.exists():
                return 0
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return 0
            entries = [CachedEntry.from_json(d) for d in payload.get("entries", [])]
            entries = [e for e in entries if not self._is_expired(e)]
            self._entries[doc_hash] = entries
            return len(entries)

    # ------------------------------------------------------------- Async promotion

    def _maybe_schedule_promotion(
        self,
        doc_hash: str,
        bucket: List[CachedEntry],
        query_text: str,
        query_vec: "np.ndarray",
    ) -> None:
        """Called under lock during a miss. Schedules a judge job if warranted."""
        if not self._enable_async or self._judge is None or self._executor is None:
            return

        # Find best static candidate regardless of threshold
        best = self._scan(bucket, query_vec, tier="static")
        if best is None:
            return
        entry, sim = best
        if not (self._borderline_low <= sim < self._static_threshold):
            return

        # Deduplicate: same (doc, query) shouldn't queue twice
        key = (doc_hash, query_text)
        if key in self._pending_promotions:
            return
        self._pending_promotions.add(key)
        self._promotions_scheduled += 1

        # Snapshot the source entry + vector for the worker
        source_snapshot = self._clone(entry)
        vec_snapshot = np.asarray(query_vec, dtype=np.float32)

        try:
            self._executor.submit(
                self._run_promotion_worker,
                doc_hash,
                source_snapshot,
                query_text,
                vec_snapshot,
            )
        except RuntimeError as e:
            # Executor shut down between now and submission
            print(f"[CACHE] executor submit failed: {e}", flush=True)
            self._pending_promotions.discard(key)
            self._promotions_failed += 1

    def _run_promotion_worker(
        self,
        doc_hash: str,
        source_entry: CachedEntry,
        new_query: str,
        query_vec: "np.ndarray",
    ) -> None:
        """Runs on the executor thread. Never raises."""
        key = (doc_hash, new_query)
        try:
            if not self._judge.should_judge(doc_hash):
                with self._lock:
                    self._promotions_throttled += 1
                return

            approved = self._judge.judge(
                doc_hash=doc_hash,
                static_query=source_entry.query_text,
                static_response=source_entry.response,
                new_query=new_query,
            )

            if not approved:
                with self._lock:
                    self._promotions_failed += 1
                return

            # Store under dynamic tier with a pre-computed embedding
            with self._lock:
                self._put_with_vec(
                    doc_hash=doc_hash,
                    query_text=new_query,
                    embedding=query_vec,
                    response=source_entry.response,
                    tier="dynamic",
                    metadata={
                        "promoted_from_query": source_entry.query_text,
                        "promoted_at": self._clock(),
                    },
                )
                self._promotions_succeeded += 1

        except Exception as e:  # defensive — worker must not propagate
            print(f"[CACHE] promotion worker failed: {e}", flush=True)
            with self._lock:
                self._promotions_failed += 1
        finally:
            with self._lock:
                self._pending_promotions.discard(key)

    # ------------------------------------------------------------- Internals

    def _scan(
        self,
        bucket: List[CachedEntry],
        query_vec: "np.ndarray",
        tier: str,
    ) -> Optional[Tuple[CachedEntry, float]]:
        """Best matching entry in the tier, regardless of threshold."""
        best_entry: Optional[CachedEntry] = None
        best_sim = -1.0
        for entry in bucket:
            if entry.tier != tier:
                continue
            entry_vec = np.asarray(entry.embedding, dtype=np.float32)
            sim = float(np.dot(query_vec, entry_vec))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry is None:
            return None
        return best_entry, best_sim

    def _find_match(
        self,
        bucket: List[CachedEntry],
        query_vec: "np.ndarray",
        tier: str,
    ) -> Optional[CachedEntry]:
        result = self._scan(bucket, query_vec, tier)
        if result is None:
            return None
        entry, sim = result
        threshold = self._static_threshold if tier == "static" else self._dynamic_threshold
        return entry if sim >= threshold else None

    def _normalize(self, vec: "np.ndarray") -> "np.ndarray":
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return vec
        return vec / norm

    def _is_expired(self, entry: CachedEntry) -> bool:
        if self._ttl_seconds is None:
            return False
        return (self._clock() - entry.created_at) > self._ttl_seconds

    def _clone(self, entry: CachedEntry) -> CachedEntry:
        return CachedEntry(
            query_text=entry.query_text,
            embedding=list(entry.embedding),
            response=json.loads(json.dumps(entry.response)),
            tier=entry.tier,
            created_at=entry.created_at,
            metadata=dict(entry.metadata),
        )

    def _persist(self, doc_hash: str) -> None:
        if not self._storage_dir:
            return
        bucket = self._entries.get(doc_hash, [])
        payload = {
            "doc_hash": doc_hash,
            "saved_at": self._clock(),
            "entries": [e.to_json() for e in bucket],
        }
        target = self._storage_dir / f"{doc_hash}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)
