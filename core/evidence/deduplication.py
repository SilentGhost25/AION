"""
AION Core Evidence — Multi-level Evidence Deduplication
========================================================
Prevents PyMuPDF and Docling double-counting identical content
as specified in Part VII of the Production Hardening Specification.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, List, Set, Tuple

logger = logging.getLogger("AION.EvidenceDeduplicator")


class EvidenceDeduplicator:
    """Deduplicates evidence chunks to prevent content duplication and word count inflation."""

    @classmethod
    def deduplicate(cls, chunks: List[Any]) -> List[Any]:
        if not chunks:
            return []

        input_count = len(chunks)
        seen_hashes: Set[str] = set()
        dedup_exact: List[Any] = []
        exact_removed = 0

        # STEP 1 — EXACT DUPLICATES BY SHA256 OF NORMALIZED TEXT
        for chunk in chunks:
            text = getattr(chunk, "text", "")
            norm_text = re.sub(r"\s+", " ", text).strip().lower()
            text_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

            if text_hash in seen_hashes:
                exact_removed += 1
                continue

            seen_hashes.add(text_hash)
            dedup_exact.append(chunk)

        # STEP 2 — NEAR-DUPLICATES (SAME PAGE & OVERLAPPING BBOX)
        final_output: List[Any] = []
        near_removed = 0

        page_groups: dict = {}
        for chunk in dedup_exact:
            p = getattr(chunk, "page_start", getattr(chunk, "page", 1))
            page_groups.setdefault(p, []).append(chunk)

        for p, p_chunks in page_groups.items():
            kept_in_page: List[Any] = []
            for c in p_chunks:
                c_bbox = getattr(c, "bbox", None)
                c_conf = getattr(c, "confidence", 1.0)
                duplicate_found = False

                if c_bbox and len(c_bbox) == 4:
                    for existing in kept_in_page:
                        e_bbox = getattr(existing, "bbox", None)
                        e_conf = getattr(existing, "confidence", 1.0)
                        if e_bbox and len(e_bbox) == 4 and cls._bbox_overlaps(c_bbox, e_bbox):
                            duplicate_found = True
                            near_removed += 1
                            # Keep higher confidence
                            if c_conf > e_conf:
                                kept_in_page.remove(existing)
                                kept_in_page.append(c)
                            break

                if not duplicate_found:
                    kept_in_page.append(c)

            final_output.extend(kept_in_page)

        logger.info(f"[DEDUP] Input chunks  : {input_count}")
        logger.info(f"[DEDUP] Exact dups    : {exact_removed}")
        logger.info(f"[DEDUP] Near dups     : {near_removed}")
        logger.info(f"[DEDUP] Output chunks : {len(final_output)}")

        return final_output

    @classmethod
    def _bbox_overlaps(cls, b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> bool:
        """Returns True if two bboxes (x0, y0, x1, y1) overlap significantly (>50%)."""
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        min_area = min(area1, area2)

        return (intersection_area / min_area) > 0.50 if min_area > 0 else False
