# core/extraction/block_classifier.py

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

LOG = logging.getLogger("aion.extraction.block_classifier")


class BlockRole(str, Enum):
    """
    10-Role Taxonomy for content-aware block classification.
    Distinguishes genuine academic body prose from scaffolding, metadata,
    exercises, and administrative boilerplate.
    """
    BODY = "BODY"
    HEADING = "HEADING"
    CAPTION = "CAPTION"
    LIST_ITEM = "LIST_ITEM"
    METADATA = "METADATA"
    ADMIN = "ADMIN"
    BIBLIOGRAPHY = "BIBLIOGRAPHY"
    TOC = "TOC"
    EXERCISE = "EXERCISE"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassifiedBlock:
    block_id: str
    text: str
    role: BlockRole
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    provisional_topic_eligible: bool = False


class ContentAwareBlockClassifier:
    """
    Multi-class classifier categorizing text blocks into the 10-role taxonomy
    using lexical, structural, and contextual window features [e_{t-1}, e_t, e_{t+1}].
    """

    CONFIDENCE_THRESHOLD = 0.70
    STARVATION_THRESHOLD = 5
    STARVATION_MIN_CHARS = 80
    DRIFT_ALERT_RATIO = 0.25

    # Compiled patterns for lexical role signaling
    _ADMIN_PATTERNS = [
        re.compile(r'\b(?:answer\s+(?:any\s+)?five|max(?:imum)?\s+marks|duration\s*:\s*\d+|usn\b|internal\s+assessment|continuous\s+internal|semester\s+end|visvesvaraya|scheme\s+of\s+evaluation|course\s+outcomes?|passing\s+standard|model\s+question\s+paper)\b', re.IGNORECASE),
        re.compile(r'\b(?:signature\s+of|invigilator|dean|hod|iqac|academic\s+year\s+\d{4})\b', re.IGNORECASE),
    ]

    _METADATA_PATTERNS = [
        re.compile(r'\b(?:author|dr\.|prof\.|isbn[-:\s0-9x]+|copyright|\(c\)\s*\d{4}|all\s+rights\s+reserved|published\s+by|publisher\b|circulated\s+exclusively|course\s+code\s*:\s*[a-z0-9]+)\b', re.IGNORECASE),
    ]

    _EXERCISE_PATTERNS = [
        re.compile(r'^(?:draw\s+figure|reproduce\s+figure|review\s+questions?|exercises?|practice\s+problems?|discussion\s+prompt|laboratory\s+exercise|self[- ]study|self[- ]assessment)\b', re.IGNORECASE),
        re.compile(r'\b(?:draw|reproduce)\s+figure\s+\d+[\.\d]*\b', re.IGNORECASE),
    ]

    _BIBLIO_PATTERNS = [
        re.compile(r'^\s*\[\d+\]\s+[A-Z]', re.IGNORECASE),
        re.compile(r'\b(?:ieee\s+trans|proceedings\s+of|springer\s+nature|acm\s+press|wiley|doi:\s*10\.\d+|http[s]?://\S+)\b', re.IGNORECASE),
    ]

    _TOC_PATTERNS = [
        re.compile(r'\b(?:table\s+of\s+contents|appendix\s+[a-z]|glossary\s+of|index\s+of\s+terms)\b', re.IGNORECASE),
        re.compile(r'\.{4,}\s*(?:page\s*)?\d+$', re.IGNORECASE),
    ]

    _CAPTION_PATTERNS = [
        re.compile(r'^(?:figure|fig\.|table|chart|graph)\s+\d+[\.\d]*\s*[:\-\.]', re.IGNORECASE),
    ]

    _LIST_PATTERNS = [
        re.compile(r'^(?:[-*•]|\(?\d+[\.\)]|\(?[a-z][\.\)]|\(?[ivx]+[\.\)])\s+', re.IGNORECASE),
    ]

    _HEADING_PATTERNS = [
        re.compile(r'^(?:chapter|module|unit|section|part)\s+[0-9ivx]+[:\s\-]', re.IGNORECASE),
        re.compile(r'^\d+\.\d+(?:\.\d+)?\s+[A-Z]', re.IGNORECASE),
    ]

    _NOISE_PATTERNS = [
        re.compile(r'^[\s\x00-\x1f\x7f-\x9f\ufffd]+$'),
        re.compile(r'^[_\-\.\=\|\*\~\#\s]{5,}$'),
    ]

    @classmethod
    def classify_block_single(cls, text: str) -> Tuple[BlockRole, float]:
        """Classify a single raw text block without window context."""
        cleaned = text.strip()
        if not cleaned:
            return BlockRole.UNKNOWN, 0.0

        if any(p.search(cleaned) for p in cls._NOISE_PATTERNS):
            return BlockRole.UNKNOWN, 0.95

        # Check explicit admin tokens
        if any(p.search(cleaned) for p in cls._ADMIN_PATTERNS):
            return BlockRole.ADMIN, 0.90

        # Check metadata
        if any(p.search(cleaned) for p in cls._METADATA_PATTERNS):
            return BlockRole.METADATA, 0.88

        # Check exercise headers
        if any(p.search(cleaned) for p in cls._EXERCISE_PATTERNS):
            return BlockRole.EXERCISE, 0.92

        # Check bibliography
        if any(p.search(cleaned) for p in cls._BIBLIO_PATTERNS):
            return BlockRole.BIBLIOGRAPHY, 0.90

        # Check TOC
        if any(p.search(cleaned) for p in cls._TOC_PATTERNS):
            return BlockRole.TOC, 0.88

        # Check Caption
        if any(p.search(cleaned) for p in cls._CAPTION_PATTERNS):
            return BlockRole.CAPTION, 0.92

        # Check List Item
        if any(p.search(cleaned) for p in cls._LIST_PATTERNS):
            return BlockRole.LIST_ITEM, 0.85

        # Check Heading
        if any(p.search(cleaned) for p in cls._HEADING_PATTERNS) or (len(cleaned.split()) <= 8 and cleaned.isupper() and len(cleaned) >= 5):
            return BlockRole.HEADING, 0.85

        # Default to BODY if substantive words and sentences
        word_count = len(cleaned.split())
        if word_count >= 10:
            return BlockRole.BODY, 0.85
        elif word_count >= 4:
            return BlockRole.BODY, 0.72
        else:
            return BlockRole.UNKNOWN, 0.60

    @classmethod
    def classify_blocks(cls, blocks: List[Any]) -> List[ClassifiedBlock]:
        """
        Classifies blocks using sliding window context [e_{t-1}, e_t, e_{t+1}].
        """
        classified: List[ClassifiedBlock] = []
        n = len(blocks)

        for i in range(n):
            block = blocks[i]
            if isinstance(block, dict):
                block_id = str(block.get("block_id", block.get("id", f"block_{i+1}")))
                text = str(block.get("text", ""))
            else:
                block_id = str(getattr(block, "block_id", getattr(block, "id", f"block_{i+1}")))
                text = str(getattr(block, "text", str(block)))

            raw_role, raw_conf = cls.classify_block_single(text)

            # Contextual feature augmentation window [e_{t-1}, e_t, e_{t+1}]
            if i > 0:
                p = blocks[i - 1]
                prev_text = p.get("text", "") if isinstance(p, dict) else str(getattr(p, "text", str(p)))
            else:
                prev_text = ""

            if i < n - 1:
                nxt = blocks[i + 1]
                next_text = nxt.get("text", "") if isinstance(nxt, dict) else str(getattr(nxt, "text", str(nxt)))
            else:
                next_text = ""

            final_role = raw_role
            final_conf = raw_conf

            # If previous was TOC and current matches TOC numbering -> reinforce TOC
            if i > 0 and classified[i - 1].role == BlockRole.TOC and re.search(r'\d+$', text.strip()):
                final_role = BlockRole.TOC
                final_conf = max(final_conf, 0.85)

            # If previous was BIBLIOGRAPHY and current starts with citation bracket -> reinforce BIBLIOGRAPHY
            if i > 0 and classified[i - 1].role == BlockRole.BIBLIOGRAPHY and text.strip().startswith("["):
                final_role = BlockRole.BIBLIOGRAPHY
                final_conf = max(final_conf, 0.90)

            # If confidence is below threshold, categorize as UNKNOWN
            if final_conf < cls.CONFIDENCE_THRESHOLD:
                final_role = BlockRole.UNKNOWN

            classified.append(ClassifiedBlock(
                block_id=block_id,
                text=text,
                role=final_role,
                confidence=final_conf,
                features={
                    "raw_role": raw_role.value,
                    "word_count": len(text.split()),
                    "has_prev": bool(prev_text),
                    "has_next": bool(next_text),
                }
            ))

        return classified

    @classmethod
    def handle_starvation(cls, blocks: List[ClassifiedBlock], module_id: int = 1) -> List[ClassifiedBlock]:
        """
        Concrete starvation fallback (R6):
        If < 5 BODY blocks exist in a module:
        - Select blocks with role in {BODY, LIST_ITEM} having length > 80 chars.
        - Mark them PROVISIONAL_TOPIC_ELIGIBLE.
        - Log [CLASSIFIER_STARVATION] with block IDs for human review routing.
        """
        body_blocks = [b for b in blocks if b.role == BlockRole.BODY]
        if len(body_blocks) >= cls.STARVATION_THRESHOLD:
            return blocks

        provisional_ids = []
        for b in blocks:
            if b.role in (BlockRole.BODY, BlockRole.LIST_ITEM) and len(b.text.strip()) >= cls.STARVATION_MIN_CHARS:
                b.provisional_topic_eligible = True
                provisional_ids.append(b.block_id)

        LOG.warning(
            f"[CLASSIFIER_STARVATION] module_{module_id}: Only {len(body_blocks)} BODY blocks identified (< {cls.STARVATION_THRESHOLD}). "
            f"Activated concrete starvation fallback on {len(provisional_ids)} blocks {provisional_ids[:10]}. Routed for review."
        )
        return blocks

    @classmethod
    def check_data_drift(cls, blocks: List[ClassifiedBlock]) -> Dict[str, Any]:
        """
        Drift Telemetry (R8):
        Computes UNKNOWN block distribution. If UNKNOWN blocks exceed 25%:
        - Triggers [DATA_DRIFT_ALERT]
        - Auto-disables TOPIC_VALIDATOR to prevent total pipeline blockage.
        """
        if not blocks:
            return {"drift_detected": False, "unknown_ratio": 0.0, "disable_topic_validator": False}

        unknown_count = sum(1 for b in blocks if b.role == BlockRole.UNKNOWN)
        total = len(blocks)
        unknown_ratio = unknown_count / total

        drift_detected = unknown_ratio > cls.DRIFT_ALERT_RATIO
        if drift_detected:
            LOG.error(
                f"[DATA_DRIFT_ALERT] UNKNOWN blocks ratio is {unknown_ratio:.1%} "
                f"({unknown_count}/{total} blocks), exceeding threshold {cls.DRIFT_ALERT_RATIO:.1%}. "
                f"Auto-disabling TOPIC_VALIDATOR safety gate."
            )

        return {
            "drift_detected": drift_detected,
            "unknown_ratio": round(unknown_ratio, 3),
            "unknown_count": unknown_count,
            "total_blocks": total,
            "disable_topic_validator": drift_detected
        }
