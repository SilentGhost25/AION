# core/generation/topic_validator.py

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("aion.generation.topic_validator")


@dataclass
class TopicValidationResult:
    is_valid: bool
    score: float
    reason: str
    assigned_domain: str
    relative_margin: float
    is_borderline: bool = False
    requires_human_review: bool = False


class MultiDomainTopicValidator:
    """
    Multi-Domain Topic Validity Gate (Phase C.4).
    Validates candidate topics against course taxonomy, enforcing a relative
    margin delta = 0.15 over foreign domains and eliminating administrative/metadata leakage.
    """

    DEFAULT_DELTA = 0.15
    DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "config" / "course_taxonomy.json"
    DEFAULT_THRESHOLDS_PATH = Path("tests/fixtures/thresholds.json")

    def __init__(self, taxonomy_path: Optional[Path] = None, delta_margin: Optional[float] = None):
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else self.DEFAULT_TAXONOMY_PATH
        self.taxonomy: Dict[str, Any] = self._load_taxonomy()
        
        # Load calibrated delta if available
        if delta_margin is not None:
            self.delta = delta_margin
        else:
            self.delta = self._load_calibrated_delta()

    def _load_taxonomy(self) -> Dict[str, Any]:
        if self.taxonomy_path.exists():
            try:
                with open(self.taxonomy_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("domains", {})
            except Exception as e:
                LOG.warning(f"[TOPIC_VALIDATOR] Failed to load taxonomy from {self.taxonomy_path}: {e}")
        return {}

    def _load_calibrated_delta(self) -> float:
        if self.DEFAULT_THRESHOLDS_PATH.exists():
            try:
                with open(self.DEFAULT_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("parameters", {}).get("delta_margin", self.DEFAULT_DELTA))
            except Exception:
                pass
        return self.DEFAULT_DELTA

    def _score_domain_overlap(self, text: str, domain_keywords: List[str]) -> float:
        """Compute keyword hit density against domain taxonomy."""
        if not text or not domain_keywords:
            return 0.0
        words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text.lower()))
        if not words:
            return 0.0

        overlap = sum(1 for k in domain_keywords if k.lower() in text.lower() or any(w.startswith(k.lower()) for w in words))
        # Normalized score
        return min(1.0, overlap / max(1, len(words) * 0.5))

    def validate_topic(
        self,
        candidate_topic: str,
        current_domain: str = "IOT_AGRICULTURE",
        drift_active: bool = False
    ) -> TopicValidationResult:
        """
        Validates candidate topic against taxonomy and anti-leakage rules.
        """
        # Feature flag check: canary-safe default false
        enabled = os.getenv("AION_ENABLE_TOPIC_VALIDATOR", "false").lower() in ("true", "1", "yes")
        if not enabled:
            return TopicValidationResult(
                is_valid=True,
                score=1.0,
                reason="Topic validator disabled via AION_ENABLE_TOPIC_VALIDATOR flag.",
                assigned_domain=current_domain,
                relative_margin=1.0,
                is_borderline=False
            )

        if drift_active:
            LOG.warning("[TOPIC_VALIDATOR] Drift active: bypassing foreign domain hard block to preserve availability.")
            return TopicValidationResult(
                is_valid=True,
                score=0.75,
                reason="Bypassed due to data drift safety mechanism.",
                assigned_domain=current_domain,
                relative_margin=0.0,
                is_borderline=True,
                requires_human_review=True
            )

        text = candidate_topic.strip()
        if not text:
            return TopicValidationResult(
                is_valid=False,
                score=0.0,
                reason="Empty topic string.",
                assigned_domain="UNKNOWN",
                relative_margin=-1.0
            )

        # 1. Hard lexical filters (Exercise headers, Figure reproductions, Admin tokens, URLs, Syllabi)
        bad_patterns = [
            r'\b(?:draw|reproduce)\s+figure\b',
            r'\b(?:diagram\s+and\s+formula|revision\s+(?:list|points?|checklist))\b',
            r'\b(?:review\s+questions?|exercises?|practice\s+problems?|discussion\s+prompt|laboratory\s+exercise|self[- ]study|self[- ]assessment)\b',
            r'\b(?:expected\s+learning\s+outcomes?|additional\s+knowledge)\b',
            r'\b(?:usn\b|course\s+code\s*:|all\s+rights\s+reserved|isbn[-:\s0-9x]+|copyright|\(c\)\s*\d{4})\b',
            r'\b(?:table\s+of\s+contents|dr\.|prof\.|author\s*:|published\s+by|publisher\b)\b',
            r'\b(?:visvesvaraya|technological\s+university|belagavi|vtu\b)\b',
            r'\b(?:answer\s+(?:any\s+)?five|max(?:imum)?\s+marks|duration\s*:\s*\d+|model\s+question\s+paper)\b',
            r'\b(?:internal\s+assessment|continuous\s+internal|semester\s+end|passing\s+standard)\b',
            r'\b(?:signature\s+of|invigilator|dean|hod|iqac|academic\s+year|scheme\s+of\s+evaluation)\b',
            r'\b(?:https?://|www\.)\S+\b',
            r'\[\d+\]\s+[A-Z]',
            r'\b(?:chapter|module|unit|section|part)\s+[0-9ivx]+\b',
            r'\b(?:figure|fig\.|table|chart|graph)\s+\d+[\.\d]*\s*[:\-\.]',
            r'\b(?:examination[- ]oriented|consolidated\s+review|m\.tech|ph\.d)\b',
        ]
        for pat in bad_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return TopicValidationResult(
                    is_valid=False,
                    score=0.0,
                    reason=f"Topic matches invalid metadata or exercise pattern: '{pat}'",
                    assigned_domain="ADMIN_METADATA",
                    relative_margin=-1.0
                )

        # 2. Check administrative taxonomy overlap
        admin_keywords = self.taxonomy.get("ADMIN_METADATA", {}).get("keywords", [])
        admin_score = self._score_domain_overlap(text, admin_keywords)
        if admin_score >= 0.50:
            return TopicValidationResult(
                is_valid=False,
                score=0.1,
                reason=f"High administrative/metadata taxonomy match ({admin_score:.2f}).",
                assigned_domain="ADMIN_METADATA",
                relative_margin=-admin_score
            )

        # 3. Domain scoring and relative margin calculation
        target_info = self.taxonomy.get(current_domain, {})
        target_keywords = target_info.get("keywords", [])
        target_score = self._score_domain_overlap(text, target_keywords)

        foreign_scores: Dict[str, float] = {}
        for dom, dom_info in self.taxonomy.items():
            if dom not in (current_domain, "ADMIN_METADATA"):
                foreign_scores[dom] = self._score_domain_overlap(text, dom_info.get("keywords", []))

        max_foreign_domain = max(foreign_scores, key=foreign_scores.get) if foreign_scores else "NONE"
        max_foreign_score = foreign_scores.get(max_foreign_domain, 0.0)

        relative_margin = target_score - max_foreign_score

        # If foreign domain strongly dominates by more than delta margin
        if max_foreign_score > 0.60 and relative_margin < -self.delta:
            return TopicValidationResult(
                is_valid=False,
                score=target_score,
                reason=f"Topic belongs to foreign domain '{max_foreign_domain}' (score={max_foreign_score:.2f} vs target={target_score:.2f}, margin={relative_margin:.2f} < -{self.delta}).",
                assigned_domain=max_foreign_domain,
                relative_margin=relative_margin
            )

        # 4. Borderline classification (U6: enforce <= 20% acceptance rate)
        words = text.split()
        borderline_keywords = [
            "overview", "summary", "introduction", "case study", "general",
            "comparison", "challenges", "advantages", "disadvantages", "benefits",
            "drawbacks", "steps", "survey", "list of", "checklist", "interfaces"
        ]
        is_borderline = (
            (len(words) <= 2)
            or any(re.search(rf'\b{re.escape(w)}\b', text, re.IGNORECASE) for w in borderline_keywords)
        )

        if is_borderline:
            # Enforce strict <= 20% acceptance; only pass if highly grounded in target domain
            if target_score >= 0.70:
                return TopicValidationResult(
                    is_valid=True,
                    score=target_score,
                    reason="Borderline topic accepted conditionally under high domain specificity.",
                    assigned_domain=current_domain,
                    relative_margin=relative_margin,
                    is_borderline=True,
                    requires_human_review=True
                )
            else:
                return TopicValidationResult(
                    is_valid=False,
                    score=target_score,
                    reason="Topic classified as borderline without sufficient technical specificity.",
                    assigned_domain=current_domain,
                    relative_margin=relative_margin,
                    is_borderline=True,
                    requires_human_review=True
                )

        return TopicValidationResult(
            is_valid=True,
            score=max(target_score, 0.75),
            reason="Topic passes domain validity gate.",
            assigned_domain=current_domain if target_score >= max_foreign_score else max_foreign_domain,
            relative_margin=relative_margin,
            is_borderline=False,
            requires_human_review=False
        )
