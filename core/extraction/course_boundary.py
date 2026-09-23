# core/extraction/course_boundary.py

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("aion.extraction.course_boundary")


class CrossCourseBoundaryDetector:
    """
    Detects foreign curriculum materials accidentally concatenated into notes or textbooks.
    Per U1 and R5: An exclusion requires BOTH centroid divergence AND an explicit
    foreign course marker (mismatched degree code, foreign syllabus header).
    """

    FOREIGN_MARKER_TEMPLATE = r'\b(?:Course|Subject)\s+(?:Code|Title)\s*:\s*(?!{})[A-Z0-9]{{5,}}\b'

    @classmethod
    def extract_course_codes(cls, text: str) -> List[str]:
        """Find course codes like 21CS72, BIC703, 18CS54 in text."""
        pattern = re.compile(r'\b(?:[0-9]{2}[A-Z]{2,4}[0-9]{2,3}|[A-Z]{2,4}[0-9]{3})\b')
        return pattern.findall(text)

    @classmethod
    def check_foreign_boundary(
        cls,
        chunk_text: str,
        current_course_code: Optional[str] = None,
        subject_keywords: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates whether a chunk belongs to an extraneous course.
        Returns:
            {
                "is_foreign": bool,
                "has_foreign_marker": bool,
                "has_centroid_divergence": bool,
                "detected_code": Optional[str],
                "reason": str
            }
        """
        cleaned = chunk_text.strip()
        current_code = (current_course_code or "").strip().upper()

        # 1. Foreign Course Marker Check (Per R5)
        has_foreign_marker = False
        detected_foreign_code = None

        if current_code:
            escaped_code = re.escape(current_code)
            marker_regex = re.compile(cls.FOREIGN_MARKER_TEMPLATE.format(escaped_code), re.IGNORECASE)
            match = marker_regex.search(cleaned)
            if match:
                has_foreign_marker = True
                detected_foreign_code = match.group(0)
        else:
            # Fallback if current_code absent: fail-open on marker check (R5 guidance)
            has_foreign_marker = False

        # 2. Vocabulary Centroid Divergence
        has_centroid_divergence = False
        if subject_keywords:
            words = set(re.findall(r'\b[a-zA-Z]{4,}\b', cleaned.lower()))
            overlap = words & {k.lower() for k in subject_keywords}
            # If chunk is substantial (> 50 words) and shares zero subject keywords
            if len(words) > 50 and len(overlap) == 0:
                has_centroid_divergence = True

        # Rule U1: Exclusion strictly requires BOTH signals
        is_foreign = has_foreign_marker and has_centroid_divergence

        reason = "Clean within-course content"
        if is_foreign:
            reason = f"Cross-course contamination: foreign marker '{detected_foreign_code}' with high vocabulary divergence."
        elif has_foreign_marker:
            reason = f"Foreign marker detected ({detected_foreign_code}) but within-domain vocabulary overlap found; preserved."
        elif has_centroid_divergence:
            reason = "Vocabulary divergence detected without foreign course marker; preserved to prevent false positive."

        return {
            "is_foreign": is_foreign,
            "has_foreign_marker": has_foreign_marker,
            "has_centroid_divergence": has_centroid_divergence,
            "detected_code": detected_foreign_code,
            "reason": reason
        }
