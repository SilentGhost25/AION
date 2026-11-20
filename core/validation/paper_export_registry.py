"""
AION Paper Export Registry
==========================
Server-authoritative registry and canonical digest engine for physical document exports (DOCX / PDF).
Enforces Tier 1 authorization: no paper is exportable unless a trusted server-side generation
record exists, is unexpired, and matches the submitted export payload digest exactly.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

# Valid paper ID regex: alphanumeric, underscores, hyphens, 8-64 characters
PAPER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")

# Default TTL: 24 hours (86400 seconds)
DEFAULT_REGISTRY_TTL_SECONDS = 86400

# Registry lookup status codes
STATUS_VALID = "VALID"
STATUS_MISSING = "MISSING"
STATUS_EXPIRED = "EXPIRED"
STATUS_CORRUPTED = "CORRUPTED"
STATUS_ERROR = "ERROR"


class RegistryError(Exception):
    """Base exception for PaperExportRegistry failures."""
    pass


class RegistryCollisionError(RegistryError):
    """Raised when an attempt is made to overwrite an existing paper ID."""
    pass


class RegistryPersistenceError(RegistryError):
    """Raised when a registry record cannot be atomically persisted to disk."""
    pass


@dataclass
class PaperExportRecord:
    paper_id: str
    canonical_hash: str
    export_eligible: bool
    paper_status: str
    created_at: str
    expires_at: str
    blocked_reasons: List[str] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)
    qa_report_summary: Dict[str, Any] = field(default_factory=dict)
    subject: str = ""
    exam_type: str = ""
    total_marks: int = 0
    module_count: int = 0
    question_count: int = 0

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            exp = datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return now > exp
        except Exception:
            return True


def generate_paper_id(prefix: str = "paper") -> str:
    """Generates a secure server-controlled paper ID: paper_{YYYYMMDD_HHMMSS}_{hex8}."""
    utc_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand_suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{utc_str}_{rand_suffix}"


def validate_paper_id(paper_id: Any) -> bool:
    """Strictly validates paper ID syntax and prevents path traversal."""
    if not isinstance(paper_id, str):
        return False
    if not paper_id.strip() or len(paper_id) < 8 or len(paper_id) > 64:
        return False
    if ".." in paper_id or "/" in paper_id or "\\" in paper_id or "\x00" in paper_id:
        return False
    return bool(PAPER_ID_REGEX.match(paper_id))


def canonicalize_paper_payload(paper_data: Dict[str, Any]) -> str:
    """
    Computes an order-sensitive SHA-256 canonical digest over all rendered fields.
    Preserves exact sequence of modules, questions, and subquestions as submitted.
    Altering any rendered header, question text, marks, CO, Bloom, or ordering alters the digest.
    """
    if not isinstance(paper_data, dict):
        raise ValueError("paper_data must be a dictionary")

    config = paper_data.get("config") if isinstance(paper_data.get("config"), dict) else {}

    # 1. Header & Institutional Metadata
    header_tuple = (
        str(config.get("subjectCode") or paper_data.get("subject_code") or "").strip().upper(),
        str(config.get("subjectName") or paper_data.get("subject") or paper_data.get("subject_name") or "").strip(),
        str(config.get("examType") or paper_data.get("exam_type") or "").strip().upper(),
        str(config.get("department") or paper_data.get("department") or "").strip(),
        str(config.get("semester") or paper_data.get("semester") or "").strip(),
        int(config.get("maxMarks") or paper_data.get("total_marks") or paper_data.get("totalMarks") or 0),
        str(config.get("duration") or paper_data.get("duration") or "").strip(),
    )

    # 2. Ordered Modules & Questions
    modules_raw = paper_data.get("modules") or []
    if not isinstance(modules_raw, list):
        modules_raw = []

    canonical_modules = []
    for mod_seq, mod in enumerate(modules_raw):
        if not isinstance(mod, dict):
            continue

        mod_title = str(mod.get("module_title") or mod.get("title") or "").strip()
        try:
            m_idx = int(mod.get("module_index") if mod.get("module_index") is not None else (mod_seq + 1))
        except (ValueError, TypeError):
            m_idx = mod_seq + 1

        questions_raw = mod.get("questions") or []
        if not isinstance(questions_raw, list):
            questions_raw = []

        canonical_questions = []
        for q_seq, q in enumerate(questions_raw):
            if not isinstance(q, dict):
                continue

            try:
                mq_idx = int(q.get("mq_index") or q.get("question_number") or q.get("qNo") or (q_seq + 1))
            except (ValueError, TypeError):
                mq_idx = q_seq + 1

            is_or = bool(q.get("is_or", False))
            try:
                raw_tot = q.get("total_marks") or q.get("marks") or 0
                total_marks = int(raw_tot) if not isinstance(raw_tot, bool) else 0
            except (ValueError, TypeError):
                total_marks = 0

            sub_qs_raw = q.get("sub_questions") or q.get("subQuestions") or q.get("subquestions") or []
            if not isinstance(sub_qs_raw, list):
                sub_qs_raw = []

            canonical_subs = []
            for s_seq, sq in enumerate(sub_qs_raw):
                if not isinstance(sq, dict):
                    continue

                letter = str(sq.get("letter") or sq.get("label") or sq.get("sub_label") or "").strip().lower()
                text_raw = str(sq.get("text") or sq.get("question_text") or "")
                text_norm = " ".join(text_raw.split())

                try:
                    raw_m = sq.get("marks")
                    sq_marks = int(raw_m) if (raw_m is not None and not isinstance(raw_m, bool)) else 0
                except (ValueError, TypeError):
                    sq_marks = 0

                co_norm = str(sq.get("co") or sq.get("coMapping") or "").strip().upper()
                bloom_norm = str(sq.get("bloom") or sq.get("bloom_level") or sq.get("rbt") or "").strip().upper()

                img_val = sq.get("image_path") or sq.get("figure_path") or sq.get("image")
                if isinstance(img_val, dict):
                    img_str = str(img_val.get("image_path") or img_val.get("path") or "").strip()
                else:
                    img_str = str(img_val or "").strip()

                img_bytes_hash = ""
                if img_str:
                    try:
                        p_img = Path(img_str)
                        if not p_img.is_file():
                            p_img = Path.cwd() / img_str
                        if p_img.is_file():
                            img_bytes_hash = hashlib.sha256(p_img.read_bytes()).hexdigest()[:16]
                    except Exception:
                        pass

                tbl_val = sq.get("table") or sq.get("table_data")
                tbl_repr = json.dumps(tbl_val, sort_keys=True) if isinstance(tbl_val, (dict, list)) else ""

                sol_raw = str(sq.get("solution") or sq.get("answer") or sq.get("model_answer") or "")
                sol_norm = " ".join(sol_raw.split())

                sub_tuple = (
                    s_seq,
                    letter,
                    text_norm,
                    sq_marks,
                    co_norm,
                    bloom_norm,
                    img_str,
                    img_bytes_hash,
                    tbl_repr,
                    sol_norm,
                )
                canonical_subs.append(sub_tuple)

            if not canonical_subs:
                # Main question without subquestions
                text_raw = str(q.get("text") or q.get("question_text") or "")
                text_norm = " ".join(text_raw.split())
                co_norm = str(q.get("co") or q.get("coMapping") or "").strip().upper()
                bloom_norm = str(q.get("bloom") or q.get("bloom_level") or q.get("rbt") or "").strip().upper()
                img_val = q.get("image_path") or q.get("figure_path") or q.get("image")
                img_str = str(img_val or "").strip()

                img_bytes_hash = ""
                if img_str:
                    try:
                        p_img = Path(img_str)
                        if not p_img.is_file():
                            p_img = Path.cwd() / img_str
                        if p_img.is_file():
                            img_bytes_hash = hashlib.sha256(p_img.read_bytes()).hexdigest()[:16]
                    except Exception:
                        pass

                tbl_val = q.get("table") or q.get("table_data")
                tbl_repr = json.dumps(tbl_val, sort_keys=True) if isinstance(tbl_val, (dict, list)) else ""
                sol_raw = str(q.get("solution") or q.get("answer") or q.get("model_answer") or "")
                sol_norm = " ".join(sol_raw.split())

                canonical_subs.append((
                    0,
                    "main",
                    text_norm,
                    total_marks,
                    co_norm,
                    bloom_norm,
                    img_str,
                    img_bytes_hash,
                    tbl_repr,
                    sol_norm,
                ))

            q_tuple = (
                q_seq,
                mq_idx,
                is_or,
                total_marks,
                canonical_subs,
            )
            canonical_questions.append(q_tuple)

        mod_tuple = (
            mod_seq,
            m_idx,
            mod_title,
            canonical_questions,
        )
        canonical_modules.append(mod_tuple)

    canonical_payload = {
        "header": header_tuple,
        "modules": canonical_modules,
    }

    serialized = json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


class PaperExportRegistry:
    """
    Filesystem-backed registry for authorized paper export records.
    Provides atomic registration, path-traversal prevention, no-overwrite protection,
    and distinct status handling for valid, missing, expired, and corrupted records.
    """

    @classmethod
    def get_registry_dir(cls) -> Path:
        env_dir = os.getenv("AION_PAPER_REGISTRY_DIR") or os.getenv("AION_REGISTRY_DIR")
        if env_dir:
            p = Path(env_dir).resolve()
        else:
            p = (Path.cwd() / "workspace" / "cache" / "paper_registry").resolve()
        p.mkdir(parents=True, exist_ok=True)
        try:
            # Set directory permissions to owner only where supported
            os.chmod(p, 0o700)
        except Exception:
            pass
        return p

    @classmethod
    def get_record_path(cls, paper_id: str) -> Optional[Path]:
        if not validate_paper_id(paper_id):
            return None
        reg_dir = cls.get_registry_dir()
        target = (reg_dir / f"{paper_id}.json").resolve()
        # Path traversal guard
        if target.parent != reg_dir:
            return None
        return target

    @classmethod
    def register(
        cls,
        *,
        paper_id: str,
        paper_data: Dict[str, Any],
        export_eligible: bool,
        paper_status: str,
        blocked_reasons: Optional[List[str]] = None,
        degraded_reasons: Optional[List[str]] = None,
        qa_report: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = DEFAULT_REGISTRY_TTL_SECONDS,
        expires_at: Optional[str] = None,
        allow_overwrite: bool = False,
    ) -> PaperExportRecord:
        """
        Atomically registers a generated paper.
        Fails closed on collisions, invalid IDs, or I/O persistence errors.
        """
        if not validate_paper_id(paper_id):
            raise RegistryError(f"Invalid paper_id syntax: {paper_id!r}")

        record_path = cls.get_record_path(paper_id)
        if record_path is None:
            raise RegistryError(f"Invalid registry target path for paper_id '{paper_id}'")

        # Collision protection: must not overwrite an existing paper record
        if record_path.exists() and not allow_overwrite:
            raise RegistryCollisionError(f"Paper record '{paper_id}' already exists in registry.")

        canonical_hash = canonicalize_paper_payload(paper_data)
        now = datetime.now(timezone.utc)
        created_at = now.isoformat()
        if expires_at is None:
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()

        config = paper_data.get("config") if isinstance(paper_data.get("config"), dict) else {}
        subject = str(config.get("subjectName") or paper_data.get("subject") or paper_data.get("subject_name") or "")
        exam_type = str(config.get("examType") or paper_data.get("exam_type") or "")
        try:
            total_marks = int(config.get("maxMarks") or paper_data.get("total_marks") or paper_data.get("totalMarks") or 0)
        except (ValueError, TypeError):
            total_marks = 0

        modules = paper_data.get("modules") or []
        module_count = len(modules) if isinstance(modules, list) else 0
        question_count = 0
        if isinstance(modules, list):
            for m in modules:
                if isinstance(m, dict):
                    qs = m.get("questions")
                    if isinstance(qs, list):
                        question_count += len(qs)

        qa_summary = {}
        if isinstance(qa_report, dict):
            qa_summary = {
                "status": qa_report.get("status"),
                "slot_count": qa_report.get("slot_count"),
                "resolved_slots": qa_report.get("resolved_slots"),
                "unresolved_slots": list(qa_report.get("unresolved_slots") or []),
            }

        record = PaperExportRecord(
            paper_id=paper_id,
            canonical_hash=canonical_hash,
            export_eligible=bool(export_eligible),
            paper_status=str(paper_status),
            created_at=created_at,
            expires_at=expires_at,
            blocked_reasons=list(blocked_reasons or []),
            degraded_reasons=list(degraded_reasons or []),
            qa_report_summary=qa_summary,
            subject=subject,
            exam_type=exam_type,
            total_marks=total_marks,
            module_count=module_count,
            question_count=question_count,
        )

        reg_dir = cls.get_registry_dir()
        tmp_name = f".tmp_{paper_id}_{uuid.uuid4().hex[:6]}.json"
        tmp_path = reg_dir / tmp_name

        try:
            raw_json = json.dumps(asdict(record), indent=2)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(raw_json)
                f.flush()
                os.fsync(f.fileno())

            # Atomic commit
            os.replace(tmp_path, record_path)
            try:
                os.chmod(record_path, 0o600)
            except Exception:
                pass
            return record
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise RegistryPersistenceError(f"Failed to persist registry record for '{paper_id}': {exc}") from exc

    @classmethod
    def get(cls, paper_id: Any) -> Tuple[Optional[PaperExportRecord], str]:
        """
        Retrieves and validates a paper record.
        Returns:
            (record, STATUS_VALID) if found, unexpired, and intact.
            (None, STATUS_MISSING) if paper_id does not exist or syntax invalid.
            (record, STATUS_EXPIRED) if record exists but has passed its expiration window.
            (None, STATUS_CORRUPTED) if JSON on disk is malformed or missing required schema fields.
            (None, STATUS_ERROR) if unexpected I/O error occurs.
        """
        if not validate_paper_id(paper_id):
            return None, STATUS_MISSING

        record_path = cls.get_record_path(paper_id)
        if record_path is None or not record_path.exists():
            return None, STATUS_MISSING

        try:
            with open(record_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return None, STATUS_CORRUPTED

            # Verify required schema keys
            required_keys = {"paper_id", "canonical_hash", "export_eligible", "paper_status", "created_at", "expires_at"}
            if not required_keys.issubset(data.keys()):
                return None, STATUS_CORRUPTED

            record = PaperExportRecord(
                paper_id=str(data["paper_id"]),
                canonical_hash=str(data["canonical_hash"]),
                export_eligible=bool(data["export_eligible"]),
                paper_status=str(data["paper_status"]),
                created_at=str(data["created_at"]),
                expires_at=str(data["expires_at"]),
                blocked_reasons=list(data.get("blocked_reasons") or []),
                degraded_reasons=list(data.get("degraded_reasons") or []),
                qa_report_summary=dict(data.get("qa_report_summary") or {}),
                subject=str(data.get("subject", "")),
                exam_type=str(data.get("exam_type", "")),
                total_marks=int(data.get("total_marks", 0)),
                module_count=int(data.get("module_count", 0)),
                question_count=int(data.get("question_count", 0)),
            )

            if record.is_expired():
                # Lazy expiry: return record with STATUS_EXPIRED so caller can format exact 403 response
                return record, STATUS_EXPIRED

            return record, STATUS_VALID
        except json.JSONDecodeError:
            return None, STATUS_CORRUPTED
        except Exception:
            return None, STATUS_ERROR
