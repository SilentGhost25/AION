"""
AION v2 Stage 1 Contract — Generation Request
=============================================
Strict input contract for all paper and question generation requests.
Handles frontend field alias normalization, validation, and reporting.

Production-safe. Zero laptop-specific code.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class GenerationRequest:
    subject:                str            = "General Academic"
    department:             str            = "Computer Science & Engineering"
    semester:               int            = 5
    exam_type:              str            = "SEE"        # IA, IA1, IA2, IA3, SEE
    difficulty:             str            = "mixed"      # easy, medium, hard, mixed
    bloom_levels:           List[str]      = field(default_factory=lambda: ["L1", "L2", "L3", "L4"])
    selected_modules:       List[int]      = field(default_factory=lambda: [1, 2, 3, 4, 5])
    question_types:         List[str]      = field(default_factory=lambda: ["conceptual", "numerical", "derivation"])
    model:                  str            = "qwen2.5:14b"
    visual_mode:            bool           = True
    constraints:            List[str]      = field(default_factory=list)
    paper_structure:        Dict[str, Any] = field(default_factory=lambda: {"sections": 5, "questions_per_module": 2})
    retrieval_mode:         str            = "hybrid"     # vector, bm25, hybrid
    grounding_mode:         str            = "strict"     # strict, permissive
    allow_external_sources: bool           = False
    file_id:                Optional[str]  = None
    file_path:              Optional[str]  = None
    notes_text:             Optional[str]  = None
    raw_payload:            Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GenerationRequest":
        """
        Normalize frontend raw payload into a canonical GenerationRequest.
        Maps all known field aliases from old and new frontends.
        """
        if not isinstance(raw, dict):
            raw = {}

        # -- Subject ----------------
        subject = (
            raw.get("subject") or
            raw.get("subjectName") or
            raw.get("subject_code") or
            raw.get("course") or
            "General Academic"
        )

        # -- Department & Semester ----
        dept = raw.get("department") or raw.get("dept") or "Computer Science & Engineering"
        try:
            sem = int(raw.get("semester") or raw.get("sem") or 5)
        except (ValueError, TypeError):
            sem = 5

        # -- Exam Type ---------------
        exam_type = (
            raw.get("exam_type") or
            raw.get("examType") or
            raw.get("exam_mode") or
            "SEE"
        )

        # -- Difficulty --------------
        diff = (
            raw.get("difficulty") or
            raw.get("difficultyLevel") or
            raw.get("difficulty_profile") or
            "mixed"
        )

        # -- Bloom Levels ------------
        bloom_input = raw.get("bloom_levels") or raw.get("bloom") or raw.get("bloomsTaxonomy") or raw.get("bloomLevels")
        if isinstance(bloom_input, str):
            bloom_levels = [b.strip() for b in bloom_input.split(",") if b.strip()]
        elif isinstance(bloom_input, list):
            bloom_levels = [str(b).strip() for b in bloom_input if str(b).strip()]
        else:
            bloom_levels = ["L1", "L2", "L3", "L4"]

        # -- Modules -----------------
        mod_input = raw.get("selected_modules") or raw.get("modules") or raw.get("selectedModules") or raw.get("sections")
        if isinstance(mod_input, list):
            selected_modules = []
            for m in mod_input:
                try:
                    if isinstance(m, dict) and "moduleNumber" in m:
                        selected_modules.append(int(m["moduleNumber"]))
                    elif isinstance(m, dict) and "number" in m:
                        selected_modules.append(int(m["number"]))
                    else:
                        selected_modules.append(int(m))
                except (ValueError, TypeError):
                    pass
            if not selected_modules:
                selected_modules = [1, 2, 3, 4, 5]
        else:
            selected_modules = [1, 2, 3, 4, 5]

        # -- Question Types ----------
        q_type_input = raw.get("question_types") or raw.get("questionTypes") or raw.get("question_type")
        if isinstance(q_type_input, str):
            question_types = [qt.strip() for qt in q_type_input.split(",") if qt.strip()]
        elif isinstance(q_type_input, list):
            question_types = [str(qt).strip() for qt in q_type_input if str(qt).strip()]
        else:
            question_types = ["conceptual", "numerical", "derivation"]

        # -- Model -------------------
        model = raw.get("model") or raw.get("production_model") or "qwen2.5:14b"

        # -- Visual Mode -------------
        visual = raw.get("visual_mode")
        if visual is None:
            visual = raw.get("includeVisual")
        if visual is None:
            visual = raw.get("useImages")
        if visual is None:
            visual = True
        visual_mode = bool(visual)

        # -- File ID / Path ----------
        file_id = raw.get("file_id") or raw.get("fileId")
        file_path = raw.get("file_path") or raw.get("filePath")
        notes_text = raw.get("notes_text") or raw.get("notesText") or raw.get("context")

        return cls(
            subject                = str(subject).strip(),
            department             = str(dept).strip(),
            semester               = sem,
            exam_type              = str(exam_type).strip().upper(),
            difficulty             = str(diff).strip().lower(),
            bloom_levels           = bloom_levels,
            selected_modules       = sorted(list(set(selected_modules))),
            question_types         = question_types,
            model                  = str(model).strip(),
            visual_mode            = visual_mode,
            file_id                = file_id,
            file_path              = file_path,
            notes_text             = notes_text,
            raw_payload            = raw
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate request parameters. Returns (is_valid, list_of_errors).
        """
        errors = []

        if not self.subject:
            errors.append("subject is required")

        if self.semester not in range(1, 9):
            errors.append(f"invalid semester '{self.semester}' (must be 1-8)")

        valid_exams = {"IA", "IA1", "IA2", "IA3", "IAT1", "IAT2", "IAT3", "SEE", "MID", "FINAL"}
        if self.exam_type not in valid_exams:
            errors.append(f"invalid exam_type '{self.exam_type}' (valid: {sorted(list(valid_exams))})")

        valid_diffs = {"easy", "medium", "hard", "mixed"}
        if self.difficulty not in valid_diffs:
            errors.append(f"invalid difficulty '{self.difficulty}' (valid: {sorted(list(valid_diffs))})")

        if not self.file_id and not self.file_path and not self.notes_text:
            errors.append("no content provided (file_id, file_path, or notes_text required)")

        return len(errors) == 0, errors

    def print_received_summary(self):
        """Format and print standard server request log box."""
        print("\n========================================================")
        print("                GENERATION REQUEST RECEIVED")
        print("========================================================")
        print(f"  Subject      : {self.subject}")
        print(f"  Dept / Sem   : {self.department} (Sem {self.semester})")
        print(f"  Exam Type    : {self.exam_type}")
        print(f"  Difficulty   : {self.difficulty}")
        print(f"  Bloom Levels : {', '.join(self.bloom_levels)}")
        print(f"  Modules      : {', '.join(map(str, self.selected_modules))}")
        print(f"  Q Types      : {', '.join(self.question_types)}")
        print(f"  Model        : {self.model}")
        print(f"  Visual Mode  : {'ON' if self.visual_mode else 'OFF'}")
        print(f"  File Source  : {self.file_id or self.file_path or ('Inline Text (' + str(len(self.notes_text or '')) + ' chars)')}")
        print("========================================================\n")
