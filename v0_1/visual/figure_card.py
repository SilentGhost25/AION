"""
AION Visual: FigureCard + FigureRegistry
Bulletproof serialization — never returns strings as cards
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisualFact:
    id:         str   = ""
    text:       str   = ""
    confidence: float = 0.5
    source:     str   = "unknown"

    @staticmethod
    def from_dict(d: dict) -> "VisualFact":
        return VisualFact(
            id         = str(d.get("id",         "")),
            text       = str(d.get("text",       "")),
            confidence = float(d.get("confidence", 0.5)),
            source     = str(d.get("source",     "unknown")),
        )

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "text":       self.text,
            "confidence": self.confidence,
            "source":     self.source,
        }


@dataclass
class FigureCard:
    # Identity
    id:           str   = ""
    document_id:  str   = ""
    module_id:    str   = "module_1"
    page:         int   = 0
    figure_index: int   = 0

    # Asset
    image_path:   str   = ""
    image_url:    str   = ""

    # Text evidence
    caption:          str        = ""
    ocr_text:         str        = ""
    preceding_text:   str        = ""
    following_text:   str        = ""
    section_title:    str        = ""
    explicit_refs:    list       = field(default_factory=list)

    # VLM analysis
    visual_type:      str        = "unknown"
    vlm_description:  str        = ""
    vlm_entities:     list       = field(default_factory=list)
    vlm_confidence:   float      = 0.0
    vlm_readable:     bool       = True

    # Facts
    facts:            list       = field(default_factory=list)

    # Eligibility
    eligible:         bool       = True
    skip_reason:      str        = ""

    # Provenance
    provenance_score: float      = 0.0

    def full_context(self) -> str:
        parts = [
            self.caption,
            self.ocr_text,
            self.section_title,
            self.preceding_text,
            self.following_text,
        ] + (self.explicit_refs or [])
        return " ".join(p for p in parts if p).strip()

    def fact_texts(self) -> list[str]:
        result = []
        for f in self.facts:
            if isinstance(f, VisualFact):
                result.append(f.text)
            elif isinstance(f, dict):
                result.append(f.get("text", ""))
        return result

    @staticmethod
    def from_dict(d: dict) -> "FigureCard":
        """
        Safe deserialization — handles missing or wrong-type fields.
        Never raises — returns a card with eligible=False on error.
        """
        try:
            raw_facts = d.get("facts", [])
            facts = []
            for f in raw_facts:
                if isinstance(f, dict):
                    facts.append(VisualFact.from_dict(f))
                elif isinstance(f, VisualFact):
                    facts.append(f)

            card = FigureCard(
                id               = str(d.get("id",               "")),
                document_id      = str(d.get("document_id",      "")),
                module_id        = str(d.get("module_id",        "module_1")),
                page             = int(d.get("page",              0)),
                figure_index     = int(d.get("figure_index",      0)),
                image_path       = str(d.get("image_path",       "")),
                image_url        = str(d.get("image_url",        "")),
                caption          = str(d.get("caption",          "")),
                ocr_text         = str(d.get("ocr_text",         "")),
                preceding_text   = str(d.get("preceding_text",   "")),
                following_text   = str(d.get("following_text",   "")),
                section_title    = str(d.get("section_title",    "")),
                explicit_refs    = list(d.get("explicit_refs",   [])),
                visual_type      = str(d.get("visual_type",      "unknown")),
                vlm_description  = str(d.get("vlm_description",  "")),
                vlm_entities     = list(d.get("vlm_entities",    [])),
                vlm_confidence   = float(d.get("vlm_confidence",  0.0)),
                vlm_readable     = bool(d.get("vlm_readable",     True)),
                eligible         = bool(d.get("eligible",         True)),
                skip_reason      = str(d.get("skip_reason",       "")),
                provenance_score = float(d.get("provenance_score", 0.0)),
                facts            = facts,
            )
            return card

        except Exception as e:
            print(f"[REGISTRY] FigureCard.from_dict error: {e}")
            bad = FigureCard()
            bad.eligible    = False
            bad.skip_reason = f"deserialization_error:{e}"
            return bad

    def to_dict(self) -> dict:
        """Safe serialization."""
        return {
            "id":               self.id,
            "document_id":      self.document_id,
            "module_id":        self.module_id,
            "page":             self.page,
            "figure_index":     self.figure_index,
            "image_path":       self.image_path,
            "image_url":        self.image_url,
            "caption":          self.caption,
            "ocr_text":         self.ocr_text,
            "preceding_text":   self.preceding_text,
            "following_text":   self.following_text,
            "section_title":    self.section_title,
            "explicit_refs":    self.explicit_refs or [],
            "visual_type":      self.visual_type,
            "vlm_description":  self.vlm_description,
            "vlm_entities":     self.vlm_entities or [],
            "vlm_confidence":   self.vlm_confidence,
            "vlm_readable":     self.vlm_readable,
            "eligible":         self.eligible,
            "skip_reason":      self.skip_reason,
            "provenance_score": self.provenance_score,
            "facts": [
                f.to_dict() if isinstance(f, VisualFact)
                else f if isinstance(f, dict)
                else {}
                for f in (self.facts or [])
            ],
        }


class FigureRegistry:

    def __init__(
        self,
        document_id: str,
        store_dir:   str = "extracted_output",
    ):
        self.document_id = document_id
        self.store_path  = Path(store_dir) / f"figures_{document_id}.json"
        self.cards: list[FigureCard] = []

    def add(self, card: FigureCard) -> None:
        if isinstance(card, FigureCard):
            self.cards.append(card)

    def add_all(self, cards: list) -> None:
        for c in cards:
            if isinstance(c, FigureCard):
                self.cards.append(c)
            else:
                print(f"[REGISTRY] Skipping non-FigureCard: {type(c)}")

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for c in self.cards:
            if isinstance(c, FigureCard):
                try:
                    data.append(c.to_dict())
                except Exception as e:
                    print(f"[REGISTRY] Save error {c.id}: {e}")

        self.store_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )
        print(f"[REGISTRY] Saved {len(data)} figures -> {self.store_path}")

    def load(self) -> bool:
        if not self.store_path.exists():
            return False

        try:
            raw  = self.store_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            if not isinstance(data, list):
                print("[REGISTRY] Bad format — expected list")
                return False

            self.cards = []
            for item in data:
                if not isinstance(item, dict):
                    print(f"[REGISTRY] Skipping non-dict item: {type(item)}")
                    continue

                card = FigureCard.from_dict(item)
                self.cards.append(card)

            valid = [c for c in self.cards if isinstance(c, FigureCard)]
            if len(valid) != len(self.cards):
                print(
                    f"[REGISTRY] Warning: "
                    f"{len(self.cards) - len(valid)} invalid cards dropped"
                )
                self.cards = valid

            print(f"[REGISTRY] Loaded {len(self.cards)} figures")
            return len(self.cards) > 0

        except Exception as e:
            print(f"[REGISTRY] Load failed: {e} — will re-extract")
            self.cards = []
            return False

    def eligible_cards(self) -> list[FigureCard]:
        """Always returns FigureCard objects."""
        result = []
        for c in self.cards:
            if not isinstance(c, FigureCard):
                print(f"[REGISTRY] Non-FigureCard found: {type(c)}")
                continue
            if not hasattr(c, "provenance_score"):
                print(f"[REGISTRY] Card missing provenance_score")
                continue
            if not hasattr(c, "eligible"):
                continue
            if c.eligible:
                result.append(c)
        return result

    def cards_for_module(self, module_id: str) -> list[FigureCard]:
        return [
            c for c in self.eligible_cards()
            if c.module_id == module_id
        ]

    def get(self, figure_id: str) -> Optional[FigureCard]:
        return next(
            (c for c in self.cards if c.id == figure_id), None
        )

    @staticmethod
    def make_document_id(file_path: str) -> str:
        p = Path(file_path)
        try:
            stat = p.stat()
            raw  = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        except Exception:
            raw  = file_path
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
