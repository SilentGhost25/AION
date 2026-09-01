"""
AION Visual: VLM Analyzer
Fixed: better prompt, longer timeout, robust JSON parsing, text parsing fallback, batch processing
"""

from __future__ import annotations

import json
import re
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from .figure_card import FigureCard, VisualFact


# -- Moondream-compatible prompt -------------------------------

_VLM_PROMPT_SIMPLE = """\
Look at this figure carefully. Answer these questions briefly:

1. What type of figure is this? (flowchart/diagram/graph/circuit/table/photo/other)
2. What are the main components or labels visible?
3. What does this figure show or represent?
4. Is the figure clear and readable? (yes/no)
5. List 3 key facts visible in this figure.

Be specific. Only describe what you can actually see."""


_VLM_PROMPT_JSON = """\
Analyze this figure and return ONLY a JSON object.
No explanation before or after the JSON.

{
  "visual_type": "flowchart or architecture or graph or circuit or table or equation or code or photo or unknown",
  "readable": true,
  "title": "visible title or empty string",
  "entities": ["list", "of", "visible", "components"],
  "key_facts": ["fact 1", "fact 2", "fact 3"],
  "warnings": [],
  "confidence": 0.8
}"""


class VLMAnalyzer:
    """
    Analyze figures using moondream or llava via Ollama.
    Robust fallback chain — never crashes the pipeline.
    """

    PREFERRED_MODELS = [
        "moondream:latest",
        "moondream",
        "llava:7b",
        "llava:latest",
        "llava",
        "bakllava",
    ]

    def __init__(
        self,
        host:    str = "http://localhost:11434",
        timeout: int = 90,   # Increased for CPU inference
    ):
        self.host    = host.rstrip("/")
        self.timeout = timeout
        self._model  = None

    # -- Model detection ---------------------------------------

    def _find_available_model(self) -> Optional[str]:
        if self._model:
            return self._model
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as r:
                data      = json.loads(r.read())
                installed = [m["name"] for m in data.get("models", [])]

            for preferred in self.PREFERRED_MODELS:
                for name in installed:
                    if preferred.split(":")[0] in name:
                        self._model = name
                        print(f"[VLM] Using model: {self._model}")
                        return self._model

            print("[VLM] No vision model found.")
            print("[VLM] Run: ollama pull moondream")
            return None

        except Exception as e:
            print(f"[VLM] Ollama check failed: {e}")
            return None

    # -- Main analyze ------------------------------------------

    def analyze(self, card: FigureCard) -> FigureCard:
        """Process one FigureCard using fast rule-based metadata without heavy VLM inference."""
        return self._rule_based_fallback(card)

    # -- Strategy 1: JSON prompt -------------------------------

    def _try_json_prompt(
        self,
        card:    FigureCard,
        img_b64: str,
        model:   str,
    ) -> Optional[FigureCard]:
        """Try to get structured JSON from VLM."""
        try:
            raw = self._call_vlm(
                model   = model,
                prompt  = _VLM_PROMPT_JSON,
                img_b64 = img_b64,
                options = {
                    "temperature": 0.05,  # Very low for JSON
                    "num_predict": 200,
                    "num_ctx":     1024,
                }
            )
            if not raw:
                return None

            data = self._extract_json(raw)
            if not data:
                return None

            return self._apply_json_result(card, data)

        except Exception as e:
            print(f"[VLM] JSON prompt error for {card.id}: {e}")
            return None

    # -- Strategy 2: Simple prompt + text parsing --------------

    def _try_simple_prompt(
        self,
        card:    FigureCard,
        img_b64: str,
        model:   str,
    ) -> Optional[FigureCard]:
        """Ask simple questions, parse the text response."""
        try:
            raw = self._call_vlm(
                model   = model,
                prompt  = _VLM_PROMPT_SIMPLE,
                img_b64 = img_b64,
                options = {
                    "temperature": 0.2,
                    "num_predict": 200,
                    "num_ctx":     1024,
                }
            )
            if not raw or len(raw.split()) < 5:
                return None

            return self._parse_text_response(card, raw)

        except Exception as e:
            print(f"[VLM] Simple prompt error for {card.id}: {e}")
            return None

    # -- VLM HTTP call -----------------------------------------

    def _call_vlm(
        self,
        model:   str,
        prompt:  str,
        img_b64: str,
        options: dict,
    ) -> Optional[str]:
        """Call Ollama /api/generate with image."""
        payload = json.dumps({
            "model":   model,
            "prompt":  prompt,
            "images":  [img_b64],
            "stream":  False,
            "options": options,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            data = json.loads(res.read())
            return data.get("response", "").strip()

    # -- JSON extraction ---------------------------------------

    def _extract_json(self, raw: str) -> Optional[dict]:
        """
        Robustly extract JSON from VLM response.
        Handles common VLM formatting issues.
        """
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        fixed = raw.replace("'", '"')
        try:
            start = fixed.find("{")
            end   = fixed.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(fixed[start:end])
        except json.JSONDecodeError:
            pass

        result = self._manual_json_extract(raw)
        if result:
            return result

        return None

    def _manual_json_extract(self, raw: str) -> Optional[dict]:
        """Extract fields manually if JSON is malformed."""
        result = {}

        type_match = re.search(
            r'"?visual_type"?\s*:\s*"?(\w+)"?', raw, re.I
        )
        if type_match:
            result["visual_type"] = type_match.group(1).lower()

        read_match = re.search(r'"?readable"?\s*:\s*(true|false)', raw, re.I)
        if read_match:
            result["readable"] = read_match.group(1).lower() == "true"

        conf_match = re.search(r'"?confidence"?\s*:\s*([\d.]+)', raw, re.I)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        ent_match = re.search(
            r'"?entities"?\s*:\s*\[([^\]]*)\]', raw, re.I
        )
        if ent_match:
            items = re.findall(r'"([^"]+)"', ent_match.group(1))
            result["entities"] = items

        facts_match = re.search(
            r'"?key_facts"?\s*:\s*\[([^\]]*)\]', raw, re.I
        )
        if facts_match:
            items = re.findall(r'"([^"]+)"', facts_match.group(1))
            result["key_facts"] = items

        return result if len(result) >= 2 else None

    # -- Apply JSON result to card -----------------------------

    def _apply_json_result(
        self,
        card: FigureCard,
        data: dict,
    ) -> FigureCard:
        """Populate FigureCard from parsed JSON data."""

        card.visual_type    = data.get("visual_type", "unknown")
        card.vlm_readable   = bool(data.get("readable", True))
        card.vlm_confidence = min(
            float(data.get("confidence", 0.5)), 1.0
        )
        card.vlm_entities   = data.get("entities", [])
        card.vlm_description = (
            data.get("title", "") + " " +
            " ".join(card.vlm_entities[:5])
        ).strip()

        warnings_text = " ".join(data.get("warnings", [])).lower()
        if any(
            w in warnings_text
            for w in ["decorative", "logo", "blank", "watermark", "icon"]
        ):
            card.eligible    = False
            card.skip_reason = "vlm_decorative"

        if not card.vlm_readable:
            card.eligible    = False
            card.skip_reason = "vlm_unreadable"

        facts = []
        for i, ft in enumerate(data.get("key_facts", []), 1):
            if ft and len(ft.split()) >= 3:
                facts.append(VisualFact(
                    id         = f"{card.id}_vf{i}",
                    text       = ft,
                    confidence = card.vlm_confidence,
                    source     = "vlm",
                ))

        if card.caption:
            facts.append(VisualFact(
                id         = f"{card.id}_cap",
                text       = card.caption,
                confidence = 0.95,
                source     = "caption",
            ))

        if card.ocr_text and len(card.ocr_text.split()) > 2:
            facts.append(VisualFact(
                id         = f"{card.id}_ocr",
                text       = card.ocr_text[:250],
                confidence = 0.80,
                source     = "ocr",
            ))

        card.facts = facts

        print(
            f"[VLM] {card.id}: "
            f"type={card.visual_type} "
            f"conf={card.vlm_confidence:.2f} "
            f"facts={len(facts)} "
            f"eligible={card.eligible}"
        )
        return card

    # -- Parse text response -----------------------------------

    def _parse_text_response(
        self,
        card: FigureCard,
        raw:  str,
    ) -> FigureCard:
        """
        Parse moondream's free-text response into structured data.
        Used when JSON prompt fails.
        """
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        vtype = "unknown"
        for line in lines:
            classified = self._classify_from_text(line)
            if classified != "unknown":
                vtype = classified
                break

        card.visual_type    = vtype
        card.vlm_readable   = not any(
            w in raw.lower()
            for w in ["cannot read", "unclear", "blurry", "unreadable"]
        )
        card.vlm_confidence = 0.55
        card.vlm_description = raw[:200]

        facts = []
        fact_lines = [
            l for l in lines
            if re.match(r"^(\d+\.|[-•*]|\(\d+\))", l)
            and len(l.split()) >= 4
        ]

        if not fact_lines:
            fact_lines = [l for l in lines if len(l.split()) >= 6]

        for i, line in enumerate(fact_lines[:5], 1):
            clean = re.sub(r"^(\d+\.|[-•*]|\(\d+\))\s*", "", line)
            if clean:
                facts.append(VisualFact(
                    id         = f"{card.id}_tf{i}",
                    text       = clean,
                    confidence = 0.55,
                    source     = "vlm_text",
                ))

        if card.caption:
            facts.append(VisualFact(
                id         = f"{card.id}_cap",
                text       = card.caption,
                confidence = 0.95,
                source     = "caption",
            ))

        if card.ocr_text:
            facts.append(VisualFact(
                id         = f"{card.id}_ocr",
                text       = card.ocr_text[:250],
                confidence = 0.80,
                source     = "ocr",
            ))

        card.facts = facts

        print(
            f"[VLM] {card.id} (text-parsed): "
            f"type={card.visual_type} "
            f"facts={len(facts)}"
        )
        return card

    def _classify_from_text(self, text: str) -> str:
        text = text.lower()
        rules = {
            "flowchart":    ["flow", "process", "step", "decision", "arrow"],
            "architecture": ["layer", "block", "component", "architecture",
                             "system", "network", "module"],
            "graph":        ["axis", "graph", "chart", "plot", "curve",
                             "bar", "histogram", "x-axis", "y-axis"],
            "circuit":      ["circuit", "resistor", "gate", "transistor",
                             "signal", "voltage"],
            "table":        ["table", "row", "column", "cell", "entry"],
            "equation":     ["equation", "formula", "mathematical"],
            "code":         ["code", "function", "syntax", "program"],
            "photo":        ["photograph", "image", "photo", "specimen"],
        }
        for vtype, keywords in rules.items():
            if any(kw in text for kw in keywords):
                return vtype
        return "unknown"

    # -- Rule-based fallback -----------------------------------

    def _rule_based_fallback(self, card: FigureCard) -> FigureCard:
        facts = []

        if card.caption:
            facts.append(VisualFact(
                id="f_cap", text=card.caption,
                confidence=0.95, source="caption"
            ))
        if card.ocr_text and len(card.ocr_text.split()) > 2:
            facts.append(VisualFact(
                id="f_ocr", text=card.ocr_text[:300],
                confidence=0.75, source="ocr"
            ))
        if card.preceding_text:
            facts.append(VisualFact(
                id="f_ctx", text=card.preceding_text[:300],
                confidence=0.60, source="context"
            ))
        if card.section_title:
            facts.append(VisualFact(
                id="f_sec", text=card.section_title[:200],
                confidence=0.70, source="section"
            ))

        if not facts:
            name = Path(card.image_path).stem if card.image_path else f"Figure {card.figure_index or 1}"
            facts.append(VisualFact(
                id=f"{card.id}_meta",
                text=f"Figure reference showing {name.replace('_', ' ')}",
                confidence=0.65,
                source="metadata"
            ))

        card.facts          = facts
        card.vlm_confidence = 0.50
        card.eligible       = True
        card.skip_reason    = ""

        return card

    # -- Batch -------------------------------------------------

    def analyze_batch(
        self,
        cards:   list[FigureCard],
        max_vlm: int = 0,
    ) -> list[FigureCard]:
        for card in cards:
            self._rule_based_fallback(card)

        eligible_after = sum(1 for c in cards if c.eligible)
        print(
            f"\n[VISUAL] Fast extraction batch complete: "
            f"{eligible_after}/{len(cards)} figures registered (VLM processing bypassed)"
        )
        return cards
