"""
AION Local Visual Engine
========================
Zero-API Multimodal Image & Diagram Hydration.
Supports:
  1. Base64 Data URI Conversion (from PyMuPDF extracted figures/images)
  2. Inline SVG Vector Diagrams
  3. Unicode / ASCII Box-Drawing Schematics (Look-angle, Transponders, AOCS, Orbits)
"""

from __future__ import annotations
import base64
import os
import re
from typing import Dict, Any, Optional

# ── PRE-BUILT LOCAL ASCII / UNICODE SCHEMATICS ────────────────────────────────
ASCII_DIAGRAMS: Dict[str, str] = {
    "look_angle": """
                  Polar Axis
                      ^
                      |
                 __--'|'--__
              .-'     |     '-.
            .'        |        '.
           /          |          \
          |           +-----------|--> Equator
          |          / \          |
           \        /   \        /
            '.     /  d  \     .'
              '-._/       \_.-'
                 /'-.__.-'\
                /     |    \
               /      |     \
        Earth Station        \ Sub-Satellite Point (SSP)
""",
    "transponder": """
+---------+     +-------+     +---------+     +------+     +-----+
| Receive | --> | Input | --> | Down    | --> | TWTA | --> | Tx  |
| Antenna |     |  BPF  |     | Converter|     | Amp  |     | Ant |
+---------+     +-------+     +---------+     +------+     +-----+
                                   ^
                                   |
                             +-----------+
                             | Local Osc |
                             +-----------+
""",
    "orbit_geometry": """
                   Apogee (r_a)
                      +---+
                 . '    |    ' .
             .          |          .
           .            |            .
          +-------------+-------------+
         Perigee   Center (C)   Focus (Earth)
          (r_p)         |<--- a ---->|
""",
    "aocs": """
           +----------------------------------+
           |     Attitude Control (AOCS)      |
           +----------------------------------+
                  /                    \
        +-------------------+    +--------------------+
        | Spin-Stabilized   |    | Three-Axis Body    |
        | (Gyroscopic Torque|    | (Reaction Wheels   |
        |  & Solar Drum)    |    |  & Solar Panels)   |
        +-------------------+    +--------------------+
""",
    "dsss": """
[Data Input] ----> (+) ----> [BPSK Modulator] ----> [RF Output]
                    ^
                    |
          [PN Code Generator]
"""
}

# ── PRE-BUILT INLINE SVG DIAGRAMS ─────────────────────────────────────────────
SVG_DIAGRAMS: Dict[str, str] = {
    "transponder": """
<svg width="420" height="70" viewBox="0 0 420 70" xmlns="http://www.w3.org/2000/svg" style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:6px; margin:8px 0;">
  <rect x="10" y="15" width="70" height="40" rx="4" fill="#3b82f6" />
  <text x="45" y="38" fill="#ffffff" font-size="10" font-family="sans-serif" text-anchor="middle">BPF / LNA</text>
  <line x1="80" y1="35" x2="110" y2="35" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="110" y="15" width="80" height="40" rx="4" fill="#10b981" />
  <text x="150" y="38" fill="#ffffff" font-size="10" font-family="sans-serif" text-anchor="middle">Mixer / Conv</text>
  <line x1="190" y1="35" x2="220" y2="35" stroke="#64748b" stroke-width="2"/>
  <rect x="220" y="15" width="80" height="40" rx="4" fill="#f59e0b" />
  <text x="260" y="38" fill="#ffffff" font-size="10" font-family="sans-serif" text-anchor="middle">TWTA Power</text>
  <line x1="300" y1="35" x2="330" y2="35" stroke="#64748b" stroke-width="2"/>
  <rect x="330" y="15" width="80" height="40" rx="4" fill="#6366f1" />
  <text x="370" y="38" fill="#ffffff" font-size="10" font-family="sans-serif" text-anchor="middle">Tx BPF</text>
</svg>
""",
    "orbit_geometry": """
<svg width="350" height="80" viewBox="0 0 350 80" xmlns="http://www.w3.org/2000/svg" style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:6px; margin:8px 0;">
  <ellipse cx="175" cy="40" rx="140" ry="30" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="4 2"/>
  <circle cx="120" cy="40" r="14" fill="#3b82f6"/>
  <text x="120" y="44" fill="#fff" font-size="8" text-anchor="middle">Earth</text>
  <circle cx="290" cy="22" r="6" fill="#ef4444"/>
  <text x="290" y="12" fill="#1e293b" font-size="9" text-anchor="middle">Satellite</text>
</svg>
"""
}


class LocalVisualEngine:
    """Zero-API Multimodal Image & Schematic Processor."""

    @staticmethod
    def file_to_base64_uri(file_path: str) -> Optional[str]:
        """Convert a local image file (PNG/JPG) to an RFC 2397 Base64 Data URI."""
        if not file_path or not os.path.exists(file_path):
            return None
        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            mime = "image/svg+xml" if ext == "svg" else f"image/{ext if ext in ('png', 'jpeg', 'gif', 'webp') else 'png'}"
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            print(f"[VISUAL ENGINE] Base64 encoding error: {e}")
            return None

    @classmethod
    def resolve_visual_asset(cls, question_text: str, figure_obj: Any = None) -> Optional[Dict[str, Any]]:
        """
        Resolve the best visual asset for a question.
        Priority:
          1. Extracted PDF Figure (converted to Base64 URI)
          2. Inline SVG Diagram (for architecture/circuits)
          3. Unicode/ASCII Schematic (for geometry/subsystems)
        """
        text_lower = question_text.lower()

        # 1. Extracted Figure Object from PyMuPDF
        if figure_obj:
            img_path = getattr(figure_obj, "path", None) or getattr(figure_obj, "image_path", None)
            if img_path and os.path.exists(img_path):
                data_uri = cls.file_to_base64_uri(img_path)
                if data_uri:
                    return {
                        "url": data_uri,
                        "type": "base64",
                        "caption": getattr(figure_obj, "caption", "Extracted Reference Diagram")
                    }

        # 2. Topic-driven SVG / ASCII Injection
        if any(w in text_lower for w in ["transponder", "block diagram", "receiver", "twta", "mixer"]):
            return {
                "svg": SVG_DIAGRAMS["transponder"],
                "ascii": ASCII_DIAGRAMS["transponder"],
                "type": "svg",
                "caption": "Figure: Satellite Transponder Functional Architecture"
            }

        if any(w in text_lower for w in ["look-angle", "look angle", "elevation", "azimuth", "geometry", "sub-satellite"]):
            return {
                "ascii": ASCII_DIAGRAMS["look_angle"],
                "type": "ascii",
                "caption": "Reference Diagram: Look-Angle & Elevation Geometry"
            }

        if any(w in text_lower for w in ["kepler", "orbital elements", "apogee", "perigee", "eccentricity", "orbit"]):
            return {
                "svg": SVG_DIAGRAMS["orbit_geometry"],
                "ascii": ASCII_DIAGRAMS["orbit_geometry"],
                "type": "svg",
                "caption": "Figure: Orbital Geometry and Focus Coordinates"
            }

        if any(w in text_lower for w in ["spin", "stabiliz", "three-axis", "reaction wheel", "aocs", "attitude"]):
            return {
                "ascii": ASCII_DIAGRAMS["aocs"],
                "type": "ascii",
                "caption": "Figure: Attitude & Orbit Control Subsystem (AOCS) Configurations"
            }

        if any(w in text_lower for w in ["dsss", "spread spectrum", "pn code", "cdma"]):
            return {
                "ascii": ASCII_DIAGRAMS["dsss"],
                "type": "ascii",
                "caption": "Figure: DSSS CDMA Transmitter Schematic"
            }

        return None
