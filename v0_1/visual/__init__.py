"""
AION Visual System — Safe imports & Safe Builder
"""

from typing import Optional

from .figure_card      import FigureCard, FigureRegistry, VisualFact
from .figure_extractor import extract_figures
from .vlm_analyzer     import VLMAnalyzer
from .visual_generator import VisualQuestionGenerator, ModuleVisualPlanner
from .verifier         import VisualVerifier
from .asset_server     import register_asset_routes


def _build_module_map(
    modules: list,
    total_pages: int = 200,
) -> dict[int, str]:
    """Map page numbers → module IDs."""
    module_map     = {}
    n              = max(len(modules), 1)
    pages_per_mod  = max(1, total_pages // n)

    for i in range(n):
        start = i * pages_per_mod + 1
        end   = (i + 1) * pages_per_mod
        for pg in range(start, end + 1):
            module_map[pg] = f"module_{i + 1}"

    return module_map


def safe_build_planner(
    file_path:        str,
    modules:          list,
    include_visual:   bool = True,
) -> Optional[ModuleVisualPlanner]:
    """
    Build visual planner safely.
    Returns None on ANY failure — never crashes pipeline.
    """
    if not include_visual:
        return None

    try:
        doc_id   = FigureRegistry.make_document_id(file_path)
        registry = FigureRegistry(doc_id)

        if not registry.load():
            print("[VISUAL] Extracting figures...")

            module_map = _build_module_map(modules)
            raw_cards  = extract_figures(
                file_path        = file_path,
                doc_id           = doc_id,
                module_map       = module_map,
                asset_dir        = "extracted_output/assets",
                image_url_prefix = "/api/assets",
            )

            valid = []
            for c in raw_cards:
                if not isinstance(c, FigureCard):
                    print(f"[VISUAL] Skipping non-FigureCard: {type(c)}")
                    continue
                if not hasattr(c, "provenance_score"):
                    print(f"[VISUAL] Skipping card without provenance_score")
                    continue
                valid.append(c)

            print(f"[VISUAL] {len(valid)} valid FigureCards")

            if valid:
                VLMAnalyzer().analyze_batch(valid, max_vlm=15)
                registry.add_all(valid)
                registry.save()
            else:
                print("[VISUAL] No valid figures → text-only")
                return None

        eligible = registry.eligible_cards()

        safe_eligible = []
        for c in eligible:
            if not isinstance(c, FigureCard):
                continue
            if not hasattr(c, "provenance_score"):
                continue
            if not hasattr(c, "eligible"):
                continue
            safe_eligible.append(c)

        if not safe_eligible:
            print("[VISUAL] No eligible figures -> text-only")
            return None

        print(f"[VISUAL] Planner ready: {len(safe_eligible)} figures")
        return ModuleVisualPlanner(registry_cards=safe_eligible)

    except Exception as e:
        import traceback
        print(f"[VISUAL] Setup failed -> text-only: {e}")
        traceback.print_exc()
        return None


__all__ = [
    "extract_figures",
    "FigureCard",
    "FigureRegistry",
    "VisualFact",
    "VLMAnalyzer",
    "VisualQuestionGenerator",
    "ModuleVisualPlanner",
    "VisualVerifier",
    "register_asset_routes",
    "safe_build_planner",
    "_build_module_map",
]
