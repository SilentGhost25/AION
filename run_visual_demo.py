#!/usr/bin/env python3
"""
AION Visual Question Generation Quickstart Demo
Run: python run_visual_demo.py <path_to_pdf_or_docx>
"""

import sys
import json
from pathlib import Path

from v0_1.visual import (
    extract_figures,
    FigureRegistry,
    VLMAnalyzer,
    VisualQuestionGenerator,
    VisualVerifier,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_visual_demo.py <path_to_pdf_or_docx>")
        print("Example: python run_visual_demo.py workspace/uploads/notes.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    path = Path(file_path)

    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"       AION Visual Question Generator Demo        ")
    print(f"==================================================\n")
    print(f"[1] Target File: {path.resolve()}")

    # 1. Generate Document ID & Registry
    doc_id = FigureRegistry.make_document_id(str(path))
    registry = FigureRegistry(document_id=doc_id)
    print(f"[2] Document Hash ID: {doc_id}")

    # 2. Extract Figures
    module_map = {p: f"module_{(p-1)//2 + 1}" for p in range(1, 100)}
    print("[3] Extracting figures and evidence bundles...")
    cards = extract_figures(
        file_path=str(path),
        doc_id=doc_id,
        module_map=module_map,
        asset_dir="extracted_output/assets",
        image_url_prefix="/api/assets"
    )

    if not cards:
        print("[INFO] No figures extracted or file contains text-only content.")
        sys.exit(0)

    print(f"    Extracted {len(cards)} visual figure cards.")

    # 3. Analyze Figures (VLM + Text Fallback)
    print("[4] Running VLM Fact Analysis (timeout=120s per figure for CPU)...")
    analyzer = VLMAnalyzer(timeout=120)
    cards = analyzer.analyze_batch(cards, max_vlm=15)
    for card in cards:
        registry.add(card)

    registry.save()

    # 4. Generate Visual Questions
    print("[5] Generating & Verifying Visual Questions...")
    generator = VisualQuestionGenerator()
    verifier = VisualVerifier()

    generated_count = 0
    for card in registry.eligible_cards():
        q_dict = generator.generate_question(card, marks=10, bloom=4)
        if not q_dict:
            continue

        passed, reason = verifier.verify(q_dict, card, target_module=card.module_id)
        if passed:
            generated_count += 1
            print(f"\n--------------------------------------------------")
            print(f"★ VERIFIED VISUAL QUESTION #{generated_count}")
            print(f"--------------------------------------------------")
            print(f"Question : {q_dict['text']}")
            print(f"Marks    : {q_dict['marks']}")
            print(f"Asset ID : {q_dict['visualEvidence'][0]['id']}")
            print(f"Asset URL: {q_dict['visualEvidence'][0]['url']}")
            print(f"Caption  : {q_dict['visualEvidence'][0]['caption'] or 'None'}")
            print(f"--------------------------------------------------")
        else:
            print(f"[VERIFIER FAIL] {card.id}: {reason} → Fail-closed to text question")

    print(f"\n[SUMMARY] Successfully generated {generated_count} verified visual questions.")


if __name__ == "__main__":
    main()
