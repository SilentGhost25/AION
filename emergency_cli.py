"""
AION Emergency CLI Mode
=======================
Command-line emergency question paper generator.
Bypasses Flask API entirely. Fast and local.
Usage:
  python emergency_cli.py <pdf_file> [n_questions]
"""

import sys
import json
from pathlib import Path
from v0_1.minimal_pipeline import emergency_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python emergency_cli.py <pdf_file> [n_questions]")
        print("\nExample:")
        print("  python emergency_cli.py notes.pdf 10")
        sys.exit(1)

    pdf_path = sys.argv[1]
    n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"PDF: {pdf_path}")
    print(f"Target: {n_questions} questions")
    print()

    try:
        result = emergency_pipeline(
            pdf_path=pdf_path,
            n_questions=n_questions
        )

        output_file = Path(pdf_path).stem + "_emergency.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Saved: {output_file}")

        print("\n" + "="*60)
        print("GENERATED QUESTIONS")
        print("="*60)

        for i, q in enumerate(result["questions"], 1):
            print(f"\nQ{i} [{q['marks']}M | {q['rbtl']}]:")
            print(f"  {q['question']}")

    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
