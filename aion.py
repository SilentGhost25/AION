#!/usr/bin/env python
r"""
AION CLI — Question Generation Runner with strict VTU Exam layouts.
"""

import argparse
import os
import sys

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="aion",
        description="AION — VTU Academic Question Generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
==============================================================
Examples:
==============================================================
  python aion.py "notes.pdf"
  python aion.py "textbook.pdf" -e see
  python aion.py "textbook.pdf" -e ia
  python aion.py "folder/Module1.pdf" -e see -o questions.txt
==============================================================
        """,
    )

    # -- Positional argument -----------------------------------
    parser.add_argument(
        "path",
        help="Path to PDF, TXT file, or directory (required)",
    )

    # -- Optional arguments ------------------------------------
    parser.add_argument(
        "-e", "--exam",
        choices=["ia", "see"],
        default="see",
        help="Exam structure: ia (10-mark units) | see (20-mark units) (default: see)",
    )

    parser.add_argument(
        "-n", "--num",
        type=int,
        default=10,
        metavar="N",
        help="Number of concepts limit (default: 10)",
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["turbo", "balanced", "deep"],
        default="turbo",
        help="Generation mode: turbo|balanced|deep (default: turbo)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Ollama model to use (default: qwen2.5:7b)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save output to file (optional)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output, show only results",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version="AION v0.1 — VTU Academic Question Generator",
    )

    # -- Parse arguments ---------------------------------------
    args = parser.parse_args()

    # -- Validate path -----------------------------------------
    file_path = Path(args.path)

    if not file_path.exists():
        print(f"[ERROR] Path not found: {args.path}")
        sys.exit(1)

    # -- Setup environment -------------------------------------
    # Change to AION directory
    aion_dir = Path(__file__).parent.resolve()
    os.chdir(aion_dir)
    sys.path.insert(0, str(aion_dir))

    # Set model
    if args.model:
        os.environ["AION_MODEL"] = args.model
    elif "AION_MODEL" not in os.environ:
        os.environ.setdefault("AION_MODEL", "qwen2.5:3b-instruct")

    # -- Print banner ------------------------------------------
    if not args.quiet:
        print()
        print("+----------------------------------------------------------+")
        print("|      AION - VTU Academic Question Generator              |")
        print("+----------------------------------------------------------+")
        print(f"|  Path   : {file_path.name[:45]:<45} |")
        print(f"|  Exam   : {args.exam.upper():<45} |")
        print(f"|  Mode   : {args.mode:<45} |")
        print(f"|  Model  : {os.environ.get('AION_MODEL', 'qwen2.5:7b'):<45} |")
        print("+----------------------------------------------------------+")
        print()

    # -- Run pipeline ------------------------------------------
    try:
        from v0_1.main import run_pipeline

        accepted, rejected = run_pipeline(
            str(file_path),
            max_concepts=args.num,
            mode=args.mode,
            exam_type=args.exam
        )

        # -- Save to file if requested -------------------------
        if args.output:
            _save_output(accepted, args.output, args.exam)
            print(f"\n[SAVED] Output written to: {args.output}")

        # -- Return code ---------------------------------------
        if len(accepted) > 0:
            sys.exit(0)
        else:
            print("[WARN] No questions generated.")
            sys.exit(1)

    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Make sure you are running from the AION directory.")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[CANCELLED] Generation interrupted by user.")
        sys.exit(130)

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _save_output(paper, output_path, exam_type):
    """Save generated questions to a file."""
    from datetime import datetime

    lines = [
        "=" * 60,
        f"AION — Generated VTU {exam_type.upper()} Question Paper",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    for mod in paper:
        lines.append(f"MODULE {mod['module_index']}: {mod['module_title'].upper()}")
        lines.append("-" * 60)
        
        # Helper to format a main question block
        def format_mq(mq):
            mq_lines = []
            prefix = f"Q{mq['mq_index']} "
            bloom_tag = f" [Bloom Level {mq['bloom_level']}: {mq['bloom_name']}]"
            
            if len(mq["sub_questions"]) == 1:
                sq = mq["sub_questions"][0]
                mq_lines.append(f"{prefix}{sq['text']} ({sq['marks']} Marks){bloom_tag}")
            else:
                mq_lines.append(f"{prefix}Answer the following subquestions:{bloom_tag}")
                for sq in mq["sub_questions"]:
                    mq_lines.append(f"   ({sq['letter']}) {sq['text']} ({sq['marks']} Marks)")
            return mq_lines

        lines.extend(format_mq(mod["questions"][0]))
        lines.append(f"{' '*30}[OR]")
        lines.extend(format_mq(mod["questions"][1]))
        
        lines.append("")
        lines.append("· " * 30)
        lines.append("")
        
        lines.extend(format_mq(mod["questions"][2]))
        lines.append(f"{' '*30}[OR]")
        lines.extend(format_mq(mod["questions"][3]))
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
