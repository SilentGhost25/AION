#!/usr/bin/env python
r"""
AION CLI — Quick Question Generation Runner

Usage:
    python aion.py <file_path>
    python aion.py <file_path> -n 20
    python aion.py <file_path> -n 15 -m balanced
    python aion.py <file_path> --mode deep --model mistral:7b
    python aion.py --help

Examples:
    python aion.py "C:\Users\Tarun J\Downloads\notes.pdf"
    python aion.py "C:\path\to\textbook.pdf" -n 20
    python aion.py "C:\path\to\textbook.pdf" -n 15 -m balanced
    python aion.py "C:\path\to\textbook.pdf" --mode deep
    python aion.py "D:\Documents\AI Module 1.pdf" -n 5 -m turbo

Modes:
    turbo     — Question only, fastest (~3-5s per question)
    balanced  — Question + Answer, moderate speed
    deep      — Full question + detailed answer + marking scheme
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="aion",
        description="AION — Academic Question Generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
==============================================================
Examples:
==============================================================
  python aion.py "notes.pdf"
  python aion.py "textbook.pdf" -n 20
  python aion.py "textbook.pdf" -n 15 -m balanced
  python aion.py "textbook.pdf" --mode deep
  python aion.py "textbook.pdf" --model mistral:7b
  python aion.py "folder/Module1.pdf" -n 10 -m turbo -o questions.txt
==============================================================
Modes:
  turbo     Fast question-only generation (~3-5s per question)
  balanced  Question + ideal answer (default quality)
  deep      Comprehensive answer + marking scheme
==============================================================
        """,
    )

    # ── Positional argument ───────────────────────────────────
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to PDF, TXT file, or directory (required)",
    )

    # ── Optional arguments ────────────────────────────────────
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=10,
        metavar="N",
        help="Number of questions to generate (default: 10)",
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
        help="Ollama model to use (default: qwen2.5:3b)",
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
        version="AION v0.1 — Academic Question Generator",
    )

    # ── Parse arguments ───────────────────────────────────────
    args = parser.parse_args()

    # ── Validate path ─────────────────────────────────────────
    if not args.path:
        parser.print_help()
        print("\n[ERROR] No file or directory path provided.")
        print("Usage: python aion.py <path_to_pdf_txt_or_directory>")
        sys.exit(1)

    file_path = Path(args.path)

    if not file_path.exists():
        print(f"[ERROR] Path not found: {args.path}")
        sys.exit(1)

    if not file_path.is_dir() and not file_path.suffix.lower() in [".pdf", ".txt", ".md", ".docx", ".pptx"]:
        print(f"[ERROR] Unsupported file type: {file_path.suffix}")
        print("Supported: .pdf, .txt, .md, .docx, .pptx, or a directory of these files")
        sys.exit(1)

    # ── Setup environment ─────────────────────────────────────
    # Change to AION directory
    aion_dir = Path(__file__).parent.resolve()
    os.chdir(aion_dir)
    sys.path.insert(0, str(aion_dir))

    # Set model
    if args.model:
        os.environ["AION_MODEL"] = args.model
    elif "AION_MODEL" not in os.environ:
        os.environ["AION_MODEL"] = "qwen2.5:3b"

    # ── Print banner ──────────────────────────────────────────
    if not args.quiet:
        print()
        print("+----------------------------------------------------------+")
        print("|           AION - Academic Question Generator             |")
        print("+----------------------------------------------------------+")
        print(f"|  Path   : {file_path.name[:45]:<45} |")
        print(f"|  Mode   : {args.mode:<45} |")
        print(f"|  Count  : {args.num:<45} |")
        print(f"|  Model  : {os.environ.get('AION_MODEL', 'qwen2.5:3b'):<45} |")
        print("+----------------------------------------------------------+")
        print()

    # ── Run pipeline ──────────────────────────────────────────
    try:
        from v0_1.main import run_pipeline

        accepted, rejected = run_pipeline(
            str(file_path),
            max_concepts=args.num,
            mode=args.mode,
        )

        # ── Save to file if requested ─────────────────────────
        if args.output:
            _save_output(accepted, rejected, args.output, args.mode)
            print(f"\n[SAVED] Output written to: {args.output}")

        # ── Return code ───────────────────────────────────────
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


def _save_output(accepted, rejected, output_path, mode):
    """Save generated questions to a file."""
    from datetime import datetime

    lines = [
        "=" * 60,
        "AION — Generated Question Paper",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {mode}",
        f"Total Questions: {len(accepted)}",
        "=" * 60,
        "",
    ]

    for i, q in enumerate(accepted, 1):
        lines.append(f"Q{i} [{q.marks} Marks | Bloom Level: {q.bloom_level}]")
        lines.append("-" * 50)
        lines.append(q.question_text)
        lines.append("")

        if q.ideal_answer:
            lines.append("Ideal Answer:")
            lines.append(q.ideal_answer)
            lines.append("")

        lines.append("")

    if rejected:
        lines.append("=" * 60)
        lines.append(f"REJECTED QUESTIONS: {len(rejected)}")
        lines.append("=" * 60)
        for q, reason in rejected[:5]:
            lines.append(f"[{reason}] {q.question_text[:80]}...")
            lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Shortcut functions for interactive use
# ─────────────────────────────────────────────────────────────

def run(path, n=10, mode="turbo"):
    """
    Quick run function for interactive Python sessions.

    Usage:
        from aion import run
        run("path/to/file.pdf")
        run("path/to/file.pdf", n=20, mode="balanced")
    """
    os.environ.setdefault("AION_MODEL", "qwen2.5:3b")

    aion_dir = Path(__file__).parent.resolve()
    os.chdir(aion_dir)
    sys.path.insert(0, str(aion_dir))

    from v0_1.main import run_pipeline
    return run_pipeline(path, max_concepts=n, mode=mode)


def turbo(path, n=10):
    """Shortcut for turbo mode."""
    return run(path, n=n, mode="turbo")


def balanced(path, n=10):
    """Shortcut for balanced mode."""
    return run(path, n=n, mode="balanced")


def deep(path, n=10):
    """Shortcut for deep mode."""
    return run(path, n=n, mode="deep")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
