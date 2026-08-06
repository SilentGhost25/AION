#!/usr/bin/env python
"""Ultra-short AION runner. Usage: python a.py "file.pdf" [n] [mode]"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
os.environ.setdefault("AION_MODEL", "qwen2.5:7b")
from v0_1.main import run_pipeline
path = sys.argv[1] if len(sys.argv) > 1 else input("File path: ")
n    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
mode = sys.argv[3] if len(sys.argv) > 3 else "turbo"
run_pipeline(path, max_concepts=n, mode=mode)
