"""Backward-compatible wrapper. Prefer: python scripts/tune_model.py --model lstm"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tune_model import main

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "lstm"]
    main()
