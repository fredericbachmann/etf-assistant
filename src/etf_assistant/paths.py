"""Project paths in one place, so every module agrees on where the data lives."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/etf_assistant/paths.py -> project root
DATA = ROOT / "data"
FACTS = DATA / "facts"
MANIFEST = ROOT / "manifest.csv"
EVALS = ROOT / "evals"
