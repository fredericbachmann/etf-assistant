"""Project paths in one place, so every module agrees on where the data lives."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/etf_assistant/paths.py -> project root
DATA = ROOT / "data"
FACTS = DATA / "facts"
MANIFEST = ROOT / "manifest.csv"
EVALS = ROOT / "evals"
PARSED = DATA / "parsed"  # cached Markdown per PDF (step 2a)
CHUNKS = DATA / "chunks.jsonl"  # all chunks, one JSON object per line (step 2b)
