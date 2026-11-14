"""Project paths in one place, so every module agrees on where the data lives."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/etf_assistant/paths.py -> project root
DATA = ROOT / "data"
FACTS = DATA / "facts"
MANIFEST = ROOT / "manifest.csv"
EVALS = ROOT / "evals"
PARSED = DATA / "parsed"  # cached Markdown per PDF
CHUNKS = DATA / "chunks.jsonl"  # all chunks, one JSON object per line
EMBEDDING_CACHE = DATA / "embedding_cache"  # passage vectors, so re-indexing needn't re-embed
CATALOG = DATA / "catalog.json"  # fund names and tickers extracted from the documents
