"""
Data setup: turn the PDFs in manifest.csv into a searchable index.

  1. ingest   parse every PDF (cached), chunk it, write data/chunks.jsonl, report sizes
  2. index    embed the chunks and (re)build the OpenSearch index from data/chunks.jsonl

The fund catalog (names, tickers) is a separate step, `uv run build-catalog`,
because it costs LLM calls; ingest uses it for the fund names in chunk headers.

Usage:
  uv run ingest            # parse (cached), chunk, report
  uv run ingest --force    # re-parse all PDFs, ignoring the cache
  uv run ingest --show IE00B4L5Y983_kid_p3_c1    # print one chunk as it will be embedded
  uv run index             # needs OpenSearch: docker compose up -d
  uv run check-chunks      # are the verified facts intact inside one chunk? (evals/check_chunks.py)
"""

import statistics
import sys
from collections import Counter

from etf_assistant.corpus.catalog import load_catalog
from etf_assistant.corpus.chunking import Chunk, chunk_pages
from etf_assistant.corpus.manifest import load_manifest, parse_us_date
from etf_assistant.corpus.parse import load_pages
from etf_assistant.paths import CATALOG, CHUNKS, DATA
from etf_assistant.retrieval.embeddings import MAX_TOKENS, count_tokens
from etf_assistant.retrieval.store import index_chunks

# Chunk text budget in embedding-model tokens. The context header (~30-60 tokens)
# is added before embedding, so leave room for it below MAX_TOKENS=512.
CHUNK_MAX_TOKENS = 420
CHUNK_MIN_TOKENS = 70


def build_chunks(force: bool = False) -> list[Chunk]:
    # Names come from the catalog (extracted from the documents), never from data/facts:
    # that's the evals' answer key. The share class keeps the twins VWRL/VWCE apart.
    catalog = load_catalog() if CATALOG.exists() else {}
    chunks = []
    for (isin, doc_type), row in sorted(load_manifest().items()):
        pdf = DATA / row["file_path"]
        print(f"  {pdf.name}", file=sys.stderr)
        chunks += chunk_pages(
            load_pages(pdf, force=force),
            isin=isin,
            fund_name=catalog[isin].display_name if isin in catalog else isin,
            doc_type=doc_type,
            as_of_date=parse_us_date(row["as_of_date"]),
            source_file=row["file_path"],
            length=count_tokens, max_len=CHUNK_MAX_TOKENS, min_len=CHUNK_MIN_TOKENS,
        )
    return chunks


def load_chunks() -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


def report(chunks: list[Chunk]) -> None:
    sizes = [len(c.text) for c in chunks]
    tokens = [count_tokens(c.embedding_text()) for c in chunks]
    per_doc = Counter((c.isin, c.doc_type) for c in chunks)
    print(f"\n{len(chunks)} chunks from {len(per_doc)} documents -> {CHUNKS.relative_to(DATA.parent)}")
    print(f"chunk size (chars):  min {min(sizes)}, median {int(statistics.median(sizes))}, max {max(sizes)}")
    print(f"embedded tokens:     min {min(tokens)}, median {int(statistics.median(tokens))}, max {max(tokens)}; "
          f"{sum(t > MAX_TOKENS for t in tokens)} over the model's {MAX_TOKENS} (would be truncated)")
    for doc_type in ("kid", "factsheet"):
        counts = [n for (_, t), n in per_doc.items() if t == doc_type]
        print(f"{doc_type:9}: {len(counts)} docs, {min(counts)}-{max(counts)} chunks each")


def ingest_cli() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--show"]:
        wanted = set(args[1:])
        for c in load_chunks():
            if c.chunk_id in wanted:
                print(f"----- {c.chunk_id} ({len(c.text)} chars)\n{c.embedding_text()}\n")
        return
    chunks = build_chunks(force="--force" in args)
    CHUNKS.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    report(chunks)


def index_cli() -> None:
    index_chunks(load_chunks())
