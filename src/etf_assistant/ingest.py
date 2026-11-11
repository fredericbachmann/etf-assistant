"""
Step 2: parse every PDF in the manifest, chunk it, write data/chunks.jsonl, and report.

Usage:
  uv run ingest            # parse (cached), chunk, report
  uv run ingest --force    # re-parse all PDFs, ignoring the cache
  uv run ingest --show IE00B4L5Y983_kid_p3_c1    # print one chunk as it will be embedded

The report ends with "fact coverage": for every quote in data/facts, is it
intact inside ONE chunk on the cited page? A quote that isn't can't be
retrieved cleanly, no matter how good the search is. It's an upper bound on
what the RAG system can get right, measured before any search exists.
"""

import json
import re
import statistics
import sys
from collections import Counter

from etf_assistant.check_facts import norm
from etf_assistant.chunking import MAX_CHARS, Chunk, chunk_pages
from etf_assistant.fund_facts import FundFacts, Sourced
from etf_assistant.manifest import load_manifest, parse_us_date
from etf_assistant.parse import load_pages
from etf_assistant.paths import CHUNKS, DATA, FACTS


def display_name(facts: FundFacts) -> str:
    """Fund name plus share class, so the twins VWRL/VWCE get different headers."""
    name = facts.fund_name.value or facts.isin
    policy = facts.distribution_policy.value
    return f"{name}, {policy} share class" if policy else name


def build_chunks(force: bool = False) -> list[Chunk]:
    facts = {f.stem: FundFacts.model_validate_json(f.read_text()) for f in FACTS.glob("*.json")}
    chunks = []
    for (isin, doc_type), row in sorted(load_manifest().items()):
        pdf = DATA / row["file_path"]
        print(f"  {pdf.name}", file=sys.stderr)
        chunks += chunk_pages(
            load_pages(pdf, force=force),
            isin=isin,
            fund_name=display_name(facts[isin]) if isin in facts else isin,
            doc_type=doc_type,
            as_of_date=parse_us_date(row["as_of_date"]),
            source_file=row["file_path"],
        )
    return chunks


def plain(md: str) -> str:
    """Markdown -> plain text for comparing with quotes: drop table pipes, bold, heading marks."""
    md = re.sub(r"<br>|\|", " ", md)
    md = re.sub(r"\*\*|^#+\s*|^-{3,}$|^- ", "", md, flags=re.M)
    return norm(md)


def fact_coverage(chunks: list[Chunk]) -> tuple[int, list[str]]:
    by_page: dict[tuple, list[str]] = {}
    for c in chunks:
        by_page.setdefault((c.isin, c.doc_type, c.page), []).append(plain(c.text))
    total, misses = 0, []
    for f in sorted(FACTS.glob("*.json")):
        facts = FundFacts.model_validate_json(f.read_text())
        for name in FundFacts.model_fields:
            s = getattr(facts, name)
            if not isinstance(s, Sourced) or not s.quote:
                continue
            total += 1
            texts = by_page.get((facts.isin, s.doc, s.page), [])
            q = norm(s.quote)
            if any(q in t for t in texts):
                continue
            nospace = lambda t: re.sub(r"\s", "", t)  # noqa: E731
            if any(nospace(q) in nospace(t) for t in texts):
                # the parser lost/added spaces or cut a word at a table cell: "hold|ing period"
                why = "damaged (spacing / cell break)"
            elif q in " ".join(texts):
                why = "split across chunks"
            else:
                why = "not in parsed text"
            misses.append(f"{facts.isin} {name:32} {s.doc} p.{s.page}  {why}: {s.quote!r}")
    return total, misses


def report(chunks: list[Chunk]) -> None:
    sizes = [len(c.text) for c in chunks]
    per_doc = Counter((c.isin, c.doc_type) for c in chunks)
    print(f"\n{len(chunks)} chunks from {len(per_doc)} documents -> {CHUNKS.relative_to(DATA.parent)}")
    print(f"chunk size (chars): min {min(sizes)}, median {int(statistics.median(sizes))}, "
          f"max {max(sizes)}; {sum(s > MAX_CHARS for s in sizes)} over MAX_CHARS={MAX_CHARS}")
    for doc_type in ("kid", "factsheet"):
        counts = [n for (_, t), n in per_doc.items() if t == doc_type]
        print(f"{doc_type:9}: {len(counts)} docs, {min(counts)}-{max(counts)} chunks each")
    total, misses = fact_coverage(chunks)
    print(f"\nfact coverage: {total - len(misses)}/{total} quotes intact in one chunk on the cited page")
    for m in misses:
        print("  MISS", m)


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--show"]:
        wanted = set(args[1:])
        for line in CHUNKS.read_text(encoding="utf-8").splitlines():
            c = Chunk.model_validate_json(line)
            if c.chunk_id in wanted:
                print(f"----- {c.chunk_id} ({len(c.text)} chars)\n{c.embedding_text()}\n")
        return
    chunks = build_chunks(force="--force" in args)
    CHUNKS.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    report(chunks)


if __name__ == "__main__":
    main()
