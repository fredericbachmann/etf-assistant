"""
Fact coverage of the chunks: for every quote in data/facts, is it intact inside
ONE chunk on the cited page?

A quote that isn't can't be retrieved cleanly, no matter how good the search
is. It's an upper bound on what retrieval can get right, measured before any
search exists. It lives here, not in the pipeline, because it reads the
answer key (data/facts), and nothing outside evals/ may do that.

Usage:
  uv run check-chunks       # after `uv run ingest`
"""

import re

from etf_assistant.corpus.chunking import Chunk
from etf_assistant.evals.fund_facts import FundFacts, Sourced
from etf_assistant.evals.textmatch import norm, plain
from etf_assistant.paths import FACTS
from etf_assistant.pipeline import load_chunks


def nospace(s: str) -> str:
    return re.sub(r"\s", "", s)


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
            if any(nospace(q) in nospace(t) for t in texts):
                # the parser lost/added spaces or cut a word at a table cell: "hold|ing period"
                why = "damaged (spacing / cell break)"
            elif q in " ".join(texts):
                why = "split across chunks"
            else:
                why = "not in parsed text"
            misses.append(f"{facts.isin} {name:32} {s.doc} p.{s.page}  {why}: {s.quote!r}")
    return total, misses


def main() -> None:
    total, misses = fact_coverage(load_chunks())
    print(f"fact coverage: {total - len(misses)}/{total} quotes intact in one chunk on the cited page")
    for m in misses:
        print("  MISS", m)


if __name__ == "__main__":
    main()
