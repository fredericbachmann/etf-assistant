"""
Step 2b: Markdown pages -> chunks.

A chunk is the unit of retrieval: the search engine returns chunks, and the
LLM only sees the chunks it gets. So a chunk should be
  - small enough to be about ONE thing (a vector of a 10-page document means
    "everything" and matches nothing well),
  - big enough to be self-contained (the cost table with its heading, not
    half a row).

Rules used here:
  1. Never cross a page: every chunk has one exact page for citations.
  2. Split along the document's structure: blocks are paragraphs/tables
     (separated by empty lines); a heading starts a new chunk; consecutive
     blocks are packed into one chunk up to MAX_CHARS.
  3. A block longer than MAX_CHARS (legal disclaimers, one long KID paragraph)
     is split by lines (tables) or sentences (prose).
  4. Each chunk gets a context header (fund, ISIN, document, page, section)
     before it is embedded. The cost table on KID p.3 never names the fund;
     without the header, the costs of all 11 funds look identical to search.

No overlap between chunks: overlap is a fix for blind fixed-size splitting
that cuts sentences in half. We split at paragraph and heading boundaries,
so there's little to repair. If the evals show facts split across chunks,
this is the place to revisit.
"""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel

MAX_CHARS = 1500  # ~350-400 tokens; leaves room for the header within a 512-token embedding model
MIN_CHARS = 300  # a heading only starts a new chunk if the current one has at least this much text

HEADING = re.compile(r"^#{1,6}\s+(.+)$")


class Chunk(BaseModel):
    chunk_id: str  # e.g. "IE00B4L5Y983_kid_p3_c1"
    isin: str
    fund_name: str
    doc_type: Literal["kid", "factsheet"]
    as_of_date: date
    page: int  # 1-based, as in the facts files
    section: str  # headings inside this chunk, or the section it continues
    source_file: str
    text: str

    def embedding_text(self) -> str:
        """What gets embedded and indexed: context header + chunk text."""
        doc = "Key Information Document (KID)" if self.doc_type == "kid" else "Factsheet"
        section = f" | Section: {self.section}" if self.section else ""
        return f"{self.fund_name} ({self.isin}) | {doc}, page {self.page}{section}\n\n{self.text}"


def heading_text(block: str) -> str | None:
    """'## **Risk Indicator**' -> 'Risk Indicator'; None if the block doesn't start with a heading."""
    m = HEADING.match(block.split("\n", 1)[0].strip())
    if not m:
        return None
    return re.sub(r"\*\*|__", "", m.group(1)).strip() or None  # bold markers, not "(*)"


def shorten(section: str, limit: int = 120) -> str:
    """Packed chunks can contain several headings; keep the header line short."""
    return section if len(section) <= limit else section[: limit - 1].rstrip() + "…"


def split_oversized(block: str) -> list[str]:
    """Split a block longer than MAX_CHARS by lines (tables, lists) or else by sentences."""
    if len(block) <= MAX_CHARS:
        return [block]
    lines = block.split("\n")
    units, sep = (lines, "\n") if len(lines) > 1 else (re.split(r"(?<=[.;])\s+", block), " ")
    pieces, cur = [], ""
    for u in units:
        if cur and len(cur) + len(sep) + len(u) > MAX_CHARS:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}{sep}{u}" if cur else u
    pieces.append(cur)
    # a single line/sentence can still exceed MAX_CHARS; rare, and we keep it whole
    return pieces


def chunk_pages(pages: list[str], *, isin: str, fund_name: str, doc_type: str,
                as_of_date: date, source_file: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""  # carried across pages: a section that starts on p.2 can continue on p.3

    for page_no, md in enumerate(pages, 1):
        buf: list[str] = []  # blocks of the chunk being built
        headings: list[str] = []  # headings inside it
        start_section = section  # section in effect when the chunk started
        n_on_page = 0

        def flush():
            nonlocal buf, headings, start_section, n_on_page
            if buf:
                n_on_page += 1
                chunks.append(Chunk(
                    chunk_id=f"{isin}_{doc_type}_p{page_no}_c{n_on_page}",
                    isin=isin, fund_name=fund_name, doc_type=doc_type, as_of_date=as_of_date,
                    page=page_no, section=shorten(" / ".join(headings) or start_section),
                    source_file=source_file, text="\n\n".join(buf),
                ))
            buf, headings, start_section = [], [], section

        for block in (b.strip() for b in re.split(r"\n\s*\n", md)):
            if not block:
                continue
            if h := heading_text(block):
                if sum(map(len, buf)) >= MIN_CHARS:
                    flush()  # rule 2: a heading starts a new chunk...
                section = h  # ...unless the current one is tiny; then it joins it
                headings.append(h)
            for piece in split_oversized(block):
                if buf and sum(map(len, buf)) + len(piece) > MAX_CHARS:
                    flush()
                buf.append(piece)
        flush()  # rule 1: end of page ends the chunk
    return chunks
