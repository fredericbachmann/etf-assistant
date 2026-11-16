"""
Markdown pages -> chunks, the units that are embedded, indexed and retrieved.

Rules:
  1. A chunk never crosses a page, so every chunk has one page to cite.
  2. Pages are split along their structure. Blocks are paragraphs and tables
     (separated by empty lines). Consecutive blocks are packed into one chunk
     up to max_len; a heading starts a new chunk once the current one has
     reached min_len.
  3. A block longer than max_len (legal disclaimers, one long KID paragraph)
     is split by lines (tables) or by sentences (prose).
  4. Each chunk is embedded and indexed with a context header: fund, ISIN,
     document, page, section (Chunk.embedding_text). The cost table on KID p.3
     never names the fund; without the header, the cost tables of all funds
     look the same to search.

Size is measured by the `length` function passed in. pipeline.py passes the
embedding model's token counter: the model reads at most 512 tokens, and
Markdown tables use far more tokens per character than prose (a 1,000-character
table can be 800 tokens), so a character limit can't guarantee that a chunk fits.

Chunks don't overlap. They end at paragraph and heading boundaries, so no
sentence is cut in half. `uv run check-chunks` reports verified facts that are
split across chunks.
"""

import re
from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import BaseModel

# Defaults in characters, for use without a tokenizer. pipeline.py passes token limits instead.
MAX_CHARS = 1500
MIN_CHARS = 300  # a heading only starts a new chunk if the current one is at least this long

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


def split_oversized(block: str, length: Callable[[str], int], max_len: int) -> list[str]:
    """Split a block longer than max_len by lines (tables, lists) or else by sentences."""
    if length(block) <= max_len:
        return [block]
    lines = block.split("\n")
    units, sep = (lines, "\n") if len(lines) > 1 else (re.split(r"(?<=[.;])\s+", block), " ")
    pieces, cur = [], ""
    for u in units:
        if cur and length(f"{cur}{sep}{u}") > max_len:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}{sep}{u}" if cur else u
    pieces.append(cur)
    # a single line or sentence can still exceed max_len; it is rare and kept whole
    return pieces


def chunk_pages(pages: list[str], *, isin: str, fund_name: str, doc_type: str,
                as_of_date: date, source_file: str,
                length: Callable[[str], int] = len,
                max_len: int = MAX_CHARS, min_len: int = MIN_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []

    def size(blocks: list[str]) -> int:
        return length("\n\n".join(blocks)) if blocks else 0

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
                if size(buf) >= min_len:
                    flush()  # rule 2: a heading starts a new chunk...
                section = h  # ...unless the current one is tiny; then it joins it
                headings.append(h)
            for piece in split_oversized(block, length, max_len):
                if buf and size(buf + [piece]) > max_len:
                    flush()
                buf.append(piece)
        flush()  # rule 1: end of page ends the chunk
    return chunks
