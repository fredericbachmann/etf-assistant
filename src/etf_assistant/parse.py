"""
PDF -> Markdown, one string per page.

Why Markdown and not plain text? Fund documents keep their facts in tables.
Plain text (what check_facts.py reads) loses the rows: "Number of stocks" and
"3,821" end up far apart. pymupdf4llm detects tables and headings and keeps
them as Markdown, so a label stays next to its value:

    |Number of stocks<br>3,821|4,203|

Why one string per page? Every chunk must know its page, so answers can cite
"KID p.3" and the evals can check that retrieval found the right page.

Parsing all PDFs takes ~30 s, so the result is cached in data/parsed/:
  <stem>.json  the pages as a JSON list (what the chunker reads)
  <stem>.md    the same, readable, so you can see what the parser produced
"""

import json
import re
from pathlib import Path

import pymupdf4llm

from etf_assistant.paths import PARSED


def clean_markdown(md: str) -> str:
    md = re.sub(r"</?(?:mark|u|sup|sub)>", "", md)  # styling tags; keep <br> (line break inside a table cell)
    md = re.sub(r"[ \t]+\n", "\n", md)  # trailing spaces
    md = re.sub(r"\n{3,}", "\n\n", md)  # runs of empty lines left by images and drawings
    return md.strip()


def parse_pdf(pdf: Path) -> list[str]:
    pages = pymupdf4llm.to_markdown(
        str(pdf),
        page_chunks=True,  # a list with one dict per page instead of one big string
        use_ocr=False,  # all our PDFs have a text layer; OCR is slow and adds noise
        show_progress=False,
    )
    return [clean_markdown(p["text"]) for p in pages]


def load_pages(pdf: Path, force: bool = False) -> list[str]:
    """Pages of `pdf` as Markdown, from the cache unless the PDF is newer (or force=True)."""
    cache = PARSED / f"{pdf.stem}.json"
    if not force and cache.exists() and cache.stat().st_mtime >= pdf.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))

    pages = parse_pdf(pdf)
    PARSED.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
    readable = "\n\n".join(f"<!-- ===== page {i} ===== -->\n\n{p}" for i, p in enumerate(pages, 1))
    cache.with_suffix(".md").write_text(readable, encoding="utf-8")
    return pages
