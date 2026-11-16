"""
PDF -> Markdown, one string per page.

Markdown, because fund documents keep their facts in tables. pymupdf4llm
detects tables and headings, so a label stays next to its value:

    |Number of stocks<br>3,821|4,203|

In plain text (which check_facts.py reads) "Number of stocks" and "3,821" can
end up far apart.

One string per page, because every chunk records its page: answers cite
"KID p.3", and check-chunks looks for each verified quote on its cited page.

Parsing all PDFs takes about 30 s, so the result is cached in data/parsed/:
  <stem>.json  the pages as a JSON list (read by the chunker)
  <stem>.md    the same pages in one readable file, for inspecting the parser's output
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
        use_ocr=False,  # all the PDFs have a text layer; OCR is slow and adds noise
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
