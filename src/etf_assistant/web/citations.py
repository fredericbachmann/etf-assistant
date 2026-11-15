"""Turn the assistant's citations into links to the cited PDF page.

The system prompt asks for [IE00B4L5Y983 KID p.3]; answers also contain variants:
  [IE00B4L5Y983 KID p.3; IE00B4L5Y983 factsheet p.1]   several sources, separated by ";"
  [IE00B4L5Y983 KID p.3; factsheet p.1, p.2]           ISIN given once, several pages
  [IE00B4L5Y983 factsheet p.1, as of 30 Sep 2026]      extra words
  [IE00B4L5Y983 KID, Objectives section]               no page
A source's ISIN and document type carry over to the next source that omits them. Each
page reference becomes a link to that page (#page=N); a source without a page links its
document type. Brackets without an ISIN, and documents without a URL, stay plain text.

Applied to the whole text each time it is shown, so while an answer streams, a citation
whose closing bracket hasn't arrived yet simply stays plain until it does.
"""

import re
from collections.abc import Callable

UrlFor = Callable[[str, str], str | None]  # (isin, doc_type) -> URL of the PDF, or None

_ISIN = r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b"
CITATION = re.compile(rf"\[([^\[\]]*{_ISIN}[^\[\]]*)\](?!\()")  # (?!\() skips existing Markdown links
ISIN = re.compile(_ISIN)
DOC_TYPE = re.compile(r"\b(KID|factsheet)\b", re.IGNORECASE)
PAGE = re.compile(r"\bp\.\s?(\d+)")


def link_citations(text: str, url_for: UrlFor) -> str:
    def link_bracket(m: re.Match) -> str:
        isin = doc_type = None
        sources = []
        for source in m.group(1).split(";"):
            if found := ISIN.search(source):
                isin = found.group()
            if found := DOC_TYPE.search(source):
                doc_type = found.group().lower()
            url = url_for(isin, doc_type) if isin and doc_type else None
            if url and PAGE.search(source):
                source = PAGE.sub(lambda p: f"[{p.group()}]({url}#page={p.group(1)})", source)
            elif url:
                source = DOC_TYPE.sub(lambda d: f"[{d.group()}]({url})", source, count=1)
            sources.append(source)
        linked = ";".join(sources)
        if linked == m.group(1):
            return m.group()
        # escaped brackets: shown as text, and can't pair up with the links inside
        return r"\[" + linked + r"\]"

    return CITATION.sub(link_bracket, text)
