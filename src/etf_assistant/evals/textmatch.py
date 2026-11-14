"""Normalising text so a quote from data/facts can be found in PDF text or in a chunk.

PDF text and parsed Markdown differ from a hand-copied quote in whitespace,
Unicode forms (ligatures, non-breaking spaces) and Markdown syntax. These
functions remove those differences before comparing.
"""

import re
import unicodedata


def norm(s: str) -> str:
    """Unicode NFKC, all whitespace runs -> one space."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def plain(md: str) -> str:
    """Markdown -> plain text for comparing with quotes: drop table pipes, bold, heading marks."""
    md = re.sub(r"<br>|\|", " ", md)
    md = re.sub(r"\*\*|^#+\s*|^-{3,}$|^- ", "", md, flags=re.M)
    return norm(md)


def squash(s: str) -> str:
    """plain() without any whitespace, lowercased: survives spacing damage from the PDF parser."""
    return re.sub(r"\s", "", plain(s)).lower()
