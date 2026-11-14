"""Read manifest.csv: one row per source PDF (ISIN, document type, as-of date, file name, URL)."""

import csv
from datetime import date, datetime

from etf_assistant.paths import MANIFEST


def load_manifest() -> dict[tuple[str, str], dict]:
    """Rows keyed by (isin, doc_type), e.g. ("IE00B4L5Y983", "kid")."""
    # utf-8-sig: the CSV starts with a byte-order mark (Excel export)
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        return {(r["isin"], r["doc_type"]): r for r in csv.DictReader(f)}


def parse_us_date(s: str) -> date:
    """Manifest dates are written US style, e.g. '9/29/26'."""
    return datetime.strptime(s, "%m/%d/%y").date()
