"""
Mechanical checks for data/facts/<ISIN>.json before (and after) hand verification.

For every Sourced field that has a value it checks that:
  - doc and page are set, and the page exists in that document (resolved via manifest.csv)
  - the quote occurs on that page (ignoring whitespace and line breaks)
  - the value can be read from the quote (numbers, dates, enum keywords)

It cannot tell you the value is *right*. It tells you the citation is real and
consistent, so the hand pass is just: open the page, read the quote, agree or fix.

Usage:
  uv run check-facts                # all files
  uv run check-facts IE00B4L5Y983   # one fund, with a full checklist
"""

import json
import re
import sys
import unicodedata
from datetime import date, datetime
from functools import cache
from pathlib import Path

import pymupdf

from etf_assistant.fund_facts import FundFacts, Sourced
from etf_assistant.manifest import load_manifest
from etf_assistant.paths import DATA, FACTS

# Keywords that must appear in the quote for enum / bool values.
KEYWORDS = {
    "distribution_policy": {
        "accumulating": r"accumulat|reinvest",
        "distributing": r"distribut|paid out",
    },
    "replication": {
        "physical_full": r"replicat",
        "physical_sampling": r"optimis|optimiz|sampl",
        "synthetic": r"synthetic|swap",
    },
}


@cache
def page_texts(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as doc:
        return [norm(p.get_text()) for p in doc]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def numbers_in(s: str) -> list[float]:
    # "1,250" -> 1250, "0.12 %" -> 0.12, "10.000 USD" is not used for values we check
    return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", s)]


DATE_FORMATS = ["%d-%b-%Y", "%d %B %Y", "%d %b %Y", "%d/%m/%Y", "%d.%m.%Y", "%B %d, %Y"]
# day-month-year ("25-Sep-2009", "22/07/2014") or US style ("september 30, 2026", Xtrackers)
DATE_RE = r"\d{1,2}[-/. ](?:[A-Za-z]+|\d{1,2})[-/. ]\d{4}|[A-Za-z]+ \d{1,2}, \d{4}"


def dates_in(s: str) -> list[date]:
    out = []
    for m in re.findall(DATE_RE, s):
        for fmt in DATE_FORMATS:
            try:
                out.append(datetime.strptime(m, fmt).date())
                break
            except ValueError:
                pass
    return out


def check_value(name: str, v, quote: str) -> str | None:
    """Return an error string if the value can't be read from the quote, else None."""
    q = norm(quote)
    if isinstance(v, bool):
        if v and not re.search(r"hedg", q, re.I):
            return "value True but quote doesn't mention hedging"
        return None
    if isinstance(v, (int, float)):
        return None if any(abs(n - v) < 1e-9 for n in numbers_in(q)) else f"{v} not found in quote"
    if isinstance(v, date):
        return None if v in dates_in(q) else f"{v.isoformat()} not found in quote"
    if name in KEYWORDS:
        pat = KEYWORDS[name][v]
        return None if re.search(pat, q, re.I) else f"quote has no keyword /{pat}/ for {v!r}"
    if name == "fund_size":
        nums = numbers_in(v)
        return None if nums and nums[0] in numbers_in(q) else "amount not found in quote"
    return None if norm(v).lower() in q.lower() else f"{v!r} not found in quote"


def check_file(path: Path, manifest: dict) -> tuple[FundFacts, list[tuple[str, str, str]]]:
    """Returns (facts, issues) where issue = (level, field, message)."""
    facts = FundFacts.model_validate_json(path.read_text())
    issues = []

    if facts.isin != path.stem:
        issues.append(("FAIL", "isin", f"file name {path.stem} != isin {facts.isin}"))

    for name in FundFacts.model_fields:
        s = getattr(facts, name)
        if not isinstance(s, Sourced):
            continue
        if s.value is None:
            if s.doc or s.page or s.quote:
                issues.append(("WARN", name, "value is None but a source is set"))
            continue
        if not (s.doc and s.page and s.quote):
            issues.append(("FAIL", name, "value without doc/page/quote"))
            continue
        row = manifest.get((facts.isin, s.doc))
        if row is None:
            issues.append(("FAIL", name, f"no {s.doc} for {facts.isin} in manifest"))
            continue
        pages = page_texts(DATA / row["file_path"])
        if not 1 <= s.page <= len(pages):
            issues.append(("FAIL", name, f"page {s.page} out of range (1-{len(pages)})"))
            continue
        if norm(s.quote) not in pages[s.page - 1]:
            other = [i + 1 for i, t in enumerate(pages) if norm(s.quote) in t]
            hint = f"; found on page(s) {other}" if other else ""
            issues.append(("FAIL", name, f"quote not on {s.doc} p.{s.page}{hint}"))
            continue
        if err := check_value(name, s.value, s.quote):
            issues.append(("FAIL", name, err))

    # factsheet_as_of should be printed on the factsheet's first page
    fs = manifest.get((facts.isin, "factsheet"))
    if facts.factsheet_as_of and fs:
        if facts.factsheet_as_of not in dates_in(page_texts(DATA / fs["file_path"])[0]):
            issues.append(("FAIL", "factsheet_as_of", "date not printed on factsheet p.1"))
    if fs and not facts.factsheet_as_of:
        issues.append(("WARN", "factsheet_as_of", "factsheet exists but as-of date is empty"))

    # Snapshot fields must come from the factsheet
    for name in ("fund_size", "holdings_count", "ter_pct"):
        s = getattr(facts, name)
        if s.value is not None and s.doc != "factsheet":
            issues.append(("WARN", name, f"sourced from {s.doc}, expected factsheet"))

    if facts.verified and any(lvl == "FAIL" for lvl, _, _ in issues):
        issues.append(("FAIL", "verified", "marked verified but checks fail"))
    return facts, issues


def checklist(facts: FundFacts, manifest: dict) -> str:
    lines = [f"\n{facts.isin}  {facts.fund_name.value}"]
    for kind in ("kid", "factsheet"):
        if row := manifest.get((facts.isin, kind)):
            lines.append(f"  {kind:9} data/{row['file_path']}")
    for name in FundFacts.model_fields:
        s = getattr(facts, name)
        if isinstance(s, Sourced):
            src = f"{s.doc} p.{s.page}" if s.doc else "-"
            lines.append(f"  [ ] {name:34} {str(s.value):42} {src}")
    if facts.notes:
        lines.append(f"  notes: {facts.notes}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    manifest = load_manifest()
    paths = [FACTS / f"{a}.json" for a in argv] or sorted(FACTS.glob("*.json"))
    n_fail = 0
    for path in paths:
        facts, issues = check_file(path, manifest)
        fails = sum(lvl == "FAIL" for lvl, _, _ in issues)
        n_fail += fails
        filled = sum(
            getattr(facts, n).value is not None
            for n in FundFacts.model_fields
            if isinstance(getattr(facts, n), Sourced)
        )
        total = sum(isinstance(getattr(facts, n), Sourced) for n in FundFacts.model_fields)
        status = "FAIL" if fails else "ok  "
        print(f"{status} {path.stem}  {filled}/{total} fields  verified={facts.verified}")
        for lvl, field, msg in issues:
            print(f"     {lvl} {field}: {msg}")
        if argv:
            print(checklist(facts, manifest))
    print(f"\n{len(paths)} files, {n_fail} failures")
    return 1 if n_fail else 0


def cli() -> None:
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
