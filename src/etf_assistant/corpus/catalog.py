"""
The fund catalog: name, share class and tickers of every fund, read from its own documents.

The agent uses it to turn "VWCE" or "the iShares MSCI World" into an ISIN. It
must not come from data/facts or evals/ (the answer key), and it must work for
a new ETF without manual steps. Issuers format their documents differently, so
an LLM, not parsing rules, reads the KID's first page and the whole factsheet
(tickers are often listed on a later page) and fills a fixed schema (structured
output).

The extraction is checked: a ticker that doesn't occur verbatim in the
documents is dropped, as check_facts.py does for quotes.

Usage:
  uv run build-catalog      # writes data/catalog.json (tracked in git: costs an LLM call per fund to rebuild)
"""

import json
import re
import sys
from typing import Literal

from pydantic import BaseModel, Field

from etf_assistant.corpus.manifest import load_manifest
from etf_assistant.corpus.parse import load_pages
from etf_assistant.llm import MODEL, Budget, call
from etf_assistant.paths import CATALOG, DATA


class Extracted(BaseModel):
    """What the LLM fills in. Field descriptions are part of the prompt."""
    name: str = Field(description="Fund name as printed, without the share class, e.g. 'iShares Core MSCI World UCITS ETF'")
    share_class: str = Field(description="Share class as printed, e.g. 'USD (Acc)', '(USD) Distributing', '1C'; '' if none")
    distribution: Literal["accumulating", "distributing", "unknown"]
    brand: str = Field(description="Product brand, e.g. 'iShares', 'Vanguard', 'Invesco', 'Xtrackers'")
    tickers: list[str] = Field(description="Exchange trading symbols as printed, e.g. 'VWRL', 'EUNL'. "
                                           "Not Bloomberg/Reuters codes ('VWRL LN', 'VWRL.L'), ISINs, SEDOLs or WKNs.")


class Fund(Extracted):
    isin: str
    documents: list[str]  # doc types available, e.g. ["factsheet", "kid"]

    @property
    def display_name(self) -> str:
        sc = f" {self.share_class}" if self.share_class else ""
        return f"{self.name}{sc}"


PROMPT = """Below are documents of one exchange-traded fund (ISIN {isin}).
Extract the fields of the schema. Use only what the documents state.

{pages}"""


MAX_COST_USD = 0.25  # hard cap for a full rebuild; 11 funds cost about a cent


def extract(isin: str, docs: dict[str, list[str]], budget: Budget) -> Extracted:
    # KID: page 1 names the product. Factsheet: all pages, the listings table can be anywhere.
    parts = []
    for doc_type, pages in docs.items():
        for n, page in enumerate(pages if doc_type == "factsheet" else pages[:1], 1):
            parts.append(f'<document type="{doc_type}" page="{n}">\n{page}\n</document>')
    pages = "\n\n".join(parts)
    response, _ = call(
        budget, parse=True,
        model=MODEL,
        max_tokens=4096,
        output_config={"effort": "low"},  # reading off a page, no reasoning needed
        messages=[{"role": "user", "content": PROMPT.format(isin=isin, pages=pages)}],
        output_format=Extracted,
    )
    return response.parsed_output


def verified_tickers(tickers: list[str], text: str) -> tuple[list[str], list[str]]:
    """Keep tickers that occur as a whole word in the documents."""
    keep, drop = [], []
    for t in dict.fromkeys(tickers):  # dedupe, keep order
        (keep if re.search(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])", text) else drop).append(t)
    return keep, drop


def build() -> list[Fund]:
    by_isin: dict[str, dict[str, list[str]]] = {}
    for (isin, doc_type), row in sorted(load_manifest().items()):
        by_isin.setdefault(isin, {})[doc_type] = load_pages(DATA / row["file_path"])

    budget = Budget(MAX_COST_USD)
    funds = []
    for isin, docs in by_isin.items():
        e = extract(isin, docs, budget)
        all_text = "\n".join(p for pages in docs.values() for p in pages)
        e.tickers, dropped = verified_tickers(e.tickers, all_text)
        fund = Fund(isin=isin, documents=sorted(docs), **e.model_dump())
        funds.append(fund)
        note = f"  (dropped, not in documents: {dropped})" if dropped else ""
        print(f"{isin}  {fund.display_name}  [{fund.distribution}]  tickers={fund.tickers}{note}", file=sys.stderr)
    print(f"cost: ${budget.spent:.4f} for {budget.calls} calls", file=sys.stderr)
    return funds


def load_catalog() -> dict[str, Fund]:
    return {f["isin"]: Fund(**f) for f in json.loads(CATALOG.read_text(encoding="utf-8"))}


def main() -> None:
    funds = build()
    CATALOG.write_text(json.dumps([f.model_dump() for f in funds], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(funds)} funds to {CATALOG.relative_to(DATA.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
