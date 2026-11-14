"""
The agent's tools: definitions the model sees + the Python functions that run them.

A tool definition is a name, a description and a JSON schema for the input.
The model never runs anything itself: it replies "please call find_fund with
{'query': 'VWCE'}", our code runs the function, and sends back the result as
text. So the description is effectively part of the prompt; it's how the model
knows when a tool is useful.

Tool results are plain text with light XML-style tags rather than JSON: easier
for the model to read, and the tags carry what it needs to cite (ISIN, doc, page).
"""

import re
from functools import cache

from etf_assistant.catalog import Fund, load_catalog
from etf_assistant.manifest import load_manifest
from etf_assistant.parse import load_pages
from etf_assistant.paths import DATA
from etf_assistant.search import search

TOOLS = [
    {
        "name": "find_fund",
        "description": (
            "Identify funds in the library from a name, ticker or ISIN, e.g. 'VWCE', "
            "'iShares Core MSCI World', 'IE00B4L5Y983'. Returns the best-matching funds with ISIN, "
            "share class and tickers. Use this before reading or searching a specific fund's documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Name, ticker or ISIN as the user wrote it"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_funds",
        "description": "List every fund in the library with ISIN, share class, tickers and available documents.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_fund_documents",
        "description": (
            "Return the full text of one fund's documents, with page markers for citing. "
            "'kid' is the Key Information Document (PRIIPs: risk indicator, costs, holding period, objectives); "
            "'factsheet' is the issuer's monthly marketing factsheet (TER, size, holdings, performance, listings; "
            "values as of the factsheet date). Best for questions about one or two specific funds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "isin": {"type": "string"},
                "doc_type": {"type": "string", "enum": ["kid", "factsheet", "both"]},
            },
            "required": ["isin", "doc_type"],
        },
    },
    {
        "name": "search_documents",
        "description": (
            "Search passages across all funds' documents (keyword + semantic search). Returns the best "
            "passages with fund, document and page. Best for questions across many funds or about a topic "
            "('which funds use swaps'). Optionally restrict to one fund (isin) or document type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for; use the documents' vocabulary"},
                "isin": {"type": "string", "description": "Only this fund"},
                "doc_type": {"type": "string", "enum": ["kid", "factsheet"]},
                "k": {"type": "integer", "description": "Number of passages, default 8, max 20"},
            },
            "required": ["query"],
        },
    },
]


TOOLS.append({
    "name": "calculate_fee_cost",
    "description": (
        "Compute exactly how much money annual fees take over time, assuming zero market return. "
        "Use it for every fee amount in currency instead of calculating by hand. "
        "mode 'lump_sum': `amount` invested once; 'monthly': `amount` invested at the start of every month. "
        "rate_pct is the annual cost rate in percent, e.g. 0.12 for TER 0.05% + swap fee 0.07%."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rate_pct": {"type": "number"},
            "amount": {"type": "number"},
            "years": {"type": "integer"},
            "mode": {"type": "string", "enum": ["lump_sum", "monthly"]},
        },
        "required": ["rate_pct", "amount", "years", "mode"],
    },
})


class ToolError(Exception):
    """An error the model should see and can react to (wrong ISIN, ...)."""


@cache
def _catalog() -> dict[str, Fund]:
    return load_catalog()


def _fund_line(f: Fund) -> str:
    return (f'<fund isin="{f.isin}" name="{f.display_name}" distribution="{f.distribution}" '
            f'tickers="{", ".join(f.tickers)}" documents="{", ".join(f.documents)}"/>')


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9€&]+", s.lower())) - {"the", "ucits", "etf", "fund", "of", "a"}


def find_fund(query: str) -> str:
    q_tokens = _tokens(query)
    scored = []
    for f in _catalog().values():
        if f.isin.lower() in query.lower():
            score = 100.0
        elif any(t.lower() in q_tokens for t in f.tickers):
            score = 50.0
        else:
            name_tokens = _tokens(f"{f.brand} {f.display_name} {f.distribution}")
            score = len(q_tokens & name_tokens) / max(len(q_tokens), 1)
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda sf: -sf[0])
    if not scored:
        return f"No fund in the library matches {query!r}. Use list_funds to see all funds."
    best = scored[0][0]
    # a clear ISIN/ticker hit is unambiguous; for name matches return the close runners-up too
    hits = [f for s, f in scored if s >= best * (1 if best >= 50 else 0.75)][:5]
    note = "" if best >= 50 else "\n(name match: if several funds fit, ask the user which one is meant)"
    return "\n".join(_fund_line(f) for f in hits) + note


def list_funds() -> str:
    return "\n".join(_fund_line(f) for f in _catalog().values())


def read_fund_documents(isin: str, doc_type: str) -> str:
    if isin not in _catalog():
        raise ToolError(f"Unknown ISIN {isin!r}. Use find_fund first.")
    manifest = load_manifest()
    types = ["kid", "factsheet"] if doc_type == "both" else [doc_type]
    parts = []
    for t in types:
        row = manifest.get((isin, t))
        if row is None:
            parts.append(f'<document isin="{isin}" type="{t}">not available in the library</document>')
            continue
        pages = load_pages(DATA / row["file_path"])
        body = "\n\n".join(f"<page number=\"{n}\">\n{p}\n</page>" for n, p in enumerate(pages, 1))
        parts.append(f'<document isin="{isin}" type="{t}" as_of="{row["as_of_date"]}">\n{body}\n</document>')
    return "\n\n".join(parts)


def search_documents(query: str, isin: str | None = None, doc_type: str | None = None, k: int = 8) -> str:
    if isin and isin not in _catalog():
        raise ToolError(f"Unknown ISIN {isin!r}. Use find_fund first.")
    hits = search(query, k=min(max(k, 1), 20), mode="hybrid", isin=isin, doc_type=doc_type)
    if not hits:
        return "No matching passages."
    return "\n\n".join(
        f'<passage isin="{h.isin}" fund="{h.fund_name}" doc="{h.doc_type}" page="{h.page}" '
        f'section="{h.section}">\n{h.text}\n</passage>'
        for h in hits
    )


def calculate_fee_cost(rate_pct: float, amount: float, years: int, mode: str) -> str:
    """Zero-return fee drag. Fees are charged continuously in reality; monthly steps are close enough."""
    if not (0 <= rate_pct < 10 and amount > 0 and 0 < years <= 100):
        raise ToolError("expected 0 <= rate_pct < 10, amount > 0, 0 < years <= 100")
    rate = rate_pct / 100
    if mode == "lump_sum":
        paid_in = amount
        value = amount * (1 - rate) ** years  # the fee is taken from what's left each year
        method = f"{amount:,.2f} invested once; each year {rate_pct}% of the remaining value is charged"
    elif mode == "monthly":
        paid_in, value = 0.0, 0.0
        for _ in range(years * 12):
            paid_in += amount
            value += amount
            value *= 1 - rate / 12  # a twelfth of the annual rate, on the current balance
        method = f"{amount:,.2f} invested at the start of each month; {rate_pct}%/12 charged monthly on the balance"
    else:
        raise ToolError(f"unknown mode {mode!r}")
    return (f"{method}, {years} years, zero market return.\n"
            f"total invested: {paid_in:,.2f}\nvalue at the end: {value:,.2f}\nfees paid: {paid_in - value:,.2f}")


HANDLERS = {
    "find_fund": find_fund,
    "list_funds": list_funds,
    "read_fund_documents": read_fund_documents,
    "search_documents": search_documents,
    "calculate_fee_cost": calculate_fee_cost,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool; returns (result text, is_error). Errors go back to the model instead of crashing."""
    try:
        return HANDLERS[name](**tool_input), False
    except ToolError as e:
        return str(e), True
    except Exception as e:  # e.g. unexpected arguments: let the model see and correct it
        return f"{type(e).__name__}: {e}", True
