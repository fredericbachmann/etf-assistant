"""
The agent's tools: definitions the model sees + the Python functions that run them.

A tool definition is a name, a description and a JSON schema for the input.
The model never runs anything itself: it replies "please call
read_fund_documents with {'isin': 'IE00BK5BQT80', 'doc_type': 'kid'}", our
code runs the function, and sends back the result as text. So the description
is effectively part of the prompt; it's how the model knows when a tool is useful.

Tool results are plain text with light XML-style tags rather than JSON: easier
for the model to read, and the tags carry what it needs to cite (ISIN, doc, page).

There is no tool to look up funds: the fund list is in the system prompt (prompt.py).
"""

from functools import cache

from etf_assistant.corpus.catalog import Fund, load_catalog
from etf_assistant.corpus.manifest import load_manifest
from etf_assistant.corpus.parse import load_pages
from etf_assistant.paths import DATA
from etf_assistant.retrieval.store import Hit, search

K_PER_FUND_DEFAULT, K_PER_FUND_MAX = 3, 5
K_GLOBAL_DEFAULT, K_GLOBAL_MAX = 8, 20

TOOLS = [
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
            "Search passages in the funds' documents (keyword + semantic search). Returns passages with fund, "
            "document and page.\n"
            "With isins: the best k passages FOR EACH of these funds, grouped by fund. Use it to compare one "
            "detail across funds ('summary risk indicator', 'number of holdings'); isins=['all'] means every "
            "fund in the library. A fund's passages may miss the answer: then read that fund's documents.\n"
            "Without isins: the best k passages across the whole library, for topics ('which funds use swaps')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What to look for; use the documents' vocabulary, e.g. "
                                         "'Number of holdings', 'Summary risk indicator', 'Total expense ratio'"},
                "isins": {"type": "array", "items": {"type": "string"},
                          "description": "Funds to search one by one, or ['all']"},
                "doc_type": {"type": "string", "enum": ["kid", "factsheet"]},
                "k": {"type": "integer",
                      "description": f"Passages per fund with isins (default {K_PER_FUND_DEFAULT}, "
                                     f"max {K_PER_FUND_MAX}); in total without (default {K_GLOBAL_DEFAULT}, "
                                     f"max {K_GLOBAL_MAX})"},
            },
            "required": ["query"],
        },
    },
    {
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
    },
]


class ToolError(Exception):
    """An error the model should see and can react to (wrong ISIN, ...)."""


@cache
def _catalog() -> dict[str, Fund]:
    return load_catalog()


def _check_isins(isins: list[str]) -> list[str]:
    if isins == ["all"]:
        return list(_catalog())
    unknown = [i for i in isins if i not in _catalog()]
    if unknown:
        raise ToolError(f"Unknown ISIN(s) {unknown}. Use the ISINs from the fund library list.")
    return list(dict.fromkeys(isins))  # dedupe, keep order


def read_fund_documents(isin: str, doc_type: str) -> str:
    _check_isins([isin])
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


def _passage(h: Hit, with_fund: bool) -> str:
    fund = f'isin="{h.isin}" fund="{h.fund_name}" ' if with_fund else ""
    return f'<passage {fund}doc="{h.doc_type}" page="{h.page}" section="{h.section}">\n{h.text}\n</passage>'


def search_documents(query: str, isins: list[str] | None = None, doc_type: str | None = None,
                     k: int | None = None) -> str:
    if not isins:
        hits = search(query, k=min(max(k or K_GLOBAL_DEFAULT, 1), K_GLOBAL_MAX), mode="hybrid", doc_type=doc_type)
        return "\n\n".join(_passage(h, with_fund=True) for h in hits) or "No matching passages."

    # One filtered search per fund: a global top-k would let a few funds with many similar
    # passages crowd out the rest. The query is embedded once (embed_query is cached).
    k = min(max(k or K_PER_FUND_DEFAULT, 1), K_PER_FUND_MAX)
    groups = []
    for isin in _check_isins(isins):
        hits = search(query, k=k, mode="hybrid", isin=isin, doc_type=doc_type)
        body = "\n".join(_passage(h, with_fund=False) for h in hits) or "no matching passages"
        groups.append(f'<fund isin="{isin}" name="{_catalog()[isin].display_name}">\n{body}\n</fund>')
    return "\n\n".join(groups)


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
