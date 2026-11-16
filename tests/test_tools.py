"""Agent tools and prompt, offline: search is replaced by a fake, so no OpenSearch is needed."""

import pytest

from etf_assistant.agent import tools
from etf_assistant.agent.prompt import system_prompt
from etf_assistant.corpus.catalog import load_catalog
from etf_assistant.retrieval.store import Hit

CATALOG = load_catalog()
ISINS = list(CATALOG)


@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    def search(query, k=5, mode="hybrid", isin=None, doc_type=None):
        calls.append({"query": query, "k": k, "isin": isin, "doc_type": doc_type})
        owner = isin or ISINS[0]
        return [Hit(chunk_id=f"{owner}_kid_p2_c{i}", score=1.0, isin=owner, fund_name=CATALOG[owner].display_name,
                    doc_type="kid", page=2, section="Risk", text=f"passage {i}") for i in range(k)]

    monkeypatch.setattr(tools, "search", search)
    return calls


def test_system_prompt_lists_every_fund_with_tickers():
    prompt = system_prompt()
    for f in CATALOG.values():
        assert f.isin in prompt
        for t in f.tickers:
            assert t in prompt


def test_per_fund_search_runs_one_filtered_search_per_fund(fake_search):
    out = tools.search_documents("Summary risk indicator", isins=["all"])
    assert [c["isin"] for c in fake_search] == ISINS
    assert all(c["k"] == tools.K_PER_FUND_DEFAULT for c in fake_search)
    assert out.count("<fund ") == len(ISINS)
    assert out.count("<passage ") == len(ISINS) * tools.K_PER_FUND_DEFAULT


def test_per_fund_search_caps_k_and_dedupes(fake_search):
    tools.search_documents("TER", isins=[*ISINS[:3], ISINS[0]], k=50)
    assert fake_search == [{"query": "TER", "k": tools.K_PER_FUND_MAX, "isin": i, "doc_type": None} for i in ISINS[:3]]


def test_global_search_has_no_fund_filter(fake_search):
    out = tools.search_documents("swap", k=4)
    assert fake_search[0]["isin"] is None and fake_search[0]["k"] == 4
    assert out.count("<passage ") == 4 and 'isin="' in out


def test_search_on_one_or_two_funds_returns_their_whole_documents(fake_search, monkeypatch):
    monkeypatch.setattr(tools, "read_fund_documents", lambda isin, doc_type: f"<document {isin} {doc_type}>")
    out = tools.search_documents("Number of holdings", isins=ISINS[:2], k=5)
    assert fake_search == []  # no passages to mistake for the documents
    assert out.endswith(f"<document {ISINS[0]} both>\n\n<document {ISINS[1]} both>")
    assert tools.search_documents("TER", isins=[ISINS[0], ISINS[0]], doc_type="kid").endswith(f"<document {ISINS[0]} kid>")
    assert "<passage " in tools.search_documents("TER", isins=ISINS[:3])


def test_unknown_isin_is_reported_to_the_model(fake_search):
    result, is_error = tools.run_tool("search_documents", {"query": "TER", "isins": ["XX0000000000"]})
    assert is_error and "XX0000000000" in result
    assert fake_search == []  # validated before any search
    result, is_error = tools.run_tool("read_fund_documents", {"isin": "XX0000000000", "doc_type": "kid"})
    assert is_error and "XX0000000000" in result


def test_fee_cost_lump_sum():
    out = tools.calculate_fee_cost(rate_pct=1.0, amount=10_000, years=2, mode="lump_sum")
    assert "fees paid: 199.00" in out  # 10,000 * (1 - 0.99^2)


def test_fee_cost_rejects_nonsense():
    _, is_error = tools.run_tool("calculate_fee_cost", {"rate_pct": 50, "amount": 1, "years": 1, "mode": "lump_sum"})
    assert is_error
