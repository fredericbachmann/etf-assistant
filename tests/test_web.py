"""The web layer, offline: citations, chat messages, budget, documents route. No Claude, no OpenSearch."""

from datetime import date
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from etf_assistant.agent.loop import Answer, TextDelta, ToolCall, ToolFinished, ToolStarted
from etf_assistant.llm import BudgetExceeded
from etf_assistant.web import app as web_app
from etf_assistant.web.budget import DailyBudget
from etf_assistant.web.chat import BUDGET_MESSAGE, ERROR_MESSAGE, Transcript, describe_tool, respond
from etf_assistant.web.citations import link_citations
from etf_assistant.web.config import Settings
from etf_assistant.web.documents import Documents

A, B = "IE00B4L5Y983", "IE00BK5BQT80"


def url_for(isin, doc_type):
    return None if isin == "IE0000000000" else f"/documents/{isin}_{doc_type}.pdf"


# citations

@pytest.mark.parametrize("text, expected", [
    (f"TER 0.20% [{A} factsheet p.1].",
     rf"TER 0.20% \[{A} factsheet [p.1](/documents/{A}_factsheet.pdf#page=1)\]."),
    (f"[{A} KID p.3; {B} KID p.2]",
     rf"\[{A} KID [p.3](/documents/{A}_kid.pdf#page=3); {B} KID [p.2](/documents/{B}_kid.pdf#page=2)\]"),
    (f"[{A} KID p.3; factsheet p.1, p.2]",  # ISIN carries over; several pages
     rf"\[{A} KID [p.3](/documents/{A}_kid.pdf#page=3); factsheet [p.1](/documents/{A}_factsheet.pdf#page=1), "
     rf"[p.2](/documents/{A}_factsheet.pdf#page=2)\]"),
    (f"[{A} factsheet p.1, as of 30 Sep 2026]",
     rf"\[{A} factsheet [p.1](/documents/{A}_factsheet.pdf#page=1), as of 30 Sep 2026\]"),
    (f"[{A} KID, Objectives section]", rf"\[{A} [KID](/documents/{A}_kid.pdf), Objectives section\]"),
])
def test_citations_become_page_links(text, expected):
    assert link_citations(text, url_for) == expected


@pytest.mark.parametrize("text", [
    "[a note without ISIN] and [1]",
    "[IE0000000000 KID p.3]",  # no URL for that document
    f"[{A} KID p.3",  # still streaming: no closing bracket yet
    f"[{A} KID p.3](/already/a/link.pdf)",
])
def test_other_text_is_left_alone(text):
    assert link_citations(text, url_for) == text


# chat messages

def describe(name, args):
    return describe_tool(name, args, {A: "iShares Core MSCI World"})


def test_describe_tool():
    assert describe("read_fund_documents", {"isin": A, "doc_type": "kid"}) == \
        "Reading the KID of iShares Core MSCI World"
    assert describe("search_documents", {"query": "TER", "isins": ["all"]}) == \
        "Searching each fund's documents for “TER”"
    assert describe("calculate_fee_cost", {"rate_pct": 0.2, "amount": 10000, "years": 20, "mode": "monthly"}) == \
        "Calculating fees: 0.2% a year on 10,000 a month, over 20 years"
    assert describe("new_tool", {}) == "Using new_tool"


def test_transcript_shows_text_and_tool_progress():
    t = Transcript(link=str.upper, describe=describe)
    t.add(TextDelta("let me "))
    t.add(TextDelta("check"))
    t.add(ToolStarted("t1", "read_fund_documents", {"isin": A, "doc_type": "both"}))
    pending = t.messages()[1]
    assert pending["metadata"]["status"] == "pending"
    assert pending["metadata"]["title"] == "Reading the KID and factsheet of iShares Core MSCI World"

    t.add(ToolFinished("t1", ToolCall("read_fund_documents", {}, 12345, False)))
    t.add(TextDelta("answer"))
    t.add(Answer(text="answer", stop_reason="end_turn"))

    msgs = t.messages()
    assert [m["content"] for m in (msgs[0], msgs[2])] == ["LET ME CHECK", "ANSWER"]  # text goes through link()
    assert msgs[1]["metadata"]["status"] == "done" and msgs[1]["metadata"]["log"] == "12,345 characters"


class FakeAgent:
    def __init__(self, events, error=None):
        self.events, self.error, self.budget = events, error, None

    def stream(self, question):
        yield from self.events
        if self.error:
            raise self.error


def test_respond_ends_failures_with_a_message():
    t = Transcript(link=str, describe=describe)
    out = list(respond(FakeAgent([TextDelta("partial")], BudgetExceeded("cap")), "q", t))
    assert out[-1][-1]["content"] == BUDGET_MESSAGE

    t = Transcript(link=str, describe=describe)
    out = list(respond(FakeAgent([], RuntimeError("API down")), "q", t))
    assert out[-1] == [{"role": "assistant", "content": ERROR_MESSAGE}]


# budget

def test_daily_budget_starts_again_each_day():
    day = [date(2026, 11, 15)]
    b = DailyBudget(1.0, today=lambda: day[0])
    b.reserve(0.9)
    b.settle(0.9, 0.9)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.2)
    day[0] = date(2026, 11, 16)
    b.reserve(0.2)
    assert b.spent == 0 and b.reserved == 0.2


# the app

@pytest.fixture
def client(tmp_path, monkeypatch):
    pdf = tmp_path / f"{A}_kid.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(web_app, "check_ready", lambda: None)
    settings = Settings(username="u", password=None, daily_budget_usd=1.0, concurrency=1,
                        max_question_chars=100, host="127.0.0.1", port=0)
    return TestClient(web_app.create_app(settings, Documents({(A, "kid"): pdf})))


def test_documents_route_serves_only_listed_pdfs(client):
    r = client.get(f"/documents/{A}_kid.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    assert client.get("/documents/catalog.json").status_code == 404
    assert client.get(f"/documents/..%2Ffacts%2F{A}.json").status_code == 404


def test_health_and_page(client):
    assert client.get("/health").json() == {"status": "ok", "search": "ok"}
    assert client.get("/").status_code == 200
