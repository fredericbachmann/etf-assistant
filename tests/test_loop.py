"""The streaming agent loop, offline: the Claude client and the tools are replaced by fakes."""

from types import SimpleNamespace as NS

import pytest

from etf_assistant import llm
from etf_assistant.agent import loop
from etf_assistant.agent.loop import Agent, Answer, TextDelta, ToolFinished, ToolStarted


def usage(inp=100, out=20):
    return NS(input_tokens=inp, output_tokens=out, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


class FakeStream:
    def __init__(self, chunks, message):
        self.text_stream = iter(chunks)
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


@pytest.fixture
def fake_claude(monkeypatch):
    """Queue up (text chunks, final message) per model turn; records the params of each request."""
    turns, requests = [], []

    def stream(**params):
        requests.append({**params, "messages": list(params["messages"])})  # the loop keeps appending to its list
        chunks, message = turns.pop(0)
        return FakeStream(chunks, message)

    monkeypatch.setattr(llm, "client", lambda: NS(messages=NS(stream=stream)))
    monkeypatch.setattr(loop, "run_tool", lambda name, input: (f"result of {name}", False))
    return turns, requests


def test_stream_yields_text_tool_events_then_answer(fake_claude):
    turns, requests = fake_claude
    turns.append((["Let me ", "check."],
                  NS(stop_reason="tool_use", usage=usage(),
                     content=[text("Let me check."), tool_use("t1", "read_fund_documents", {"isin": "X"})])))
    turns.append((["TER is ", "0.22%."],
                  NS(stop_reason="end_turn", usage=usage(), content=[text("TER is 0.22%.")])))

    events = list(Agent().stream("TER of X?"))

    assert events[:2] == [TextDelta("Let me "), TextDelta("check.")]
    assert events[2] == ToolStarted("t1", "read_fund_documents", {"isin": "X"})
    assert isinstance(events[3], ToolFinished) and events[3].id == "t1"
    assert events[4:6] == [TextDelta("TER is "), TextDelta("0.22%.")]
    answer = events[-1]
    assert isinstance(answer, Answer) and len(events) == 7
    assert answer.text == "TER is 0.22%."  # the preamble is not part of the answer
    assert answer.stop_reason == "end_turn"
    assert [c.name for c in answer.tool_calls] == ["read_fund_documents"]
    assert answer.input_tokens == 200 and answer.output_tokens == 40
    # the second request carries the tool result back to the model
    assert requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "t1"


def test_ask_returns_the_answer_and_settles_the_budget(fake_claude):
    turns, _ = fake_claude
    turns.append((["Hi."], NS(stop_reason="end_turn", usage=usage(), content=[text("Hi.")])))
    budget = llm.Budget(cap_usd=1.0)

    answer = Agent(budget=budget).ask("hello")

    assert answer.text == "Hi."
    assert budget.reserved == 0
    assert budget.spent == pytest.approx(answer.cost_usd) and budget.calls == 1


def test_cut_off_answer_streams_the_note_too(fake_claude):
    turns, _ = fake_claude
    turns.append((["Part"], NS(stop_reason="max_tokens", usage=usage(), content=[text("Part")])))

    events = list(Agent().stream("q"))

    streamed = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert streamed == events[-1].text and "cut off" in streamed


def test_abandoned_stream_keeps_the_reservation_as_spent(fake_claude):
    turns, _ = fake_claude
    turns.append((["a", "b"], NS(stop_reason="end_turn", usage=usage(), content=[text("ab")])))
    budget = llm.Budget(cap_usd=1.0)

    events = Agent(budget=budget).stream("q")
    next(events)
    events.close()  # e.g. the browser disconnected mid-answer

    assert budget.reserved == 0 and budget.spent > 0 and budget.calls == 1


def test_unfinished_turn_is_removed_from_the_history(fake_claude):
    turns, _ = fake_claude
    turns.append((["A."], NS(stop_reason="end_turn", usage=usage(), content=[text("A.")])))
    turns.append(([], NS(stop_reason="tool_use", usage=usage(), content=[tool_use("t1", "search_documents", {})])))
    agent = Agent()
    agent.ask("first")
    finished = list(agent.messages)

    # "second" gets a tool round, then the next model call fails (no turn queued: the fake raises)
    with pytest.raises(IndexError):
        agent.ask("second")

    assert agent.messages == finished


def test_closed_stream_is_removed_from_the_history(fake_claude):
    turns, _ = fake_claude
    turns.append((["a", "b"], NS(stop_reason="end_turn", usage=usage(), content=[text("ab")])))
    agent = Agent()

    events = agent.stream("q")
    next(events)
    events.close()

    assert agent.messages == []
