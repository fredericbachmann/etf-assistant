"""One question's answer as chat messages, updated event by event.

Messages are plain dicts in the common chat format, {"role", "content", "metadata"}:
Gradio's Chatbot shows them as they are (metadata with a title makes a collapsible
box with a spinner while "status" is "pending"), and nothing here imports Gradio.

  text the model writes            -> an assistant message, citations linked
  a tool call                      -> a collapsible box: what it does, the raw call, time taken
  a failure (budget, API error)    -> an assistant message saying so
"""

import logging
import time
from collections.abc import Callable, Iterator

from etf_assistant.agent.loop import Agent, Answer, TextDelta, ToolFinished, ToolStarted
from etf_assistant.llm import BudgetExceeded

log = logging.getLogger(__name__)

Message = dict
BUDGET_MESSAGE = "The assistant has reached its spending limit for today. Please try again tomorrow."
ERROR_MESSAGE = "Sorry, something went wrong while answering. Please try again."

DOC_TYPES = {"kid": "KID", "factsheet": "factsheet", "both": "KID and factsheet"}


def describe_tool(name: str, args: dict, fund_names: dict[str, str]) -> str:
    """A tool call in words, e.g. 'Reading the KID of iShares Core MSCI World'."""
    def fund(isin):
        return fund_names.get(isin, isin)

    match name:
        case "read_fund_documents":
            return f"Reading the {DOC_TYPES.get(args.get('doc_type'), 'documents')} of {fund(args.get('isin'))}"
        case "search_documents":
            query, isins = args.get("query", ""), args.get("isins") or []
            where = ("all documents" if not isins else "each fund's documents" if isins == ["all"]
                     else fund(isins[0]) if len(isins) == 1 else f"{len(isins)} funds")
            return f"Searching {where} for “{query}”"
        case "calculate_fee_cost":
            amount = f"{args.get('amount', 0):,.0f}"
            invested = f"{amount} a month" if args.get("mode") == "monthly" else f"{amount} invested once"
            return f"Calculating fees: {args.get('rate_pct')}% a year on {invested}, over {args.get('years')} years"
    return f"Using {name}"


class Transcript:
    """The assistant's messages for one question; messages() renders them as they are now."""

    def __init__(self, link: Callable[[str], str], describe: Callable[[str, dict], str]):
        self._link = link  # text -> Markdown with citations linked
        self._describe = describe  # (tool name, input) -> words
        self._items: list[dict] = []  # {"text": raw} or {"tool": ...}
        self._tools: dict[str, dict] = {}  # tool call id -> its item

    def add(self, event) -> None:
        match event:
            case TextDelta(text) if text:
                if self._items and "text" in self._items[-1]:
                    self._items[-1]["text"] += text
                else:
                    self._items.append({"text": text})
            case ToolStarted(id, name, args):
                shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
                item = {"tool": self._describe(name, args), "call": f"{name}({shown})",
                        "started": time.monotonic(), "duration": None, "result": None}
                self._items.append(item)
                self._tools[id] = item
            case ToolFinished(id, call):
                item = self._tools[id]
                item["duration"] = time.monotonic() - item["started"]
                item["result"] = "error" if call.is_error else f"{call.result_chars:,} characters"

    def fail(self, message: str) -> None:
        self._items.append({"text": message, "plain": True})

    def messages(self) -> list[Message]:
        out = []
        for item in self._items:
            if "tool" in item:
                done = item["duration"] is not None
                meta = {"title": ("⚠️ " if item["result"] == "error" else "") + item["tool"],
                        "status": "done" if done else "pending"}
                if done:
                    meta["duration"] = round(item["duration"], 1)
                    meta["log"] = item["result"]
                out.append({"role": "assistant", "content": f"`{item['call']}`", "metadata": meta})
            else:
                text = item["text"] if item.get("plain") else self._link(item["text"])
                out.append({"role": "assistant", "content": text})
        return out


def respond(agent: Agent, question: str, transcript: Transcript) -> Iterator[list[Message]]:
    """Answer one question, yielding the assistant's messages so far after every event.

    Failures end the answer with a message instead of an exception; the agent has already
    removed the unfinished turn from its history, so the conversation can go on."""
    try:
        for event in agent.stream(question):
            transcript.add(event)
            if isinstance(event, Answer):
                budget = agent.budget
                log.info("answered: %d tool calls, %s tokens in / %s out, $%.4f, %.1fs%s",
                         len(event.tool_calls), f"{event.input_tokens:,}", f"{event.output_tokens:,}",
                         event.cost_usd, event.seconds,
                         f"; today ${budget.spent:.2f} of ${budget.cap:.2f}" if budget else "")
            yield transcript.messages()
    except BudgetExceeded as e:
        log.warning("budget: %s", e)
        transcript.fail(BUDGET_MESSAGE)
        yield transcript.messages()
    except Exception:
        log.exception("answering failed")
        transcript.fail(ERROR_MESSAGE)
        yield transcript.messages()
