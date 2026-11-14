"""
Command-line front ends for the assistant.

Usage:
  uv run ask "What's the ongoing charge of VWCE?"       # one question, with a trace of tool calls
  uv run chat                                           # conversation; follow-up questions keep context
"""

import argparse
import sys

from etf_assistant.agent.loop import Agent, Answer
from etf_assistant.retrieval.store import check_ready


def _warn_if_search_down() -> None:
    if problem := check_ready():
        print(f"warning: search_documents will fail. {problem}\n", file=sys.stderr)


def _print(answer: Answer) -> None:
    print(answer.text)
    print(f"\n[{len(answer.tool_calls)} tool calls, {answer.input_tokens:,} in "
          f"({answer.cache_read_tokens:,} cached) / {answer.output_tokens:,} out tokens, "
          f"${answer.cost_usd:.4f}, {answer.seconds:.1f}s]", file=sys.stderr)


def ask_cli() -> None:
    ap = argparse.ArgumentParser(description="Ask the ETF assistant one question")
    ap.add_argument("question")
    ap.add_argument("-q", "--quiet", action="store_true", help="don't show tool calls")
    a = ap.parse_args()
    _warn_if_search_down()
    _print(Agent(verbose=not a.quiet).ask(a.question))


def chat_cli() -> None:
    agent = Agent(verbose=True)
    _warn_if_search_down()
    print("ETF assistant. Ask about the funds in the library; empty line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        _print(agent.ask(q))
        print()
