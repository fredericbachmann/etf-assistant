"""
Command-line front ends for the assistant.

Usage:
  uv run ask "What's the ongoing charge of VWCE?"       # one question, with a trace of tool calls
  uv run chat                                           # conversation; follow-up questions keep context
"""

import argparse
import sys

from etf_assistant.agent.loop import Agent, Answer, TextDelta, ToolFinished, ToolStarted
from etf_assistant.retrieval.store import check_ready


def _warn_if_search_down() -> None:
    if problem := check_ready():
        print(f"warning: search_documents will fail. {problem}\n", file=sys.stderr)


def _run(agent: Agent, question: str, show_tools: bool = True) -> None:
    """Print the answer as it is written (stdout), tool calls and a cost line (stderr)."""
    line_open = False  # text printed since the last newline
    for event in agent.stream(question):
        match event:
            case TextDelta(text):
                print(text, end="", flush=True)
                line_open = not text.endswith("\n")
            case ToolStarted(_, name, args) if show_tools:
                if line_open:
                    print(flush=True)
                    line_open = False
                shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"  -> {name}({shown})", end="", file=sys.stderr, flush=True)
            case ToolFinished(_, call) if show_tools:
                print(f"  [{call.result_chars:,} chars{', ERROR' if call.is_error else ''}]", file=sys.stderr)
            case Answer() as answer:
                print()
                print(f"\n[{len(answer.tool_calls)} tool calls, {answer.input_tokens:,} in "
                      f"({answer.cache_read_tokens:,} cached) / {answer.output_tokens:,} out tokens, "
                      f"${answer.cost_usd:.4f}, {answer.seconds:.1f}s]", file=sys.stderr)


def ask_cli() -> None:
    ap = argparse.ArgumentParser(description="Ask the ETF assistant one question")
    ap.add_argument("question")
    ap.add_argument("-q", "--quiet", action="store_true", help="don't show tool calls")
    a = ap.parse_args()
    _warn_if_search_down()
    _run(Agent(), a.question, show_tools=not a.quiet)


def chat_cli() -> None:
    agent = Agent()
    _warn_if_search_down()
    print("ETF assistant. Ask about the funds in the library; empty line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        _run(agent, q)
        print()
