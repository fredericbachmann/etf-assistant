"""
The ETF assistant: Claude + tools in an agentic loop.

The loop, by hand so every step is visible:
  1. send the conversation (+ tool definitions) to the model
  2. stop_reason "tool_use": the reply contains tool calls -> run them, append
     the results as the next user message, go to 1
  3. stop_reason "end_turn": the reply is the answer
The SDK's Tool Runner (client.beta.messages.tool_runner) automates exactly this;
writing it once by hand shows what such frameworks do for you.

Usage:
  uv run ask "What's the ongoing charge of VWCE?"       # one question, with a trace of tool calls
  uv run chat                                           # conversation; follow-up questions keep context
"""

import argparse
import sys
import time
from dataclasses import dataclass, field

from etf_assistant.llm import MODEL, Budget, call, text_of
from etf_assistant.search import check_ready
from etf_assistant.tools import TOOLS, run_tool

SYSTEM = """You answer questions about the exchange-traded funds in a document library: each fund's \
Key Information Document (KID) and, for most, its issuer factsheet. You have tools to identify funds, \
read their documents and search across them.

How to work
- Identify the fund(s) first (find_fund, or list_funds for questions about all funds). If a name or \
ticker is ambiguous or not in the library, say so instead of guessing.
- For one or two specific funds, read their documents (read_fund_documents). For questions across \
many funds or about a topic, use search_documents, restricted to a fund with isin when that helps.
- Base every fund-specific statement on the documents, never on general knowledge, and cite it like \
[IE00B4L5Y983 KID p.3] or [IE00B4L5Y983 factsheet p.1].
- If the documents don't contain the answer (current prices, future returns, tax treatment, broker \
offerings, funds outside the library), say that the documents don't cover it. You may explain general \
concepts (what accumulating means) without citations.

Costs
- TER: the factsheet's total expense ratio / ongoing charges figure / all-in fee. A question about the \
TER or ongoing charge is answered with that figure alone; any swap fee is a side note and never changes \
which fund has the lowest TER.
- Swap fee (synthetic, swap-based funds): a real cost of the index exposure. Factsheets may say it is \
"not charged separately" because it is deducted inside the swap rather than from the fund; it is still \
paid. For total or all-in cost, comparisons "including the swap fee", and fee calculations, use TER + \
swap fee and show the sum (e.g. 0.05% + 0.07% = 0.12%). Don't net it against an "enhanced return" the \
documents don't quantify.
- KID cost lines (management/operating costs, transaction costs) are calculated differently; quote them \
when asked about the KID and say which document a figure comes from. Transaction costs are estimates: \
leave them out of fee calculations and mention them separately.
- Fee amounts in currency: always use calculate_fee_cost (one call per fund and rate) instead of calculating by hand; state the rate you used and why, and give the result per fund.

Getting it right
- Factsheet figures (fund size, holdings, performance) are snapshots: give the as-of date.
- Share classes of the same fund (same fund name, different ISIN) hold the same portfolio and track the \
same index; when comparing them, say they are share classes of one fund and name what differs \
(distribution, currency).

Advice
- Don't give personal investment advice: don't tell the user what to buy, sell or how to allocate. \
Comparing facts across funds ("which has the lowest TER?") is not advice; answer it.

Style: concise and direct; lead with the answer, then the supporting figures with citations."""

MAX_TURNS = 12  # safety net: a model stuck in a tool loop shouldn't run up a bill


@dataclass
class ToolCall:
    name: str
    input: dict
    result_chars: int
    is_error: bool


@dataclass
class Answer:
    text: str
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


class Agent:
    def __init__(self, verbose: bool = False, budget: Budget | None = None):
        self.messages: list[dict] = []  # the whole conversation; only ever appended to
        self.verbose = verbose
        self.budget = budget  # if set, a call that could exceed it raises BudgetExceeded

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def ask(self, question: str) -> Answer:
        self.messages.append({"role": "user", "content": question})
        answer = Answer(text="", stop_reason="")
        t0 = time.time()

        for _ in range(MAX_TURNS):
            response, cost = call(
                self.budget,
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=self.messages,
                output_config={"effort": "medium"},
                cache_control={"type": "ephemeral"},  # re-sent history (documents!) is billed at 10% on the next turn
            )
            u = response.usage
            # input_tokens counts only uncached tokens; add cache reads/writes for the full prompt size
            prompt = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            answer.input_tokens += prompt
            answer.cache_read_tokens += u.cache_read_input_tokens or 0
            answer.output_tokens += u.output_tokens
            answer.cost_usd += cost

            # Append the reply unchanged, thinking blocks included: the API requires them back
            # with the tool results, and the history must stay append-only.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result, is_error = run_tool(block.name, block.input)
                    args = ", ".join(f"{k}={v!r}" for k, v in block.input.items())
                    self._log(f"  -> {block.name}({args})  [{len(result):,} chars{', ERROR' if is_error else ''}]")
                    answer.tool_calls.append(ToolCall(block.name, dict(block.input), len(result), is_error))
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": result, "is_error": is_error})
                # all results of one turn go back in ONE user message
                self.messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "pause_turn":
                continue  # the server paused a long turn; sending the history again resumes it

            answer.stop_reason = response.stop_reason
            answer.text = text_of(response)
            if response.stop_reason == "refusal":
                answer.text = answer.text or "The model declined to answer this request."
            elif response.stop_reason == "max_tokens":
                answer.text += "\n\n[answer cut off: output token limit reached]"
            break
        else:
            answer.stop_reason = "max_turns"
            answer.text = "Stopped: too many tool calls without an answer."

        answer.seconds = time.time() - t0
        return answer


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
