"""
The ETF assistant: Claude + tools in an agentic loop.

The loop, by hand so every step is visible:
  1. send the conversation (+ tool definitions) to the model
  2. stop_reason "tool_use": the reply contains tool calls -> run them, append
     the results as the next user message, go to 1
  3. stop_reason "end_turn": the reply is the answer
The SDK's Tool Runner (client.beta.messages.tool_runner) automates exactly this;
writing it once by hand shows what such frameworks do for you.

The command-line front ends (`uv run ask`, `uv run chat`) are in cli.py.
"""

import sys
import time
from dataclasses import dataclass, field

from etf_assistant.agent.prompt import system_prompt
from etf_assistant.agent.tools import TOOLS, run_tool
from etf_assistant.llm import MODEL, Budget, call, text_of

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
                system=system_prompt(),
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
