"""
The ETF assistant: Claude + tools in an agentic loop.

The loop, by hand so every step is visible:
  1. send the conversation (+ tool definitions) to the model
  2. stop_reason "tool_use": the reply contains tool calls -> run them, append
     the results as the next user message, go to 1
  3. stop_reason "end_turn": the reply is the answer
The SDK's Tool Runner (client.beta.messages.tool_runner) automates exactly this;
writing it once by hand shows what such frameworks do for you.

Agent.stream() runs the loop and yields events as they happen, for front ends that
show the answer while it is written and the tool calls while they run:

  TextDelta      a piece of text, as the model writes it
  ToolStarted    a tool is about to run (its input is complete)
  ToolFinished   that tool has returned
  Answer         always the last event: the final answer, tool calls, tokens, cost

The text deltas include what the model writes before calling tools ("Let me check
the KID..."). A ToolStarted after text means that text was a preamble, not the
answer; Answer.text holds the final answer alone. Agent.ask() runs the same loop
and returns just the Answer.

The command-line front ends (`uv run ask`, `uv run chat`) are in cli.py.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from etf_assistant.agent.prompt import system_prompt
from etf_assistant.agent.tools import TOOLS, run_tool
from etf_assistant.llm import MODEL, Budget, StreamedCall, text_of

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


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolStarted:
    id: str  # pairs it with its ToolFinished
    name: str
    input: dict


@dataclass
class ToolFinished:
    id: str
    call: ToolCall


Event = TextDelta | ToolStarted | ToolFinished | Answer


class Agent:
    def __init__(self, budget: Budget | None = None):
        self.messages: list[dict] = []  # the whole conversation; only ever appended to
        self.budget = budget  # if set, a call that could exceed it raises BudgetExceeded

    def ask(self, question: str) -> Answer:
        """Run the loop to the end and return the answer."""
        for event in self.stream(question):
            if isinstance(event, Answer):
                return event
        raise AssertionError("stream() ended without an Answer")

    def stream(self, question: str) -> Iterator[Event]:
        """Run the loop, yielding events as they happen; the last one is the Answer.

        A turn that doesn't finish (an error, the budget, or the consumer stops iterating)
        is removed from the history, so the conversation can go on as if it hadn't been asked:
        a question without an answer, or a tool call without its result, would make the
        next request invalid."""
        start = len(self.messages)
        try:
            yield from self._run(question)
        except BaseException:  # includes GeneratorExit, raised here when the consumer closes the stream
            del self.messages[start:]
            raise

    def _run(self, question: str) -> Iterator[Event]:
        self.messages.append({"role": "user", "content": question})
        answer = Answer(text="", stop_reason="")
        t0 = time.time()

        for _ in range(MAX_TURNS):
            streamed = StreamedCall(
                self.budget,
                model=MODEL,
                max_tokens=16000,
                system=system_prompt(),
                tools=TOOLS,
                messages=self.messages,
                output_config={"effort": "medium"},
                cache_control={"type": "ephemeral"},  # re-sent history (documents!) is billed at 10% on the next turn
            )
            for text in streamed:
                yield TextDelta(text)
            response = streamed.response

            u = response.usage
            # input_tokens counts only uncached tokens; add cache reads/writes for the full prompt size
            prompt = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            answer.input_tokens += prompt
            answer.cache_read_tokens += u.cache_read_input_tokens or 0
            answer.output_tokens += u.output_tokens
            answer.cost_usd += streamed.cost

            # Append the reply unchanged, thinking blocks included: the API requires them back
            # with the tool results, and the history must stay append-only.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    yield ToolStarted(block.id, block.name, dict(block.input))
                    result, is_error = run_tool(block.name, block.input)
                    call = ToolCall(block.name, dict(block.input), len(result), is_error)
                    answer.tool_calls.append(call)
                    yield ToolFinished(block.id, call)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": result, "is_error": is_error})
                # all results of one turn go back in ONE user message
                self.messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "pause_turn":
                continue  # the server paused a long turn; sending the history again resumes it

            answer.stop_reason = response.stop_reason
            answer.text = text_of(response)
            note = ""
            if response.stop_reason == "refusal" and not answer.text:
                note = "The model declined to answer this request."
            elif response.stop_reason == "max_tokens":
                note = "\n\n[answer cut off: output token limit reached]"
            if note:
                answer.text += note
                yield TextDelta(note)
            break
        else:
            answer.stop_reason = "max_turns"
            answer.text = "Stopped: too many tool calls without an answer."
            yield TextDelta(answer.text)

        answer.seconds = time.time() - t0
        yield answer
