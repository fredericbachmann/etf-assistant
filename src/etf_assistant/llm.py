"""The Claude client and model, cost accounting, and a hard spending cap.

Credentials: anthropic.Anthropic() finds them itself. Locally that's the
`ant auth login` profile; on a server it will be the platform's mechanism.
The code never reads an API key, so it doesn't change between the two.

Budget: every call goes through call() or StreamedCall. Before sending, it reserves the call's
WORST-CASE cost (prompt size estimated generously from its characters, plus
the full max_tokens of output, at cache-write prices). If that reservation
doesn't fit under the cap, the call is never made. After the response, the
reservation is replaced by the real cost from response.usage. So the cap holds
even with several calls in flight at once.
"""

import json
import threading
from functools import cache

import anthropic

MODEL = "claude-haiku-5-5"

# Claude Haiku 5.5, $ per million tokens. Two rate cards, chosen by the prompt length of each call.
LONG_PROMPT = 100_000
RATES = {"short": (0.10, 0.50), "long": (0.50, 2.50)}  # (input, output)
CACHE_WRITE, CACHE_READ = 1.25, 0.10  # multipliers on the input rate
CHARS_PER_TOKEN_SAFE = 2.5  # English text is ~4 chars/token; 2.5 overestimates on purpose


@cache
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def warm_up() -> None:
    """One free request (model info) before starting parallel calls.

    With `ant auth login` credentials, an expired access token is renewed on the
    next request, and the refresh token is single-use. If several threads find the
    token expired at once, they all try to renew it and all but one are rejected
    ("Refresh token not found or invalid"). Renewing once up front avoids that."""
    client().models.retrieve(MODEL)


def text_of(message) -> str:
    """The text of a response. Read blocks by type: with thinking on, the first block can be a thinking block."""
    return "".join(b.text for b in message.content if b.type == "text")


def cost_of(usage) -> float:
    """Actual cost of one response in USD, from its usage block."""
    read = usage.cache_read_input_tokens or 0
    written = usage.cache_creation_input_tokens or 0
    prompt = usage.input_tokens + read + written
    rate_in, rate_out = RATES["long" if prompt > LONG_PROMPT else "short"]
    return (usage.input_tokens * rate_in + written * rate_in * CACHE_WRITE
            + read * rate_in * CACHE_READ + usage.output_tokens * rate_out) / 1e6


def worst_case_cost(params: dict) -> float:
    """Upper bound for a request: generous prompt estimate, all of it cache-written, plus max_tokens of output."""
    def default(o):
        return o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o)
    chars = len(json.dumps({k: params.get(k) for k in ("system", "tools", "messages")}, default=default))
    prompt = chars / CHARS_PER_TOKEN_SAFE
    rate_in, rate_out = RATES["long" if prompt > LONG_PROMPT else "short"]
    return (prompt * rate_in * CACHE_WRITE + params["max_tokens"] * rate_out) / 1e6


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, cap_usd: float):
        self.cap = cap_usd
        self.spent = 0.0  # settled, real costs
        self.reserved = 0.0  # worst cases of calls in flight
        self.calls = 0
        self._lock = threading.Lock()

    def reserve(self, amount: float) -> None:
        with self._lock:
            if self.spent + self.reserved + amount > self.cap:
                raise BudgetExceeded(f"next call could cost up to ${amount:.4f}; spent ${self.spent:.4f} "
                                     f"+ in flight ${self.reserved:.4f} would exceed the ${self.cap:.3f} cap")
            self.reserved += amount

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += actual
            self.calls += 1


def call(budget: "Budget | None" = None, *, parse: bool = False, **params):
    """messages.create (or .parse for structured output), within the budget if one is given.

    Returns (response, cost_usd)."""
    reserved = worst_case_cost(params) if budget else 0.0
    if budget:
        budget.reserve(reserved)
    response, cost = None, 0.0
    try:
        api = client().messages
        response = (api.parse if parse else api.create)(**params)
        cost = cost_of(response.usage)
        return response, cost
    finally:
        if budget:
            # if the call failed, nothing was billed for it... except it may have been: keep the
            # reservation as spent when we can't know (no response), to stay on the safe side
            budget.settle(reserved, cost if response is not None else reserved)


class StreamedCall:
    """messages.stream within the budget: iterate it for the text as the model writes it.

    Same budget rules as call(): the worst case is reserved before sending, and replaced
    by the real cost once the stream completes. A stream that fails or is abandoned midway
    (the consumer stops iterating) keeps the full reservation as spent.

        streamed = StreamedCall(budget, model=..., messages=...)
        for text in streamed:
            ...
        streamed.response, streamed.cost   # set once iteration has finished
    """

    def __init__(self, budget: "Budget | None" = None, **params):
        self.budget = budget
        self.params = params
        self.response = None
        self.cost = 0.0

    def __iter__(self):
        reserved = worst_case_cost(self.params) if self.budget else 0.0
        if self.budget:
            self.budget.reserve(reserved)
        try:
            with client().messages.stream(**self.params) as stream:
                yield from stream.text_stream
                self.response = stream.get_final_message()
            self.cost = cost_of(self.response.usage)
        finally:
            if self.budget:
                self.budget.settle(reserved, self.cost if self.response is not None else reserved)
