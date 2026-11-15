"""Settings of the web app, from environment variables (a platform's secrets end up there too)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    username: str
    password: str | None  # None: no login (fine on localhost, never on a public server)
    daily_budget_usd: float  # Claude spending of all conversations together, per day (UTC)
    concurrency: int  # questions answered at the same time; more wait in a queue
    max_question_chars: int
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            username=env("APP_USERNAME", "etf"),
            password=env("APP_PASSWORD") or None,
            daily_budget_usd=float(env("DAILY_BUDGET_USD", "2.0")),
            concurrency=int(env("CONCURRENCY", "4")),
            max_question_chars=int(env("MAX_QUESTION_CHARS", "1000")),
            host=env("HOST", "127.0.0.1"),
            port=int(env("PORT", "7860")),
        )
