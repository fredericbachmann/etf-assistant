"""The server's spending cap: one Budget for all conversations, starting again from zero every day."""

from collections.abc import Callable
from datetime import UTC, date, datetime

from etf_assistant.llm import Budget


def utc_today() -> date:
    return datetime.now(UTC).date()


class DailyBudget(Budget):
    """A Budget whose spending is reset at the first reservation of a new day (UTC).

    Calls in flight across midnight keep their reservation and are settled into the new day."""

    def __init__(self, cap_usd: float, today: Callable[[], date] = utc_today):
        super().__init__(cap_usd)
        self._today = today
        self.day = today()

    def reserve(self, amount: float) -> None:
        with self._lock:
            if (today := self._today()) != self.day:
                self.day, self.spent, self.calls = today, 0.0, 0
        super().reserve(amount)
