"""
Schema of the verified facts files (data/facts/<ISIN>.json): the answer key of the evals.

The assistant never reads these files. Only evals/ may, and tests/test_layers.py
checks it; otherwise the end-to-end eval would measure copying from the answer key.

Every value records its source: document, page and a verbatim quote. The
citations are checked automatically (check_facts.py, check_chunks.py,
eval_retrieval.py), and hand verification starts at the cited page.

How a facts file is made:
  1. An LLM fills FundFacts from the parsed text of the fund's KID and factsheet
     (structured output with FundFacts.model_json_schema()).
  2. The result is saved as data/facts/<ISIN>.json with verified=False.
  3. Every field is checked against the PDF by hand and corrected; then
     verified is set to True.
Evals read only verified files (load_verified_facts).
"""

from datetime import date
from pathlib import Path
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from etf_assistant.paths import FACTS

T = TypeVar("T")


class Sourced(BaseModel, Generic[T]):
    value: Optional[T] = Field(None, description="None if the documents don't state it")
    doc: Optional[Literal["kid", "factsheet"]] = None
    page: Optional[int] = Field(None, description="1-based page number in that document")
    quote: Optional[str] = Field(None, description="the short text span the value was read from")


class FundFacts(BaseModel):
    isin: str
    fund_name: Sourced[str]
    issuer: Sourced[str]
    index_name: Sourced[str]

    # Costs. The factsheet TER and the KID cost lines are separate fields: the figures can differ,
    # and the golden set asks about the difference.
    ter_pct: Sourced[float] = Field(description="TER / ongoing charge from the factsheet, in % p.a.")
    swap_fee_pct: Sourced[float] = Field(description="only for swap-based funds, else value=None")
    kid_ongoing_costs_pct: Sourced[float] = Field(description="KID 'management fees and other administrative or operating costs', % p.a.")
    kid_transaction_costs_pct: Sourced[float] = Field(description="KID 'transaction costs', % p.a.")

    # Risk (KID)
    sri: Sourced[int] = Field(description="summary risk indicator, 1-7")
    recommended_holding_period_years: Sourced[float]

    # Structure
    distribution_policy: Sourced[Literal["accumulating", "distributing"]]
    distribution_frequency: Sourced[str] = Field(description="e.g. quarterly, semi-annual; None if accumulating")
    replication: Sourced[Literal["physical_full", "physical_sampling", "synthetic"]]
    currency_hedged: Sourced[bool]
    domicile: Sourced[str]
    fund_currency: Sourced[str]
    launch_date: Sourced[date]

    # Snapshot values: valid only for the factsheet's as-of date
    fund_size: Sourced[str] = Field(description="as printed, incl. currency, e.g. 'USD 57,079.99m'")
    holdings_count: Sourced[int]
    factsheet_as_of: Optional[date] = None

    verified: bool = False
    notes: str = ""


def load_verified_facts(facts_dir: Path = FACTS) -> dict[str, FundFacts]:
    """ISIN -> facts, for verified files only: the answer key every eval grades against.

    Unverified files are skipped, so a fund whose facts aren't checked yet is
    simply missing (check-golden reports questions that need it). check-facts
    is the one reader of unverified files: it exists to help verify them.
    """
    facts = (FundFacts.model_validate_json(f.read_text(encoding="utf-8")) for f in sorted(facts_dir.glob("*.json")))
    return {f.isin: f for f in facts if f.verified}


if __name__ == "__main__":
    import json

    print(json.dumps(FundFacts.model_json_schema(), indent=2)[:1500], "...")