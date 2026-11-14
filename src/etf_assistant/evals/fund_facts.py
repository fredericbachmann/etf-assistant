"""
Schema for the verified facts table: your exact answer key for the evals
and the data behind the agent's lookup_fund tool.

Every value carries where it came from (document + page). That way:
  - the agent can cite exact values, not only retrieved passages
  - you can check citations automatically ("did the answer cite the page the fact is on?")
  - when you hand-verify, you open the exact page instead of hunting for it

Workflow:
  1. Have an LLM fill FundFacts from the parsed text of one fund's KID + factsheet
     (use structured output / JSON schema = FundFacts.model_json_schema()).
  2. Save to data/facts/<ISIN>.json with verified=False.
  3. Check every field against the PDF by hand, fix it, set verified=True.
  4. Evals only ever read verified files.
"""

from datetime import date
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

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

    # Costs: keep the factsheet TER and the KID cost lines separate. They differ, and that's a test case.
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


if __name__ == "__main__":
    import json

    print(json.dumps(FundFacts.model_json_schema(), indent=2)[:1500], "...")