"""
LLM-as-a-judge: decide whether an answer meets a golden question's expectation.

Why an LLM and not string matching? "0.2" in the answer key may appear as
"0.20%", "physical_sampling" as "optimised sampling", and a decline of advice
can be phrased a hundred ways. A judge reads the answer like a person would.

Why that's risky: the judge is a model too, and can be wrong or lenient. Three
habits keep it honest:
  - it gets the EXACT expected result (resolved from the verified facts), so it
    compares instead of knowing the answer itself;
  - it writes its reasoning before the verdict (fields in that order), and
    checks each criterion separately;
  - you read its reasoning on the failures (and spot-check some passes): the
    eval report prints them for that reason.
"""

import json
from typing import Literal

from pydantic import BaseModel, Field

from etf_assistant.llm import MODEL, Budget, call

BEHAVIORS = {
    "answer": "gives a substantive factual answer to the question (it must NOT refuse or deflect it as advice)",
    "decline_advice": (
        "does not give a personal recommendation (no 'buy X', no allocation, no prediction of which will do best), "
        "but is still useful: offers relevant facts from the documents or what to consider. "
        "A bare refusal without help fails."
    ),
    "not_in_docs": (
        "says the documents don't contain this information and does not invent it or answer it from general "
        "knowledge as if it were documented. It may offer related facts that the documents do contain."
    ),
}

OPS = {
    "argmin": "the fund(s) with the LOWEST value; if several funds tie, all of them must be named",
    "argmax": "the fund(s) with the HIGHEST value; if several funds tie, all of them must be named",
    "diff": "the difference first fund minus second fund, in percentage points",
    "equal": "whether the funds' values are equal (true/false)",
    "sort_asc": "the funds ordered from lowest to highest value; ties may appear in any order",
    "filter_equals": "exactly the funds whose value equals the given value; none missing, none extra",
    "fee_drag_lump": "fees paid per fund in currency; accept each figure within 10% and the comparison must be right",
    "fee_drag_monthly": "fees paid per fund in currency; accept each figure within 10% and the comparison must be right",
}

INSTRUCTIONS = """You grade answers of an assistant that answers questions about ETFs from their KIDs and \
factsheets. You get the question, what a correct answer must contain (from a verified answer key), and the \
answer. Grade only against the expectation; do not use your own knowledge of the funds.

Conventions in the expectation: percentages are in percent (0.2 means 0.20%); a list of fields means their \
sum (TER + swap fee); funds are given as ISIN with name. Formatting, rounding to the precision shown in the \
documents and extra correct detail are fine. A wrong value, a contradiction, or a missing required element fails.

What is NOT a contradiction: related figures the documents also state, clearly labelled as something else \
(share-class vs fund-level size, fund vs share-class launch date, KID vs factsheet cost figures), and caveats \
or footnotes the answer attributes to the documents. Fail for a contradiction only if the answer gives a \
different value or conclusion for the thing that was asked, or does not commit to the expected result."""


class Verdict(BaseModel):
    # order matters: the model writes reasoning before deciding
    reasoning: str = Field(description="Check each requirement in turn, briefly, quoting the answer")
    behavior: Literal["answered", "declined_advice", "said_not_in_docs", "asked_clarification", "bare_refusal"] = Field(
        description="What the answer mainly does")
    passed: bool


def describe_expectation(q: dict, names: dict[str, str]) -> str:
    """Turn a resolved golden question into plain instructions for the judge."""
    e, v = q["expected"], q["expected_value"]

    def fund(isin: str) -> str:
        return f"{isin} ({names.get(isin, '?')})"

    lines = []
    if e["type"] == "fact":
        lines.append(f"The answer must state, for fund {fund(e['isin'])}: {json.dumps(v)} "
                     f"(field{'s' if isinstance(e['field'], list) else ''}: {e['field']}). "
                     "null means the documents don't state it.")
    elif e["type"] == "compute":
        lines.append(f"Field: {e['field']}. The answer must give {OPS[e['op']]}.")
        lines.append("Funds considered: " + "; ".join(fund(i) for i in (q.get("_isins") or [])))
        if isinstance(v, dict) and "funds" in v:
            lines.append(f"Correct result: {', '.join(fund(i) for i in v['funds'])} with value {v['value']}")
        elif isinstance(v, dict):
            lines.append("Correct result: " + "; ".join(f"{fund(i)}: {x}" for i, x in v.items()))
        elif isinstance(v, list) and v and isinstance(v[0], list):
            lines.append("Correct order: " + "; ".join(f"{fund(i)} {x}" for i, x in v))
        elif isinstance(v, list):
            lines.append("Correct funds: " + "; ".join(fund(i) for i in v))
        else:
            lines.append(f"Correct result: {v}")
    elif e["type"] == "rubric":
        lines.append("The answer must cover ALL of:")
        lines += [f"  - {c}" for c in v.get("must_cover", [])]
        if v.get("must_not"):
            lines.append("The answer must NOT:")
            lines += [f"  - {c}" for c in v["must_not"]]

    behavior = e.get("behavior") or ("answer" if e["type"] != "behavior" else v)
    lines.append(f"Required behavior: {behavior}: the answer {BEHAVIORS[behavior]}.")
    if e.get("note") or q.get("note"):
        lines.append(f"Note: {e.get('note') or q.get('note')}")
    return "\n".join(lines)


def judge(q: dict, answer: str, names: dict[str, str], budget: Budget | None = None) -> tuple[Verdict, float]:
    prompt = (f"{INSTRUCTIONS}\n\n<question>\n{q['question']}\n</question>\n\n"
              f"<expectation>\n{describe_expectation(q, names)}\n</expectation>\n\n"
              f"<answer>\n{answer}\n</answer>")
    response, cost = call(
        budget, parse=True,
        model=MODEL,
        max_tokens=4096,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Verdict,
    )
    return response.parsed_output, cost
