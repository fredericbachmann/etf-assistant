"""
The LLM judge: decides whether an answer meets a golden question's expectation.

Answers can't be graded by string matching: "0.2" in the answer key appears as
"0.20%", "physical_sampling" as "optimised sampling", and declining to give
advice has no fixed wording.

The judge is a model too and can be wrong or lenient. Safeguards:
  - It receives the exact expected result, resolved from the verified facts, so
    it compares the answer with that result and doesn't need to know the funds.
  - It writes its reasoning before the verdict (the order of the Verdict
    fields) and checks each requirement separately.
  - eval-e2e prints the reasoning for every failure, so a wrong verdict can be
    told from a wrong answer. The reasoning for passes is in the results file,
    for spot checks.

Grounding. The expectation covers what was asked, not what else the answer
says: an answer can decline advice correctly and quote a wrong return on the
way, or claim the documents don't give a figure they do give. So the judge also
gets the evidence (build_evidence) and lists every statement it contradicts. An
answer passes only if it meets the expectation and that list is empty.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from etf_assistant.agent.tools import read_fund_documents
from etf_assistant.evals.fund_facts import FundFacts, Sourced
from etf_assistant.llm import MODEL, Budget, call

ISIN = re.compile(r"\b[A-Z]{2}[0-9A-Z]{9}[0-9]\b")
MAX_FULL_DOCS = 4  # funds whose whole KID and factsheet the judge reads (about 8k tokens each)

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

FEE_DRAG = "fees paid per fund in currency; accept each figure within 10% and the comparison must be right"

OPS = {
    "argmin": "the fund(s) with the LOWEST value; if several funds tie, all of them must be named",
    "argmax": "the fund(s) with the HIGHEST value; if several funds tie, all of them must be named",
    "diff": "the difference first fund minus second fund, in percentage points",
    "equal": "whether the funds' values are equal (true/false)",
    "sort_asc": "the funds ordered from lowest to highest value; ties may appear in any order",
    "filter_equals": "exactly the funds whose value equals the given value; none missing, none extra",
    "fee_drag_lump": FEE_DRAG,
    "fee_drag_monthly": FEE_DRAG,
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
different value or conclusion for the thing that was asked, or does not commit to the expected result.

Grounding: if <evidence> is given, also check the rest of the answer against it. List in `unsupported` \
every statement that would leave a reader with a wrong figure or a wrong fact about a fund:
- a wrong value, or a figure attributed to the wrong period, table column, fund or document. For every \
figure the answer takes from a table (performance, holdings, costs), find the row in the evidence, match \
the row's values to the column headers in order, and compare the header with what the answer calls the \
figure: a value from the "since inception" column given as the 10-year return is wrong. Tables were \
extracted from PDFs, so a header or a number may be split over several cells; the order of the values \
still follows the order of the headers;
- a claim that the documents don't state something (or that the answer has no figure for it) when the \
evidence states it, including a figure for the fund that the answer says is missing;
- a statement about several funds that is false for one of them, such as a "lowest" that leaves out a \
lower one.
Give each as the answer's statement, then the evidence's wording that contradicts it. Before listing a \
statement, read it again in its context and take the reading under which it is true, if there is one. Do \
NOT list: rounding; different words for the same thing (a fund that samples or optimises still holds its \
securities physically; exchange ticker, RIC and Bloomberg code); which ticker or name the answer uses \
for a fund (those come from the library list, which you don't have); general explanations that aren't about a \
specific fund; arithmetic and comparisons on documented figures; statements the answer marks as general \
knowledge, an assumption or its own inference; a wrong page number in a citation; statements you can't \
check because the evidence doesn't include that fund, document or passage. Where only verified key facts \
are given for a fund, they are short excerpts: check values against them, but what else that fund's \
documents say or don't say can't be checked. Grounding is separate from the expectation: decide \
`meets_expectation` as if there were no evidence. Don't judge dates by your own sense of the current date."""


class Verdict(BaseModel):
    # order matters: the model writes reasoning before deciding
    reasoning: str = Field(description="Check each requirement in turn, briefly, quoting the answer; "
                                       "then check the answer's other statements against the evidence")
    behavior: Literal["answered", "declined_advice", "said_not_in_docs", "asked_clarification", "bare_refusal"] = Field(
        description="What the answer mainly does")
    meets_expectation: bool
    unsupported: list[str] = Field(
        description="Statements in the answer that the evidence contradicts, each with the contradicting "
                    "wording; empty if there are none or no evidence was given")

    @property
    def passed(self) -> bool:
        return self.meets_expectation and not self.unsupported


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
        lines.append("Funds considered: " + "; ".join(fund(i) for i in q["funds_in_scope"]))
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


def key_facts(isin: str, facts: FundFacts) -> str:
    lines = []
    for name in FundFacts.model_fields:
        s = getattr(facts, name)
        if isinstance(s, Sourced):
            where = f' ({s.doc} p.{s.page}: "{s.quote}")' if s.quote else ""
            lines.append(f"  {name}: {'not stated in the documents' if s.value is None else s.value}{where}")
    return f'<key_facts isin="{isin}">\n' + "\n".join(lines) + "\n</key_facts>"


def build_evidence(q: dict, answer: str, names: dict[str, str], facts: dict[str, FundFacts]) -> str:
    """What the judge checks the answer's statements against: the documents of the funds it is about.

    The funds are those the question is about (from the answer key) and those the answer names
    by ISIN. Up to MAX_FULL_DOCS funds: their whole KID and factsheet, as the assistant reads
    them. More than that (rankings across the library): the verified key facts instead, which
    is what such answers state, at a fraction of the size. Empty if no fund is involved.
    """
    asked = [q["expected"]["isin"]] if q["expected"].get("isin") else q.get("funds_in_scope") or []
    isins = [i for i in dict.fromkeys([*asked, *ISIN.findall(answer)]) if i in names]
    if len(isins) <= MAX_FULL_DOCS:
        return "\n\n".join(read_fund_documents(i, "both") for i in isins)
    return "\n\n".join(key_facts(i, facts[i]) for i in isins if i in facts)


def judge(q: dict, answer: str, names: dict[str, str], budget: Budget | None = None,
          model: str = MODEL, evidence: str = "") -> tuple[Verdict, float]:
    prompt = (f"{INSTRUCTIONS}\n\n<question>\n{q['question']}\n</question>\n\n"
              f"<expectation>\n{describe_expectation(q, names)}\n</expectation>\n\n"
              + (f"<evidence>\n{evidence}\n</evidence>\n\n" if evidence else "")
              + f"<answer>\n{answer}\n</answer>")
    response, cost = call(
        budget, parse=True,
        model=model,
        max_tokens=4096,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Verdict,
    )
    if response.parsed_output is None:  # e.g. the model declined: there is no verdict to read
        raise RuntimeError(f"judge returned no verdict (stop_reason {response.stop_reason})")
    return response.parsed_output, cost
