"""
End-to-end evaluation: run the golden questions through the agent and grade the answers.

Each question gets a fresh agent (no shared conversation). Its answer is graded
by the LLM judge (judge.py) against the expected result, resolved from the
verified facts at start-up (check_golden.resolve_golden). If any question
can't be resolved, the run doesn't start: `uv run check-golden` lists why.
The judge also reads the documents of the funds an answer is about and fails
an answer that states something they contradict, whatever the expectation.

Cost safety: agent and judge share one Budget with a hard cap. A call is only
sent if its worst-case cost still fits under the cap; otherwise the run stops,
and questions not reached are reported as "not run", never silently dropped.

Consistency: --repeat N asks every selected question N times, each with a fresh
agent, and reports how many of the N runs passed per question. A question that
passes some runs and fails others is flaky: a single run doesn't show whether a
change fixed it. Questions tagged `consistency` in the golden set are the ones
known or expected to vary, so the usual call is --tag consistency --repeat 5.

Usage:
  uv run eval-e2e                        # all questions, cap $0.50
  uv run eval-e2e --max-cost 0.10 --only f01,c03,a01
  uv run eval-e2e --category advice
  uv run eval-e2e --tag consistency --repeat 5 --max-cost 0.30
  uv run eval-e2e --rejudge evals/results/e2e_<time>.json   # grade saved answers again (judge only)
  uv run eval-e2e --rejudge evals/results/e2e_<time>.json --judge-model claude-sonnet-5-5 --max-cost 1

--rejudge separates grading from answering: after a change to the judge, the
same answers are graded again without re-running the agent. Questions that are
not in the results file (added to the golden set since) are skipped.
Old and new grades on identical answers show what the judge change did.

--judge-model grades with another model than the agent's (about 20x the cost
with Sonnet). With --rejudge it gives a second judge's verdicts on the same
answers: how often the two agree says how much a pass rate depends on the judge.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

from etf_assistant.agent.loop import Agent
from etf_assistant.corpus.catalog import load_catalog
from etf_assistant.evals.check_golden import Golden, resolve_golden
from etf_assistant.evals.fund_facts import FundFacts, load_verified_facts
from etf_assistant.evals.judge import Verdict, build_evidence, judge
from etf_assistant.llm import MODEL, RATES, Budget, BudgetExceeded, warm_up
from etf_assistant.paths import EVALS
from etf_assistant.retrieval.store import check_ready

WORKERS = 4  # questions in flight at once


@dataclass
class Grader:
    """What every judge call needs besides the question and the answer."""
    budget: Budget
    names: dict[str, str]  # ISIN -> fund name
    facts: dict[str, FundFacts]
    model: str

    def grade(self, q: dict, answer: str) -> tuple[Verdict, float]:
        return judge(q, answer, self.names, self.budget, self.model,
                     evidence=build_evidence(q, answer, self.names, self.facts))


def graded(verdict: Verdict, judge_cost: float) -> dict:
    return {"passed": verdict.passed, "meets_expectation": verdict.meets_expectation,
            "unsupported": verdict.unsupported, "behavior": verdict.behavior,
            "judge_reasoning": verdict.reasoning, "judge_cost": judge_cost}


def load_golden() -> Golden:
    g = resolve_golden()
    if g.problems:
        sys.exit(f"{len(g.problems)} golden question(s) can't be resolved (first: {g.problems[0]}): "
                 "run `uv run check-golden` for the full list")
    return g


def rejudge_one(prev: dict, q: dict, grader: Grader) -> dict:
    out = dict(prev)
    if prev.get("status") != "done":
        return out
    try:
        verdict, judge_cost = grader.grade(q, prev["answer"])
        out.update(graded(verdict, judge_cost), previous_passed=prev["passed"], agent_cost=0.0)
    except BudgetExceeded as e:
        out.update(status="not_run", error=str(e))
    except Exception as e:  # one failed judge call shouldn't lose the verdicts already paid for
        out.update(status="error", error=f"{type(e).__name__}: {e}")
    return out


def run_one(q: dict, run: int, grader: Grader) -> dict:
    out = {"id": q["id"], "run": run, "category": q["category"], "tags": q.get("tags") or [],
           "question": q["question"]}
    try:
        a = Agent(budget=grader.budget).ask(q["question"])
        out.update(answer=a.text, stop_reason=a.stop_reason, seconds=round(a.seconds, 1),
                   tool_calls=[{"name": t.name, "input": t.input, "error": t.is_error} for t in a.tool_calls],
                   agent_cost=a.cost_usd)
        verdict, judge_cost = grader.grade(q, a.text)
        out.update(graded(verdict, judge_cost), status="done")
    except BudgetExceeded as e:
        out.update(status="not_run", error=str(e))
    except Exception as e:  # an API or tool failure shouldn't kill the whole run
        out.update(status="error", error=f"{type(e).__name__}: {e}")
    return out


def label(r: dict) -> str:
    """Question id, plus the run number when a question was asked more than once."""
    return f"{r['id']}#{r['run'] + 1}" if r.get("repeated") else r["id"]


def consistency(results: list[dict]) -> dict[str, tuple[int, int]]:
    """Question id -> (runs passed, runs graded), for questions graded more than once."""
    by_id = defaultdict(list)
    for r in results:
        if r["status"] == "done":
            by_id[r["id"]].append(r["passed"])
    return {i: (sum(p), len(p)) for i, p in by_id.items() if len(p) > 1}


def summarize(results: list[dict]) -> None:
    done = [r for r in results if r["status"] == "done"]

    def rate(rs):
        return f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "-"

    unit = "runs passed" if any(r.get("repeated") for r in results) else "passed"
    print(f"\n{'category':14}{unit:>12}")
    by_cat = defaultdict(list)
    for r in done:
        by_cat[r["category"]].append(r)
    for cat, rs in by_cat.items():
        print(f"{cat:14}{rate(rs):>12}")
    print(f"{'TOTAL':14}{rate(done):>12}")
    ungrounded = [r for r in done if r.get("unsupported")]
    if ungrounded:
        print(f"\ngrounding: {len(ungrounded)} answers state something the documents contradict; "
              f"{sum(r['meets_expectation'] for r in ungrounded)} of them met the expectation otherwise")

    by_tag = defaultdict(list)
    for r in done:
        for t in r["tags"]:
            by_tag[t].append(r)
    if by_tag:
        print("\nby tag: " + ", ".join(f"{t} {rate(rs)}" for t, rs in sorted(by_tag.items())))

    if cons := consistency(results):
        flaky = [i for i, (p, n) in cons.items() if 0 < p < n]
        print(f"\nconsistency ({len(cons)} questions asked repeatedly): "
              f"{sum(p == n for p, n in cons.values())} always pass, "
              f"{sum(p == 0 for p, _ in cons.values())} always fail, {len(flaky)} flaky")
        for i, (p, n) in cons.items():
            print(f"  {i:28} {p}/{n}" + ("  flaky" if i in flaky else ""))

    other = [r for r in results if r["status"] != "done"]
    if other:
        print(f"\n{len(other)} not graded: " + ", ".join(f"{label(r)} ({r['status']})" for r in other))
        print(f"  first reason: {other[0]['error']}")

    if done:
        secs = sorted(r["seconds"] for r in done)
        calls = sum(len(r["tool_calls"]) for r in done) / len(done)
        print(f"\nlatency median {secs[len(secs) // 2]}s, max {secs[-1]}s; {calls:.1f} tool calls per question")

    failed = [r for r in done if not r["passed"]]
    if failed:
        print("\n--- failures (read the judge's reasoning: is the answer wrong, or the judge?) ---")
        for r in failed:
            print(f"\n{label(r)} [{r['category']}] {r['question']}")
            print(f"  behavior: {r['behavior']}; tools: {[t['name'] for t in r['tool_calls']]}")
            print(f"  judge: {r['judge_reasoning']}")
            for u in r.get("unsupported") or []:
                print(f"  unsupported: {u}")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end eval of the ETF assistant")
    ap.add_argument("--max-cost", type=float, default=0.50, help="hard cap in USD for agent + judge (default 0.50)")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--category")
    ap.add_argument("--tag", help="only questions with this tag, e.g. consistency")
    ap.add_argument("--repeat", type=int, default=1, help="ask each question N times (fresh agent each time)")
    ap.add_argument("--rejudge", help="results file whose saved answers should be graded again")
    ap.add_argument("--judge-model", default=MODEL, choices=sorted(RATES),
                    help=f"model that grades the answers (default {MODEL}, the agent's)")
    a = ap.parse_args()
    if a.repeat < 1:
        ap.error("--repeat must be at least 1")
    if a.rejudge and a.repeat > 1:
        ap.error("--rejudge grades the saved runs; --repeat doesn't apply")

    golden = load_golden()
    questions = golden.questions
    if a.only:
        wanted = set(a.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if a.category:
        questions = [q for q in questions if q["category"] == a.category]
    if a.tag:
        questions = [q for q in questions if a.tag in (q.get("tags") or [])]
    by_id = {q["id"]: q for q in questions}
    names = {i: f.display_name for i, f in load_catalog().items()}

    if not a.rejudge and (problem := check_ready()):
        sys.exit(f"preflight failed: {problem}")  # don't measure a half-broken system
    warm_up()  # renew credentials once, before threads race to do it
    budget = Budget(a.max_cost)
    grader = Grader(budget, names, load_verified_facts(), a.judge_model)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(WORKERS) as pool:
        if a.rejudge:
            prev = [r for r in json.loads(open(a.rejudge, encoding="utf-8").read())["results"] if r["id"] in by_id]
            for r in prev:
                r.setdefault("run", 0)  # files from before --repeat
            print(f"rejudging {len(prev)} saved answers with {a.judge_model}, {WORKERS} at a time, "
                  f"hard cap ${budget.cap:.3f}")
            jobs = pool.map(lambda r: rejudge_one(r, by_id[r["id"]], grader), prev)
        else:
            # run-major order: if the budget runs out, every question has had its first runs
            work = [(q, run) for run in range(a.repeat) for q in questions]
            print(f"running {len(questions)} questions x {a.repeat} = {len(work)} runs, "
                  f"{WORKERS} at a time, judge {a.judge_model}, hard cap ${budget.cap:.3f}")
            jobs = pool.map(lambda qr: run_one(qr[0], qr[1], grader), work)
        for r in jobs:
            if a.repeat > 1:
                r["repeated"] = True
            mark = {"done": "PASS" if r.get("passed") else "FAIL"}.get(r["status"], r["status"].upper())
            print(f"  {label(r):28} {mark:7} ${budget.spent:.3f} spent", flush=True)
            results.append(r)

    summarize(results)
    flipped = [r for r in results if "previous_passed" in r and r["previous_passed"] != r["passed"]]
    if a.rejudge:
        print(f"\nrejudged {a.rejudge}: {len(flipped)} verdicts changed: "
              + ", ".join(f"{label(r)} {'FAIL->PASS' if r['passed'] else 'PASS->FAIL'}" for r in flipped))
    agent_cost = sum(r.get("agent_cost", 0) for r in results)
    judge_cost = sum(r.get("judge_cost", 0) for r in results)
    print(f"\ncost: ${budget.spent:.4f} total (agent ${agent_cost:.4f}, judge ${judge_cost:.4f}; "
          f"{budget.calls} API calls) in {time.time() - t0:.0f}s; cap ${budget.cap:.3f}")

    run_at = datetime.now().isoformat(timespec="seconds")
    out = EVALS / "results" / f"e2e_{run_at.replace(':', '-')}.json"
    out.write_text(json.dumps({"run_at": run_at, "rejudged_from": a.rejudge, "max_cost": a.max_cost,
                               "golden_version": golden.version, "agent_model": MODEL, "judge_model": a.judge_model,
                               "repeat": a.repeat, "tag": a.tag, "spent": budget.spent,
                               "consistency": {i: {"passed": p, "runs": n} for i, (p, n) in consistency(results).items()},
                               "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(EVALS.parent)}")


if __name__ == "__main__":
    main()
