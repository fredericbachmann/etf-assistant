"""
End-to-end evaluation: run the golden questions through the agent and grade the answers.

Each question gets a fresh agent (no shared conversation). Its answer is graded
by the LLM judge (judge.py) against the expected result resolved from the
verified facts (evals/golden_v1.resolved.json, made by `uv run check-golden`).

Cost safety: agent and judge share one Budget with a hard cap. A call is only
sent if its worst-case cost still fits under the cap; otherwise the run stops,
and questions not reached are reported as "not run", never silently dropped.

Usage:
  uv run eval-e2e                        # all questions, cap $0.50
  uv run eval-e2e --max-cost 0.10 --only f01,c03,a01
  uv run eval-e2e --category advice
  uv run eval-e2e --rejudge evals/results/e2e_<time>.json   # grade saved answers again (judge only)

--rejudge separates grading from answering: after changing the judge, the same
answers are graded again for ~2 cents, without re-running the agent. Comparing
old and new grades on identical answers shows exactly what the judge change did.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from etf_assistant.agent import Agent
from etf_assistant.catalog import load_catalog
from etf_assistant.judge import judge
from etf_assistant.llm import Budget, BudgetExceeded
from etf_assistant.paths import EVALS, FACTS

RESOLVED = EVALS / "golden_v1.resolved.json"
WORKERS = 4  # questions in flight at once


def load_questions() -> tuple[list[dict], dict]:
    sources = [EVALS / "golden_v1.yaml", *FACTS.glob("*.json")]
    if not RESOLVED.exists() or any(p.stat().st_mtime > RESOLVED.stat().st_mtime for p in sources):
        sys.exit(f"{RESOLVED.name} is missing or older than the golden set / facts: run `uv run check-golden` first")
    data = json.loads(RESOLVED.read_text(encoding="utf-8"))
    funds = data["funds"]
    for q in data["questions"]:  # resolve "all" / "group:x" so the judge sees which funds were in scope
        sel = q["expected"].get("isins")
        if sel == "all":
            q["_isins"] = list(funds)
        elif isinstance(sel, str) and sel.startswith("group:"):
            q["_isins"] = [i for i, f in funds.items() if f.get("group") == sel.split(":", 1)[1]]
        else:
            q["_isins"] = sel or []
    return data["questions"], funds


def rejudge_one(prev: dict, q: dict, budget: Budget, names: dict[str, str]) -> dict:
    out = dict(prev)
    if prev.get("status") != "done":
        return out
    try:
        verdict, judge_cost = judge(q, prev["answer"], names, budget)
        out.update(previous_passed=prev["passed"], passed=verdict.passed, behavior=verdict.behavior,
                   judge_reasoning=verdict.reasoning, judge_cost=judge_cost, agent_cost=0.0)
    except BudgetExceeded as e:
        out.update(status="not_run", error=str(e))
    return out


def run_one(q: dict, budget: Budget, names: dict[str, str]) -> dict:
    out = {"id": q["id"], "category": q["category"], "tags": q.get("tags") or [], "question": q["question"]}
    try:
        a = Agent(budget=budget).ask(q["question"])
        out.update(answer=a.text, stop_reason=a.stop_reason, seconds=round(a.seconds, 1),
                   tool_calls=[{"name": t.name, "input": t.input, "error": t.is_error} for t in a.tool_calls],
                   agent_cost=a.cost_usd)
        verdict, judge_cost = judge(q, a.text, names, budget)
        out.update(passed=verdict.passed, behavior=verdict.behavior, judge_reasoning=verdict.reasoning,
                   judge_cost=judge_cost, status="done")
    except BudgetExceeded as e:
        out.update(status="not_run", error=str(e))
    except Exception as e:  # an API or tool failure shouldn't kill the whole run
        out.update(status="error", error=f"{type(e).__name__}: {e}")
    return out


def summarize(results: list[dict]) -> None:
    done = [r for r in results if r["status"] == "done"]

    def rate(rs):
        return f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "-"

    print(f"\n{'category':14}{'passed':>8}")
    by_cat = defaultdict(list)
    for r in done:
        by_cat[r["category"]].append(r)
    for cat, rs in by_cat.items():
        print(f"{cat:14}{rate(rs):>8}")
    print(f"{'TOTAL':14}{rate(done):>8}")

    by_tag = defaultdict(list)
    for r in done:
        for t in r["tags"]:
            by_tag[t].append(r)
    if by_tag:
        print("\nby tag: " + ", ".join(f"{t} {rate(rs)}" for t, rs in sorted(by_tag.items())))

    other = [r for r in results if r["status"] != "done"]
    if other:
        print(f"\n{len(other)} not graded: " + ", ".join(f"{r['id']} ({r['status']})" for r in other))
        print(f"  first reason: {other[0]['error']}")

    if done:
        secs = sorted(r["seconds"] for r in done)
        calls = sum(len(r["tool_calls"]) for r in done) / len(done)
        print(f"\nlatency median {secs[len(secs) // 2]}s, max {secs[-1]}s; {calls:.1f} tool calls per question")

    failed = [r for r in done if not r["passed"]]
    if failed:
        print("\n--- failures (read the judge's reasoning: is the answer wrong, or the judge?) ---")
        for r in failed:
            print(f"\n{r['id']} [{r['category']}] {r['question']}")
            print(f"  behavior: {r['behavior']}; tools: {[t['name'] for t in r['tool_calls']]}")
            print(f"  judge: {r['judge_reasoning']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end eval of the ETF assistant")
    ap.add_argument("--max-cost", type=float, default=0.50, help="hard cap in USD for agent + judge (default 0.50)")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--category")
    ap.add_argument("--rejudge", help="results file whose saved answers should be graded again")
    a = ap.parse_args()

    questions, funds = load_questions()
    if a.only:
        wanted = set(a.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if a.category:
        questions = [q for q in questions if q["category"] == a.category]
    names = {i: f.display_name for i, f in load_catalog().items()}

    budget = Budget(a.max_cost)
    print(f"running {len(questions)} questions, {WORKERS} at a time, hard cap ${budget.cap:.3f}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(WORKERS) as pool:
        if a.rejudge:
            prev = {r["id"]: r for r in json.loads(open(a.rejudge, encoding="utf-8").read())["results"]}
            questions = [q for q in questions if q["id"] in prev]
            jobs = pool.map(lambda q: rejudge_one(prev[q["id"]], q, budget, names), questions)
        else:
            jobs = pool.map(lambda q: run_one(q, budget, names), questions)
        for r in jobs:
            mark = {"done": "PASS" if r.get("passed") else "FAIL"}.get(r["status"], r["status"].upper())
            print(f"  {r['id']:28} {mark:7} ${budget.spent:.3f} spent", flush=True)
            results.append(r)

    summarize(results)
    flipped = [r for r in results if "previous_passed" in r and r["previous_passed"] != r["passed"]]
    if a.rejudge:
        print(f"\nrejudged {a.rejudge}: {len(flipped)} verdicts changed: "
              + ", ".join(f"{r['id']} {'FAIL->PASS' if r['passed'] else 'PASS->FAIL'}" for r in flipped))
    agent_cost = sum(r.get("agent_cost", 0) for r in results)
    judge_cost = sum(r.get("judge_cost", 0) for r in results)
    print(f"\ncost: ${budget.spent:.4f} total (agent ${agent_cost:.4f}, judge ${judge_cost:.4f}; "
          f"{budget.calls} API calls) in {time.time() - t0:.0f}s; cap ${budget.cap:.3f}")

    run_at = datetime.now().isoformat(timespec="seconds")
    out = EVALS / "results" / f"e2e_{run_at.replace(':', '-')}.json"
    out.write_text(json.dumps({"run_at": run_at, "rejudged_from": a.rejudge, "max_cost": a.max_cost, "spent": budget.spent,
                               "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(EVALS.parent)}")


if __name__ == "__main__":
    main()
