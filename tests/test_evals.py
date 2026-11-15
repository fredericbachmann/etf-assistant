"""Offline checks of the eval plumbing: which facts count, and how repeated runs are summarised."""

import json

from etf_assistant.evals.eval_e2e import consistency
from etf_assistant.evals.fund_facts import load_verified_facts
from etf_assistant.paths import FACTS


def test_load_verified_facts_skips_unverified(tmp_path):
    src = json.loads(next(iter(sorted(FACTS.glob("*.json")))).read_text(encoding="utf-8"))
    (tmp_path / "A.json").write_text(json.dumps({**src, "isin": "A", "verified": True}), encoding="utf-8")
    (tmp_path / "B.json").write_text(json.dumps({**src, "isin": "B", "verified": False}), encoding="utf-8")
    assert list(load_verified_facts(tmp_path)) == ["A"]


def test_consistency_counts_only_repeated_graded_questions():
    def r(qid, passed, status="done"):
        return {"id": qid, "status": status, "passed": passed}

    results = [r("c03", True), r("c03", False), r("c03", True),
               r("f01", True),                       # asked once: not a consistency measurement
               r("c05", True), r("c05", None, "not_run")]
    assert consistency(results) == {"c03": (2, 3)}
