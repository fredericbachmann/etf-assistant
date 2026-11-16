"""Offline checks of the eval plumbing: which facts count, how repeated runs are summarised,
and what the judge is given and how its verdict becomes a pass."""

import json

from etf_assistant.evals import judge
from etf_assistant.evals.check_golden import resolve_golden
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


def test_golden_set_resolves_and_keeps_the_v1_ids():
    g = resolve_golden()
    assert g.problems == []
    ids = {q["id"] for q in g.questions}
    v1 = [f"f{n:02}" for n in range(1, 21)] + [f"c{n:02}" for n in range(1, 9)] + ["k01", "k02", "k03"] \
        + [f"o{n:02}" for n in range(1, 11)] + [f"a{n:02}" for n in range(1, 15)] + [f"u{n:02}" for n in range(1, 8)]
    assert not set(v1) - ids  # old results files stay comparable and can be rejudged


def test_a_verdict_passes_only_if_expected_and_grounded():
    def v(meets, unsupported):
        return judge.Verdict(reasoning="", behavior="answered", meets_expectation=meets, unsupported=unsupported)

    assert v(True, []).passed
    assert not v(False, []).passed
    assert not v(True, ["10-year return 818.71%: the table gives 311.46%"]).passed


def test_evidence_is_whole_documents_for_a_few_funds_and_key_facts_for_many(monkeypatch):
    monkeypatch.setattr(judge, "read_fund_documents", lambda isin, doc_type: f"<document {isin} {doc_type}>")
    facts = load_verified_facts()
    names = dict.fromkeys(facts, "a fund")
    a, b = list(facts)[:2]

    def q(isin=None, scope=()):
        return {"expected": {"isin": isin} if isin else {}, "funds_in_scope": list(scope)}

    # the fund asked about comes first, then funds the answer names; unknown ISINs are ignored
    few = judge.build_evidence(q(isin=a), f"see [{b} KID p.1], not IE00XXXXXXX1", names, facts)
    assert few == f"<document {a} both>\n\n<document {b} both>"

    many = judge.build_evidence(q(scope=facts), "a ranking of all funds", names, facts)
    assert "<document" not in many and many.count("<key_facts") == len(facts)
    assert 'ter_pct: 0.14 (factsheet p.1: "Ongoing Charges Figure† 0.14%")' in many
    assert "swap_fee_pct: not stated in the documents" in many

    assert judge.build_evidence(q(), "I can't pick a fund for you.", names, facts) == ""
