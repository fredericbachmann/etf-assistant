"""Per-field retrieval metrics (evals/eval_retrieval.py), on made-up ranks."""

from etf_assistant.evals.eval_retrieval import Query, Result, metrics_by_field


def result(field: str, rank: int | None) -> Result:
    return Result(Query(id=f"t_{field}_{rank}", set="templated", question="?", isin="X", quotes={field: "q"}), rank)


def test_metrics_by_field():
    by = metrics_by_field([result("sri", 1), result("sri", 4), result("ter_pct", None)])
    assert by["sri"]["n"] == 2 and by["sri"]["R@1"] == 0.5 and by["sri"]["R@5"] == 1.0
    assert by["ter_pct"]["R@10"] == 0.0 and by["ter_pct"]["MRR"] == 0.0
