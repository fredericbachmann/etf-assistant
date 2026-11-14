"""
Retrieval evaluation: does search put the chunk that contains the answer near the top?

No LLM is involved. If retrieval misses the evidence, the LLM can't answer from
it, so this is measured first: it's fast, free and exactly reproducible.

Query sets
  golden     the single-fact questions of evals/golden_v1 (realistic wording,
             tickers like "VWCE"). Small (21), so one question moves the score ~5 points.
  templated  one generated question per verified fact ("What is the TER of the
             {fund}?"), ~170 questions. Plain wording, but enough of them to see
             differences between configurations.
  per_fund   the templated questions without the fund name ("What is the TER of
             the fund?"), searched within the right fund only. This is what the
             agent's per-fund search mode sends: one query for all funds, so no
             fund name in it. (With the filter, a fund name in the query only adds
             noise: it matches every chunk that repeats the name.)

Relevance
  A retrieved chunk is relevant to a fact if it belongs to the right fund AND
  contains the fact's quote from data/facts (ignoring spaces, case and Markdown).
  Content-based, so the same statement on another page counts too.
  A question asking for several facts needs all of them.

Metrics (per query: rank = position of the first relevant chunk, or of the
last fact to be covered when several are needed)
  Recall@k  share of queries whose evidence is within the top k chunks.
            The RAG prompt will contain the top k, so Recall@5 is the headline.
  MRR       mean of 1/rank (0 if not in the top 10): rewards ranking it first.

Configurations
  bm25 / vector / hybrid, each without and with an ISIN filter. "+isin" is an
  oracle: we pass the correct fund, as if the agent had already resolved
  "EIMI" -> IE00BKM4GZ66. It shows how much fund identification is worth.

  "hybrid +isin" on the per_fund set is exactly what the agent's per-fund
  search mode does (search_documents with isins), so its per-field table shows
  which details that mode finds reliably and which need a fallback to reading
  documents.

Usage:
  uv run eval-retrieval                 # all configurations and sets, plus the per-field table
  uv run eval-retrieval --show-misses   # also list golden questions that miss at k=5
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

import yaml

from etf_assistant.evals.fund_facts import FundFacts, Sourced
from etf_assistant.evals.textmatch import squash
from etf_assistant.paths import EVALS, FACTS
from etf_assistant.retrieval.embeddings import MODEL_NAME
from etf_assistant.retrieval.store import Hit, search

K_VALUES = (1, 3, 5, 10)
K_MAX = max(K_VALUES)
MODES = ("bm25", "vector", "hybrid")

TEMPLATES = {
    "issuer": "Which company issues the {name}?",
    "index_name": "Which index does the {name} track?",
    "ter_pct": "What is the TER of the {name}?",
    "swap_fee_pct": "What swap fee does the {name} charge?",
    "kid_ongoing_costs_pct": "What management and operating costs does the KID state for the {name}?",
    "kid_transaction_costs_pct": "What transaction costs does the KID state for the {name}?",
    "sri": "What is the summary risk indicator of the {name}?",
    "recommended_holding_period_years": "What is the recommended holding period of the {name}?",
    "distribution_policy": "Does the {name} distribute or accumulate income?",
    "distribution_frequency": "How often does the {name} pay dividends?",
    "replication": "How does the {name} replicate its index?",
    "currency_hedged": "Is the {name} currency hedged?",
    "domicile": "Where is the {name} domiciled?",
    "fund_currency": "What is the base currency of the {name}?",
    "launch_date": "When was the {name} launched?",
    "fund_size": "How large is the {name}?",
    "holdings_count": "How many holdings does the {name} have?",
}


@dataclass
class Query:
    id: str
    set: str
    question: str
    isin: str
    quotes: dict[str, str]  # field -> quote that must be retrieved


@dataclass
class Result:
    query: Query
    rank: int | None  # None = evidence not in the top K_MAX
    top_ids: list[str] = field(default_factory=list)


def load_facts() -> dict[str, FundFacts]:
    return {f.stem: FundFacts.model_validate_json(f.read_text()) for f in sorted(FACTS.glob("*.json"))}


def evidence(facts: FundFacts, name: str) -> str | None:
    s = getattr(facts, name)
    if not isinstance(s, Sourced) or not s.quote:
        return None
    if name == "currency_hedged" and s.value is False:
        return None  # "not hedged" is inferred from silence; its quote isn't evidence
    return s.quote


def build_queries(facts: dict[str, FundFacts]) -> list[Query]:
    spec = yaml.safe_load((EVALS / "golden_v1.yaml").read_text(encoding="utf-8"))
    queries = []
    for q in spec["questions"]:
        e = q["expected"]
        if e["type"] != "fact":
            continue
        fields = e["field"] if isinstance(e["field"], list) else [e["field"]]
        quotes = {f: evidence(facts[e["isin"]], f) for f in fields}
        if all(quotes.values()):
            queries.append(Query(q["id"], "golden", q["question"], e["isin"], quotes))

    for isin, fund in spec["funds"].items():
        for name, template in TEMPLATES.items():
            if quote := evidence(facts[isin], name):
                queries.append(Query(f"t_{isin}_{name}", "templated",
                                     template.format(name=fund["name"]), isin, {name: quote}))
                queries.append(Query(f"p_{isin}_{name}", "per_fund",
                                     template.format(name="fund"), isin, {name: quote}))
    return queries


def rank_of(query: Query, hits: list[Hit]) -> int | None:
    ranks = []
    for quote in query.quotes.values():
        q = squash(quote)
        r = next((i for i, h in enumerate(hits, 1) if h.isin == query.isin and q in squash(h.text)), None)
        if r is None:
            return None
        ranks.append(r)
    return max(ranks)


def run(queries: list[Query], mode: str, isin_filter: bool) -> list[Result]:
    out = []
    for q in queries:
        hits = search(q.question, k=K_MAX, mode=mode, isin=q.isin if isin_filter else None)
        out.append(Result(q, rank_of(q, hits), [h.chunk_id for h in hits]))
    return out


def metrics(results: list[Result]) -> dict[str, float]:
    n = len(results)
    m = {f"R@{k}": sum(r.rank is not None and r.rank <= k for r in results) / n for k in K_VALUES}
    m["MRR"] = sum(1 / r.rank for r in results if r.rank) / n
    return m


def field_of(query: Query) -> str:
    """Templated queries ask for one field; golden ones may ask for several."""
    return "+".join(query.quotes)


def metrics_by_field(results: list[Result]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(field_of(r.query), []).append(r)
    return {f: {"n": len(rs), **metrics(rs)} for f, rs in groups.items()}


def print_by_field(by_field: dict[str, dict[str, float]], title: str) -> None:
    """Weakest fields first: where retrieval needs work."""
    print(f"\n{title}")
    print(f"  {'field':36} {'n':>3} " + " ".join(f"{f'R@{k}':>6}" for k in K_VALUES))
    for f, m in sorted(by_field.items(), key=lambda fm: (fm[1]["R@5"], fm[1]["R@10"])):
        print(f"  {f:36} {m['n']:3.0f} " + " ".join(f"{m[f'R@{k}']:6.2f}" for k in K_VALUES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    queries = build_queries(load_facts())
    sets = {s: [q for q in queries if q.set == s] for s in ("golden", "templated", "per_fund")}
    print(f"embedding model: {MODEL_NAME}")
    print("queries: " + ", ".join(f"{s} {len(qs)}" for s, qs in sets.items()) + "\n")

    cols = [f"R@{k}" for k in K_VALUES] + ["MRR"]
    print(f"{'config':14}" + "".join(f"| {s} ({len(qs)})".ljust(36) for s, qs in sets.items()))
    print(f"{'':14}" + ("| " + " ".join(f"{c:>6}" for c in cols) + "  ") * len(sets))

    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "embedding_model": MODEL_NAME, "configs": {}}
    misses = {}
    t = time.time()
    for isin_filter in (False, True):
        for mode in MODES:
            name = mode + (" +isin" if isin_filter else "")
            row, report["configs"][name] = f"{name:14}", {}
            for set_name, qs in sets.items():
                if set_name == "per_fund" and not isin_filter:
                    row += "| " + " ".join(f"{'-':>6}" for _ in cols) + "  "  # no fund named, no filter: meaningless
                    continue
                results = run(qs, mode, isin_filter)
                m = metrics(results)
                report["configs"][name][set_name] = {
                    "metrics": m,
                    "by_field": metrics_by_field(results),
                    "ranks": {r.query.id: r.rank for r in results},
                }
                row += "| " + " ".join(f"{m[c]:6.2f}" for c in cols) + "  "
                if set_name == "golden":
                    misses[name] = [r for r in results if not (r.rank and r.rank <= 5)]
            print(row)
    print(f"\n({time.time() - t:.0f}s)")
    # The configuration behind search_documents(isins=...): one filtered search per fund.
    print_by_field(report["configs"]["hybrid +isin"]["per_fund"]["by_field"],
                   "per field, hybrid +isin, per_fund (what the agent's per-fund search mode sees):")

    out = EVALS / "results" / f"retrieval_{report['run_at'].replace(':', '-')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(EVALS.parent)}")

    if args.show_misses:
        for name in ("hybrid", "hybrid +isin"):
            print(f"\ngolden misses at k=5, {name}:")
            for r in misses[name]:
                q = r.query
                print(f"  {q.id} rank={r.rank}  {q.question}")
                print(f"      needs {q.isin} {list(q.quotes)}: {list(q.quotes.values())}")
                print(f"      got   {r.top_ids[:3]}")


if __name__ == "__main__":
    main()
