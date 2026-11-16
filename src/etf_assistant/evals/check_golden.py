"""
Resolve the golden test set against the verified facts and report problems.

Usage:
    uv run check-golden                                  # defaults below
    uv run check-golden evals/golden_v2.yaml data/facts/

resolve_golden() turns every question into one with its concrete expected
answer filled in. eval-e2e calls it directly at start-up, so there is no file
to keep in sync. This command is the lint step: it also writes
evals/golden_v2.resolved.json for reading, and exits non-zero if anything
can't be resolved (missing or unverified facts file, null value where a fact is
expected, duplicate ids).
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from etf_assistant.evals.fund_facts import load_verified_facts
from etf_assistant.paths import EVALS, FACTS

GOLDEN = EVALS / "golden_v2.yaml"  # golden_v1.yaml stays: the set older results files were run on
PLACEHOLDER = re.compile(r"\{(IE[0-9A-Z]{10})\.(\w+)\}")


class Resolver:
    def __init__(self, spec: dict, facts: dict[str, dict]):
        self.funds = spec["funds"]
        self.facts = facts  # ISIN -> verified facts as JSON-like dicts (dates as ISO strings)
        self.problems: list[str] = []

    # --- helpers -------------------------------------------------------
    def isins(self, sel) -> list[str]:
        if sel == "all":
            return list(self.funds)
        if isinstance(sel, str) and sel.startswith("group:"):
            g = sel.split(":", 1)[1]
            return [i for i, f in self.funds.items() if f.get("group") == g]
        return list(sel or [])

    def raw(self, qid: str, isin: str, field: str):
        fund = self.facts.get(isin)
        if fund is None:
            self.problems.append(f"{qid}: no verified facts file for {isin}")
            return None
        entry = fund.get(field)
        if not isinstance(entry, dict):
            self.problems.append(f"{qid}: field '{field}' missing for {isin}")
            return None
        return entry.get("value")

    def value(self, qid: str, isin: str, field, required=True):
        """Single field -> its value. List of fields -> numeric sum (null counts as 0, e.g. no swap fee)."""
        if isinstance(field, list):
            vals = [self.raw(qid, isin, f) for f in field]
            if vals[0] is None and required:
                self.problems.append(f"{qid}: {isin}.{field[0]} is null")
            return round(sum(v or 0 for v in vals), 6)
        v = self.raw(qid, isin, field)
        if v is None and required:
            self.problems.append(f"{qid}: {isin}.{field} is null")
        return v

    # --- expected types ------------------------------------------------
    def resolve(self, q: dict):
        e, qid = q["expected"], q["id"]
        t = e["type"]
        if t == "fact":
            fields = e["field"] if isinstance(e["field"], list) else [e["field"]]
            return {f: self.value(qid, e["isin"], f) for f in fields} if len(fields) > 1 else self.value(qid, e["isin"], fields[0])
        if t == "compute":
            return self.compute(qid, e)
        if t == "rubric":
            return {k: [self.fill(qid, s) for s in v] if isinstance(v, list) else v for k, v in e.items() if k != "type"}
        if t == "behavior":
            return e["behavior"]
        self.problems.append(f"{qid}: unknown expected.type {t}")

    def compute(self, qid: str, e: dict):
        op, field = e["op"], e["field"]
        ids = self.isins(e["isins"])
        # allow_missing: funds whose value is null are skipped instead of reported as a problem
        vals = {i: self.value(qid, i, field, required=not e.get("allow_missing")) for i in ids}
        known = {i: v for i, v in vals.items() if v is not None}
        if op in ("argmin", "argmax"):
            if not known:
                return None
            best = (min if op == "argmin" else max)(known.values())
            return {"funds": [i for i, v in known.items() if v == best], "value": best}
        if op == "diff":
            a, b = (vals[i] for i in ids)
            return None if a is None or b is None else round(a - b, 6)
        if op == "equal":
            return len(set(known.values())) == 1 and len(known) == len(ids)
        if op == "sort_asc":
            return sorted(known.items(), key=lambda kv: kv[1])
        if op == "filter_equals":
            return [i for i, v in vals.items() if v == e["value"]]
        if op in ("fee_drag_lump", "fee_drag_monthly"):
            # Zero market return; c = TER (+ swap fee where stated).
            # Lump sum A: cost = A * (1 - (1 - c)^years).
            # Monthly plan: contribute C at the start of each month, then deduct c/12;
            # cost = total contributed - final balance.
            out = {}
            for i, c in vals.items():
                if c is None:
                    continue
                rate = c / 100
                if op == "fee_drag_lump":
                    out[i] = round(e["amount"] * (1 - (1 - rate) ** e["years"]), 2)
                else:
                    bal, paid = 0.0, 0.0
                    for _ in range(e["years"] * 12):
                        bal += e["amount"]
                        paid += e["amount"]
                        bal *= 1 - rate / 12
                    out[i] = round(paid - bal, 2)
            return out
        self.problems.append(f"{qid}: unknown op {op}")

    def fill(self, qid: str, text: str) -> str:
        def sub(m):
            v = self.value(qid, m.group(1), m.group(2))
            return "?" if v is None else str(v)
        if not isinstance(text, str):
            self.problems.append(f"{qid}: non-text rubric item {text!r} (quote it in the YAML)")
            return str(text)
        return PLACEHOLDER.sub(sub, text)

    def generated(self, gen: dict) -> list[dict]:
        """One question per (fund, field) whose value is null in the verified facts."""
        out = []
        for isin, fund in self.funds.items():
            facts = self.facts.get(isin, {})
            for field, label in gen["fields"].items():
                entry = facts.get(field)
                if isinstance(entry, dict) and entry.get("value") is None:
                    out.append({
                        "id": f"{gen['id_prefix']}_{isin}_{field}",
                        "category": gen["category"],
                        "question": gen["template"].format(field_label=label, fund_name=fund["name"]),
                        "expected": gen["expected"],
                        "tags": ["generated"],
                    })
        return out


@dataclass
class Golden:
    version: int
    funds: dict[str, dict]  # ISIN -> {name, group}
    questions: list[dict]  # each with expected_value and funds_in_scope (ISINs) added
    problems: list[str]


def resolve_golden(spec_path: Path = GOLDEN, facts_dir: Path = FACTS) -> Golden:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    facts = {i: f.model_dump(mode="json") for i, f in load_verified_facts(facts_dir).items()}
    r = Resolver(spec, facts)

    ids = [q["id"] for q in spec["questions"]]
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        r.problems.append(f"duplicate id {d}")

    questions = list(spec["questions"])
    for gen in spec.get("generated", []):
        questions += r.generated(gen)

    resolved = []
    for q in questions:
        q = dict(q)
        q["expected_value"] = r.resolve(q)
        q["funds_in_scope"] = r.isins(q["expected"].get("isins"))  # "all" / "group:x" spelled out for the judge
        resolved.append(q)
    return Golden(spec["version"], spec["funds"], resolved, list(dict.fromkeys(r.problems)))


def main() -> None:
    if len(sys.argv) not in (1, 3):
        sys.exit(__doc__)
    spec_path, facts_dir = (Path(a) for a in sys.argv[1:]) if len(sys.argv) == 3 else (GOLDEN, FACTS)
    g = resolve_golden(spec_path, facts_dir)

    out = spec_path.with_suffix(".resolved.json")
    out.write_text(json.dumps({"version": g.version, "funds": g.funds, "questions": g.questions},
                              indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    by_cat: dict[str, int] = {}
    for q in g.questions:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    print("questions:", len(g.questions), by_cat)
    print("wrote", out)
    if g.problems:
        print(f"\n{len(g.problems)} problem(s):")
        for p in g.problems:
            print("  -", p)
        sys.exit(1)
    print("all expected answers resolved")


if __name__ == "__main__":
    main()
