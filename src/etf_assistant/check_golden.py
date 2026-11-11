"""
Resolve the golden test set against your verified facts and report problems.

Usage:
    uv run check-golden                                  # defaults below
    uv run check-golden evals/golden_v1.yaml data/facts/

Writes evals/golden_v1.resolved.json: every question with its concrete expected
answer filled in. The eval runner reads that file, never the YAML directly, so
re-run this script whenever a facts file changes.

Exits non-zero if anything can't be resolved (missing or unverified facts file,
null value where a fact is expected, duplicate ids/aliases).
"""

import json
import re
import sys
from pathlib import Path

import yaml

from etf_assistant.paths import EVALS, FACTS

PLACEHOLDER = re.compile(r"\{(IE[0-9A-Z]{10})\.(\w+)\}")


def load_facts(facts_dir: Path) -> dict:
    facts = {}
    for f in sorted(facts_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        facts[d["isin"]] = d
    return facts


class Resolver:
    def __init__(self, spec: dict, facts: dict):
        self.funds = spec["funds"]
        self.facts = facts
        self.problems: list[str] = []

    # --- helpers -------------------------------------------------------
    def isins(self, sel) -> list[str]:
        if sel == "all":
            return list(self.funds)
        if isinstance(sel, str) and sel.startswith("group:"):
            g = sel.split(":", 1)[1]
            return [i for i, f in self.funds.items() if f.get("group") == g]
        return list(sel)

    def raw(self, qid: str, isin: str, field: str):
        fund = self.facts.get(isin)
        if fund is None:
            self.problems.append(f"{qid}: no facts file for {isin}")
            return None
        if not fund.get("verified"):
            self.problems.append(f"{qid}: facts for {isin} not verified")
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


def main() -> None:
    if len(sys.argv) not in (1, 3):
        sys.exit(__doc__)
    spec_path, facts_dir = (Path(a) for a in sys.argv[1:]) if len(sys.argv) == 3 else (EVALS / "golden_v1.yaml", FACTS)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    r = Resolver(spec, load_facts(facts_dir))

    # structural checks
    ids = [q["id"] for q in spec["questions"]]
    for d in {i for i in ids if ids.count(i) > 1}:
        r.problems.append(f"duplicate id {d}")
    seen = {}
    for isin, f in spec["funds"].items():
        for a in f.get("aliases", []):
            if a.lower() in seen:
                r.problems.append(f"alias '{a}' used by {seen[a.lower()]} and {isin}")
            seen[a.lower()] = isin

    questions = list(spec["questions"])
    for gen in spec.get("generated", []):
        questions += r.generated(gen)

    resolved = []
    for q in questions:
        q = dict(q)
        q["expected_value"] = r.resolve(q)
        resolved.append(q)

    out = spec_path.with_suffix(".resolved.json")
    out.write_text(json.dumps({"version": spec["version"], "funds": spec["funds"], "questions": resolved},
                              indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    by_cat: dict[str, int] = {}
    for q in resolved:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    print("questions:", len(resolved), by_cat)
    print("wrote", out)
    if r.problems:
        print(f"\n{len(r.problems)} problem(s):")
        for p in dict.fromkeys(r.problems):
            print("  -", p)
        sys.exit(1)
    print("all expected answers resolved")


if __name__ == "__main__":
    main()