"""The package's layering rule, checked from the source: each layer imports only from the ones it may.

Most important: nothing outside evals/ imports evals/, so the system under test
can never read the answer key (data/facts) it is graded against.
"""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "etf_assistant"

# layer -> layers it must NOT import
FORBIDDEN = {
    "paths": {"corpus", "retrieval", "pipeline", "agent", "evals"},
    "llm": {"corpus", "retrieval", "pipeline", "agent", "evals"},
    "corpus": {"retrieval", "pipeline", "agent", "evals"},
    "retrieval": {"pipeline", "agent", "evals"},
    "pipeline": {"agent", "evals"},
    "agent": {"pipeline", "evals"},
    "evals": set(),
}


def layer_of(path: Path) -> str:
    rel = path.relative_to(PKG)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def imported_layers(path: Path) -> set[str]:
    layers = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        for n in names:
            parts = n.split(".")
            if parts[0] == "etf_assistant" and len(parts) > 1:
                layers.add(parts[1])
    return layers


MODULES = [p for p in sorted(PKG.rglob("*.py")) if p.name != "__init__.py" or p.parent != PKG]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PKG)))
def test_module_respects_layers(path: Path):
    layer = layer_of(path)
    assert layer in FORBIDDEN, f"{path.relative_to(PKG)}: unknown layer {layer!r}, add it to FORBIDDEN"
    bad = imported_layers(path) & FORBIDDEN[layer]
    assert not bad, f"{path.relative_to(PKG)} ({layer}) imports {sorted(bad)}"
