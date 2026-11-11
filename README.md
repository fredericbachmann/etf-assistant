# ETF Assistant

A chatbot that answers questions about ETFs from their official documents
(KIDs and factsheets) using retrieval-augmented generation (RAG), plus an
evaluation pipeline that scores its answers against a hand-verified answer key.

## Setup

```bash
uv sync                      # create .venv and install dependencies from uv.lock
```

The PDFs are not in git. Download them from the `source_url` column of
`manifest.csv` into `data/`, named as in the `file_path` column.

## Data and evaluation

| Path | What it is |
|---|---|
| `manifest.csv` | One row per PDF: ISIN, document type, as-of date, source URL |
| `data/facts/<ISIN>.json` | Verified facts per fund; every value cites document, page and quote |
| `evals/golden_v1.yaml` | Test questions; expected answers are resolved from the facts files |

```bash
uv run check-facts               # check every citation in the facts files against the PDFs
uv run check-facts IE00B4L5Y983  # same for one fund, plus a hand-verification checklist
uv run check-golden              # resolve expected answers -> evals/golden_v1.resolved.json
```
