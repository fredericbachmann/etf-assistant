# ETF Assistant

A chatbot that answers questions about ETFs from their official documents
(KIDs and factsheets) using retrieval-augmented generation (RAG), plus an
evaluation pipeline that scores its answers against a hand-verified answer key.

## Project layout

```
src/etf_assistant/
  paths.py, llm.py   shared: file locations; Claude client, cost accounting, budget cap
  corpus/            the documents: manifest, PDF -> Markdown, chunking, fund catalog
  retrieval/         embeddings and the OpenSearch index (BM25, vector, hybrid)
  pipeline.py        data setup: ingest (PDFs -> chunks) and index (chunks -> OpenSearch)
  agent/             the assistant: system prompt, tools, agentic loop, CLI
  evals/             answer key schema and checks, golden set, retrieval and end-to-end evals
tests/               offline tests, including the layering rule below
```

Each layer imports only from the ones above it in this list, and nothing outside
`evals/` imports `evals/`: the assistant can never read the answer key it is
graded against. `tests/test_layers.py` enforces this.

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

## RAG pipeline

```bash
uv run ingest                   # PDFs -> Markdown (cached in data/parsed/) -> data/chunks.jsonl
uv run check-chunks             # are the verified facts intact inside one chunk? (upper bound for retrieval)
docker compose up -d            # start OpenSearch on localhost:9200
uv run index                    # embed chunks (BAAI/bge-small-en-v1.5) and index them
uv run search "What's the risk score of EIMI?"                      # hybrid (BM25 + vector)
uv run search "swap fee" --mode bm25 --isin IE00B3YCGJ38 -k 3      # keyword only, one fund
```

## Assistant

Uses Claude Haiku via the Anthropic SDK. Locally, log in once with the `ant` CLI
(`brew install anthropics/tap/ant && ant auth login`); no API key in the project.

```bash
uv run build-catalog                                   # fund names + tickers, extracted from the documents by Haiku
uv run ask "What's the ongoing charge of VWCE?"        # one question, shows the tool calls
uv run chat                                            # conversation with follow-up questions
```

## Evaluation

```bash
uv run eval-retrieval                 # Recall@k / MRR for BM25, vector, hybrid (with and without ISIN filter)
uv run eval-retrieval --show-misses   # plus the golden questions whose evidence isn't in the top 5
uv run eval-e2e                       # all golden questions through the agent, graded by an LLM judge (cap $0.50)
uv run eval-e2e --only f01,c03 --max-cost 0.05
uv run eval-e2e --rejudge evals/results/e2e_<time>.json   # grade saved answers again after changing the judge
```

Results are written to `evals/results/` so runs can be compared over time.
