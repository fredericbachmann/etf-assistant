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

The system prompt contains the fund library (ISIN, name, share class, tickers from
`data/catalog.json`), so the model resolves "VWCE" to an ISIN itself. It has three tools:

| Tool | Use |
|---|---|
| `read_fund_documents` | Full text of one fund's KID and/or factsheet, with page markers: questions about one or two funds |
| `search_documents` | Hybrid search. With `isins`: the top passages *per fund*, to compare one detail across funds. Without: top passages across the library, for topics |
| `calculate_fee_cost` | Exact fee amounts over time, so the model never does compounding in its head |

## Evaluation

```bash
uv run eval-retrieval                 # Recall@k / MRR for BM25, vector, hybrid (with and without ISIN filter)
uv run eval-retrieval --show-misses   # plus the golden questions whose evidence isn't in the top 5
uv run eval-e2e                       # all golden questions through the agent, graded by an LLM judge (cap $0.50)
uv run eval-e2e --only f01,c03 --max-cost 0.05
uv run eval-e2e --rejudge evals/results/e2e_<time>.json   # grade saved answers again after changing the judge
```

Results are written to `evals/results/` so runs can be compared over time.

## Design decisions

Each decision is backed by an eval result in `evals/results/`.

Effect of the last two decisions below (full end-to-end run, 64 questions,
`e2e_2026-11-14T15-03-01` before, `e2e_2026-11-14T16-03-06` after):

| | lookup tools, documents for comparisons | fund list in prompt, per-fund search |
|---|---|---|
| passed | 63/64 | 62/64 |
| tool calls per question | 3.2 | 1.2 |
| latency, median / max | 5.5 s / 16.3 s | 4.1 s / 13.7 s |
| agent cost per question | $0.0024 | $0.0017 |
| comparisons: tool calls per question | 8.1 | 1.5 |

The two failures after the change are headline/body contradictions (correct figures,
wrong one-line summary); both questions passed in a separate run minutes earlier.

**The fund list lives in the system prompt, not behind a lookup tool.** With
`find_fund`/`list_funds` tools, 72 of 154 tool calls in a full eval run (47%)
were fund lookups, and 60 of 64 questions started with one. Matching a ticker
or nickname to a fund is what language models do well; the cost was the extra
round-trip. For 11 funds the list is about 400 tokens. With hundreds of funds
it no longer fits, and lookup moves into a search tool over a fund index.

**Questions about one or two funds read the whole documents instead of retrieved
chunks.** A KID plus factsheet fits easily in the context window, and reading it is
near-complete, while the best retrieval setting (hybrid, filtered to the right fund)
finds the evidence in the top 5 chunks for 95% of the golden questions but only 70%
of the templated ones.

**Comparisons across funds use one filtered search per fund.** Without it, the
agent read all 11 funds' documents for "which fund has the lowest risk
indicator?" (13 tool calls). Fund-filtered search is reliable for KID fields
(costs, risk indicator, holding period: 9-11 of 11 in the top 5) and weak for
factsheet key facts (number of holdings, index name, domicile, fund size: 3-5 of
9-11), so the agent is told to read a fund's documents when its passages miss
the answer. `uv run eval-retrieval` prints this per-field table.

**Hybrid search (BM25 + vectors) with synonyms.** Hybrid has the best recall on
both query sets (golden R@5 0.76 vs 0.62 BM25 and 0.67 vector; templated R@10 0.80
vs 0.71 and 0.55); on templated questions BM25 alone ranks slightly better at the
very top (MRR 0.39 vs 0.36). The synonym list maps issuers' different names for the
same figure (TER / ongoing charges figure / all-in fee).

**The answer key is out of the assistant's reach.** Fund names come from an LLM
extraction of the documents themselves (`build-catalog`, tickers verified against
the text), never from `data/facts`, and `tests/test_layers.py` checks that no
module outside `evals/` imports the evals package.

