# Development commands

The full command reference. The [README](../README.md#running-it-locally) covers setup
and the commands most people need.

## Data and search

```bash
uv run ingest                   # PDFs -> Markdown (cached in data/parsed/) -> data/chunks.jsonl
docker compose up -d            # start OpenSearch on localhost:9200
uv run index                    # embed the chunks and index them
uv run build-catalog            # rebuild data/catalog.json after the documents change (one Claude call per fund)

uv run search "What's the risk score of EIMI?"                      # hybrid (BM25 + vector)
uv run search "swap fee" --mode bm25 --isin IE00B3YCGJ38 -k 3      # keyword only, one fund
```

## Asking questions

```bash
uv run ask "What's the ongoing charge of VWCE?"        # one question, shows the tool calls
uv run chat                                            # conversation with follow-up questions
uv run web                                             # the chat page on http://127.0.0.1:7860
```

## Checks on the answer key and test set

```bash
uv run check-facts               # check every citation in the facts files against the PDFs
uv run check-facts IE00B4L5Y983  # same for one fund, plus a hand-verification checklist
uv run check-golden              # resolve expected answers and list problems
uv run check-chunks              # are the verified facts intact inside one chunk? (upper bound for retrieval)
```

## Evals

```bash
uv run eval-retrieval                 # Recall@k / MRR for BM25, vector, hybrid, with and without fund filter
uv run eval-retrieval --show-misses   # plus the golden questions whose evidence isn't in the top 5

uv run eval-e2e                       # all golden questions through the agent, graded by the judge (cap $0.50)
uv run eval-e2e --only f01,c03 --max-cost 0.05
uv run eval-e2e --tag consistency --repeat 5 --max-cost 0.30   # pass rate per question over 5 runs
uv run eval-e2e --rejudge evals/results/e2e_<time>.json        # grade saved answers again after changing the judge
uv run eval-e2e --rejudge evals/results/e2e_<time>.json --judge-model claude-sonnet-5-5 --max-cost 3   # second opinion from a stronger judge
```
