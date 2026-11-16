# ETF Assistant

A chatbot that answers questions about exchange-traded funds (ETFs) from the funds'
own official documents, and cites the page each figure comes from. It comes with an
evaluation pipeline that grades its answers against a hand-verified answer key, so
every design choice in it is backed by a measurement.

**Live demo: <https://etf-assistant.duckdns.org/>** (login required; credentials are
sent on request)

The library holds 11 European (UCITS) ETFs from iShares, Vanguard, Invesco and
Xtrackers. Each fund has two documents:

- the **KID** (Key Information Document), the standard EU disclosure with the risk
  indicator, the costs and the recommended holding period
- the **factsheet**, the issuer's monthly summary with the TER (total expense ratio,
  the annual fee), fund size, holdings and performance

The assistant is a tool-using agent on Claude Haiku 5.5 with hybrid search (keyword
plus vector) in OpenSearch. It runs as a command-line tool and as a web app, deployed
on AWS on one small EC2 instance.

## What it does

A question from the test set, with the tool calls the agent made and its answer
(the notes below the table are left out here):

> **Same €200 a month for 10 years: how do total fees compare between the Invesco S&P 500 and CSP1?**

```
-> read_fund_documents(isin='IE00B3YCGJ38', doc_type='both')
-> read_fund_documents(isin='IE00B5BMR087', doc_type='both')
-> calculate_fee_cost(rate_pct=0.12, amount=200, years=10, mode='monthly')
-> calculate_fee_cost(rate_pct=0.07, amount=200, years=10, mode='monthly')
```

> **Invesco S&P 500 (SPXS) costs more: about €145 in fees over 10 years vs. about €85 for CSP1.**
> This assumes €200 invested at the start of each month for 10 years (€24,000 total) and zero market return.
>
> | | Invesco S&P 500 UCITS ETF Acc (IE00B3YCGJ38) | iShares Core S&P 500 UCITS ETF USD Acc, CSP1 (IE00B5BMR087) |
> |---|---|---|
> | Total expense ratio | 0.05% p.a. [IE00B3YCGJ38 factsheet p.1, as of 30 Sep 2026] | 0.07% p.a. [IE00B5BMR087 factsheet p.1, as of 30 Sep 2026] |
> | Swap fee | 0.07% p.a. [IE00B3YCGJ38 factsheet p.1] | None listed; the fund is physically replicated [IE00B5BMR087 factsheet p.1] |
> | All-in rate used | 0.12% (0.05% + 0.07%) | 0.07% |
> | Fees paid over 10 years | €144.63 | €84.50 |
> | Value at end, before market returns | €23,855.37 | €23,915.50 |

This question is a trap: the Invesco fund has the lower headline fee, but it is a
synthetic fund that also pays a swap fee, which its factsheet describes as "not
charged separately". The assistant recognises the ticker CSP1, reads both funds'
documents, adds the swap fee, and leaves the compounding to a calculator tool. In the
web app, each citation is a link that opens the PDF at the cited page.

The assistant also knows what not to answer:

- **Investment advice.** "Should I buy VWCE or EUNL?" gets a factual comparison of
  the two funds and no recommendation.
- **Questions the documents don't cover.** Current prices, future returns, taxes and
  funds outside the library get "the documents don't cover this", not an answer from
  the model's general knowledge.

## Results

From the latest full run, stored in [evals/results/](evals/results/): the 79 questions
of 16 November 2026 (`e2e_2026-11-16T09-53-59`), answered by Claude Haiku 5.5 and graded
by a Claude Haiku 5.5 judge that also checks each answer against the documents.

| | |
|---|---|
| End-to-end test questions passed | 74 of 79 |
| Passed by category | facts 21/22, comparisons 11/11, fee calculations 3/3, open questions 13/14, details beyond the answer key 5/5, advice handling 12/14, not in the documents 9/10 |
| Failures | 1 misses a requirement of the question, 4 meet it but state something the documents contradict |
| Latency per question | 4.3 s median, 13.0 s maximum |
| Tool calls per question | 1.2 on average |
| Claude cost per question | $0.0018 |
| Cost of one full evaluation run | $0.28 (agent $0.14, judge $0.13) |
| Retrieval: evidence in the top 5 passages (hybrid search, filtered to the right fund) | 91% of the hand-written fact questions, 70% of the generated ones |

Three verdicts come from follow-up runs: one question hit an API rate limit and was run
again on its own (`e2e_2026-11-16T09-54-58`, passed), and two answers were graded again
after a rubric that asked for more than its question and a judge rule were corrected
(`e2e_2026-11-16T09-54-49`, one pass, one fail).

The five failures:

- **A figure from the wrong table column.** Asked whether now is a good time to buy
  the S&P 500, the assistant declines correctly, then quotes a fund's return since
  inception (818.71%) as its 10-year return (311.46%). Asked for the 10-year return
  directly, it reads the same table correctly.
- **"The documents don't say" when they do, three times.** After a search, the
  assistant reports that the passages it retrieved lack a figure (number of holdings,
  dividend yield, payout schedule) instead of reading the document, which states it.
- **A weak example.** Asked why KID costs can differ from the TER, it illustrates with
  two funds where they are identical.

Four of these five answers pass without the judge's check against the documents. The
same check, applied to 15 saved answers of the previous run on the 64 questions of
version 1 (63 of 64 with the Haiku judge, 64 of 64 with a Claude Sonnet 5.5 judge, both
grading against the answer key only), fails five of the 15, among them the same wrong
return in two answers (`e2e_2026-11-16T09-50-51`).

In an earlier run on version 1 (`e2e_2026-11-14T16-03-06`, 62 of 64) both failures were comparison
answers whose one-line summary contradicted the correct figures listed below it. Over
repeated runs this happens in about 2 of 40 comparison answers.

## How it works

```mermaid
flowchart LR
    subgraph setup [Data setup, run once]
        PDF[22 PDFs] --> MD[Markdown per page]
        MD --> CH[351 chunks, each with fund, document and page]
        CH --> OS[(OpenSearch: BM25 + vectors)]
        MD --> CAT[Fund catalog: names and tickers]
    end
    subgraph answer [Answering a question]
        Q[Question] --> AG[Agent loop on Claude Haiku]
        AG <--> T1[read_fund_documents]
        AG <--> T2[search_documents]
        AG <--> T3[calculate_fee_cost]
        AG --> A[Answer with page citations]
    end
    MD -.-> T1
    OS -.-> T2
    CAT -.-> AG
```

**Data setup.** Each PDF is converted to Markdown page by page, so tables stay intact
and every passage knows its page. Pages are split into chunks along headings and
paragraphs, never across a page. Each chunk gets a header naming its fund, document
and page before it is embedded: the cost table in a KID never names the fund, so
without the header the cost tables of all 11 funds look the same to search. Chunks are
embedded locally (`BAAI/bge-small-en-v1.5`) and indexed in OpenSearch for keyword
search, vector search and a hybrid of the two.

**The agent.** The agentic loop is written by hand in about 170 lines
([agent/loop.py](src/etf_assistant/agent/loop.py)), without a framework. The system
prompt contains the fund list, so the model itself resolves "VWCE" or "the Invesco
one" to a fund. It has three tools:

| Tool | Use |
|---|---|
| `read_fund_documents` | Full text of one fund's KID and/or factsheet, with page markers. For questions about one or two funds |
| `search_documents` | Hybrid search. With a list of funds: the top passages per fund, to compare one detail across funds. Without: the top passages across the library, for topics |
| `calculate_fee_cost` | Exact fee amounts over time, so the model never does compounding in its head |

**The web app.** A FastAPI server with a Gradio chat page. It streams the answer,
shows each tool call while it runs, and links every citation to the cited PDF page.
It has a login, a daily spending cap shared by all conversations, and a limit on
questions answered at the same time.

**Cost control.** Every Claude call goes through a budget
([llm.py](src/etf_assistant/llm.py)). Before a call is sent, its worst-case cost is
reserved; if that doesn't fit under the cap, the call is never made. The cap therefore
holds even with several calls in flight. The evals and the web server both use it.

## How it is evaluated

**A verified answer key.** For each fund, [data/facts/](data/facts/) holds about 16
facts (TER, risk indicator, replication method, and so on). Every value carries its
document, page and a verbatim quote. `check-facts` confirms mechanically that each
quote is on that page and that the value can be read from it. Each file was then
checked by hand against the PDF and marked verified. Evals read only verified files.

**A test set that can't disagree with the answer key.** The 79 questions in
[evals/golden_v2.yaml](evals/golden_v2.yaml) don't contain expected answers. They
contain references ("the TER of this fund", "the fund with the lowest risk indicator
among these") that are resolved from the facts files when an eval starts. The exception
is five questions about figures the facts files don't cover; their expected values are
written in the test set with the page they were read from.

| Category | Questions | Example |
|---|---|---|
| Facts | 22 | "What's the risk score of EIMI?" |
| Comparisons | 11 | "Rank all the funds you cover by TER, cheapest first." |
| Fee calculations | 3 | "I put €200 a month into VWCE for 10 years. Ignoring returns, roughly how much goes to fees?" |
| Open questions | 14 | "Why does the hedged MSCI World ETF cost more, and what does the hedging actually do?", "Why is VWCE's TER 0.22%?" (false premise: it is 0.14%) |
| Details beyond the answer key | 5 | "How much did the Invesco S&P 500 ETF return over the last 10 years, compared with its index?" |
| Advice handling | 14 | "Just tell me which one you'd pick. I won't hold you to it." (must decline), "Which of these funds has the lowest TER?" (must answer) |
| Not in the documents | 10 | "How is VWRL taxed in Austria?" |

Version 2 adds 15 questions to the 64 of
[version 1](evals/golden_v1.yaml), which every answer passed with the stronger judge.
Reading those 64 answers against the documents showed what version 1 didn't test:
answers found beyond page 1 of a document, questions with a false premise, a fund name
that fits two share classes, and comparisons that need every fund's KID. No version 1
question was reworded or removed, so older runs stay comparable on those 64.

**Retrieval is measured first, without an LLM.** `eval-retrieval` checks whether
search puts the passage containing the verified quote in the top results (Recall@k and
MRR), for keyword, vector and hybrid search, with and without a fund filter. It runs
on the hand-written fact questions and on about 170 questions generated from the
facts. It is free and exactly reproducible.

**End-to-end answers are graded by an LLM judge.** `eval-e2e` runs every question
through a fresh agent. A judge model receives the exact expected result and must write
its reasoning before its verdict. The judge also reads the documents of the funds an
answer is about and lists every statement they contradict: a figure from the wrong
table column, or "the documents don't give this" when they do. An answer passes only
if it meets the expectation and that list is empty. Without this check, an answer that
correctly declines to give advice passes even when it quotes a wrong return on the
way. Saved answers can be graded again after a change to
the judge (`--rejudge`), and questions can be repeated to find the ones that pass only
some of the time (`--repeat`). The judge is Claude Haiku 5.5 by default;
`--judge-model` grades with Claude Sonnet 5.5 instead, and together with `--rejudge`
it shows how much a pass rate depends on the judge. Each results file records the
agent and judge model.

**The assistant can't see the answer key.** Nothing outside the `evals` package
imports it, and a test enforces that
([tests/test_layers.py](tests/test_layers.py)). Fund names and tickers come from a
separate extraction of the documents, never from the facts files.

## Design decisions

Each decision is backed by an eval result in [evals/results/](evals/results/).

**The fund list lives in the system prompt, not behind a lookup tool.** With
`find_fund` and `list_funds` tools, 72 of 154 tool calls in a full eval run (47%) were
fund lookups, and 60 of 64 questions started with one. Matching a ticker or nickname
to a fund is what language models do well; the cost was the extra round-trip. For 11
funds the list is about 400 tokens. With hundreds of funds it no longer fits, and
lookup moves into a search tool over a fund index.

**Questions about one or two funds read the whole documents instead of retrieved
chunks.** A KID plus factsheet fits easily in the context window, and reading it is
near-complete. The best retrieval setting (hybrid, filtered to the right fund) finds
the evidence in the top 5 chunks for 91% of the hand-written questions but only 70% of
the generated ones.

**Comparisons across funds use one filtered search per fund.** Without it, the agent
read all 11 funds' documents for "which fund has the lowest risk indicator?" (13 tool
calls). Fund-filtered search is reliable for KID fields (costs, risk indicator, holding
period: found in the top 5 for 9 to 11 of 11 funds) and weak for factsheet key facts
(number of holdings, index name, domicile, fund size: 3 to 5 of 9 to 11). The agent is
therefore told to read a fund's documents when its passages miss the answer.
`uv run eval-retrieval` prints this per-field table.

**Hybrid search (BM25 plus vectors) with synonyms.** Hybrid has the best recall on
both question sets:

| Evidence found | BM25 | Vector | Hybrid |
|---|---|---|---|
| Hand-written questions, top 5 | 0.57 | 0.65 | 0.74 |
| Generated questions, top 10 | 0.71 | 0.55 | 0.80 |

On the generated questions, BM25 alone ranks slightly better at the very top (MRR 0.39
against 0.36). The synonym list maps issuers' different names for the same figure (TER,
ongoing charges figure, all-in fee).

**Effect of the first and third decision** (full end-to-end runs of 64 questions,
`e2e_2026-11-14T15-03-01` before and `e2e_2026-11-14T16-03-06` after):

| | Lookup tools, whole documents for comparisons | Fund list in prompt, per-fund search |
|---|---|---|
| Passed | 63/64 | 62/64 |
| Tool calls per question | 3.2 | 1.2 |
| Latency, median / maximum | 5.5 s / 16.3 s | 4.1 s / 13.7 s |
| Agent cost per question | $0.0024 | $0.0017 |
| Comparisons: tool calls per question | 8.1 | 1.5 |

The two failures after the change are the summary contradictions described under
Results (the earlier run); both questions passed in a separate run minutes earlier. The style rule in the
prompt now asks for a headline "worked out from the figures you cite". Three runs
before and three after that rewording all passed 24/24, so its effect is too small to
measure at this sample size. The rule stays because it costs nothing.

## Tech stack

| | |
|---|---|
| Language and tooling | Python 3.13, uv, pytest |
| Model | Claude Haiku 5.5 through the Anthropic SDK (streaming, tool use, prompt caching, structured output) |
| PDF parsing | pymupdf4llm |
| Embeddings | BAAI/bge-small-en-v1.5 through fastembed, running locally on CPU |
| Search | OpenSearch 3 (BM25, k-NN vectors, hybrid) |
| Web | FastAPI, Gradio |
| Deployment | Docker Compose with Caddy on one EC2 instance, CloudFormation, GitHub Actions |

## Project layout

```
src/etf_assistant/
  paths.py, llm.py   shared: file locations; Claude client, cost accounting, budget cap
  corpus/            the documents: manifest, PDF -> Markdown, chunking, fund catalog
  retrieval/         embeddings and the OpenSearch index (BM25, vector, hybrid)
  pipeline.py        data setup: ingest (PDFs -> chunks) and index (chunks -> OpenSearch)
  agent/             the assistant: system prompt, tools, agentic loop, CLI
  web/               the chat page: FastAPI + Gradio over the agent, citations linked to PDF pages
  evals/             answer key schema and checks, golden set, retrieval and end-to-end evals
tests/               offline tests, including the layering rule below
deploy/              CloudFormation stack, production Compose file, deploy scripts
```

Each layer imports only from the ones above it in this list, and nothing outside
`evals/` imports `evals/`. `tests/test_layers.py` enforces this.

| Path | What it is |
|---|---|
| `manifest.csv` | One row per PDF: ISIN, document type, as-of date, source URL |
| `data/catalog.json` | Fund names, share classes and tickers, extracted from the documents |
| `data/facts/<ISIN>.json` | Verified facts per fund; every value cites document, page and quote |
| `evals/golden_v2.yaml` | Test questions; expected answers are resolved from the facts files (`golden_v1.yaml`: the earlier set) |
| `evals/results/` | Saved eval runs, so runs can be compared over time |

## Running it locally

You need [uv](https://docs.astral.sh/uv/), Docker, and access to the Claude API.

```bash
uv sync                      # create .venv and install dependencies from uv.lock
```

**Documents.** The PDFs are the issuers' and are not in git. Download them from the
`source_url` column of `manifest.csv` into `data/`, named as in the `file_path` column.

**Claude credentials.** The Anthropic SDK finds them itself: either set
`ANTHROPIC_API_KEY`, or log in once with the `ant` CLI
(`brew install anthropics/tap/ant && ant auth login`).

**Build the index.**

```bash
uv run ingest                   # PDFs -> Markdown (cached in data/parsed/) -> data/chunks.jsonl
docker compose up -d            # start OpenSearch on localhost:9200
uv run index                    # embed the chunks and index them
uv run search "What's the risk score of EIMI?"                      # hybrid (BM25 + vector)
uv run search "swap fee" --mode bm25 --isin IE00B3YCGJ38 -k 3      # keyword only, one fund
```

`data/catalog.json` is in git. To rebuild it after the documents change, run
`uv run build-catalog` (one Claude call per fund).

**Ask questions.**

```bash
uv run ask "What's the ongoing charge of VWCE?"        # one question, shows the tool calls
uv run chat                                            # conversation with follow-up questions
uv run web                                             # the chat page on http://127.0.0.1:7860
```

**Run the tests and evals.**

```bash
uv run pytest                    # offline tests: no OpenSearch, no Claude calls

uv run check-facts               # check every citation in the facts files against the PDFs
uv run check-facts IE00B4L5Y983  # same for one fund, plus a hand-verification checklist
uv run check-golden              # resolve expected answers and list problems
uv run check-chunks              # are the verified facts intact inside one chunk? (upper bound for retrieval)

uv run eval-retrieval                 # Recall@k / MRR for BM25, vector, hybrid, with and without fund filter
uv run eval-retrieval --show-misses   # plus the golden questions whose evidence isn't in the top 5
uv run eval-e2e                       # all golden questions through the agent, graded by the judge (cap $0.50)
uv run eval-e2e --only f01,c03 --max-cost 0.05
uv run eval-e2e --tag consistency --repeat 5 --max-cost 0.30   # pass rate per question over 5 runs
uv run eval-e2e --rejudge evals/results/e2e_<time>.json        # grade saved answers again after changing the judge
uv run eval-e2e --rejudge evals/results/e2e_<time>.json --judge-model claude-sonnet-5-5 --max-cost 3   # second opinion from a stronger judge
```

## Web app settings

Settings are environment variables:

| Variable | Default | |
|---|---|---|
| `APP_PASSWORD` | unset | Login password (user name `APP_USERNAME`, default `etf`); unset means no login, so set it on any public server |
| `DAILY_BUDGET_USD` | `2.0` | Claude spending of all conversations together per day (UTC); then questions are refused until the next day |
| `CONCURRENCY` | `4` | Questions answered at the same time; more wait in a queue |
| `MAX_QUESTION_CHARS` | `1000` | Longest question accepted |
| `HOST`, `PORT` | `127.0.0.1`, `7860` | Where the server listens; `0.0.0.0` in a container |

`GET /health` returns 503 when search isn't working, for uptime checks.
`/documents/<file>` serves only the PDFs listed in `manifest.csv`, never the rest of `data/`.

## Deployment (AWS)

One EC2 instance (t4g.small, ARM) runs `deploy/docker-compose.prod.yml`: Caddy for
HTTPS, the web app, and OpenSearch without the plugins the app doesn't use. Everything
in AWS is one CloudFormation stack, `deploy/aws/stack.yaml`, named `etf-assistant`.

| In AWS | For |
|---|---|
| EC2 instance + Elastic IP | the server; ports 80 and 443 only, shell access through Session Manager |
| ECR repositories `etf-assistant/app`, `etf-assistant/opensearch` | the container images |
| S3 bucket | `data.tar.gz` (the documents) and the deploy files |
| Parameter Store `/etf-assistant/*` | the app's secrets: `APP_PASSWORD`, `ANTHROPIC_API_KEY`; optional settings such as `DAILY_BUDGET_USD` |

```bash
export AWS_PROFILE=etf
deploy/push-data.sh       # after the PDFs changed (and ingest + index locally): upload the documents
deploy/deploy.sh          # build and push the images of the current commit, update the server
```

`deploy.sh` runs `deploy/remote.sh` on the server through Systems Manager. That script
fetches the documents and secrets, pulls the images, rebuilds the search index and waits
for `/health`. To roll back, check out an earlier commit and run `deploy.sh` again.

```bash
# set or change a secret, then deploy again to apply it
aws ssm put-parameter --name /etf-assistant/ANTHROPIC_API_KEY --type SecureString --overwrite --value ...
# read the login password (user name: etf)
aws ssm get-parameter --name /etf-assistant/APP_PASSWORD --with-decryption --query Parameter.Value --output text
# a shell on the server (needs the Session Manager plugin); the app is in /opt/etf-assistant
aws ssm start-session --target $(aws cloudformation describe-stacks --stack-name etf-assistant \
    --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
```

Every push to any branch runs the tests (`.github/workflows/ci.yml`). Pushes to `main`
also deploy through `.github/workflows/deploy.yml`, which runs the same `deploy.sh` as
the stack's `etf-assistant-github-deploy` user. Its access key is stored as the
repository secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. To replace the key:

```bash
aws iam create-access-key --user-name etf-assistant-github-deploy   # then: gh secret set ... for both values
aws iam delete-access-key --user-name etf-assistant-github-deploy --access-key-id <the old one>
```

To remove everything: delete that user's access key, run `aws s3 rm s3://<bucket> --recursive`, then
`aws cloudformation delete-stack --stack-name etf-assistant`, then delete the parameters
under `/etf-assistant/`.

## Disclaimer

A learning and portfolio project. Nothing the assistant says is investment advice.
The fund documents belong to their issuers and are not redistributed in this repository.
