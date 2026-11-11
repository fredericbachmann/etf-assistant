"""
The OpenSearch index and the three ways to search it.

Everything OpenSearch-specific lives in this file. The rest of the project
only calls index_chunks() and search(), so the store can be swapped later
(e.g. for something cheaper on AWS) without touching anything else.

Search modes:
  bm25    keyword search. Scores chunks by how often the query's words occur
          in them, weighted by how rare those words are across all chunks.
          Strong for exact tokens: ISINs, tickers, "swap fee".
  vector  semantic search. Embeds the query and finds the chunks whose
          vectors are closest (k-nearest neighbours). Strong for paraphrases:
          "how risky is it?" -> the Summary Risk Indicator section.
  hybrid  runs both, rescales each result list to 0..1 (min-max), and averages
          the scores. Aims for the best of both; the evals will tell.

Usage:
  uv run index                                       # (re)build the index from data/chunks.jsonl
  uv run search "what does EIMI cost per year?"      # hybrid, top 5
  uv run search "risk indicator" --mode vector --isin IE00B4L5Y983 -k 3
"""

import argparse
import os
import time
from typing import Literal

from opensearchpy import OpenSearch, helpers
from pydantic import BaseModel

from etf_assistant.chunking import Chunk
from etf_assistant.embeddings import DIM, MODEL_NAME, embed_passages, embed_query
from etf_assistant.paths import CHUNKS

INDEX = "etf-chunks"
HYBRID_PIPELINE = "etf-hybrid"
HYBRID_WEIGHTS = [0.5, 0.5]  # [bm25, vector]; a knob to tune with the evals

Mode = Literal["bm25", "vector", "hybrid"]


def client() -> OpenSearch:
    return OpenSearch(hosts=[os.environ.get("OPENSEARCH_URL", "http://localhost:9200")])


# --------------------------------------------------------------------------- index

# The mapping tells OpenSearch what each field is and how to index it.
INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True,  # enable the k-NN (vector) plugin for this index
            "number_of_shards": 1,  # a few hundred documents: one shard, no replicas
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            # keyword = exact value, for filtering ("only IE00B4L5Y983")
            "chunk_id": {"type": "keyword"},
            "isin": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "page": {"type": "integer"},
            "as_of_date": {"type": "date"},
            "fund_name": {"type": "keyword"},
            "section": {"type": "keyword"},
            # the original chunk text, returned with hits; stored but not searched
            "text": {"type": "text", "index": False},
            # what BM25 searches: header + text. The "english" analyzer lowercases,
            # drops stop words and stems ("charges" -> "charg", "holding" -> "hold").
            "search_text": {"type": "text", "analyzer": "english"},
            # what vector search searches: the embedding of header + text
            "embedding": {
                "type": "knn_vector",
                "dimension": DIM,
                "method": {
                    "name": "hnsw",  # graph index for fast approximate nearest-neighbour search
                    "engine": "lucene",  # supports filtering during the search (by isin, doc_type)
                    "space_type": "cosinesimil",  # compare vector directions, not lengths
                },
            },
        }
    },
}

# A search pipeline post-processes results. Hybrid search needs one: BM25 scores
# (e.g. 3-25) and cosine scores (0-1) are on different scales, so each list is
# rescaled to 0..1 before they're combined.
HYBRID_PIPELINE_BODY = {
    "description": "min-max normalise BM25 and k-NN scores, then take a weighted mean",
    "phase_results_processors": [{
        "normalization-processor": {
            "normalization": {"technique": "min_max"},
            "combination": {"technique": "arithmetic_mean", "parameters": {"weights": HYBRID_WEIGHTS}},
        }
    }],
}


def index_chunks(chunks: list[Chunk], batch_size: int = 64) -> None:
    os_ = client()
    if os_.indices.exists(index=INDEX):
        os_.indices.delete(index=INDEX)  # a full rebuild is simplest at this size
    os_.indices.create(index=INDEX, body=INDEX_BODY)
    os_.transport.perform_request("PUT", f"/_search/pipeline/{HYBRID_PIPELINE}", body=HYBRID_PIPELINE_BODY)

    texts = [c.embedding_text() for c in chunks]
    t = time.time()
    vectors = embed_passages(texts)
    print(f"embedded {len(texts)} chunks with {MODEL_NAME} in {time.time() - t:.1f}s")

    actions = (
        {
            "_index": INDEX,
            "_id": c.chunk_id,
            **c.model_dump(mode="json"),
            "search_text": text,
            "embedding": vec,
        }
        for c, text, vec in zip(chunks, texts, vectors)
    )
    ok, errors = helpers.bulk(os_, actions, chunk_size=batch_size)
    os_.indices.refresh(index=INDEX)  # make the documents searchable now
    print(f"indexed {ok} chunks into '{INDEX}'" + (f", {len(errors)} errors" if errors else ""))


# --------------------------------------------------------------------------- search

class Hit(BaseModel):
    chunk_id: str
    score: float
    isin: str
    fund_name: str
    doc_type: str
    page: int
    section: str
    text: str


def _filters(isin: str | None, doc_type: str | None) -> list[dict]:
    f = []
    if isin:
        f.append({"term": {"isin": isin}})
    if doc_type:
        f.append({"term": {"doc_type": doc_type}})
    return f


def _bm25_query(query: str, filters: list[dict]) -> dict:
    return {"bool": {"must": {"match": {"search_text": query}}, "filter": filters}}


def _knn_query(query: str, k: int, filters: list[dict]) -> dict:
    knn = {"vector": embed_query(query), "k": k}
    if filters:
        knn["filter"] = {"bool": {"filter": filters}}
    return {"knn": {"embedding": knn}}


def search(query: str, k: int = 5, mode: Mode = "hybrid",
           isin: str | None = None, doc_type: str | None = None) -> list[Hit]:
    filters = _filters(isin, doc_type)
    candidates = max(k * 4, 20)  # hybrid combines two candidate lists; look a bit deeper than k
    params = {}
    if mode == "bm25":
        query_body = _bm25_query(query, filters)
    elif mode == "vector":
        query_body = _knn_query(query, k, filters)
    elif mode == "hybrid":
        query_body = {"hybrid": {"queries": [_bm25_query(query, filters), _knn_query(query, candidates, filters)]}}
        params["search_pipeline"] = HYBRID_PIPELINE
    else:
        raise ValueError(f"unknown mode {mode!r}")

    body = {"size": k, "query": query_body, "_source": {"excludes": ["embedding", "search_text"]}}
    res = client().search(index=INDEX, body=body, params=params)
    return [Hit(score=h["_score"], **{f: h["_source"][f] for f in Hit.model_fields if f != "score"})
            for h in res["hits"]["hits"]]


# --------------------------------------------------------------------------- CLI

def index_cli() -> None:
    chunks = [Chunk.model_validate_json(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    index_chunks(chunks)


def search_cli() -> None:
    ap = argparse.ArgumentParser(description="Search the ETF document index")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    ap.add_argument("--isin")
    ap.add_argument("--doc-type", choices=["kid", "factsheet"])
    ap.add_argument("--full", action="store_true", help="print whole chunks, not a preview")
    a = ap.parse_args()

    for i, h in enumerate(search(a.query, a.k, a.mode, a.isin, a.doc_type), 1):
        print(f"{i}. [{h.score:.3f}] {h.chunk_id}  ({h.fund_name}, {h.doc_type} p.{h.page})")
        print(f"   section: {h.section}")
        body = h.text if a.full else h.text[:300].replace("\n", " ") + ("…" if len(h.text) > 300 else "")
        print(f"   {body}\n")
