"""
Text -> vector (embedding).

An embedding model maps a text to a fixed-length list of numbers (here 384)
such that texts with similar MEANING get vectors pointing in similar
directions. "What are the ongoing charges?" lands near a chunk containing
"Management fees and other administrative or operating costs 0.20%", even
though they share almost no words. That's what keyword search can't do.

Models run locally via fastembed (ONNX, CPU). Default: BAAI/bge-small-en-v1.5
(~67 MB); larger ones are listed in MODELS and chosen with EMBEDDING_MODEL.

Two details that matter:
  - Max 512 tokens. Anything longer is silently cut off, so the chunker
    measures chunks with count_tokens() from this module.
  - Asymmetric retrieval: BGE was trained to embed short QUESTIONS with an
    instruction prefix and long PASSAGES without one. fastembed doesn't add
    the prefix for us, so embed_query() does.
"""

import hashlib
import json
import os
from functools import cache, lru_cache

from fastembed import TextEmbedding
from tokenizers import Tokenizer

from etf_assistant.paths import EMBEDDING_CACHE

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# model -> (vector length, prefix for queries). All of these share BERT's
# tokenizer and 512-token limit, so the chunks are identical for every model
# and a comparison changes only the model.
MODELS = {
    "BAAI/bge-small-en-v1.5": (384, BGE_QUERY_PREFIX),  # 33M parameters
    "BAAI/bge-base-en-v1.5": (768, BGE_QUERY_PREFIX),  # 110M
    "BAAI/bge-large-en-v1.5": (1024, BGE_QUERY_PREFIX),  # 335M
    "mixedbread-ai/mxbai-embed-large-v1": (1024, BGE_QUERY_PREFIX),  # 335M, trained with the same prompt
}

# Pick the model with an environment variable, e.g.
#   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 uv run index
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
if MODEL_NAME not in MODELS:
    raise SystemExit(f"EMBEDDING_MODEL={MODEL_NAME!r} not configured; choose from {list(MODELS)}")
DIM, QUERY_PREFIX = MODELS[MODEL_NAME]  # the OpenSearch index must be created with this DIM
MAX_TOKENS = 512


@cache  # load the model once per process (first run downloads it to a cache)
def _model() -> TextEmbedding:
    return TextEmbedding(MODEL_NAME)


@cache
def _counting_tokenizer() -> Tokenizer:
    # an independent copy of the model's tokenizer, with truncation and padding off,
    # so it reports true lengths (the original stays as the model needs it)
    tok = Tokenizer.from_str(_model().model.tokenizer.to_str())
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(text: str) -> int:
    """Number of tokens the embedding model sees for `text` (incl. 2 special tokens)."""
    return len(_counting_tokenizer().encode(text).ids)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Vectors for `texts`. Cached on disk by text hash: only new or changed chunks are embedded."""
    path = EMBEDDING_CACHE / f"{MODEL_NAME.replace('/', '__')}.json"
    cached: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cached]
    if todo:
        # small batches: at the default 256, 512-token passages need over 1 GB of memory
        vectors = _model().passage_embed([t for _, t in todo], batch_size=32)
        cached.update({k: v.tolist() for (k, _), v in zip(todo, vectors)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached))
    return [cached[k] for k in keys]


@lru_cache(maxsize=4096)  # evals send the same question once per search mode
def embed_query(query: str) -> list[float]:
    return next(iter(_model().query_embed(QUERY_PREFIX + query))).tolist()
