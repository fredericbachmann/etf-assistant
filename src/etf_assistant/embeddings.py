"""
Text -> vector (embedding).

An embedding model maps a text to a fixed-length list of numbers (here 384)
such that texts with similar MEANING get vectors pointing in similar
directions. "What are the ongoing charges?" lands near a chunk containing
"Management fees and other administrative or operating costs 0.20%", even
though they share almost no words. That's what keyword search can't do.

Model: BAAI/bge-small-en-v1.5, run locally via fastembed (ONNX, CPU, ~67 MB).
Free, fast, and a well-known baseline; a bigger model is an experiment for the
evals to judge later.

Two details that matter:
  - Max 512 tokens. Anything longer is silently cut off, so the chunker
    measures chunks with count_tokens() from this module.
  - Asymmetric retrieval: BGE was trained to embed short QUESTIONS with an
    instruction prefix and long PASSAGES without one. fastembed doesn't add
    the prefix for us, so embed_query() does.
"""

from functools import cache

from fastembed import TextEmbedding
from tokenizers import Tokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384  # vector length; the OpenSearch index must be created with the same number
MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
    return [v.tolist() for v in _model().passage_embed(texts)]


def embed_query(query: str) -> list[float]:
    return next(iter(_model().query_embed(QUERY_PREFIX + query))).tolist()
