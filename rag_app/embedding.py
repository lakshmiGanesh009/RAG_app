"""Turn text into vectors so similar meanings end up close together.

Runs locally on CPU, so no API key is needed for this step.

Usage:

    from rag_app.embedding import embed_passages, embed_query

    vectors = embed_passages(["some chunk text", "another chunk"])
    question = embed_query("how much did we spend?")

Documents and questions are embedded differently on purpose. BGE models are
trained for asymmetric search, where a short question has to match a longer
passage, and their authors recommend prefixing the question with a short
instruction. Measured on the sample documents, the prefix left top-1 accuracy
unchanged at 8/8 but widened the gap to the runner-up, most usefully on the
closest call (margin 0.046 with the prefix against 0.030 without).

Note that fastembed's own `query_embed` does not add this prefix, so we add it
here. Prefixes are model-specific; see QUERY_PREFIXES.

The model is about 70 MB and downloads the first time it is used.
"""

from __future__ import annotations

import logging

from fastembed import TextEmbedding

from rag_app.config import Settings, get_settings

log = logging.getLogger(__name__)

# Instruction to put in front of a question, per model. Only questions get one;
# stored passages never do. Models not listed here get no prefix, which is the
# safe default: a prefix meant for a different model makes retrieval worse.
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def embed_passages(
    texts: list[str],
    settings: Settings | None = None,
) -> list[list[float]]:
    """Embed document chunks, ready to be stored.

    Args:
        texts: The chunk texts to embed.
        settings: Overrides the app settings, mainly for tests.

    Returns:
        One vector per text, in the same order.
    """
    if not texts:
        return []
    model = _model(settings or get_settings())
    return [vector.tolist() for vector in model.embed(texts)]


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a question, ready to search with.

    The model's query instruction is added first, if it has one.
    """
    settings = settings or get_settings()
    model = _model(settings)
    prefixed = QUERY_PREFIXES.get(settings.embedding_model, "") + text
    return next(iter(model.embed([prefixed]))).tolist()


def embedding_dimensions(settings: Settings | None = None) -> int:
    """Return the length of the vectors this model produces.

    Worth checking against your store: mixing vector lengths, or changing model
    without re-ingesting, gives silently meaningless search results.
    """
    return _model(settings or get_settings()).embedding_size


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

# Loading a model takes a few seconds, so keep one per model name.
_cache: dict[str, TextEmbedding] = {}


def _model(settings: Settings) -> TextEmbedding:
    """Return the embedding model named in the settings."""
    name = settings.embedding_model
    if name not in _cache:
        log.info("Loading embedding model '%s'", name)
        _cache[name] = TextEmbedding(model_name=name)
    return _cache[name]
