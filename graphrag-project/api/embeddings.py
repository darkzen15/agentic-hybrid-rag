from typing import List, Optional

from config import settings
from retry import with_retry, is_retryable_ollama_error
from ollama_client import get_sync_client

_dim_cache: int = None


def embed_texts(texts: List[str], prefix: Optional[str] = None) -> List[List[float]]:
    """
    `prefix` matters for nomic-embed-text specifically: per Nomic's model
    card, it was trained with task prefixes ("search_document: " for
    indexed passages, "search_query: " for queries) and retrieval quality
    is measurably worse without them. Callers pass the right one via
    settings.EMBEDDING_DOC_PREFIX / EMBEDDING_QUERY_PREFIX — set either to
    "" in .env if you swap to a model that doesn't use this convention.

    Retries transient failures (connection errors, timeouts, 5xx) up to
    OLLAMA_RETRY_ATTEMPTS times with backoff - a single blip shouldn't
    fail an entire large-file ingest job. A 4xx (e.g. the model isn't
    pulled) is never retried since retrying can't fix that.
    """
    if not texts:
        return []
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]

    def _do_request():
        client = get_sync_client()
        resp = client.post(
            f"{settings.OLLAMA_BASE_URL}/api/embed",
            json={"model": settings.EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()

    data = with_retry(
        _do_request,
        attempts=settings.OLLAMA_RETRY_ATTEMPTS,
        base_delay=settings.OLLAMA_RETRY_BASE_DELAY,
        retryable=is_retryable_ollama_error,
        label="embed_texts",
    )
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(
            f"Ollama returned no embeddings for model '{settings.EMBEDDING_MODEL}'. "
            f"Has it been pulled? (docker compose exec ollama ollama pull "
            f"{settings.EMBEDDING_MODEL}). Response: {data}"
        )
    return embeddings


def embed_text(text: str, prefix: Optional[str] = None) -> List[float]:
    return embed_texts([text], prefix=prefix)[0]


def vector_size() -> int:
    """Dimension of the configured embedding model, determined once by
    embedding a short probe string and caching the result."""
    global _dim_cache
    if _dim_cache is None:
        _dim_cache = len(embed_text("dimension probe", prefix=settings.EMBEDDING_DOC_PREFIX))
    return _dim_cache
