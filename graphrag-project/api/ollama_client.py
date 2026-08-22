"""
Shared, reused httpx clients for talking to Ollama.

Every Ollama call in this codebase (embeddings, entity extraction,
retrieval's LLM-assist steps, chat) previously opened a brand new
httpx.Client/AsyncClient - and therefore a brand new TCP connection - on
every single call. That's real avoidable overhead, especially during
ingestion where GRAPH_EXTRACTION_CONCURRENCY runs several extraction calls
concurrently against the same Ollama host, and during interactive chat
where every user turn re-embeds a query.

httpx.Client and httpx.AsyncClient both manage an internal connection pool
and are safe to reuse across many calls (Client is documented as
thread-safe for concurrent requests, which matters here since
ingest.py's ThreadPoolExecutor shares one client across worker threads).
Reusing a single instance for the process lifetime lets httpx keep
connections alive and reuse them instead of paying a fresh TCP handshake
per call.
"""
import httpx

from config import settings

_sync_client: httpx.Client = None
_async_client: httpx.AsyncClient = None


def get_sync_client() -> httpx.Client:
    """Shared client for synchronous Ollama calls (embeddings.py,
    graph_extraction.py, retrieval.py's LLM-assist helpers)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(timeout=settings.OLLAMA_TIMEOUT)
    return _sync_client


def get_async_client() -> httpx.AsyncClient:
    """Shared client for async Ollama calls (llm.py)."""
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT)
    return _async_client


def close_sync_client():
    """Called on FastAPI shutdown (main.py) to release the pooled sync
    connection cleanly rather than relying on process teardown."""
    global _sync_client
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None


async def close_async_client():
    """Async counterpart, awaited from main.py's shutdown handler."""
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
