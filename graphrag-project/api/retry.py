import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, TypeVar

import httpx

logger = logging.getLogger("graphrag")

T = TypeVar("T")


def is_retryable_ollama_error(e: Exception) -> bool:
    """
    Shared retryability check for direct httpx calls to Ollama (used by
    embeddings.py and graph_extraction.py, which call resp.raise_for_status()
    themselves, producing httpx.HTTPStatusError on non-2xx). 4xx responses
    (unknown model, bad request) are config problems - retrying can't fix
    them and just wastes time. Connection errors, timeouts, and 5xx are
    often transient and worth a retry.
    """
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500
    if isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
        return True
    return False


def with_retry(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    retryable: Optional[Callable[[Exception], bool]] = None,
    label: str = "operation",
) -> T:
    """
    Calls fn() with up to `attempts` tries and exponential backoff
    (base_delay, base_delay*2, base_delay*4, ...) between attempts. Used
    around Ollama HTTP calls during ingestion so a single transient blip
    doesn't kill an entire large-file ingest job or silently drop graph
    data for one chunk.

    `retryable` decides whether a given exception should trigger a retry
    (defaults to retrying everything). Callers should pass a narrower
    check where possible - see is_retryable_ollama_error above.
    """
    attempts = max(1, attempts)  # attempts=0 would mean the loop body
    # never runs and last_exc stays None, so `raise last_exc` at the end
    # would raise None (TypeError) instead of a meaningful error.
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            should_retry = retryable(e) if retryable else True
            if not should_retry or attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"{label}: attempt {attempt}/{attempts} failed ({e}), retrying in {delay:.1f}s")
            time.sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above


async def with_retry_async(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 1.0,
    retryable: Optional[Callable[[Exception], bool]] = None,
    label: str = "operation",
) -> T:
    """Async counterpart to with_retry - same semantics, awaits fn() and
    uses asyncio.sleep for backoff instead of blocking the event loop."""
    attempts = max(1, attempts)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            should_retry = retryable(e) if retryable else True
            if not should_retry or attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"{label}: attempt {attempt}/{attempts} failed ({e}), retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above
