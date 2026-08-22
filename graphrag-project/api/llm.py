import json
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from config import settings
from retry import with_retry_async
from ollama_client import get_async_client


class OllamaError(RuntimeError):
    """Raised when Ollama returns a non-2xx response, with its actual error
    body included — httpx's default message is just the HTTP status line,
    which hides the useful part (e.g. 'model requires more system memory
    than is available'). Carries the original status_code so callers (the
    retry helper below) can tell a transient 5xx from a 4xx config
    problem that retrying can't fix."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


async def _raise_with_body(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        raw = await resp.aread()
        text = raw.decode(errors="ignore")
    except Exception:
        text = ""
    try:
        body = json.loads(text)
        message = body.get("error", text)
    except Exception:
        message = text or f"HTTP {resp.status_code}"
    raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {message}", status_code=resp.status_code)


def _is_retryable_llm_error(e: Exception) -> bool:
    """ollama_chat wraps HTTP errors as OllamaError (not raw
    httpx.HTTPStatusError), so it needs its own retryability check rather
    than retry.is_retryable_ollama_error, which expects the httpx
    exception type directly."""
    if isinstance(e, OllamaError):
        return e.status_code is not None and e.status_code >= 500
    if isinstance(e, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
        return True
    return False


def build_rag_messages(
    query: str, context: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """Builds the message list sent to Ollama: system prompt + retrieved
    context, optional prior turns, then the user's question."""
    context_block = context.strip() or "(no relevant context was found)"
    system_content = f"{settings.RAG_SYSTEM_PROMPT}\n\nContext:\n{context_block}"

    messages = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def build_options(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Maps OpenAI-style generation params to Ollama's `options` object.
    Only includes keys that were actually provided."""
    opts: Dict = {}
    if temperature is not None:
        opts["temperature"] = temperature
    if top_p is not None:
        opts["top_p"] = top_p
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    if seed is not None:
        opts["seed"] = seed
    return opts


async def ollama_chat(
    messages: List[Dict[str, str]], model: Optional[str] = None, options: Optional[Dict] = None
) -> str:
    """Non-streaming chat call. Returns the full assistant reply text.
    `options` maps directly to Ollama's options object, e.g.
    {"temperature": 0.2, "top_p": 0.9, "num_predict": 512}.

    Retries transient failures (connection errors, timeouts, 5xx) with
    backoff - streaming (ollama_chat_stream below) deliberately does NOT
    get this treatment, since retrying a request that's already streamed
    partial output to the user isn't well-defined."""
    payload = {
        "model": model or settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options

    async def _do_request():
        client = get_async_client()
        resp = await client.post(f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload)
        await _raise_with_body(resp)
        data = resp.json()
        return data.get("message", {}).get("content", "")

    return await with_retry_async(
        _do_request,
        attempts=settings.OLLAMA_RETRY_ATTEMPTS,
        base_delay=settings.OLLAMA_RETRY_BASE_DELAY,
        retryable=_is_retryable_llm_error,
        label="ollama_chat",
    )


async def ollama_chat_stream(
    messages: List[Dict[str, str]], model: Optional[str] = None, options: Optional[Dict] = None
) -> AsyncGenerator[str, None]:
    """Streaming chat call. Yields text deltas as Ollama produces them."""
    payload = {
        "model": model or settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
    }
    if options:
        payload["options"] = options
    client = get_async_client()
    async with client.stream(
        "POST", f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload
    ) as resp:
        await _raise_with_body(resp)
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break


async def list_ollama_models() -> List[str]:
    """Lists model names currently pulled in the connected Ollama instance.
    Uses a short per-request timeout override (not the shared client's
    default OLLAMA_TIMEOUT, which can be minutes) - this is used by
    /health/detailed and /v1/models, both of which should fail fast if
    Ollama is unreachable rather than hang."""
    client = get_async_client()
    resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]
