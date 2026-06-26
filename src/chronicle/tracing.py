"""Langfuse tracing for LLM calls.

Design rules:
- Every LLM call is wrapped in a generation span (token + call counts, latency,
  cost where Langfuse knows the model's pricing). This is a first-class metric,
  not an afterthought.
- Tracing is BEST-EFFORT and degrades to a no-op when Langfuse is not
  configured or the SDK is missing. Observability must never crash the primary
  path. (Contrast with diff parsing, which is strict-parse-or-crash.)
- The langfuse client is created lazily and cached; CLI processes must call
  `flush_traces()` before exit so batched spans are actually sent.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .config import get_settings
from .logging_config import get_logger

log = get_logger("chronicle.tracing")

_client: Any = None
_client_resolved = False


def _get_client() -> Any:
    """Return a cached Langfuse client, or None if unavailable/unconfigured."""
    global _client, _client_resolved
    if _client_resolved:
        return _client
    _client_resolved = True

    settings = get_settings()
    if not settings.langfuse_enabled:
        log.debug("Langfuse not configured (no keys); tracing is a no-op.")
        _client = None
        return None
    try:
        from langfuse import Langfuse  # lazy import

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        log.debug("Langfuse client initialized (host=%s).", settings.langfuse_host)
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("Langfuse unavailable, tracing disabled: %s", exc)
        _client = None
    return _client


def reset_client() -> None:
    """Drop the cached client (used by tests that flip configuration)."""
    global _client, _client_resolved
    _client = None
    _client_resolved = False


class GenerationHandle:
    """Caller records the LLM output and token usage onto this handle.

    A no-op when there is no underlying Langfuse generation.
    """

    __slots__ = ("output", "input_tokens", "output_tokens", "_gen")

    def __init__(self, gen: Any = None) -> None:
        self._gen = gen
        self.output: Optional[str] = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def record(self, *, output: str, input_tokens: int, output_tokens: int) -> None:
        self.output = output
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


@contextmanager
def trace_generation(
    *,
    name: str,
    model: str,
    input: Any,
    metadata: Optional[dict] = None,
) -> Iterator[GenerationHandle]:
    """Open a Langfuse generation span around an LLM call.

    Usage:
        with trace_generation(name="ollama.complete", model=m, input=msgs) as h:
            resp = call_model(...)
            h.record(output=resp.text, input_tokens=a, output_tokens=b)
    """
    client = _get_client()
    if client is None:
        yield GenerationHandle(None)
        return

    gen = None
    try:
        gen = client.generation(
            name=name,
            model=model,
            input=input,
            metadata=metadata or {},
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to start Langfuse generation: %s", exc)

    handle = GenerationHandle(gen)
    try:
        yield handle
    finally:
        if gen is not None:
            try:
                gen.end(
                    output=handle.output,
                    usage={
                        "input": handle.input_tokens,
                        "output": handle.output_tokens,
                        "total": handle.input_tokens + handle.output_tokens,
                        "unit": "TOKENS",
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Failed to end Langfuse generation: %s", exc)


def flush_traces() -> None:
    """Flush batched spans. Call before a short-lived process exits."""
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to flush Langfuse: %s", exc)
