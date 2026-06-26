"""Tracing degrades to a no-op when Langfuse is unconfigured."""

import chronicle.tracing as tracing


def test_trace_generation_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing.reset_client()

    with tracing.trace_generation(name="t", model="m", input="hi") as handle:
        handle.record(output="ok", input_tokens=1, output_tokens=1)

    assert handle.output == "ok"
    assert tracing._get_client() is None
    # flush must be safe even when there is no client.
    tracing.flush_traces()
