"""ModelClient protocol + the Ollama backend's HTTP behavior (no network)."""

import httpx

from chronicle.models import OllamaClient
from chronicle.models.base import LLMResponse, ModelClient


def test_llmresponse_fields():
    r = LLMResponse(text="x", model="m", input_tokens=1, output_tokens=2, latency_ms=3.0)
    assert (r.text, r.model, r.input_tokens, r.output_tokens) == ("x", "m", 1, 2)


def test_protocol_runtime_check():
    class Dummy:
        name = "dummy"

        def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=None):
            return LLMResponse("ok", "dummy", 0, 0, 0.0)

    assert isinstance(Dummy(), ModelClient)
    assert not isinstance(object(), ModelClient)


def _mock_ollama() -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        if request.url.path == "/api/chat":
            body = request.read().decode()
            assert '"model"' in body and "qwen2.5:3b" in body
            assert '"stream": false' in body or '"stream":false' in body
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "pong"},
                    "prompt_eval_count": 11,
                    "eval_count": 3,
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test")
    return OllamaClient(base_url="http://test", model="qwen2.5:3b", client=client)


def test_ollama_tags():
    oc = _mock_ollama()
    tags = oc.tags()
    assert tags["models"][0]["name"] == "qwen2.5:3b"


def test_ollama_complete_parses_usage():
    oc = _mock_ollama()
    resp = oc.complete("ping", system="be terse")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "pong"
    assert resp.input_tokens == 11
    assert resp.output_tokens == 3
    assert resp.model == "qwen2.5:3b"
    assert resp.latency_ms >= 0.0
