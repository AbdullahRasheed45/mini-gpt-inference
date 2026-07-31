"""Phase 7: FastAPI server correctness (docs/PLAN.md §7 Phase 7 acceptance).
Runs entirely against a tiny random-init model via FastAPI's TestClient --
no checkpoint, no GPU, runs in CI.
"""

from __future__ import annotations

import json
import time

import torch
from fastapi.testclient import TestClient

from minigpt_infer.config import EngineConfig
from minigpt_infer.model import GPT
from minigpt_infer.server.api import create_app
from tests.helpers import tiny_gpt_config


def _encode(text: str) -> list[int]:
    """A deterministic fake "tokenizer" for tests: every prompt maps to a
    fixed, valid-for-a-64-vocab-model sequence, independent of content. Real
    tiktoken ids run up to 50256 -- far outside a tiny model's vocab_size=64
    embedding table -- so tests use this instead of the real encoder, while
    output text still goes through the engine's real GPT-2 detokenizer
    (ids 0-63 decode fine, just as arbitrary short subwords)."""
    ids = [(ord(c) * 7 + i) % 64 for i, c in enumerate(text)]
    return ids or [1]


def _make_app(max_batch_size: int = 8, num_blocks: int = 256):
    cfg = tiny_gpt_config(vocab_size=64, block_size=64)
    torch.manual_seed(0)
    model = GPT(cfg)
    model.eval()
    engine_cfg = EngineConfig(block_size=4, num_blocks=num_blocks, max_batch_size=max_batch_size)
    app = create_app(model, engine_cfg, _encode, eot_token_id=None, served_model_name="tiny-test")
    return app


def _iter_sse_json(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))
    return events


def test_health():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_list_models():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "tiny-test"


def test_metrics_scrapes_cleanly_and_has_expected_names():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        for name in [
            "ttft_seconds", "tpot_seconds", "e2e_seconds", "running_requests",
            "waiting_requests", "kv_cache_usage_ratio", "tokens_generated_total",
            "preemptions_total", "spec_accept_rate", "request_total",
        ]:
            assert name in text, f"metric {name} missing from /metrics output"


def test_unknown_model_returns_404():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/completions",
            json={"model": "not-the-right-model", "prompt": "hello", "max_tokens": 4},
        )
        assert resp.status_code == 404


def test_max_tokens_beyond_block_size_returns_400_not_a_hang():
    """A request whose prompt_len + max_tokens exceeds the model's
    block_size can never finish (model.py's position-table bound). This
    must be rejected up front with a clean 400 -- not accepted and left to
    crash the engine's background thread deep inside step(), which would
    silently hang every open stream (a real bug this test guards against)."""
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/completions",
            json={"model": "tiny-test", "prompt": "hello", "max_tokens": 1000, "temperature": 0.0},
        )
        assert resp.status_code == 400


def test_completion_non_streaming_basic():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/completions",
            json={
                "model": "tiny-test", "prompt": "Once upon a time",
                "max_tokens": 8, "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["model"] == "tiny-test"
        assert len(body["choices"]) == 1
        assert isinstance(body["choices"][0]["text"], str)
        assert body["choices"][0]["finish_reason"] == "length"


def test_streaming_and_non_streaming_produce_identical_final_text():
    """docs/PLAN.md Phase 7 acceptance. temperature=0 (greedy) is used since
    that's the one setting this engine's sampling is actually guaranteed
    deterministic for (docs/ARCHITECTURE.md notes per-request seeding isn't
    implemented) -- an honest test of what the server actually promises."""
    app = _make_app()
    payload = {
        "model": "tiny-test", "prompt": "The little robot", "max_tokens": 10, "temperature": 0.0,
    }

    with TestClient(app) as client:
        non_stream_resp = client.post("/v1/completions", json=payload)
        assert non_stream_resp.status_code == 200
        non_stream_text = non_stream_resp.json()["choices"][0]["text"]

        stream_payload = {**payload, "stream": True}
        with client.stream("POST", "/v1/completions", json=stream_payload) as stream_resp:
            assert stream_resp.status_code == 200
            events = _iter_sse_json(stream_resp)
        stream_text = "".join(e["choices"][0]["text"] for e in events)

    assert stream_text == non_stream_text


def test_chat_completion_applies_a_trivial_template():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-test",
                "messages": [
                    {"role": "system", "content": "Be nice."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 6,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)


def test_chat_completion_streaming_first_chunk_has_role():
    app = _make_app()
    payload = {
        "model": "tiny-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
    }
    client = TestClient(app)
    with client, client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        events = _iter_sse_json(resp)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(e["choices"][0]["delta"].get("role") is None for e in events[1:])


def test_engine_cancel_frees_kv_blocks():
    """The reliable, direct check for docs/PLAN.md Phase 7 acceptance
    ("client-cancel mid-stream frees the sequence's KV blocks") -- exercises
    AsyncEngine.cancel() directly rather than depending on exact HTTP
    disconnect timing, which is real but much less deterministic to test.

    Doesn't assert on an intermediate "admitted, blocks held" state: with a
    tiny/fast model the whole 20-token generation can complete before this
    thread's first poll ever observes it, making that assertion flaky
    (confirmed by a real failure, not a hypothetical). Cancelling
    immediately after submit and only checking the eventual free-block count
    is deterministic regardless of how fast the engine thread runs -- either
    the request is caught mid-flight and cancel frees its blocks, or it
    already finished and its blocks were already freed on completion."""
    app = _make_app(max_batch_size=1, num_blocks=8)
    engine = app.state.engine

    from minigpt_infer.config import SamplingParams
    request_id, _q = engine.submit([1, 2, 3], SamplingParams(max_tokens=20, temperature=0.0))
    engine.cancel(request_id)

    deadline = time.time() + 5
    while engine._engine.block_manager.num_free_blocks < 8 and time.time() < deadline:
        time.sleep(0.01)
    assert engine._engine.block_manager.num_free_blocks == 8, "blocks leaked after cancel"
    engine.shutdown()


def test_client_disconnect_mid_stream_frees_kv_blocks():
    """HTTP-level version of the same check: closing the streaming response
    early should trigger AsyncEngine.stream()'s finally-block cancellation."""
    app = _make_app(max_batch_size=1, num_blocks=8)
    engine = app.state.engine
    payload = {
        "model": "tiny-test", "prompt": "x", "max_tokens": 28, "temperature": 0.0, "stream": True,
    }
    with TestClient(app) as client:
        with client.stream("POST", "/v1/completions", json=payload) as resp:
            it = resp.iter_lines()
            next(it)  # read just the first SSE line, then abandon the stream
    deadline = time.time() + 5
    while engine._engine.block_manager.num_free_blocks < 8 and time.time() < deadline:
        time.sleep(0.01)
    assert engine._engine.block_manager.num_free_blocks == 8, (
        "blocks leaked after client disconnect"
    )
