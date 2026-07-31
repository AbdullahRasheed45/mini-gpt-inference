"""OpenAI-compatible FastAPI server (docs/PLAN.md §7 Phase 7).

`create_app(...)` is a factory, not a module-level app: tests build one
around a tiny random-init model (no checkpoint, no GPU, runs in CI), while
`scripts/serve.py` builds one around the real trained checkpoint. Every
endpoint drives an `AsyncEngine`, which runs the actual `LLMEngine.step()`
loop on a dedicated thread (see engine/async_engine.py's module docstring).

**Chat endpoint honesty** (docs/PLAN.md Phase 7 point 2): this model is a
base model pretrained on TinyStories, never instruction-tuned. The chat
endpoint exists for OpenAI API *shape* compatibility and applies a trivial,
un-trained-for template (`_apply_chat_template`) -- it is not a real chat
model and chat quality is not the point of this project. Said plainly here
in code and again in the README, not just implied.
"""

from __future__ import annotations

import queue
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from minigpt_infer.config import EngineConfig, SamplingParams
from minigpt_infer.engine.async_engine import AsyncEngine
from minigpt_infer.model import GPT
from minigpt_infer.server import metrics
from minigpt_infer.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatCompletionStreamDelta,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamChunk,
    ModelInfo,
    ModelsResponse,
)


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _apply_chat_template(messages: list[ChatMessage]) -> str:
    """Trivial, un-trained-for template -- see module docstring. Just labels
    each turn by role and leaves an open "Assistant:" for the model to
    continue; nothing this base model was ever trained to recognize as a
    turn boundary."""
    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines)


def create_app(
    model: GPT,
    engine_cfg: EngineConfig,
    encode_fn: Callable[[str], list[int]],
    eot_token_id: int | None = None,
    vocab_mask: torch.Tensor | None = None,
    served_model_name: str = "minigpt-infer",
) -> FastAPI:
    """No `decode_fn` parameter: output text always comes from
    `RequestOutput.text_delta`, which `LLMEngine` already produces via its
    own `IncrementalDetokenizer` (real GPT-2 BPE, docs/PLAN.md Phase 7 point
    5) -- there is nothing left for this layer to decode itself. Only the
    *prompt* needs an explicit encode function.
    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield
        app.state.engine.shutdown()

    app = FastAPI(title="minigpt-infer", version="0.1.0", lifespan=_lifespan)
    app.state.engine = AsyncEngine(
        model, engine_cfg, vocab_mask=vocab_mask, eot_token_id=eot_token_id,
    )
    app.state.served_model_name = served_model_name

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        snap = app.state.engine.metrics_snapshot()
        metrics.running_requests.set(snap.running_requests)
        metrics.waiting_requests.set(snap.waiting_requests)
        metrics.kv_cache_usage_ratio.set(snap.kv_cache_usage_ratio)
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models() -> ModelsResponse:
        return ModelsResponse(data=[ModelInfo(id=served_model_name)])

    async def _consume(
        request_id: str, q: queue.Queue, on_token: Callable[[str, str | None], None],
    ) -> None:
        t_start = time.perf_counter()
        first_token_at: float | None = None
        prev_at = t_start
        async for out in app.state.engine.stream(request_id, q):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
                metrics.ttft_seconds.observe(now - t_start)
            else:
                metrics.tpot_seconds.observe(now - prev_at)
            prev_at = now
            metrics.tokens_generated_total.inc(len(out.new_token_ids))
            on_token(out.text_delta, out.finish_reason)
        metrics.e2e_seconds.observe(time.perf_counter() - t_start)

    def _sampling_params(max_tokens: int, temperature: float, top_p, top_k, stop) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, stop=_normalize_stop(stop),
        )

    def _submit_or_400(prompt_ids: list[int], sp: SamplingParams) -> tuple[str, queue.Queue]:
        """LLMEngine.add_request() asserts prompt_len + max_tokens fits under
        the model's block_size -- a foreseeable, client-caused error (asking
        for too many tokens), which deserves a clean 400 here rather than
        propagating as an unhandled 500."""
        try:
            return app.state.engine.submit(prompt_ids, sp)
        except AssertionError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # ---- completions ------------------------------------------------

    async def _stream_completion_response(request_id: str, q: queue.Queue, model_name: str):
        chunk_id = f"cmpl-{uuid.uuid4().hex[:24]}"

        async def gen():
            async for out in app.state.engine.stream(request_id, q):
                chunk = CompletionStreamChunk(
                    id=chunk_id, model=model_name,
                    choices=[CompletionStreamChoice(
                        index=0, text=out.text_delta, finish_reason=out.finish_reason,
                    )],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/completions", response_model=None)
    async def create_completion(req: CompletionRequest):
        if req.model != served_model_name:
            raise HTTPException(status_code=404, detail=f"unknown model '{req.model}'")
        prompt_ids = encode_fn(req.prompt)
        sp = _sampling_params(req.max_tokens, req.temperature, req.top_p, req.top_k, req.stop)
        request_id, q = _submit_or_400(prompt_ids, sp)

        if req.stream:
            return await _stream_completion_response(request_id, q, req.model)

        text_parts: list[str] = []
        finish_reason: str | None = None

        def on_token(delta: str, fr: str | None) -> None:
            nonlocal finish_reason
            text_parts.append(delta)
            finish_reason = fr

        await _consume(request_id, q, on_token)
        metrics.request_total.labels(status="success").inc()
        choice = CompletionChoice(index=0, text="".join(text_parts), finish_reason=finish_reason)
        return CompletionResponse(model=req.model, choices=[choice])

    # ---- chat completions ---------------------------------------------

    async def _stream_chat_response(request_id: str, q: queue.Queue, model_name: str):
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        async def gen():
            first = True
            async for out in app.state.engine.stream(request_id, q):
                delta = ChatCompletionStreamDelta(
                    role="assistant" if first else None, content=out.text_delta,
                )
                first = False
                chunk = ChatCompletionStreamChunk(
                    id=chunk_id, model=model_name,
                    choices=[ChatCompletionStreamChoice(
                        index=0, delta=delta, finish_reason=out.finish_reason,
                    )],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(req: ChatCompletionRequest):
        if req.model != served_model_name:
            raise HTTPException(status_code=404, detail=f"unknown model '{req.model}'")
        prompt_text = _apply_chat_template(req.messages)
        prompt_ids = encode_fn(prompt_text)
        sp = _sampling_params(req.max_tokens, req.temperature, req.top_p, req.top_k, req.stop)
        request_id, q = _submit_or_400(prompt_ids, sp)

        if req.stream:
            return await _stream_chat_response(request_id, q, req.model)

        text_parts: list[str] = []
        finish_reason: str | None = None

        def on_token(delta: str, fr: str | None) -> None:
            nonlocal finish_reason
            text_parts.append(delta)
            finish_reason = fr

        await _consume(request_id, q, on_token)
        metrics.request_total.labels(status="success").inc()
        return ChatCompletionResponse(
            model=req.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="".join(text_parts)),
                finish_reason=finish_reason,
            )],
        )

    return app
