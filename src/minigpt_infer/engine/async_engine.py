"""AsyncEngine: runs LLMEngine.step() in a dedicated background thread
(docs/PLAN.md §7 Phase 7 point 4), not the asyncio event loop.

vLLM interleaves engine stepping into the async loop; a separate thread is
simpler to get right from scratch and avoids starving the event loop during
a long prefill -- `model.forward()` is a synchronous, CPU/GPU-bound call,
and running it directly on the event loop would block every other
coroutine (every other in-flight request's streaming, health checks, metric
scrapes) for its entire duration.

Each request gets its own thread-safe `queue.Queue`; the engine thread
pushes `RequestOutput`s onto it as they're produced (plus a `_SENTINEL` when
the request finishes or is cancelled), and the async side drains it via
`loop.run_in_executor` so waiting for the next item never blocks the event
loop -- the standard sync-queue/async-coroutine bridge pattern.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import torch

from minigpt_infer.config import EngineConfig, SamplingParams
from minigpt_infer.engine.engine import LLMEngine
from minigpt_infer.engine.request import Request, RequestOutput
from minigpt_infer.model import GPT

_SENTINEL = object()  # marks end-of-stream on a request's queue


class EngineCrashedError(RuntimeError):
    """Raised on the consumer side when the engine's background thread hit
    an unexpected exception mid-step. Without this, a crash inside step()
    would kill the thread silently and every open request's queue.get()
    would block forever -- confirmed by a real test hang, not a
    hypothetical (see tests/test_server.py's history)."""


@dataclass
class EngineMetricsSnapshot:
    running_requests: int
    waiting_requests: int
    kv_cache_usage_ratio: float


class AsyncEngine:
    def __init__(
        self,
        model: GPT,
        engine_cfg: EngineConfig,
        vocab_mask: torch.Tensor | None = None,
        eot_token_id: int | None = None,
    ) -> None:
        self._engine = LLMEngine(model, engine_cfg, vocab_mask=vocab_mask)
        self._eot_token_id = eot_token_id
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._new_request_event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="minigpt-engine")
        self._thread.start()

    def submit(
        self, prompt_token_ids: list[int], sampling_params: SamplingParams,
    ) -> tuple[str, queue.Queue]:
        """Registers a new request. Returns (request_id, its queue) --
        callers MUST pass the queue itself into stream(), not just the id:
        for a fast (e.g. tiny/CPU) model, the request can finish and get
        popped from self._queues before the caller ever calls stream(); a
        second by-id lookup at that point would find nothing and silently
        yield zero outputs (confirmed by a real test failure, not a
        hypothetical -- see tests/test_server.py's streaming test). Holding
        the queue object directly sidesteps the lookup (and the race)
        entirely: queue.Queue buffers every item regardless of whether a
        consumer has started draining it yet.
        """
        request_id = uuid.uuid4().hex
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._queues[request_id] = q
            request = Request(
                request_id, prompt_token_ids, sampling_params, eot_token_id=self._eot_token_id,
            )
            self._engine.add_request(request)
        self._new_request_event.set()
        return request_id, q

    async def stream(self, request_id: str, q: queue.Queue) -> AsyncIterator[RequestOutput]:
        loop = asyncio.get_event_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise EngineCrashedError("engine background thread crashed") from item
                yield item
        finally:
            # Covers both normal completion (already popped, no-op) and a
            # client disconnecting mid-stream (async generators get
            # .aclose()'d on early exit, landing here) -- docs/PLAN.md Phase 7
            # acceptance: cancellation must free KV blocks immediately.
            self.cancel(request_id)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self._engine.cancel_request(request_id)
            q = self._queues.pop(request_id, None)
        if q is not None:
            q.put(_SENTINEL)

    def metrics_snapshot(self) -> EngineMetricsSnapshot:
        with self._lock:
            running = len(self._engine.scheduler.running)
            waiting = len(self._engine.scheduler.waiting)
            total_blocks = self._engine.block_manager.num_blocks
            free_blocks = self._engine.block_manager.num_free_blocks
        usage = 1.0 - (free_blocks / total_blocks if total_blocks else 0.0)
        return EngineMetricsSnapshot(running, waiting, usage)

    def shutdown(self) -> None:
        self._stop.set()
        self._new_request_event.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                has_work = self._engine.has_unfinished_requests()
            if not has_work:
                self._new_request_event.wait(timeout=0.1)
                self._new_request_event.clear()
                continue

            try:
                with self._lock:
                    outputs = self._engine.step()
            except Exception as e:  # noqa: BLE001 - deliberately broad, see class docstring
                # The engine's internal state may now be inconsistent (a
                # partially-written cache, a scheduler mid-mutation) -- there
                # is no safe way to know which in-flight requests were
                # affected, so fail ALL of them loudly and stop, rather than
                # silently hang every open stream forever (what happened
                # before this existed).
                with self._lock:
                    pending = list(self._queues.items())
                    self._queues.clear()
                for _request_id, q in pending:
                    q.put(e)
                return

            for out in outputs:
                with self._lock:
                    q = self._queues.get(out.request_id)
                if q is None:
                    continue
                q.put(out)
                if out.finished:
                    q.put(_SENTINEL)
                    with self._lock:
                        self._queues.pop(out.request_id, None)
