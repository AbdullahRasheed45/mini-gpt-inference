"""Prometheus metrics (docs/PLAN.md §7 Phase 7 point 6).

A dedicated CollectorRegistry (not prometheus_client's global default) so
this module can be imported by multiple app instances (e.g. one per test in
tests/test_server.py) without "Duplicated timeseries in CollectorRegistry"
errors on re-registration.

`spec_accept_rate` is defined and scrapeable but stays at its default (0)
in this server: Phase 6's speculative decoding is a standalone driver
(engine.spec_decode.speculative_generate), not yet wired into the
continuous-batching LLMEngine this server runs on. The metric exists so
`/metrics` has the name Phase 7 asks for and future engine+spec-decode
integration has nothing left to add here -- not a stand-in for real
integration work.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

registry = CollectorRegistry()

ttft_seconds = Histogram(
    "ttft_seconds", "Time to first token (arrival to first output token)", registry=registry,
)
tpot_seconds = Histogram(
    "tpot_seconds", "Mean inter-token latency during decode, excluding the first token",
    registry=registry,
)
e2e_seconds = Histogram(
    "e2e_seconds", "End-to-end request latency (arrival to final token)", registry=registry,
)
running_requests = Gauge(
    "running_requests", "Requests currently in the engine's running set", registry=registry,
)
waiting_requests = Gauge(
    "waiting_requests", "Requests waiting for admission", registry=registry,
)
kv_cache_usage_ratio = Gauge(
    "kv_cache_usage_ratio", "Fraction of KV cache blocks currently allocated", registry=registry,
)
tokens_generated_total = Counter(
    "tokens_generated_total", "Total tokens generated across all requests", registry=registry,
)
preemptions_total = Counter(
    "preemptions_total", "Total scheduler preemptions", registry=registry,
)
spec_accept_rate = Gauge(
    "spec_accept_rate",
    "Speculative decoding draft acceptance rate (0: not wired into this server's engine yet)",
    registry=registry,
)
request_total = Counter(
    "request_total", "Total requests by terminal status", ["status"], registry=registry,
)


def render() -> bytes:
    return generate_latest(registry)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "e2e_seconds",
    "kv_cache_usage_ratio",
    "preemptions_total",
    "registry",
    "render",
    "request_total",
    "running_requests",
    "spec_accept_rate",
    "tokens_generated_total",
    "tpot_seconds",
    "ttft_seconds",
    "waiting_requests",
]
