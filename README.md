# mini-gpt-inference

**Status: planned, not yet implemented.** See [`docs/PLAN.md`](docs/PLAN.md) for the
full implementation plan and [`docs/CHECKLIST.md`](docs/CHECKLIST.md) for the
ordered execution list.

Project B of a three-project ladder targeting ML/research engineer roles at
frontier labs: (A) distributed pretraining, (B) inference engine, (C) GRPO
post-training with a vLLM rollout server.

## What this will be

A from-scratch LLM inference engine serving the 50.9M-parameter GPT trained in
[Project A (mini-gpt-ddp)](https://github.com/AbdullahRasheed45/mini-gpt-ddp),
with every optimization measured against a naive reference implementation:

| | what gets built | why it matters |
|---|---|---|
| **KV cache** | O(n) decode replacing O(n²) recompute | the foundational inference optimization |
| **Static batching** | left-padded batched decode | finds the throughput/latency knee |
| **Paged KV cache** | block pool + block tables + allocator | PagedAttention's core mechanism, from scratch |
| **Continuous batching** | admission/preemption scheduler | what actually makes a server efficient |
| **CUDA graphs / `torch.compile`** | launch-overhead elimination | expected largest single win at this model size |
| **Triton kernels** | paged decode attention, online softmax | SDPA can't read block tables |
| **INT8 quantization** | W8A16 weight-only + fused dequant GEMV | memory-bandwidth reduction |
| **Speculative decoding** | prompt-lookup + self-speculative drafts | >1 token per target forward |
| **OpenAI-compatible server** | FastAPI, SSE streaming, Prometheus | the part that makes it a *system* |

## Why a 50M-parameter model is an interesting inference target

Small models are *harder* to serve efficiently than large ones, in an instructive way.
At batch size 1 this model needs a 102 MB weight read per token — a 0.32 ms floor on
a T4 — while its arithmetic intensity (~1 FLOP/byte) sits ~200x below the T4's
roofline ridge point. Decode is not compute-bound or even really cache-bound; it is
bound by **Python dispatch and kernel launch overhead**. That makes this an unusually
clean setting to measure what CUDA graphs and kernel fusion are actually worth.

Full derivation of these numbers, and nine falsifiable predictions to test them
against, are in [`docs/PLAN.md`](docs/PLAN.md) §1 and §6.

## Hardware

Free-tier only: Kaggle (Tesla P100, sm60) and Lightning AI (T4, sm75). Neither
supports bf16 or FlashAttention-2; Triton needs sm70+ so it is T4-only. The P100
has 2.3x the T4's memory bandwidth but no fp16 tensor cores, which sets up a
genuine crossover experiment — P100 should win memory-bound bs=1 decode, T4 should
win compute-bound batched decode. Measuring that is Phase 8.

## Layout

```
docs/PLAN.md        # authoritative implementation plan
docs/CHECKLIST.md   # ordered task list with per-phase gates
src/minigpt_infer/  # the engine
bench/              # benchmark harness; every published number regenerates from here
tests/              # correctness ladder (see PLAN §8)
```
