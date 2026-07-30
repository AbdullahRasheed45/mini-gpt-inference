# Execution Checklist

Ordered task list derived from [PLAN.md](PLAN.md). Work top to bottom. Do not
start a phase until the previous phase's **Gate** passes.

Read PLAN.md §5 (core abstractions), §8 (testing ladder), and §10 (benchmark
rules) **before** writing any code. They apply to every phase.

---

## Phase 0 — Foundation

- [ ] `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml` — already
      committed; verify `pip install -e ".[dev]"` works locally
- [ ] `src/minigpt_infer/__init__.py`
- [ ] `config.py`: `GPTConfig` (copy Project A's exactly), `EngineConfig`, `SamplingParams`
- [ ] `model.py`: verbatim copy of Project A's model — **commit this alone first**,
      so the caching diff is clean and reviewable
- [ ] `reference.py`: Project A's `generate()` unchanged + "never optimize" docstring
- [ ] `loader.py`: HF download, `weights_only=True` attempt w/ fallback, prefix
      stripping, config reconstruction, **weight-tying assert**, `strict=True` load
- [ ] `tokenizer.py`: tiktoken gpt2; **mask logits ≥ 50257 to `-inf`** (rows
      50257–50303 are unreachable padding — the model can otherwise emit
      undecodable ids)
- [ ] `scripts/download_checkpoint.py`
- [ ] `tests/test_loader.py`, `tests/test_reference.py` (tiny random model, CPU, fp32)
- [ ] Record golden greedy output for `"Once upon a time"` → `tests/golden/`
- [ ] **Gate**: CI green; reference vs model logits match `atol=1e-5`

## Phase 1 — KV cache

- [ ] `batch.py`: `ForwardBatch` with **all** fields from PLAN §5 (paged ones
      `None` for now — the model signature must not change again after this)
- [ ] `cache/base.py`: `KVCacheBase` protocol
- [ ] `cache/static.py`: `(B, n_head, max_len, head_dim)` per layer + seq lengths
- [ ] `model.py`: prefill/decode split, thread `position_ids` through
- [ ] ⚠ decode uses `is_causal=False` (see PLAN §7 Phase 1 pitfalls — `True` is
      silently, catastrophically wrong when `q_len != kv_len`)
- [ ] Assert `position_ids.max() < block_size`
- [ ] `tests/test_cache.py`: **exact** greedy id match vs reference, ≥20 prompts
- [ ] `bench/common.py` (implement PLAN §10 rules once, reuse everywhere)
- [ ] `bench/bench_kvcache.py`: N ∈ {16,32,64,128,256,512}, naive vs cached
- [ ] **Gate**: exact greedy match; P1 recorded in `docs/BENCHMARKS.md`

## Phase 2 — Static batching

- [ ] Left padding + `position_ids = (mask.cumsum(-1) - 1).clamp(min=0)`
- [ ] Combined causal+padding additive mask; use `torch.finfo(dtype).min`, **not**
      `float("-inf")` (fully-masked rows → NaN)
- [ ] Assert no row is entirely masked
- [ ] `sampling.py`: batched greedy / temperature / top-k / top-p / repetition
      penalty, vectorized over `(B, vocab)`, no Python loop over batch
- [ ] Per-sequence `finished` flags; never sample into a finished slot
- [ ] `tests/test_batching.py`: **batch invariance** — in-batch output == solo output
- [ ] `bench/bench_batching.py`: bs ∈ {1,2,4,8,16,32,64,128,256}
- [ ] **Gate**: batch invariance passes; throughput knee identified (P4)

## Phase 3 — Paged KV + continuous batching ★

- [ ] `cache/paged.py`: block pool `(num_blocks, block_size, n_head, head_dim)`,
      `block_size=16`, fixed `num_blocks` from config (not "all free memory")
- [ ] `BlockManager`: free list, `allocate` / `append_slot` / `free`, frag stats
- [ ] `block_tables` (B, max_blocks) int32; `slot_mapping` scatter writes
- [ ] Read path (a): gather + SDPA — the reference for the Triton kernel
- [ ] `engine/request.py`: `Request`, `SequenceState`, `RequestOutput`
- [ ] `engine/scheduler.py`: waiting/running queues, admission, prefill-priority,
      finish detection, **free blocks in the same step a sequence finishes**
- [ ] Preemption by recompute when allocation fails; count preemptions
- [ ] Invariant asserted every step: `next_position == seq_len`
- [ ] Stop strings matched on **decoded text**, handling a stop split across tokens
- [ ] `engine/engine.py`: `LLMEngine.step()`
- [ ] `tests/test_paged.py`: paged == static, exact, greedy, ≥20 prompts
- [ ] `tests/test_scheduler.py`: 1000 randomized alloc/free cycles leak **zero**
      blocks; preemption actually exercised with a tiny pool
- [ ] `bench/bench_paged.py`: static vs continuous on uniform **and** high-variance
      output lengths (the gap should appear only in the latter)
- [ ] `docs/ARCHITECTURE.md`
- [ ] **Gate**: exact paged==static; no block leaks; P5 and P6 recorded

## Phase 4 — Kernels and launch overhead ★

- [ ] **First**: profile one decode step (`torch.profiler`) — record kernel launch
      count, total kernel time, wall time. The kernel-time/wall-time gap is the
      prize and motivates this whole phase. Put it in the writeup.
- [ ] `torch.compile` with `mode="reduce-overhead"`; bucket batch sizes; drive
      steady-state recompiles to zero (verify via `TORCH_LOGS=recompiles`)
- [ ] `graphs.py`: manual CUDA graph capture per batch bucket
  - [ ] pre-allocated static input/output buffers at fixed addresses
  - [ ] warm up on a side stream before capture
  - [ ] never reallocate the KV pool after capture
- [ ] `attention/triton_paged.py`: paged decode kernel, online softmax
  - [ ] guard on compute capability ≥ sm70; fall back to SDPA on P100
  - [ ] try both elementwise-reduce (GEMV) and `tl.dot`; keep the faster
- [ ] `tests/test_triton.py` (`@pytest.mark.gpu`): matches gather+SDPA; identical
      greedy tokens
- [ ] `bench/bench_kernels.py`
- [ ] **Gate**: Triton parity; launch count before/after recorded; P2, P3 recorded

## Phase 5 — Quantization

- [ ] `quant/int8.py`: per-output-channel symmetric int8 for `qkv`, `attn.proj`,
      `mlp.fc`, `mlp.proj`; leave tied `lm_head`/`tok_emb` in fp16 (document why)
- [ ] Measure naive dequant-then-matmul — **expect it to be slower (P7); publish
      that negative result**
- [ ] Fused Triton dequant GEMV; measure again
- [ ] Perplexity fp16 vs int8 on held-out TinyStories
- [ ] `tests/test_quant.py`: round-trip error bound
- [ ] `bench/bench_quant.py`
- [ ] **Gate**: memory reduction + perplexity delta recorded; P7 resolved

## Phase 6 — Speculative decoding

- [ ] `engine/spec_decode.py`, prompt-lookup (n-gram) draft first — no model needed
- [ ] Verification loop: accept w.p. `min(1, p/q)`; on reject sample from
      `normalize(max(0, p - q))`; bonus token when all γ accepted
- [ ] Self-speculative variant (first 4 of 8 layers as draft)
- [ ] *(stretch)* trained 2L/2H/128d draft — reuse Project A's `train.py` unchanged
- [ ] `tests/test_spec_decode.py`:
  - [ ] **greedy exactness**: temp=0 spec output == target greedy, identical ids
  - [ ] **chi-square**: ≥100k samples at temp=1, no rejection at α=0.01
- [ ] `bench/bench_specdec.py`: α and mean accepted length vs γ ∈ {1..8};
      measured tokens/step vs theoretical `(1-α^(γ+1))/(1-α)`
- [ ] **Gate**: both correctness tests pass; P8 recorded

## Phase 7 — Serving

- [ ] `server/protocol.py`: pydantic OpenAI-shaped schemas
- [ ] `server/api.py`: `/v1/completions`, `/v1/chat/completions`, `/v1/models`,
      `/health`, `/metrics`
- [ ] SSE streaming: `data: {json}\n\n`, terminated `data: [DONE]\n\n`
- [ ] `engine/async_engine.py`: engine in a **dedicated thread** + thread-safe
      per-request queues (do not run `step()` on the asyncio loop)
- [ ] Incremental detokenizer: prefix offset + trailing-window diff; hold back
      incomplete UTF-8 bytes. **Never re-decode the whole sequence per step.**
- [ ] `server/metrics.py`: Prometheus (see PLAN §7 Phase 7 for the metric list)
- [ ] `tests/test_server.py`: `TestClient`, tiny random model, CPU, runs in CI
  - [ ] streaming final text == non-streaming, same seed
  - [ ] **cancel mid-stream frees KV blocks** (leaks in naive implementations)
- [ ] **Gate**: CI green including server tests; `/metrics` scrapes

## Phase 8 — Benchmarks and hardware study

- [ ] `bench/load_test.py`: Poisson arrivals, configurable λ, realistic length
      distributions, per-request TTFT/TPOT/E2E
- [ ] Latency-vs-QPS knee curve; max sustainable QPS under a **stated** SLO
- [ ] `bench/bench_hardware.py`: identical suite on T4 (Lightning) and P100
      (Kaggle) — the crossover study (P9)
- [ ] Every run writes JSON with git SHA, torch/CUDA version, GPU name, config,
      **raw samples**
- [ ] `bench/plot.py` → `docs/img/`
- [ ] `scripts/run_all_benchmarks.py`
- [ ] `notebooks/{kaggle,lightning}_bench.ipynb`
- [ ] **Gate**: every published number regenerable from one script; P9 recorded

## Phase 9 — Docs

- [ ] `README.md` per PLAN §9 structure — results-forward
- [ ] Optimization ladder table (naive → … → spec-decode)
- [ ] Hardware crossover section
- [ ] **Predictions vs measured (§6), including the wrong ones**
- [ ] Non-goals (§12) stated plainly
- [ ] Reproduction instructions
- [ ] Final pass against PLAN §15 Definition of Done
