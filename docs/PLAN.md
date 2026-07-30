# Project B — Inference Engine: Implementation Plan

**Status**: planning complete, implementation not started.
**Audience**: the engineer (human or agent) implementing this repo end to end.
**Rule**: this document is authoritative. If an instruction here conflicts with a
habit or a default, follow this document, or change this document first and say why.

---

## 1. What this project is

Build a from-scratch LLM inference engine that serves the model trained in
[Project A (mini-gpt-ddp)](https://github.com/AbdullahRasheed45/mini-gpt-ddp), and
measure every optimization rigorously.

The point is **not** to reimplement vLLM. The point is to build the five things a
modern inference stack is actually made of — KV caching, paged memory, continuous
batching, kernel-level latency work, and speculative decoding — with honest,
reproducible measurements for each, on hardware that is genuinely constrained.

### Why this model is an interesting inference target

The Project A model is small (51.2M params, 102 MB in fp16). That is a *feature*
for this project, not a limitation:

- At batch size 1, decode is **overwhelmingly memory-bandwidth-bound**, not
  compute-bound. Arithmetic intensity is ~1 FLOP/byte; the T4's roofline ridge
  point is ~203 FLOP/byte. You are ~200x below it.
- Therefore **Python and kernel-launch overhead dominate wall clock**. A naive
  PyTorch decode step issues 100+ kernels, each with microseconds of launch
  overhead, against a theoretical floor of 0.32 ms/token. This is exactly the
  regime where CUDA graphs and `torch.compile` produce dramatic, easily-measured
  wins — larger than they would on a 7B model.
- KV cache capacity is a non-issue (8 MiB per full 512-token sequence, ~1700
  sequences fit in 14 GiB). So paged attention here is implemented to demonstrate
  and measure the *mechanism*, not because memory forces it. **Say so in the
  writeup.** Claiming you needed paging for a 50M model would be dishonest.

### Verified model facts (do not re-derive, do not guess)

Computed from Project A's `model.py`, and cross-checked against its training log
line `model parameters: 50.9M`:

| quantity | value |
|---|---|
| architecture | 8 layers, 8 heads, `n_embd=512`, `head_dim=64` |
| `block_size` (max context) | 512 |
| `vocab_size` | 50304 (GPT-2's 50257 padded to a multiple of 64) |
| bias | `False` everywhere (Linear and LayerNorm) |
| activation | GELU, `approximate="tanh"` |
| norm placement | pre-norm (`x + attn(ln1(x))`, `x + mlp(ln2(x))`) |
| position embedding | **learned absolute** (`nn.Embedding(block_size, n_embd)`) |
| weight tying | `lm_head.weight is tok_emb.weight` |
| total params | 51,192,320 |
| non-position-embedding params | 50,930,176 ("50.9M") |
| fp16 weight bytes | 102.4 MB |
| KV bytes / token / layer (fp16) | 2,048 B |
| KV bytes / token, all 8 layers | 16 KiB |
| KV bytes / 512-token sequence | 8 MiB |

Learned absolute position embeddings are the single most important architectural
fact for this project. They mean **every code path must thread an explicit
`position_ids` through the model**. Rotary embeddings would let you cheat; these
do not. Get this wrong and you get subtly-degraded output with no crash.

### The trained checkpoint

- HF repo: `Abdullahrasheed45/mini-gpt-ddp` (**private** — needs a read token)
- Files: `checkpoints/ddp_2gpu.pt` (614,378,551 bytes),
  `checkpoints/ddp_2gpu.meta.json` (`{"iter": 5999, "total_iters_target": 6000}`)
- The `.pt` is a full training checkpoint, not just weights. Structure:
  `{"model": state_dict, "optimizer": ..., "scaler": ..., "iter": int,
    "config": vars(argparse_namespace), "cpu_rng": ..., "cuda_rng": ...}`
- `ckpt["config"]` carries `n_layer/n_head/n_embd/block_size` but **not**
  `vocab_size` — that came from `GPTConfig`'s default. Reconstruct config from
  `ckpt["config"]` and fall back to defaults for anything absent.
- Quality reference: final train loss ~1.25–1.30, val loss 1.3953 at iter 3000,
  still descending at the end. It writes coherent TinyStories-style English.

---

## 2. Non-negotiable design principles

1. **A reference implementation is sacred.** `reference.py` holds the naive,
   obviously-correct, unoptimized generate loop, ported verbatim from Project A.
   It is never optimized. Every optimization is validated against it.
2. **Every optimization must be measured, and the measurement must be
   reproducible.** No claim in the README without a script in `bench/` that
   regenerates it and a JSON artifact in `bench/results/`.
3. **Correctness before speed, always.** A faster wrong engine is worth nothing.
   Each phase's tests must pass before its benchmarks are believed.
4. **Be honest about hardware.** Project A's README earned credibility by stating
   plainly that the requested 2×T4 never materialized. Hold that standard. If an
   optimization doesn't help, or only helps on one GPU, publish that.
5. **CPU-testable core.** All correctness tests must run on CPU in CI with a tiny
   random-init config. GPU tests are marked and skipped when unavailable.
6. **No secret ever enters the repo.** The HF token lives in a GitHub secret and
   the local environment only. Tests must not require it (see §9).

---

## 3. Hardware reality

Available (both free tier, both already wired up from Project A):

| | Kaggle | Lightning AI |
|---|---|---|
| GPU actually granted | Tesla P100 (observed on every API-triggered run) | T4 |
| Compute capability | sm60 (Pascal) | sm75 (Turing) |
| fp16 tensor cores | **no** — fp16 runs at ~fp32 speed | yes |
| int8 tensor cores | no | yes |
| bf16 | no | no |
| FlashAttention-2 | no (needs Ampere+) | no (needs Ampere+) |
| Triton | **likely unsupported** (`tl.dot` needs sm70+) | yes |
| memory bandwidth | 732 GB/s | 320 GB/s |
| bs=1 decode floor (102.4 MB weight read) | 0.14 ms/tok (~7150 tok/s) | 0.32 ms/tok (~3125 tok/s) |

Consequences you must design around:

- **All Triton work targets the T4 (Lightning AI).** Verify Triton actually
  compiles on the granted GPU before investing in kernels; if Kaggle hands you a
  P100, Triton phases must be skipped there, not debugged there.
- **P100 has 2.3x the bandwidth of the T4 but no fp16 tensor cores.** For
  memory-bound bs=1 decode, the P100 may genuinely be *faster* than the T4. For
  batched/compute-bound decode, the T4 should pull ahead. **This is a real,
  publishable experiment** — run the same benchmark on both and explain the
  crossover. It is the single most interesting hardware result available here.
- Kaggle also enforces a 12h/session cap and 30 GPU-h/week; Lightning bills
  credits per second. Benchmarks are minutes, so neither is a real constraint —
  unlike Project A. Do not port Project A's orchestrator. The one exception is
  training a draft model (Phase 6), which is a real training job.

---

## 4. Target repository layout

```
mini-gpt-inference/
├── README.md                       # results-forward; written last, updated per phase
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .github/workflows/ci.yml
├── docs/
│   ├── PLAN.md                     # this file
│   ├── CHECKLIST.md                # ordered, tickable execution list
│   ├── ARCHITECTURE.md             # written in Phase 3, once shapes settle
│   ├── BENCHMARKS.md               # results tables, appended each phase
│   └── img/                        # generated plots (committed)
├── src/minigpt_infer/
│   ├── __init__.py
│   ├── config.py                   # GPTConfig, EngineConfig, SamplingParams
│   ├── model.py                    # cache-aware GPT
│   ├── reference.py                # naive oracle — NEVER optimize
│   ├── loader.py                   # HF Hub -> nn.Module
│   ├── tokenizer.py                # tiktoken gpt2 + incremental detokenizer
│   ├── batch.py                    # ForwardBatch / attention metadata
│   ├── sampling.py                 # batched greedy/temp/top-k/top-p/rep-penalty
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py                 # KVCacheBase protocol
│   │   ├── static.py               # contiguous per-sequence cache
│   │   └── paged.py                # block pool + BlockManager + block tables
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── sdpa.py                 # SDPA prefill + decode (gather path)
│   │   └── triton_paged.py         # Triton paged decode kernel
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── request.py              # Request, SequenceState, RequestOutput
│   │   ├── scheduler.py            # continuous batching + preemption
│   │   ├── engine.py               # LLMEngine.step()
│   │   ├── async_engine.py         # thread + queues for the server
│   │   └── spec_decode.py          # speculative decoding
│   ├── quant/
│   │   ├── __init__.py
│   │   └── int8.py                 # W8A16 weight-only
│   ├── graphs.py                   # CUDA graph capture/replay for decode
│   └── server/
│       ├── __init__.py
│       ├── api.py                  # FastAPI, OpenAI-compatible
│       ├── protocol.py             # pydantic schemas
│       └── metrics.py              # Prometheus
├── bench/
│   ├── common.py                   # timing harness (see §10 — use everywhere)
│   ├── bench_kvcache.py
│   ├── bench_batching.py
│   ├── bench_paged.py
│   ├── bench_kernels.py
│   ├── bench_quant.py
│   ├── bench_specdec.py
│   ├── bench_hardware.py           # T4 vs P100 comparison
│   ├── load_test.py                # Poisson load generator against the server
│   ├── plot.py
│   └── results/                    # JSON artifacts, committed
├── tests/
├── scripts/
│   ├── download_checkpoint.py
│   └── run_all_benchmarks.py
└── notebooks/
    ├── kaggle_bench.ipynb
    └── lightning_bench.ipynb
```

---

## 5. Core abstractions (design these first, in Phase 0/1)

Getting these interfaces right up front prevents a painful rewrite at Phase 3.
**Introduce `ForwardBatch` in Phase 1 even though paged fields stay `None` until
Phase 3.** The model signature must not change again after Phase 1.

```python
# src/minigpt_infer/batch.py
@dataclass
class ForwardBatch:
    """Everything the model needs for one forward pass, prefill or decode."""
    input_ids: Tensor            # prefill: (B, T);  decode: (B, 1)
    position_ids: Tensor         # same shape as input_ids. ABSOLUTE positions.
    is_prefill: bool
    cache: KVCacheBase | None = None
    attn_mask: Tensor | None = None      # (B, 1, Tq, Tk) additive float mask, or None
    # --- paged fields, unused until Phase 3 ---
    block_tables: Tensor | None = None   # (B, max_blocks) int32, -1 = unallocated
    slot_mapping: Tensor | None = None   # (B*T,) int32, flat write index into pool
    seq_lens: Tensor | None = None       # (B,) int32, tokens currently in cache per seq
```

```python
# src/minigpt_infer/cache/base.py
class KVCacheBase(Protocol):
    def write(self, layer_idx: int, k: Tensor, v: Tensor, batch: ForwardBatch) -> None: ...
    def read(self, layer_idx: int, batch: ForwardBatch) -> tuple[Tensor, Tensor]: ...
```

```python
# src/minigpt_infer/config.py
@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0      # 0.0 => greedy
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    n: int = 1
```

Model forward becomes `model(batch: ForwardBatch) -> Tensor  # logits`.
For decode it must return only the last position's logits, shape `(B, vocab)`.

---

## 6. Predictions to validate

State these **before** measuring, then report measured-vs-predicted. Being wrong
in public with an explanation is worth more than a table of unexamined numbers.

| # | prediction | reasoning |
|---|---|---|
| P1 | Naive generate is O(n²) in tokens; cached is O(n). At 512 tokens expect **10–40x** wall-clock speedup. | Naive recomputes ~131k token-forwards vs 512. Wall-clock gain < FLOP gain because naive batches each step into one big kernel. |
| P2 | bs=1 decode with plain PyTorch lands at **5–15 ms/token**, vs a 0.32 ms floor on T4 → **>90% of wall clock is overhead, not math**. | ~100+ kernel launches/step at µs each, plus Python dispatch. |
| P3 | CUDA graphs / `torch.compile(mode="reduce-overhead")` give **3–10x** on bs=1 decode — the largest single win in the project. | It removes exactly the overhead P2 identifies. |
| P4 | Throughput scales near-linearly with batch size to **bs≈64–128**, then bends. | Ridge point is ~203 FLOP/byte on T4; below that, extra batch is nearly free. |
| P5 | Continuous batching beats static batching by **2–4x** on a workload with high output-length variance, and by ~0 on uniform lengths. | Static batches idle until the longest sequence finishes. |
| P6 | Paged attention costs a few % vs static caching at this scale, and buys ~0 in capacity. | Gather/indirection overhead is real; 8 MiB/seq means capacity was never binding. |
| P7 | Weight-only int8 with a **naive dequant-then-matmul is slower** than fp16. Only a fused dequant GEMV kernel wins. | Naive dequant materializes an fp16 copy — strictly more traffic than just reading fp16. |
| P8 | Speculative decoding with prompt-lookup gets acceptance α≈0.3–0.6 on TinyStories. | The corpus is small-vocabulary and highly repetitive; n-gram hits should be frequent. |
| P9 | **P100 beats T4 at bs=1 decode; T4 beats P100 at large batch.** | P100 has 2.3x bandwidth (wins memory-bound) but no fp16 tensor cores (loses compute-bound). |

---

## 7. Phases

Each phase: **do not start the next phase until this phase's acceptance criteria
pass.** Commit at each phase boundary with benchmarks recorded.

---

### Phase 0 — Foundation and the correctness oracle

**Goal**: a repo that loads the real checkpoint and reproduces Project A's
generation exactly, with CI green.

**Files**: `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`,
`src/minigpt_infer/{config,model,reference,loader,tokenizer}.py`,
`scripts/download_checkpoint.py`, `tests/test_loader.py`, `tests/test_reference.py`

**Details**:

1. **Port the model.** Copy Project A's `model.py` verbatim first, commit that,
   *then* modify. This gives a clean diff showing exactly what caching changed —
   valuable for the writeup. Add a header comment crediting Project A.
2. **`reference.py`** holds Project A's `generate()` unchanged (no KV cache, full
   re-forward each step, `idx[:, -block_size:]` cropping). Add a module docstring:
   *"Correctness oracle. Never optimize this file."*
3. **Loader** (`loader.py`):
   - `download_checkpoint(repo_id, filename, token) -> Path` via `hf_hub_download`.
   - Try `torch.load(..., weights_only=True)` first; fall back to
     `weights_only=False` with a logged warning. (The checkpoint holds only
     tensors and primitive dicts, so `True` may work — but verify, don't assume.)
   - Defensively strip `module.` and `_orig_mod.` key prefixes.
   - Rebuild `GPTConfig` from `ckpt["config"]`, defaulting missing keys.
   - **Assert weight tying survived**:
     `assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()`
   - Assert `load_state_dict(..., strict=True)` returns no missing/unexpected keys.
4. **Tokenizer**: `tiktoken.get_encoding("gpt2")`. EOT is 50256. Note the
   vocab/embedding mismatch: embedding rows 50257–50303 are trained-but-unreachable
   padding. Sampling must never emit them — **mask logits ≥ 50257 to -inf**. This
   is a real bug source; the model can otherwise emit an id tiktoken can't decode.

**Pitfalls**:
- `weights_only=False` on an untrusted checkpoint is arbitrary code execution.
  Fine here (you trained it), but say so in a comment.
- Don't let `.pt` files into git. `.gitignore` already covers it — verify.

**Acceptance**:
- [ ] `python -c "from minigpt_infer.loader import load_model; load_model()"` works.
- [ ] Greedy generation from the fixed prompt `"Once upon a time"` produces
      readable English, and the token ids are recorded as a golden file.
- [ ] `tests/` pass on CPU with a tiny random-init config; CI green.
- [ ] Logit-parity test: tiny random model, fp32, CPU — `reference` forward
      matches `model` forward to `atol=1e-5`.

---

### Phase 1 — KV cache

**Goal**: O(n) decode that is *bit-comparable* to the oracle.

**Files**: `cache/base.py`, `cache/static.py`, `batch.py`, modify `model.py`,
`bench/bench_kvcache.py`, `tests/test_cache.py`

**Details**:

1. `StaticKVCache` allocates, per layer, `k`/`v` of shape
   `(B, n_head, max_len, head_dim)` and tracks per-sequence lengths.
2. Split the forward into **prefill** (T>1, writes T slots) and **decode**
   (T=1, appends 1 slot, attends over `seq_len` keys).
3. Position handling: `position_ids` is *absolute*. At decode step for a sequence
   that has consumed `L` tokens, the new token's position is `L`. Never `0`.
4. Assert `position_ids.max() < block_size` — the learned table has exactly 512
   rows and indexing past it silently errors or wraps depending on path.

**Pitfalls — read this twice**:

- **`is_causal=True` is WRONG for decode.** PyTorch's SDPA aligns the causal mask
  to the **top-left** when `q_len != kv_len`. With `q_len=1, kv_len=L`, that masks
  everything except position 0 — catastrophically wrong, and it does not crash.
  Use `is_causal=False` for decode (a single query legitimately attends to all
  cached keys). Use `is_causal=True` only for prefill into an *empty* cache where
  `q_len == kv_len`.
- You cannot pass both `is_causal=True` and `attn_mask` to SDPA. For padded or
  chunked prefill, build one combined additive mask (causal ∧ padding) yourself.
- Writing into a cache slice returns a view, not a copy — ensure writes land in
  the backing tensor (`cache.k[:, :, pos] = k`), not a detached temporary.

**Benchmark** (`bench_kvcache.py`): generate N∈{16,32,64,128,256,512} tokens,
naive vs cached, bs=1. Record ms/token and total latency. Plot both curves; the
naive one should visibly bend upward.

**Acceptance**:
- [ ] **Exact greedy match**: for ≥20 prompts, cached greedy token ids ==
      reference greedy token ids, element for element. Not "similar" — identical.
- [ ] Logit closeness: max abs diff cached vs reference < `1e-4` (fp32) / `2e-2` (fp16).
- [ ] P1 validated or refuted with numbers in `docs/BENCHMARKS.md`.

---

### Phase 2 — Static batching

**Goal**: many sequences per forward; find the throughput knee.

**Files**: modify `model.py`/`sampling.py`, `bench/bench_batching.py`,
`tests/test_batching.py`

**Details**:

1. **Use left padding.** With left padding all sequences end at the same index, so
   decode appends at one shared offset. Right padding forces per-sequence offsets
   and is much harder to get right with a shared cache.
2. Positions with left padding: `position_ids = (mask.cumsum(-1) - 1).clamp(min=0)`.
   Pad positions are junk but masked out; still clamp so you never index -1.
3. Padding mask: additive `-inf` (use `torch.finfo(dtype).min`, not `float("-inf")`,
   to avoid NaN when a row is fully masked) on pad key columns, combined with the
   causal mask into one `(B, 1, Tq, Tk)` tensor.
4. **Batched sampling** in `sampling.py`: top-k/top-p must operate per-row on a
   `(B, vocab)` tensor without a Python loop. Sort-based top-p; `torch.topk` for
   top-k. Per-request differing params: either group by params or apply the union
   and mask — document the choice.

**Pitfalls**:
- A fully-masked row (all-pad) produces NaN after softmax and silently poisons the
  batch. Never construct one; assert no row is entirely masked.
- Sampling from padded/finished slots wastes compute and can emit garbage into a
  finished sequence's output. Track a per-sequence `finished` flag and ignore them.

**Benchmark**: throughput (tok/s) and per-request latency vs
bs∈{1,2,4,8,16,32,64,128,256}. Produce the classic throughput-vs-latency tradeoff plot.

**Acceptance**:
- [ ] **Batch invariance**: each sequence's output in a batch of 8 is identical to
      running it alone (greedy). This is the test that catches every padding and
      mask bug. Do not skip it.
- [ ] P4 validated/refuted with the knee identified.

---

### Phase 3 — Paged KV cache + continuous batching ★ flagship

**Goal**: the vLLM core mechanism, from scratch, measured.

**Files**: `cache/paged.py`, `engine/{request,scheduler,engine}.py`,
`bench/bench_paged.py`, `tests/test_paged.py`, `tests/test_scheduler.py`,
`docs/ARCHITECTURE.md`

**Details**:

1. **Block pool**: per layer, `(num_blocks, block_size, n_head, head_dim)`.
   `block_size` = 16 tokens (tune later). Size `num_blocks` from a configured
   cache budget, not "all free memory" — deterministic benchmarks need fixed sizing.
2. **BlockManager**: free-list allocator. `allocate(seq, n_tokens)`,
   `append_slot(seq)` (grows by a block when the last is full), `free(seq)`.
   Track fragmentation stats for the writeup.
3. **`block_tables`**: `(B, max_blocks)` int32, `-1` for unallocated.
   **`slot_mapping`**: for each new token, `block_id * block_size + offset`;
   writing is a scatter: `k_pool.view(-1, H, D)[slot_mapping] = k_new`.
4. **Two read paths, both required**:
   - (a) *gather + SDPA* — build a dense `(B, H, max_len, D)` from block tables,
     mask past `seq_len`, run SDPA. Simple, correct, the reference for (b).
   - (b) the Triton kernel in Phase 4, validated against (a).
5. **Scheduler**, one `step()`:
   1. Admit waiting requests if blocks are free and `len(running) < max_batch`.
   2. Run one forward (prefill-priority: if any admitted this step, prefill them;
      else decode the running set).
   3. Sample; append tokens; incrementally detokenize; emit deltas.
   4. Check finish (EOS / `max_tokens` / stop strings); free blocks immediately.
   5. If allocation fails mid-flight: **preempt by recompute** — evict the newest
      sequence, free its blocks, return it to `waiting`. Count preemptions.

**Pitfalls**:
- Freeing blocks late is the #1 cause of phantom OOM. Free in the same step the
  sequence finishes.
- Off-by-one between `seq_len` (tokens in cache) and `position_ids` (next
  position). Write down the invariant and assert it every step:
  `next_position == seq_len`.
- Stop-string detection must run on *decoded text*, not token ids, and must handle
  a stop string split across two tokens.

**Benchmark**: static vs continuous batching on (a) uniform output lengths and
(b) high-variance lengths (lognormal, σ large). Report throughput, mean queue
wait, GPU idle fraction. The gap should appear only in (b) — that's the whole point.

**Acceptance**:
- [ ] Paged output == static-cache output, exactly, greedy, for ≥20 prompts.
- [ ] BlockManager unit tests: allocate/free/fragment/exhaust; **no leaked blocks
      after 1000 randomized alloc/free cycles** (assert free-list size returns to initial).
- [ ] A preemption is actually exercised in a test (set a tiny pool and force it).
- [ ] P5 and P6 validated/refuted.

---

### Phase 4 — Kernel and launch-overhead optimization ★ biggest expected win

**Goal**: attack the overhead identified in P2.

**Files**: `graphs.py`, `attention/triton_paged.py`, `bench/bench_kernels.py`,
`tests/test_triton.py`

**Details**:

1. **Establish the overhead baseline first.** Profile one decode step with
   `torch.profiler` and record: number of kernel launches, total kernel time,
   wall time. The gap between kernel time and wall time *is* the prize. Put this
   number in the writeup — it motivates everything else in this phase.
2. **`torch.compile`**: try `mode="reduce-overhead"` (which uses CUDA graphs
   internally) and `mode="max-autotune"`. Bucket batch sizes to avoid recompiles;
   log `torch._dynamo` recompile reasons and drive them to zero for steady state.
3. **Manual CUDA graphs** for the decode step — more control, and it teaches the
   real constraint:
   - Every input must live at a **fixed address**. Pre-allocate static buffers,
     `copy_()` real data in, `replay()`, read from the static output buffer.
   - Capture one graph per batch-size bucket {1,2,4,8,16,32,...}; pad to bucket.
   - Warm up on a side stream before capture (cuBLAS/cuDNN lazily allocate
     workspaces; capturing that allocation poisons the graph).
   - The KV pool must be captured by address — never reallocate it after capture.
4. **Triton paged decode attention** (T4 only): grid `(num_seqs, num_heads)`; each
   program loads its query vector, walks the block table, and accumulates with
   **online softmax** (FlashAttention-style running max/sum) for fp16 stability.
   Note: decode attention is a GEMV, not a GEMM — an elementwise-multiply +
   reduction is simpler and likely faster than forcing `tl.dot`. Try both.
5. Optional: fused residual+LayerNorm kernel. Lower value; do only if time allows.

**Pitfalls**:
- Benchmarking a CUDA graph without `torch.cuda.synchronize()` measures the
  `replay()` call returning, not the work. See §10.
- `torch.compile` silently falling back to eager per-step wipes out the gain and
  looks like "compile didn't help." Verify with `TORCH_LOGS=recompiles`.
- Triton on sm60 (Kaggle P100) will likely fail to compile. Detect capability at
  import and degrade to the SDPA path rather than crashing.

**Acceptance**:
- [ ] Triton kernel matches gather+SDPA within fp16 tolerance, and greedy token
      sequences are identical.
- [ ] Kernel-launch count per decode step recorded before and after.
- [ ] P2 and P3 validated/refuted with profiler evidence, not vibes.

---

### Phase 5 — Quantization

**Goal**: W8A16 weight-only int8, honestly evaluated.

**Files**: `quant/int8.py`, `bench/bench_quant.py`, `tests/test_quant.py`

**Details**:

1. Per-output-channel symmetric int8:
   `scale = W.abs().amax(dim=1) / 127; W_q = (W / scale[:, None]).round().to(int8)`.
2. Quantize the big Linears (`qkv`, `attn.proj`, `mlp.fc`, `mlp.proj`). **Leave
   `lm_head`/`tok_emb` in fp16** — it's tied, it's half the parameters, and
   quantizing an embedding table you also read as weights is a separate problem.
   Document that choice.
3. **A naive dequant-then-matmul will be slower than fp16 (P7).** Measure it
   anyway and publish the negative result — then write the fused Triton dequant
   GEMV that actually wins, and show both. The contrast is the interesting part.
4. Quality: perplexity on a held-out TinyStories slice, fp16 vs int8. Target
   <1% relative degradation.

**Acceptance**:
- [ ] Round-trip test: dequantized weights within expected error of the original.
- [ ] Memory reduction measured (expect ~102 MB → ~55 MB given fp16 `lm_head`).
- [ ] Perplexity delta reported. P7 validated/refuted.

---

### Phase 6 — Speculative decoding

**Goal**: >1 token per target forward, with a *proof* of distribution preservation.

**Files**: `engine/spec_decode.py`, `bench/bench_specdec.py`, `tests/test_spec_decode.py`

**Details** — implement in this order:

1. **Prompt-lookup / n-gram decoding first** (no draft model, zero training):
   search the current context for the last n tokens; if found, propose the
   following k tokens as the draft. TinyStories is repetitive, so hit rate should
   be real (P8). This gets the whole verification path working for free.
2. **Self-speculative (layer skip)**: run the first 4 of 8 layers as the draft.
   Free, and an interesting quality/acceptance datapoint.
3. **Trained draft model** (stretch): a 2L/2H/128d GPT on TinyStories. **Reuse
   Project A's `train.py` unchanged** — it already takes `--n_layer/--n_head/
   --n_embd`. Nice cross-project link; ~1 GPU-hour.

**The algorithm** (Leviathan et al. / Chen et al.) — implement exactly:
1. Draft proposes γ tokens `x_1..x_γ` with probabilities `q(x_i)`.
2. Target runs **one** forward over `[prefix, x_1..x_γ]`, yielding `p` at γ+1 positions.
3. For i in 1..γ: accept `x_i` with probability `min(1, p(x_i)/q(x_i))`.
   On first rejection, sample from `normalize(max(0, p - q))` and stop.
4. If all γ accepted, sample a bonus token from `p` at position γ+1.

**Correctness — this is the phase's real contribution**:
- **Greedy exactness**: at `temperature=0`, speculative output must be *identical*
  to target-only greedy. Clean, deterministic, no statistics needed. Make this a
  hard test.
- **Distributional**: at `temperature=1`, sample the next token N≥100k times with
  and without speculation and run a chi-square test over the token histogram. It
  must not reject at α=0.01. This empirically demonstrates the theorem that
  speculative decoding is distribution-preserving — most projects claim it and
  never check.

**Benchmark**: acceptance rate α and mean accepted length vs γ∈{1..8}; wall-clock
speedup vs γ. Compare measured tokens/step against the theoretical
`(1 - α^(γ+1)) / (1 - α)`.

**Acceptance**:
- [ ] Greedy exactness test passes.
- [ ] Chi-square distributional test passes.
- [ ] α, mean accepted length, and speedup-vs-γ curve recorded. P8 validated/refuted.

---

### Phase 7 — Serving layer

**Goal**: an OpenAI-compatible streaming server on top of the engine.

**Files**: `server/{api,protocol,metrics}.py`, `engine/async_engine.py`,
`tests/test_server.py`

**Details**:

1. Endpoints: `POST /v1/completions`, `POST /v1/chat/completions`, `GET /v1/models`,
   `GET /health`, `GET /metrics`. Pydantic schemas mirroring the OpenAI shapes.
2. **Chat endpoint honesty**: this model is a base model, not instruction-tuned.
   Provide the endpoint for API compatibility, apply a trivial template, and say
   plainly in the README that chat quality is not the point.
3. **SSE streaming**: `data: {json}\n\n` per delta, terminated by `data: [DONE]\n\n`.
4. **Concurrency architecture**: run `LLMEngine.step()` in a **dedicated thread**,
   not the asyncio loop. Each request gets a thread-safe queue; the engine thread
   pushes deltas, the endpoint coroutine drains them. (vLLM interleaves in one
   loop; a separate thread is simpler to get right from scratch and avoids
   starving the event loop during a long prefill.)
5. **Incremental detokenization** — a genuine trap. Do **not** re-decode the whole
   sequence each step (O(n²), and byte-level BPE can split a multi-byte UTF-8
   character across tokens, producing `�`). Keep a decoded-prefix offset; decode a
   small trailing window and diff; hold back bytes that don't yet form a complete
   character. Unit-test with a prompt that generates multi-byte characters.
6. **Prometheus metrics**: `ttft_seconds` (histogram), `tpot_seconds` (histogram),
   `e2e_seconds`, `running_requests` / `waiting_requests` (gauges),
   `kv_cache_usage_ratio`, `tokens_generated_total`, `preemptions_total`,
   `spec_accept_rate`, `request_total{status}`.

**Acceptance**:
- [ ] `tests/test_server.py` uses FastAPI `TestClient` against a tiny random-init
      model — runs in CI, no GPU, no checkpoint.
- [ ] Streaming and non-streaming return identical final text for the same seed.
- [ ] Client-cancel mid-stream frees the sequence's KV blocks (test it — this leaks
      in naive implementations).
- [ ] `/metrics` scrapes cleanly.

---

### Phase 8 — Benchmark harness and the hardware study

**Goal**: publication-quality numbers.

**Files**: `bench/common.py`, `bench/load_test.py`, `bench/bench_hardware.py`,
`bench/plot.py`, `scripts/run_all_benchmarks.py`, `docs/BENCHMARKS.md`

**Details**:

1. **Load generator**: Poisson arrivals at rate λ; prompt and output lengths drawn
   from a configurable distribution (sample real TinyStories prompts). Record
   per-request TTFT / TPOT / E2E.
2. Sweep λ; plot latency percentiles vs QPS — the classic knee. Report **max
   sustainable QPS under an SLO** (e.g. TTFT p95 < 200 ms AND TPOT p95 < 50 ms).
   Define the SLO explicitly; a "goodput" number without a stated SLO is meaningless.
3. **The T4-vs-P100 study (P9)**: run the identical suite on both, and explain the
   crossover in terms of bandwidth vs tensor cores. This is the most distinctive
   result available in this project — give it its own section.
4. Every benchmark writes JSON to `bench/results/<name>_<gpu>_<timestamp>.json`
   including: git SHA, torch/CUDA version, GPU name, all config, raw samples.
   `plot.py` reads only those JSONs. No number reaches the README except through
   this pipeline.

**Acceptance**:
- [ ] `python scripts/run_all_benchmarks.py` regenerates every published number.
- [ ] Every README figure traces to a committed JSON.
- [ ] P9 validated/refuted.

---

### Phase 9 — Documentation and writeup

**Files**: `README.md`, `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, `docs/img/*`

**README structure** (results-forward, following Project A's tone):
1. One-paragraph what/why + headline result table.
2. Architecture diagram (request → scheduler → paged cache → model → sampler).
3. Optimization ladder table: naive → +KV → +batching → +paged/continuous →
   +CUDA graphs → +int8 → +spec-decode, with tok/s and latency at each rung.
4. The hardware study (T4 vs P100 crossover).
5. **Predictions vs measurements** (§6) — including the ones you got wrong.
6. What's *not* implemented and why (see §12). Name the limits explicitly.
7. Reproduction instructions.

---

## 8. Testing strategy — the correctness ladder

Each rung catches a specific class of bug. Implement them as you reach each phase.

| rung | test | catches |
|---|---|---|
| 1 | reference vs model logits, tiny fp32 CPU | porting errors |
| 2 | cached greedy == reference greedy, exact ids | KV cache indexing, causal-mask misuse |
| 3 | batch invariance (in-batch == alone) | padding, position_ids, mask bugs |
| 4 | paged == static, exact | block tables, slot mapping |
| 5 | Triton == gather+SDPA | kernel bugs |
| 6 | 1000 randomized alloc/free cycles leak no blocks | allocator leaks |
| 7 | spec-decode greedy == target greedy, exact | verification-loop bugs |
| 8 | spec-decode chi-square vs target sampling | rejection-sampling math |
| 9 | int8 perplexity delta within threshold | quantization quality |
| 10 | streaming == non-streaming text | detokenizer bugs |
| 11 | cancel mid-stream frees blocks | resource leaks |

Conventions: `@pytest.mark.gpu` for anything needing CUDA, auto-skipped when
unavailable. `@pytest.mark.slow` for >10s tests, excluded from default CI.
Seed everything. Prefer exact-equality assertions where the math permits them —
`allclose` hides the bugs that matter most here.

---

## 9. CI plan

GitHub Actions, CPU-only runners. Copy Project A's working CI and extend it.

- **CI must never need the HF checkpoint or any secret.** All tests construct a
  tiny random-init model (e.g. 2L/2H/64d, `vocab_size=512`, `block_size=64`).
  Downloading a 614 MB private checkpoint in CI would be slow, secret-dependent,
  and pointless.
- Jobs: `ruff check .` → `pytest -m "not gpu and not slow"` → build check.
- Python 3.11. Pin torch to a CPU wheel in CI to keep installs fast.
- **Known trap inherited from Project A**: flat-layout `setuptools` auto-discovery
  breaks once there are multiple top-level directories. This repo uses a
  `src/` layout, which avoids it — keep it that way, and set
  `[tool.setuptools.packages.find] where = ["src"]`.
- Optional later: a self-hosted or manually-triggered GPU workflow for
  `-m gpu`. Not required.

---

## 10. Benchmark methodology — rules, not suggestions

Put these in `bench/common.py` and use it everywhere. A benchmark harness written
once and reused is itself a signal of engineering maturity.

1. **Warm up** ≥10 iterations before timing (CUDA context, autotuning, compile).
2. **`torch.cuda.synchronize()`** immediately before starting and before stopping
   the timer. CUDA is async; without this you are timing kernel *launches*.
3. Use `time.perf_counter()`. Report **median and IQR**, not mean — a single
   scheduler hiccup skews the mean.
4. ≥30 repeats for latency; ≥3 independent runs for throughput.
5. Fixed seeds; identical prompts across compared configurations.
6. Record in every JSON: git SHA, torch version, CUDA version, GPU name, driver,
   full config, and **all raw samples** (not just the summary).
7. Report `torch.cuda.max_memory_allocated()` alongside latency; a speedup bought
   with 3x memory is a different result.
8. Never compare numbers across GPUs without labeling the GPU. Never compare
   across code versions without labeling the SHA.

---

## 11. Metrics glossary (use these definitions exactly)

- **TTFT** — time to first token: request arrival → first token emitted. Includes
  queue wait and prefill.
- **TPOT** — time per output token: mean inter-token latency during decode,
  excluding the first.
- **E2E latency** — arrival → final token.
- **Throughput** — system-wide output tokens/second across all concurrent requests.
- **Goodput** — requests/second that satisfy a *stated* SLO. Always state the SLO.
- **Acceptance rate (α)** — fraction of speculative draft tokens accepted.
- **KV cache utilization** — allocated blocks / total blocks.

---

## 12. Explicit non-goals (state these in the README)

Naming what you deliberately did not build is a credibility signal, not a weakness.

- **Tensor/pipeline parallelism** — a 102 MB model on a 16 GB card needs neither.
- **FlashAttention-2** — requires Ampere+; unavailable on T4/P100.
- **Multi-node serving** — out of scope.
- **Prefix caching / RadixAttention** — high value for shared system prompts;
  our workload has none. Note as future work.
- **Chunked prefill** — worth it for long prompts; ours cap at 512 tokens.
- **FP8** — requires Hopper.
- **Instruction tuning / chat quality** — that's Project C.

---

## 13. Effort estimate and critical path

| phase | estimate | on critical path? |
|---|---|---|
| 0 Foundation | 0.5 day | yes |
| 1 KV cache | 1 day | yes |
| 2 Static batching | 0.5 day | yes |
| 3 Paged + continuous ★ | 2–3 days | yes |
| 4 Kernels/CUDA graphs ★ | 2–3 days | yes |
| 5 Quantization | 1–2 days | no |
| 6 Speculative decoding | 2 days | no |
| 7 Serving | 1–2 days | yes |
| 8 Benchmarks + hardware study | 1–2 days | yes |
| 9 Docs/writeup | 1 day | yes |

**Minimum coherent project** (if time is short): 0 → 1 → 2 → 3 → 4 → 8 → 9.
That alone is a strong portfolio piece: paged attention, continuous batching, and
CUDA graphs, all measured. Phases 5, 6, 7 are high-value additions, in that order
of technical interest: **6 (speculative) > 7 (serving) > 5 (quantization)**.

---

## 14. Risks and open questions

| risk | mitigation |
|---|---|
| Triton won't compile on Kaggle's P100 (sm60) | Detect capability at import; fall back to SDPA. Do all Triton work on Lightning's T4. |
| `torch.compile` recompiles thrash with dynamic batch sizes | Bucket and pad batch sizes; verify with `TORCH_LOGS=recompiles`. |
| CUDA graph capture fails on lazy cuBLAS workspace allocation | Warm up on a side stream before capture. |
| Naive int8 is slower than fp16 (P7) | Expected. Publish the negative result, then fix it with a fused kernel. |
| Prompt-lookup acceptance too low to help | Fall back to self-speculative (layer skip) or the trained draft. |
| Private HF checkpoint blocks reproduction by others | Consider making the model repo public at Phase 9, or publish weights-only + a config. |
| Benchmarks contaminated by a noisy shared GPU | Report IQR; re-run outliers; note the platform in every JSON. |

---

## 15. Definition of done

- [ ] Every phase's acceptance criteria met, or explicitly waived in the README
      with a reason.
- [ ] CI green; `pytest -m "not gpu"` passes on CPU.
- [ ] `scripts/run_all_benchmarks.py` regenerates every published number.
- [ ] README leads with a results table where every figure traces to a committed
      JSON artifact.
- [ ] All nine predictions in §6 marked validated or refuted, **including the
      ones that were wrong**.
- [ ] Non-goals (§12) stated plainly.
- [ ] No secret, checkpoint, or generated binary committed.
