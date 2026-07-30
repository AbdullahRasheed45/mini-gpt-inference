# Benchmarks

Every number here traces to a committed JSON artifact under `bench/results/`
and was produced by `bench/common.py`'s harness (warmup, `torch.cuda.synchronize()`,
median/IQR over repeated runs, full environment stamp) per `docs/PLAN.md` §10.
Predictions are stated in `docs/PLAN.md` §6 *before* being checked here.

---

## Phase 1 — KV cache (Prediction P1)

**Prediction**: naive (no-cache) generate is O(n²) in tokens; cached is O(n).
At 512 tokens, expect **10–40x** wall-clock speedup.

**Setup**: `bench/bench_kvcache.py`, `GPTConfig()` defaults (8 layers, 8 heads,
512 embd, block_size=512 — the real Project A architecture, random-init
weights since latency doesn't depend on weight values), bs=1, greedy decode,
prompt_len=1. N=512 is clamped to 511 effective tokens so `prompt_len + N`
stays under `block_size`. 10 repeats, 3 warmup iterations, fixed seed.

### CPU results (Apple Silicon, `bench/results/bench_kvcache_cpu_20260730T091915Z.json`)

| N (tokens) | naive (median) | cached (median) | speedup |
|---:|---:|---:|---:|
| 16  |    237.9 ms |    172.7 ms | 1.38x |
| 32  |    478.9 ms |    334.7 ms | 1.43x |
| 64  |   1038.0 ms |    656.1 ms | 1.58x |
| 128 |   2341.5 ms |   1303.1 ms | 1.80x |
| 256 |   6124.6 ms |   2611.1 ms | 2.35x |
| 512 (511 eff.) | 17777.1 ms | 5183.1 ms | **3.43x** |

![naive vs cached latency, CPU](img/bench_kvcache_cpu.png)

**Verdict: P1 directionally confirmed, magnitude not yet met — pending a GPU
re-run.** The shape is exactly as predicted: naive visibly bends upward (it's
doing ~511 forward passes over a growing sequence, i.e. genuinely
super-linear work), cached stays close to linear. But the *measured* speedup
(3.4x at N=512) falls well short of the predicted 10–40x range.

**Why the CPU number undershoots the prediction**: P1's 10–40x range was
derived for GPU decode, where a single-token matmul (`M=1`) is *overhead-bound*
— its wall-clock cost is dominated by fixed per-kernel-launch cost, not by the
FLOPs it does, so shrinking the FLOPs (what caching buys you) barely moves the
needle... except caching *also* removes the O(n) recomputation of the
attention over the whole growing prefix, which is where the big GPU win comes
from. On CPU, single-core GEMM has its own large fixed per-call overhead
(thread-pool dispatch, no SIMD-friendly tall-and-thin shapes) that's actually
*proportionally larger* relative to the FLOPs it's saving, so part of the
predicted gain is "spent" on CPU-specific overhead in both branches equally —
consistent with the shape matching but the ratio being compressed. This is
exactly the phenomenon Prediction P2 names directly (bs=1 decode is
overhead-, not math-, dominated) — P1 and P2 are entangled, and P2 can only be
properly measured on the target GPU (T4), not CPU.

**Status**: keeping this CPU result as the honest, currently-reproducible
number (§10 rule 8: always label the hardware). A T4 re-run via Lightning AI
is planned for Phase 4/8 alongside the CUDA-graph and hardware-study work,
where P1 will be re-evaluated on the actual target hardware described in
`docs/PLAN.md` §3.

---

## Phase 2 — Static batching (Prediction P4)

**Prediction**: throughput scales near-linearly with batch size up to
bs≈64–128, then bends (the ridge point where compute goes from memory-bound to
compute-bound).

**Setup**: `bench/bench_batching.py`, same `GPTConfig()` architecture as
Phase 1, bs∈{1,2,4,8,16,32,64,128,256}, uniform-length prompts (8 tokens),
32 new tokens/request, greedy, 10 repeats, 3 warmup iterations.

### CPU results (Apple Silicon, 8 torch threads, `bench/results/bench_batching_cpu_20260730T094245Z.json`)

| bs | total latency (median) | throughput | throughput / bs (efficiency) |
|---:|---:|---:|---:|
| 1   |  341.1 ms |    93.8 tok/s | 93.8 |
| 2   |  616.1 ms |   103.9 tok/s | 52.0 *(noisy outlier — see below)* |
| 4   |  505.5 ms |   253.2 tok/s | 63.3 |
| 8   |  527.5 ms |   485.3 tok/s | 60.7 |
| 16  |  531.2 ms |   963.8 tok/s | 60.2 |
| 32  |  549.1 ms |  1864.8 tok/s | 58.3 |
| 64  |  685.0 ms |  2990.0 tok/s | 46.7 |
| 128 |  822.1 ms |  4982.3 tok/s | 38.9 |
| 256 | 1279.8 ms |  6401.1 tok/s | 25.0 |

![throughput vs batch size and the latency/throughput tradeoff, CPU](img/bench_batching_cpu.png)

**Verdict: P4 confirmed, on this hardware's own terms.** The raw
throughput-vs-batch-size curve (left panel) looks like it's still climbing at
bs=256, but that's the expected shape of *linear* scaling on a log-x axis —
the plot alone doesn't show the bend. The **efficiency column (throughput /
bs)** does: it's flat at ~58–63 tok/s per unit of batch from bs=4 through
bs=32 (extra batch is nearly free — total latency barely moves, 505ms →
549ms, while throughput goes up 7.4x), then drops monotonically from bs=64
onward (46.7 → 38.9 → 25.0) as this CPU's ~8-10 threads saturate and added
batch stops being free. The knee lands around **bs≈32–64** here — earlier
than the predicted bs≈64–128, which is expected given a CPU has an order of
magnitude fewer parallel lanes than a T4's SMs. bs=1 and bs=2 are noisy in
absolute terms (fixed Python/thread-pool startup cost dominates at that
scale, not the model's actual compute) and shouldn't be read as part of the
trend.

**Status**: qualitative shape confirmed (linear-then-bend); the *exact* knee
location is hardware-specific and will be re-measured on T4 in Phase 8's
hardware study, per `docs/PLAN.md` §10 rule 8 (never compare bend points
across hardware without labeling it).

**Acceptance**: batch invariance verified in `tests/test_batching.py` — 8
ragged-length prompts generated together produce byte-identical greedy output
to running each one alone (after fixing a real per-row position-id bug this
test caught: decode position_ids were computed from the shared cache length,
which includes each row's own left-pad count, instead of that row's true
token count — see `generation.py`'s `pad_lens` offset).

---

## Phase 3 — Paged KV cache + continuous batching (Predictions P5, P6)

**Predictions**:
- P5: continuous batching beats static by **2–4x** on high-variance output
  lengths, and ~0 on uniform lengths.
- P6: paged attention costs a **few %** vs static caching at this scale, and
  buys ~0 in capacity.

**Setup**: `bench/bench_paged.py`, same `GPTConfig()` architecture, 32
requests, max_batch_size=8, prompt_len=4, greedy. Static = sequential
non-overlapping groups of 8 (the realistic naive deployment: each group waits
for its own longest member). Continuous = one `LLMEngine`, all 32 submitted
at once, admitted/decoded/replaced as slots free. Uniform workload: every
request wants 16 new tokens. High-variance: lengths drawn from
`lognormal(mu=ln(10), sigma=1.1)`, clamped to `[1, 64]` (observed range in
the run below: 2–64 tokens). 10 repeats, 3 warmup iterations.

### CPU results (`bench/results/bench_paged_cpu_20260730T100309Z.json`)

| workload | static (median) | continuous (median) | measured speedup | theoretical compute ceiling |
|---|---:|---:|---:|---:|
| uniform | 1050.9 ms | 1861.0 ms | **0.56x** | 1.00x |
| high_variance | 2577.7 ms | 2243.1 ms | **1.15x** | 2.27x |

**P6**: same uniform workload, single batch of 8 (everyone admitted at once,
nobody finishes early — isolates cache *implementation* overhead from
batching *policy*): static-cache 256.1 ms vs paged-cache 462.8 ms — paged is
**81% slower**, not "a few %".

**Verdict: P5 and P6 both refuted in magnitude on this hardware, but the
*mechanism* is validated by a number the naive comparison hides.**
`bench_paged.py` also computes the **theoretical compute ceiling**: total
(sequence, timestep) forward evaluations static performs (every group runs
`max(group)` steps at full batch width) divided by the number continuous
performs (exactly `sum(lengths)`, no waste) — ignoring all per-step engine
overhead. For the high-variance workload that ceiling is **2.27x**, squarely
inside the predicted 2–4x range. The *measured* speedup (1.15x) is far
below it, and the uniform workload is measurably **slower** under continuous
batching (0.56x) even though its ceiling is exactly 1.00x (no compute to
save by construction). Both facts point to the same cause: continuous
batching's own bookkeeping — Python-level per-request state, per-step block
table/slot mapping construction, one-sequence-at-a-time prefill (see
`docs/ARCHITECTURE.md` §5) — costs *more* wall-clock than this tiny model
saves by avoiding wasted compute, on a CPU, at this scale. This is the same
overhead-dominated-regime story as P1/P2: the theoretical win is real and
provably present (2.27x ceiling), but only shows up in wall-clock once the
useful compute per step is large enough (bigger model, real GPU, larger
batches) to amortize the engine's own fixed per-step cost. P6's 81% overhead
is consistent — `PagedKVCache`'s gather/indirection is genuinely more
expensive in Python+small-tensor terms than `StaticKVCache`'s direct slice,
and at this scale that cost isn't yet swamped by the actual attention
compute the way it would be on a real workload.

**Status**: correctness (not performance) was this phase's hard gate — see
Acceptance below, fully met. Performance is re-measured on the target T4 in
Phase 8's hardware study, where the compute-per-step is large enough for the
mechanism's real advantage to plausibly show through the overhead.

**Acceptance**:
- Paged output == static-cache output, exactly, greedy: verified for 20
  single-request prompts *and* 6 concurrent ragged-length requests with
  staggered admission (`tests/test_paged.py`). The concurrent case caught two
  real bugs beyond the single-request case (double-counting
  `num_computed_tokens` across prefill+decode, and a "mutate list while
  iterating" bug that silently dropped a request's token whenever another
  request in the same batch finished) — see `docs/ARCHITECTURE.md` §4.
- BlockManager: allocate/free/append-slot/exhaustion covered in
  `tests/test_scheduler.py`, including **1000 randomized alloc/free cycles
  with zero leaked blocks**.
- A preemption is genuinely exercised (not just unit-tested in isolation):
  `tests/test_paged.py::test_preemption_recovers_correct_output_under_a_tiny_pool`
  forces one mid-decode preemption via a deliberately undersized pool and
  verifies the preempted request's final output still matches the reference
  exactly after resuming — this is what caught the third bug, `_prefill_one`
  discarding a resumed sequence's already-generated tokens on re-admission.

---

<!-- Phases 4-9 append their sections here, in order, as they complete. -->
