# Architecture

Written in Phase 3, once the core shapes settled (per `docs/PLAN.md`'s repo
layout comment) and updated at each phase boundary from here on as new
components land. This is the "how does it fit together" companion to
`docs/PLAN.md` (why) and `docs/BENCHMARKS.md` (what was measured).

---

## 1. The one interface everything is built around

```python
# src/minigpt_infer/batch.py
@dataclass
class ForwardBatch:
    input_ids: Tensor            # prefill: (B, T);  decode: (B, 1)
    position_ids: Tensor         # same shape as input_ids. ABSOLUTE positions.
    is_prefill: bool
    cache: KVCacheBase | None = None
    attn_mask: Tensor | None = None      # (B, 1, Tq, Tk) additive float mask, or None
    block_tables: Tensor | None = None   # (B, max_blocks) int, -1 = unallocated (paged only)
    slot_mapping: Tensor | None = None   # flat pool index per new token (paged only)
    seq_lens: Tensor | None = None       # (B,) tokens cached per seq, post-write (paged only)
```

`model.py`'s forward signature has been `model(batch: ForwardBatch) -> Tensor`
(logits at the last position only, `(B, vocab)`) since Phase 1 and **has not
changed since** — not even to add paged attention. `CausalSelfAttention`
only ever calls `batch.cache.write(...)` then `batch.cache.read(...)`; it has
no idea whether it's talking to a contiguous `StaticKVCache` or a
block-table-indexed `PagedKVCache`. This is the payoff of introducing the
paged-only fields in Phase 1 even though they stayed `None` until Phase 3.

The one thing `CausalSelfAttention` *does* branch on is whether `attn_mask`
is provided:

- `attn_mask is None` → plain SDPA `is_causal=(kv_len == q_len)`. Used by
  Phase 1's single/uniform-length decode.
- `attn_mask is not None` → explicit additive mask, `is_causal=False`. Used
  by Phase 2's padded batches and Phase 3's paged batches (padding-by-seqlen
  instead of padding-by-prompt-length).

**The pitfall this encodes** (docs/PLAN.md Phase 1): SDPA's `is_causal=True`
aligns the causal mask to the *top-left* when `q_len != kv_len` — a decode
step (`q_len=1`) would silently attend to only position 0 of a non-empty
cache. `is_causal=True` is only ever passed when `kv_len == q_len` (a fresh
prefill into an empty cache); every other case passes an explicit mask.

---

## 2. Two cache implementations, one protocol

```python
class KVCacheBase(Protocol):
    def write(self, layer_idx, k, v, batch) -> None: ...
    def read(self, layer_idx, batch) -> tuple[Tensor, Tensor]: ...
```

**`StaticKVCache`** (Phase 1-2): one contiguous `(max_batch_size, n_head,
max_seq_len, head_dim)` tensor per layer per k/v. Assumes every row in the
batch shares the same current length before each `write()` — true for
synchronous batched generation (every request advances one token per step
together), false for continuous batching.

**`PagedKVCache`** (Phase 3): a fixed `(num_blocks, block_size, n_head,
head_dim)` pool per layer per k/v, shared across *all* sequences. `write()`
scatters new tokens by absolute slot index (`block_id * block_size +
offset`, via `ForwardBatch.slot_mapping`). `read()` gathers a dense
`(B, n_head, max_len, head_dim)` tensor from each row's block table — this is
docs/PLAN.md's "(a) gather + SDPA" path; "(b)" is Phase 4's Triton kernel,
validated against this one.

Block *ownership* (which block ids belong to which sequence) is tracked
entirely separately, by `BlockManager` — a free-list allocator that never
touches a tensor. `PagedKVCache` doesn't know which sequence owns a block; it
just indexes whatever `block_tables`/`slot_mapping` it's given. This split is
why `tests/test_scheduler.py` can exercise 1000 randomized alloc/free cycles
without a model or even torch being loaded.

### Why `read()` can return garbage columns, safely

`PagedKVCache.read()` sizes its gather off `seq_lens.max()` across the whole
batch, so shorter rows get columns beyond their real length — sometimes
genuinely unwritten pool memory, sometimes leftover bytes from a different,
since-freed sequence that used to own that block. This is safe *only*
because the caller always supplies a per-row `attn_mask`
(`batch.build_paged_decode_mask`) that masks every column at or beyond that
row's own `seq_lens[i]`, using `torch.finfo(dtype).min` rather than literal
`-inf` (see docs/PLAN.md Phase 2 point 3) — even a query row that's entirely
inside a masked region resolves to a harmless uniform softmax, not NaN.

---

## 3. Batching: two eras

**Phase 2 (static, left-padded).** All requests in a call are known up
front, padded to the batch's max prompt length with `pad_token_id` at the
*front* (`batch.build_left_padded_batch`), decoded together for a fixed
number of steps. `position_ids` for the left-padded prefill are
`(real_mask.cumsum(-1) - 1).clamp(min=0)` — pad slots get position 0 (junk,
masked out anyway). The subtlety that cost a real bug during development:
**decode position_ids must subtract each row's own left-pad count** from the
shared cache length — the cache's `seq_lens` is uniform across the batch
(includes padding), but each row's *true* position excludes its own padding.
See `generation.py`'s `pad_lens` offset.

**Phase 3 (continuous, block-table gather).** Requests arrive and finish at
different times. There is no shared padding prefix — each sequence's cache
content starts at its own position 0. Padding here means "columns beyond
this row's own `seq_lens[i]` in the batch's gathered max-length tensor," not
a fixed prefix, and is entirely a decode-time concern (Phase 3's prefill
processes one sequence at a time — see §5).

---

## 4. The engine

```
src/minigpt_infer/engine/
├── request.py     # Request (immutable ask), SequenceState (mutable bookkeeping)
├── scheduler.py    # admission + preemption bookkeeping, no tensors
└── engine.py       # LLMEngine.step() -- the only place tensors and scheduling meet
```

### The one invariant that matters most

> `next_position == num_computed_tokens`, always.

`SequenceState.num_computed_tokens` is "tokens already written into the KV
cache." The token about to be written goes at exactly that position. Every
caller that writes N tokens into the cache must advance the counter by
exactly N — no more, no less — and `_run_decode` asserts this explicitly
every step. A real bug during development violated this without crashing:
`_sample_and_advance` (shared by both prefill and decode) unconditionally
incremented the counter by 1, on top of `_prefill_one` *already* having set
it to the full prompt length T. Every sequence's very first decode step
silently wrote to (and read from) position `T+1` instead of `T`, corrupting
one cache slot per sequence with no error, no crash — caught only by an
exact-match test against the Phase 1 static-cache reference, not by shape or
type checks.

### `Scheduler.step()`'s prefill-priority policy

One `LLMEngine.step()` is *either*:
- a **prefill** step: admit as many waiting requests as fit (blocks free
  and `max_batch_size` not exceeded), run each admitted request through its
  own single-sequence forward (see §5 for why one at a time), or
- a **decode** step (nothing was admitted this cycle): advance every
  currently-running sequence by exactly one token, batched together in one
  forward call regardless of how differently-progressed they are.

No step ever mixes prefill and decode in the same forward call — chunked
prefill is an explicit non-goal (docs/PLAN.md §12).

### Preemption (`ensure_capacity_for_decode_step`)

Before a decode step, every running sequence may need a fresh block (its
current tail block might be exactly full). If none are free,
**preempt-by-recompute**: evict the *newest* running sequence (least sunk
cost), free its blocks, return it to the front of `waiting`. When it's later
re-admitted, `_prefill_one` reprocesses `prompt + output_token_ids so far` —
**not just the original prompt** — since a preempted sequence has already
committed real output tokens that must be rebuilt into the cache before
generation can continue from where it left off. (This, too, was a real bug
during development: the original `_prefill_one` only ever looked at
`request.prompt_token_ids`, silently discarding a resumed sequence's prior
progress and requesting the wrong number of blocks for it.)

### A second, subtler bug this phase's tests caught

`LLMEngine.step()` passes `self.scheduler.running` into `_run_decode`, which
iterates it while sampling. `Scheduler.finish()` (called mid-iteration for
any sequence that just hit EOS/`max_tokens`) mutates that *same* list via
`list.remove()`. Iterating a list while removing from it mid-loop silently
skips whichever element comes right after the one just removed — a classic
Python footgun that here meant "whichever request happens to sit right
after one that finishes in the same decode batch never gets its token
sampled or returned that step." `step()` now passes `list(self.scheduler.running)`
— a snapshot — into `_run_decode`, so the live list can be safely mutated by
`finish()` without perturbing the batch already in flight.

---

## 5. Deliberate simplifications (see `engine.py`'s module docstring)

- **Prefill is one sequence at a time**, even when several are admitted in
  the same scheduler cycle. Batched/packed ragged prefill into a shared
  paged cache is real complexity (either over-allocate blocks to cover
  padding, defeating paged memory efficiency, or a varlen/packed attention
  path) that isn't needed to demonstrate this phase's actual point:
  continuous batching across heterogeneous-length *running* sequences during
  decode, which is fully batched. Chunked prefill is an explicit non-goal.
- **`SamplingParams.n` (multiple completions/request) is not supported** —
  asserted in `add_request`. **`.seed`** (per-request reproducible sampling)
  isn't threaded through either: batched multinomial sampling shares one RNG
  stream across the whole decode batch, so per-row-seeded sampling would
  need a Python-level loop, defeating the vectorization `sampling.py` exists
  for.
- **Stop-string detection** checks the full accumulated decoded text every
  step (not per-token, so a string split across token boundaries is still
  caught) and truncates the delta in the step where it's found. It does
  *not* retroactively un-emit text from an earlier step if the stop string's
  opening characters were already flushed before its closing characters
  arrived — a true streaming server needs a lookback buffer for that; that's
  Phase 7's job, not this synchronous engine's.

---

## 6. What Phase 3's benchmarks actually showed (see `docs/BENCHMARKS.md`)

The headline numbers on this CPU dev rig undershoot the plan's GPU-oriented
predictions (P5: continuous batching beats static by 2-4x on high-variance
lengths) for the same reason Phase 1/2's did: at this scale, the *engine's
own* per-step Python bookkeeping overhead is comparable to or larger than
the compute it saves. `bench_paged.py` computes the **theoretical compute
ceiling** directly (total static (sequence, timestep) evaluations ÷ total
continuous ones, ignoring all overhead) alongside the measured wall-clock
number specifically to make this gap visible rather than hidden in a single
speedup figure — the ceiling lands in the predicted 2-4x range even though
the measured speedup doesn't yet, which is itself evidence that the
*mechanism* works and the shortfall is overhead, not a broken idea.

---

## 7. CUDA graphs: two host-sync traps, both real, both found by running

`graphs.py`'s `CUDAGraphRunner` and `attention/triton_paged.py` were written
entirely without the ability to run or even syntax-check them locally (this
dev machine has no CUDA device). Both worked essentially as designed once
actually run on a T4 via Lightning; the one real design gap was a **second**
host-sync, found only by letting the capture crash for real:

1. **`StaticKVCache.write()`/`read()`'s `.item()`** (known from Phase 1) —
   `CUDAGraphRunner` avoids this entirely by not reusing `StaticKVCache` for
   the graph-captured path. Instead, `_FixedPositionCache` bakes the decode
   position in as a plain Python int at construction time, since this
   benchmark's whole point is many replays of the same shape/position, not
   an advancing generation loop (see `graphs.py`'s module docstring).
2. **`model.py`'s own bounds-check assert** —
   `assert batch.position_ids.max() < self.cfg.block_size`. This looks like
   pure Python/host code, but comparing a CUDA tensor to an int and
   asserting on the result calls `Tensor.__bool__`, which is a
   device-to-host sync exactly like `.item()` — and CUDA graph capture
   forbids *any* sync inside the capturing region, full stop:
   `torch.AcceleratorError: CUDA error: operation not permitted when stream
   is capturing`. `_FixedPositionCache` was designed with (1) in mind and
   missed (2) entirely, because the assert lives in `model.py`, not the
   cache — it isn't the kind of thing that looks graph-related.

   Fixed by skipping the assert specifically when
   `torch.cuda.is_current_stream_capturing()` is true. This is safe, not a
   weakened check: `CUDAGraphRunner.capture()` always runs several warmup
   iterations in plain eager mode (outside any capturing stream) on the
   *exact same* static `position_ids` buffer immediately before capture
   begins, so the invariant is already verified on that tensor moments
   earlier — skipping the redundant re-check during capture skips nothing
   new.

**The general lesson**: any tensor comparison that feeds a Python-level
`assert`, `if`, or exception message (`f"... {t.item()} ..."`,
`t.max() < x`, `bool(t)`, `len()` on a tensor-shaped condition) is a
candidate host sync, and CUDA graph capture will refuse to record it
wherever it sits in the call stack — the assert doesn't have to be inside
the "obviously graph-related" code (a cache) to break capture; it broke here
from three call frames away, inside a completely generic, always-on safety
check that every other phase relies on unmodified.

---

## 8. Speculative decoding: the two changes it needed from everything before it

Phase 6 is the first phase to genuinely need something Phase 1's `ForwardBatch`
didn't anticipate: verifying several draft tokens against a *non-empty* cache
in one forward call. Two small, additive changes, both preserving every
existing caller's behavior exactly:

1. **`ForwardBatch.return_all_logits: bool = False`.** Every phase through 5
   only ever needed the last position's logits (the next token to sample).
   Verification needs logits at every one of the `gamma+1` proposed
   positions, to check each draft token against the target distribution
   that was current *when it was proposed* — not just the final one.
   `model.py`'s forward returns `(B, T, vocab)` instead of `(B, vocab)` only
   when this flag is set; every other caller leaves it `False` and sees no
   change.
2. **`ForwardBatch`'s decode assertion relaxed from "T must equal 1" to "T
   can be >1, but only with an explicit `attn_mask`."** The original
   assertion existed to catch a caller that meant to do ordinary
   single-token decode but forgot something. Speculative verification
   legitimately needs `T=gamma` (multiple new tokens continuing a non-empty
   cache in one call) — `batch.build_speculative_verify_mask` builds the
   required mask (full visibility into the cached prefix, causal only among
   the new block), and the relaxed assertion still catches the original
   mistake: multi-token decode *without* a mask still fails loudly.

Both `StaticKVCache.write()`/`read()` already handled `T>1` writes into a
non-empty cache correctly with no changes at all — they were only ever
gated by `ForwardBatch`'s stricter assertion, not by any real limitation in
the cache itself. This is the same "the abstraction was already general
enough" story as Phase 3's paged cache needing zero changes to
`CausalSelfAttention`.

### The verification algorithm's bookkeeping, spelled out

Each round carries `pending_logits` forward: the target's distribution for
the position right after the cache's current end, computed as a byproduct
of the *previous* round (or the initial prefill, for round one). Given
`gamma` draft tokens, only `gamma` *new* target distributions need
computing (not `gamma+1`) — combined with the carried-over `pending_logits`,
that's `gamma+1` total, matching `verify_and_accept`'s contract. Whichever
token is accepted last each round (the resample correction on rejection, or
the bonus token if every draft token was accepted) was never written into
the cache by the verification forward — only the `gamma` *draft* tokens
were — so every round ends with exactly one more ordinary decode step to
write it and produce the next round's `pending_logits`. This mirrors the
paged engine's `cache.advance(num_accepted, ...)` pattern from Phase 3
(rejected proposals stay physically written but logically ignored, and get
overwritten next round) rather than needing any explicit "undo."

### Why greedy exactness needs no special-casing

Representing a distribution as one-hot at its argmax (`temperature<=0`) and
feeding it through the *same* general `verify_and_accept` used for
stochastic sampling reproduces plain greedy decoding exactly: a one-hot
target `p` makes `min(1, p(x)/q(x))` either `1` (draft matches target's
argmax → always accept) or `0` (draft differs → always reject), and the
rejection residual `max(0, p-q)` collapses to a one-hot vector at the
target's own argmax regardless of what the draft's distribution `q` looked
like. `tests/test_spec_decode.py`'s greedy-exactness tests exercise this
directly against `generation.greedy_generate_cached` across both drafters.
