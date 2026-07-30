"""Phase 4 benchmark (docs/PLAN.md §7 Phase 4, predictions P2 & P3, §10 rules).

GPU-only end to end -- every section requires a real CUDA device (the
profiler's kernel-launch counts are meaningless on CPU, torch.compile's
CUDA-graph backends and manual `torch.cuda.graph` don't exist without CUDA,
and Triton needs sm70+ to even compile this kernel). Run on the target T4 via
Lightning AI, not locally.

Four sections, matching the phase's own structure:
  1. Profiler baseline (P2): one decode step, torch.profiler, kernel-launch
     count + total kernel time vs wall time. The gap IS the P2 number.
  2. torch.compile: reduce-overhead and max-autotune, on both the realistic
     StaticKVCache decode path (has a `.item()` host sync in cache read/write
     -- expect graph breaks) and the graph-safe fixed-position path from
     graphs.py (should compile cleanly) -- the comparison itself is the
     interesting result (docs/PLAN.md Phase 4 pitfall: silent eager fallback
     looks like "compile didn't help").
  3. Manual CUDA graphs (P3): eager vs graph-replay, across batch-size
     buckets.
  4. Triton paged decode attention vs the gather+SDPA baseline, at a few
     scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bench.common import default_results_path, save_json, timeit_repeated
from minigpt_infer.batch import ForwardBatch
from minigpt_infer.cache.static import StaticKVCache
from minigpt_infer.config import GPTConfig
from minigpt_infer.graphs import CUDAGraphRunner, _FixedPositionCache, graphs_supported
from minigpt_infer.model import GPT

KV_LENGTH = 128
BATCH_SIZE_BUCKETS = [1, 4, 16]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bench_kernels.py requires a CUDA device -- run on Lightning AI's T4, "
            "not locally (see docs/PLAN.md Phase 4)."
        )


def _make_static_decode_batch(model: GPT, batch_size: int, kv_length: int):
    dtype = next(model.parameters()).dtype
    cache = StaticKVCache(
        model.cfg.n_layer, model.cfg.n_head, model.head_dim,
        max_batch_size=batch_size, max_seq_len=kv_length + 1,
        device="cuda", dtype=dtype,
    )
    for layer in range(model.cfg.n_layer):
        shape = (batch_size, model.cfg.n_head, kv_length, model.head_dim)
        cache.k[layer][:, :, :kv_length, :] = torch.randn(shape, device="cuda", dtype=dtype)
        cache.v[layer][:, :, :kv_length, :] = torch.randn(shape, device="cuda", dtype=dtype)
    cache.seq_lens[:batch_size] = kv_length

    input_ids = torch.randint(0, model.cfg.vocab_size, (batch_size, 1), device="cuda")
    position_ids = torch.full((batch_size, 1), kv_length, dtype=torch.long, device="cuda")
    batch = ForwardBatch(
        input_ids=input_ids, position_ids=position_ids, is_prefill=False, cache=cache,
    )
    return batch


def run_profiler_baseline(model: GPT, batch_size: int = 1, kv_length: int = KV_LENGTH) -> dict:
    batch = _make_static_decode_batch(model, batch_size, kv_length)

    with torch.no_grad():
        for _ in range(10):
            model(batch)
        torch.cuda.synchronize()

    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(20):
            model(batch)
        torch.cuda.synchronize()

    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_type == torch.profiler.DeviceType.CUDA]
    kernel_launch_count = sum(e.count for e in cuda_events)
    total_cuda_time_us = sum(e.self_cuda_time_total for e in cuda_events)

    with torch.no_grad():
        timing = timeit_repeated(lambda: model(batch), device="cuda", warmup=10, repeats=30)

    per_step_kernel_us = total_cuda_time_us / 20
    per_step_wall_us = timing.median_s * 1e6
    overhead_pct = (1 - per_step_kernel_us / per_step_wall_us) * 100 if per_step_wall_us else 0.0

    result = {
        "batch_size": batch_size,
        "kv_length": kv_length,
        "kernel_launches_per_step": kernel_launch_count / 20,
        "cuda_kernel_time_us_per_step": per_step_kernel_us,
        "wall_time_us_per_step": per_step_wall_us,
        "overhead_pct": overhead_pct,
        "timing": timing.to_dict(),
    }
    print(
        f"P2 baseline bs={batch_size}: {kernel_launch_count / 20:.0f} kernel launches/step, "
        f"kernel_time={per_step_kernel_us:.1f}us wall={per_step_wall_us:.1f}us "
        f"overhead={overhead_pct:.1f}%"
    )
    return result


def run_torch_compile(model: GPT, batch_size: int = 1, kv_length: int = KV_LENGTH) -> dict:
    results = {}

    # (a) realistic StaticKVCache path -- expect a `.item()`-induced graph break.
    batch = _make_static_decode_batch(model, batch_size, kv_length)
    eager = timeit_repeated(lambda: model(batch), device="cuda", warmup=10, repeats=30)
    torch._dynamo.reset()
    compiled_model = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        for _ in range(3):
            compiled_model(batch)  # warm up / trigger compilation
        compiled = timeit_repeated(
            lambda: compiled_model(batch), device="cuda", warmup=10, repeats=30,
        )
    results["static_cache_path"] = {
        "eager": eager.to_dict(), "compiled": compiled.to_dict(),
        "speedup": eager.median_s / compiled.median_s,
    }
    static_speedup = eager.median_s / compiled.median_s
    print(
        f"torch.compile (StaticKVCache path): eager={eager.median_s * 1e6:.1f}us "
        f"compiled={compiled.median_s * 1e6:.1f}us speedup={static_speedup:.2f}x"
    )

    # (b) graph-safe fixed-position path (graphs.py) -- no host sync, should compile cleanly.
    runner = CUDAGraphRunner(model, batch_size, kv_length, device="cuda")
    fixed_batch = ForwardBatch(
        input_ids=runner.static_input_ids, position_ids=runner.static_position_ids,
        is_prefill=False, cache=runner.cache,
    )
    eager2 = timeit_repeated(lambda: model(fixed_batch), device="cuda", warmup=10, repeats=30)
    torch._dynamo.reset()
    compiled_model2 = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        for _ in range(3):
            compiled_model2(fixed_batch)
        compiled2 = timeit_repeated(
            lambda: compiled_model2(fixed_batch), device="cuda", warmup=10, repeats=30,
        )
    results["graph_safe_path"] = {
        "eager": eager2.to_dict(), "compiled": compiled2.to_dict(),
        "speedup": eager2.median_s / compiled2.median_s,
    }
    graph_safe_speedup = eager2.median_s / compiled2.median_s
    print(
        f"torch.compile (graph-safe path):     eager={eager2.median_s * 1e6:.1f}us "
        f"compiled={compiled2.median_s * 1e6:.1f}us speedup={graph_safe_speedup:.2f}x"
    )
    return results


@torch.no_grad()
def _eager_forward(model: GPT, batch: ForwardBatch) -> torch.Tensor:
    return model(batch)


def _graph_replay(runner: CUDAGraphRunner, input_ids: torch.Tensor) -> torch.Tensor:
    return runner.replay(input_ids)


def run_cuda_graphs(model: GPT, kv_length: int = KV_LENGTH) -> dict:
    results = {}
    for bs in BATCH_SIZE_BUCKETS:
        cache = _FixedPositionCache(
            model.cfg.n_layer, model.cfg.n_head, model.head_dim, bs, kv_length,
            "cuda", next(model.parameters()).dtype,
        )
        eager_input = torch.randint(0, model.cfg.vocab_size, (bs, 1), device="cuda")
        eager_pos = torch.full((bs, 1), kv_length, dtype=torch.long, device="cuda")
        eager_batch = ForwardBatch(
            input_ids=eager_input, position_ids=eager_pos, is_prefill=False, cache=cache,
        )
        eager = timeit_repeated(
            lambda batch=eager_batch: _eager_forward(model, batch),
            device="cuda", warmup=10, repeats=30,
        )

        runner = CUDAGraphRunner(model, bs, kv_length, device="cuda")
        runner.capture()
        replay_input = torch.randint(0, model.cfg.vocab_size, (bs, 1), device="cuda")
        graphed = timeit_repeated(
            lambda runner=runner, inp=replay_input: _graph_replay(runner, inp),
            device="cuda", warmup=10, repeats=30,
        )

        speedup = eager.median_s / graphed.median_s
        results[str(bs)] = {
            "eager": eager.to_dict(), "graphed": graphed.to_dict(), "speedup": speedup,
        }
        print(
            f"P3 CUDA graph bs={bs:>3}: eager={eager.median_s * 1e6:8.1f}us "
            f"graphed={graphed.median_s * 1e6:8.1f}us speedup={speedup:5.2f}x"
        )
    return results


class _GatherBatch:
    """Minimal stand-in for ForwardBatch -- PagedKVCache.read() only ever
    touches .block_tables and .seq_lens."""

    def __init__(self, block_tables: torch.Tensor, seq_lens: torch.Tensor) -> None:
        self.block_tables = block_tables
        self.seq_lens = seq_lens


def _gather_sdpa_decode(cache, batch: _GatherBatch, q: torch.Tensor, scale: float) -> torch.Tensor:
    k, v = cache.read(0, batch)
    max_len = k.shape[2]
    col = torch.arange(max_len, device=q.device).unsqueeze(0)
    valid = col < batch.seq_lens.unsqueeze(1)
    bias = torch.where(
        valid, torch.tensor(0.0, device=q.device, dtype=q.dtype), torch.finfo(q.dtype).min,
    )
    mask = bias.unsqueeze(1).unsqueeze(1)
    q4 = q.unsqueeze(2)
    return torch.nn.functional.scaled_dot_product_attention(q4, k, v, attn_mask=mask, scale=scale)


def run_triton_vs_gather(model: GPT) -> dict:
    import math

    from minigpt_infer.attention.triton_paged import (
        triton_paged_attention_available,
        triton_paged_decode_attention,
    )
    from minigpt_infer.cache.paged import PagedKVCache

    if not triton_paged_attention_available():
        print("Triton unavailable on this GPU (needs sm70+) -- skipping kernel benchmark.")
        return {"available": False}

    results: dict = {"available": True, "scales": {}}
    n_head, head_dim, block_size = model.cfg.n_head, model.head_dim, 16
    scale = 1.0 / math.sqrt(head_dim)

    for num_seqs in [8, 32, 128]:
        torch.manual_seed(0)
        seq_lens = [((i % 8) + 1) * 16 for i in range(num_seqs)]
        max_blocks = max((n + block_size - 1) // block_size for n in seq_lens)
        num_blocks = sum((n + block_size - 1) // block_size for n in seq_lens)

        q = torch.randn(num_seqs, n_head, head_dim, device="cuda", dtype=torch.float16)
        k_pool = torch.randn(
            num_blocks, block_size, n_head, head_dim, device="cuda", dtype=torch.float16,
        )
        v_pool = torch.randn(
            num_blocks, block_size, n_head, head_dim, device="cuda", dtype=torch.float16,
        )
        block_tables = torch.full((num_seqs, max_blocks), -1, dtype=torch.long, device="cuda")
        nb_idx = 0
        for i, n in enumerate(seq_lens):
            nb = (n + block_size - 1) // block_size
            for j in range(nb):
                block_tables[i, j] = nb_idx
                nb_idx += 1
        seq_lens_t = torch.tensor(seq_lens, device="cuda", dtype=torch.long)

        cache = PagedKVCache.__new__(PagedKVCache)
        cache.block_size = block_size
        cache.num_blocks = num_blocks
        cache.k_pool, cache.v_pool = [k_pool], [v_pool]
        gather_batch = _GatherBatch(block_tables, seq_lens_t)

        gather_timing = timeit_repeated(
            lambda cache=cache, batch=gather_batch, q=q: (
                _gather_sdpa_decode(cache, batch, q, scale)
            ),
            device="cuda", warmup=10, repeats=30,
        )
        triton_timing = timeit_repeated(
            lambda q=q, k=k_pool, v=v_pool, bt=block_tables, sl=seq_lens_t: (
                triton_paged_decode_attention(q, k, v, bt, sl, block_size, scale)
            ),
            device="cuda", warmup=10, repeats=30,
        )
        speedup = gather_timing.median_s / triton_timing.median_s
        results["scales"][str(num_seqs)] = {
            "gather_sdpa": gather_timing.to_dict(), "triton": triton_timing.to_dict(),
            "speedup": speedup,
        }
        print(
            f"Triton vs gather+SDPA, num_seqs={num_seqs:>4}: "
            f"gather={gather_timing.median_s * 1e6:8.1f}us "
            f"triton={triton_timing.median_s * 1e6:8.1f}us speedup={speedup:5.2f}x"
        )
    return results


def run(out_path: Path) -> dict:
    _require_cuda()
    assert graphs_supported()
    cfg = GPTConfig()
    torch.manual_seed(0)
    model = GPT(cfg).to("cuda").eval()

    payload = {
        "benchmark": "bench_kernels",
        "predictions": {
            "P2": "bs=1 decode overhead is >90% of wall clock, not math.",
            "P3": "CUDA graphs / torch.compile(reduce-overhead) give 3-10x on bs=1 decode.",
        },
        "config": {"gpt_config": vars(cfg), "kv_length": KV_LENGTH},
        "profiler_baseline": run_profiler_baseline(model),
        "torch_compile": run_torch_compile(model),
        "cuda_graphs": run_cuda_graphs(model),
        "triton_vs_gather": run_triton_vs_gather(model),
    }
    save_json(out_path, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path, default=None, help="default: bench/results/<name>_<gpu>_<ts>.json",
    )
    args = ap.parse_args()
    out = args.out if args.out is not None else default_results_path("bench_kernels")
    run(out)


if __name__ == "__main__":
    main()
