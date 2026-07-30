"""Phase 3 benchmark (docs/PLAN.md §7 Phase 3, predictions P5 & P6, §10 rules).

Two separate comparisons, deliberately not conflated:

  P5 (batching POLICY): static batching (Phase 2's batched_greedy_generate --
  a fixed group runs for as long as its longest member needs) vs continuous
  batching (Phase 3's LLMEngine -- a slot frees and a new request is admitted
  the instant its own sequence finishes), on two workloads:
    - uniform: every request wants the same number of new tokens. Static's
      "wait for the longest" penalty is ~0 by construction here.
    - high_variance: lengths drawn from a lognormal with large sigma -- a
      few short requests get stuck riding along with one long one.
  Static runs the request pool as sequential, non-overlapping groups of
  max_batch_size (the realistic naive deployment); continuous runs the same
  pool through one engine with staggered admission.

  P6 (cache IMPLEMENTATION): PagedKVCache (gather/indirection through block
  tables) vs StaticKVCache (contiguous) at otherwise-IDENTICAL batching --
  same uniform-length workload, same batch size, everyone admitted at once so
  nobody finishes early (no continuous-batching advantage in play). Any
  throughput gap here is attributable to paged gather overhead alone.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from bench.common import default_results_path, save_json, timeit_repeated
from minigpt_infer.config import EngineConfig, GPTConfig, SamplingParams
from minigpt_infer.engine.engine import LLMEngine
from minigpt_infer.engine.request import Request
from minigpt_infer.generation import batched_greedy_generate
from minigpt_infer.model import GPT

PROMPT_LEN = 4
NUM_REQUESTS = 32
MAX_BATCH_SIZE = 8
UNIFORM_TOKENS = 16
LOGNORMAL_MU = math.log(10)
LOGNORMAL_SIGMA = 1.1
MAX_TOKENS_CAP = 64


def _sample_lengths(workload: str, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if workload == "uniform":
        return [UNIFORM_TOKENS] * n
    if workload == "high_variance":
        raw = [rng.lognormvariate(LOGNORMAL_MU, LOGNORMAL_SIGMA) for _ in range(n)]
        return [max(1, min(MAX_TOKENS_CAP, int(x))) for x in raw]
    raise ValueError(workload)


def _make_prompts(n: int, vocab_size: int, seed: int) -> list[list[int]]:
    torch.manual_seed(seed)
    return [torch.randint(0, vocab_size, (PROMPT_LEN,)).tolist() for _ in range(n)]


def _static_thunk(model: GPT, prompts: list[list[int]], lengths: list[int], max_batch_size: int):
    @torch.no_grad()
    def thunk():
        for start in range(0, len(prompts), max_batch_size):
            group_p = prompts[start:start + max_batch_size]
            group_n = lengths[start:start + max_batch_size]
            batched_greedy_generate(model, [list(p) for p in group_p], max(group_n), pad_token_id=0)
    return thunk


def _continuous_thunk(
    model: GPT, prompts: list[list[int]], lengths: list[int], engine_cfg: EngineConfig
):
    @torch.no_grad()
    def thunk():
        engine = LLMEngine(model, engine_cfg)
        for i, (p, n) in enumerate(zip(prompts, lengths, strict=True)):
            engine.add_request(Request(f"r{i}", p, SamplingParams(temperature=0.0, max_tokens=n)))
        while engine.has_unfinished_requests():
            engine.step()
    return thunk


def _theoretical_compute_ratio(lengths: list[int], max_batch_size: int) -> float:
    """Upper bound on continuous's speedup from compute savings alone: total
    (sequence, timestep) forward evaluations static performs (every group
    runs max(group) steps at full batch width) divided by the number
    continuous performs (exactly sum(lengths), no waste). Real per-step
    engine overhead (Python bookkeeping, block-table construction, etc.) is
    NOT modeled here -- comparing this to the measured speedup is exactly
    what shows how much of the theoretical gain overhead ate.
    """
    groups = [lengths[i:i + max_batch_size] for i in range(0, len(lengths), max_batch_size)]
    static_compute = sum(max(g) * len(g) for g in groups)
    return static_compute / sum(lengths)


def run_p5(device: str, warmup: int, repeats: int, cfg: GPTConfig, model: GPT) -> dict:
    engine_cfg = EngineConfig(block_size=8, num_blocks=4096, max_batch_size=MAX_BATCH_SIZE)
    results = {}
    for workload in ["uniform", "high_variance"]:
        prompts = _make_prompts(NUM_REQUESTS, cfg.vocab_size, seed=0)
        lengths = _sample_lengths(workload, NUM_REQUESTS, seed=1)
        ceiling = _theoretical_compute_ratio(lengths, MAX_BATCH_SIZE)

        static = timeit_repeated(
            _static_thunk(model, prompts, lengths, MAX_BATCH_SIZE),
            device=device, warmup=warmup, repeats=repeats,
        )
        continuous = timeit_repeated(
            _continuous_thunk(model, prompts, lengths, engine_cfg),
            device=device, warmup=warmup, repeats=repeats,
        )
        speedup = static.median_s / continuous.median_s
        results[workload] = {
            "lengths": lengths,
            "theoretical_compute_ceiling": ceiling,
            "static": static.to_dict(),
            "continuous": continuous.to_dict(),
            "continuous_speedup": speedup,
        }
        print(
            f"P5 {workload:>14}: static={static.median_s * 1000:8.1f}ms "
            f"continuous={continuous.median_s * 1000:8.1f}ms speedup={speedup:5.2f}x "
            f"(compute ceiling {ceiling:.2f}x)"
        )
    return results


def run_p6(device: str, warmup: int, repeats: int, cfg: GPTConfig, model: GPT) -> dict:
    prompts = _make_prompts(MAX_BATCH_SIZE, cfg.vocab_size, seed=2)
    # uniform, all admitted at once -> no early finishers, isolates cache overhead
    lengths = [UNIFORM_TOKENS] * MAX_BATCH_SIZE

    static = timeit_repeated(
        _static_thunk(model, prompts, lengths, MAX_BATCH_SIZE),
        device=device, warmup=warmup, repeats=repeats,
    )
    engine_cfg = EngineConfig(block_size=8, num_blocks=4096, max_batch_size=MAX_BATCH_SIZE)
    paged = timeit_repeated(
        _continuous_thunk(model, prompts, lengths, engine_cfg),
        device=device, warmup=warmup, repeats=repeats,
    )
    overhead_pct = (paged.median_s / static.median_s - 1.0) * 100
    print(
        f"P6 uniform/same-batch: static_cache={static.median_s * 1000:8.1f}ms "
        f"paged_cache={paged.median_s * 1000:8.1f}ms overhead={overhead_pct:+5.1f}%"
    )
    return {
        "static_cache": static.to_dict(),
        "paged_cache": paged.to_dict(),
        "paged_overhead_pct": overhead_pct,
    }


def run(device: str, warmup: int, repeats: int, out_path: Path) -> dict:
    cfg = GPTConfig()
    torch.manual_seed(0)
    model = GPT(cfg).to(device).eval()

    p5 = run_p5(device, warmup, repeats, cfg, model)
    p6 = run_p6(device, warmup, repeats, cfg, model)

    payload = {
        "benchmark": "bench_paged",
        "predictions": {
            "P5": "continuous batching beats static by 2-4x on high-variance, ~0 on uniform.",
            "P6": "paged attention costs a few % vs static caching at this scale.",
        },
        "config": {
            "gpt_config": vars(cfg), "prompt_len": PROMPT_LEN, "num_requests": NUM_REQUESTS,
            "max_batch_size": MAX_BATCH_SIZE, "uniform_tokens": UNIFORM_TOKENS,
            "warmup": warmup, "repeats": repeats, "device": device,
        },
        "P5": p5,
        "P6": p6,
    }
    save_json(out_path, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument(
        "--out", type=Path, default=None, help="default: bench/results/<name>_<gpu>_<ts>.json",
    )
    args = ap.parse_args()
    out = args.out if args.out is not None else default_results_path("bench_paged")
    run(args.device, args.warmup, args.repeats, out)


if __name__ == "__main__":
    main()
