"""Phase 8 hardware study (docs/PLAN.md §7 Phase 8 point 3, prediction P9, §10 rules).

Runs the IDENTICAL decode-only benchmark against the real Project A
checkpoint (102.4 MB fp16 weights -- the same number docs/PLAN.md §3's own
P9 arithmetic is built on) on whatever CUDA device is available. Run once on
Lightning's T4 and once on Kaggle's P100; `bench/plot.py` (not this script)
overlays the two resulting JSONs into the crossover chart, per §10 rule 8
("always label the hardware when comparing").

  - **bs=1 decode**: memory-bandwidth-bound -- one token needs to read
    almost the entire weight tensor once, with negligible matmul work to
    hide that read behind. P100 has 2.3x the T4's memory bandwidth and
    nothing to lose from lacking fp16 tensor cores at bs=1 (there's no
    batched matmul to accelerate), so P100 should win here.
  - **batched decode at increasing batch size**: increasingly compute-bound
    as the same weight read amortizes over more sequences' matmuls per
    step. T4 has fp16 tensor cores; P100 (Pascal, sm60) does not -- fp16
    runs at ~fp32 speed there. T4 should pull ahead as batch size grows.

The crossover *point* between these two curves -- not just "which GPU wins
overall" -- is P9, and docs/PLAN.md §3 calls it out as the single most
interesting hardware result available in this project.

Decode is isolated from prefill by design: prefill runs once per repeat,
untimed; only the per-token decode steps that follow are timed. Otherwise a
fixed per-request prefill cost would dilute the bandwidth-vs-compute signal
this benchmark exists to measure.

Usage:
    HF_TOKEN=... python -m bench.bench_hardware                       # full spec
    HF_TOKEN=... python -m bench.bench_hardware --repeats 3 --warmup 1 --max-bs 8
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import torch

from bench.common import REPO_ROOT, TimingResult, default_results_path, save_json, sync_if_cuda
from minigpt_infer.batch import ForwardBatch
from minigpt_infer.cache.static import StaticKVCache
from minigpt_infer.model import GPT
from minigpt_infer.tokenizer import padding_vocab_mask

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
PROMPT_LEN = 8
DECODE_STEPS = 32


def _iqr(sorted_samples: list[float]) -> float:
    n = len(sorted_samples)
    lower = sorted_samples[: n // 2]
    upper = sorted_samples[(n + 1) // 2:]
    q1 = statistics.median(lower) if lower else sorted_samples[0]
    q3 = statistics.median(upper) if upper else sorted_samples[-1]
    return q3 - q1


def _prefill(model: GPT, prompt_ids: torch.Tensor) -> tuple[StaticKVCache, torch.Tensor]:
    B, T0 = prompt_ids.shape
    device = prompt_ids.device
    cache = StaticKVCache(
        n_layer=model.cfg.n_layer, n_head=model.cfg.n_head, head_dim=model.head_dim,
        max_batch_size=B, max_seq_len=model.cfg.block_size,
        device=device, dtype=next(model.parameters()).dtype,
    )
    pos = torch.arange(T0, device=device).unsqueeze(0).expand(B, -1)
    batch = ForwardBatch(input_ids=prompt_ids, position_ids=pos, is_prefill=True, cache=cache)
    logits = model(batch)
    cache.advance(T0, B)
    return cache, logits


def _decode_n_steps(
    model: GPT, cache: StaticKVCache, logits: torch.Tensor, n_steps: int,
    vocab_mask: torch.Tensor | None,
) -> None:
    B = logits.shape[0]
    device = logits.device
    for _ in range(n_steps):
        masked = logits.masked_fill(vocab_mask, float("-inf")) if vocab_mask is not None else logits
        next_id = torch.argmax(masked, dim=-1, keepdim=True)
        cur_len = int(cache.seq_lens[0].item())
        pos = torch.full((B, 1), cur_len, device=device, dtype=torch.long)
        batch = ForwardBatch(input_ids=next_id, position_ids=pos, is_prefill=False, cache=cache)
        logits = model(batch)
        cache.advance(1, B)


@torch.no_grad()
def _time_decode_only(
    model: GPT, prompt_ids: torch.Tensor, decode_steps: int,
    vocab_mask: torch.Tensor | None, device: str, warmup: int, repeats: int,
) -> TimingResult:
    """Like bench.common.timeit_repeated, but only the decode portion of
    each repeat is inside the timing window -- prefill is untimed setup,
    run fresh every repeat since decode mutates the cache in place."""
    dev = torch.device(device)

    def one_rep() -> float:
        cache, logits = _prefill(model, prompt_ids)
        sync_if_cuda(dev)
        t0 = time.perf_counter()
        _decode_n_steps(model, cache, logits, decode_steps, vocab_mask)
        sync_if_cuda(dev)
        return time.perf_counter() - t0

    for _ in range(warmup):
        one_rep()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    samples = [one_rep() for _ in range(repeats)]
    samples_sorted = sorted(samples)
    max_mem = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    return TimingResult(
        samples_s=samples,
        median_s=statistics.median(samples_sorted),
        iqr_s=_iqr(samples_sorted),
        min_s=samples_sorted[0],
        max_s=samples_sorted[-1],
        max_memory_bytes=max_mem,
    )


def run(
    device: str, warmup: int, repeats: int, max_bs: int, out_path: Path, plot_path: Path | None,
    repo_id: str | None, filename: str | None,
) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set -- skipping (P9 needs the real checkpoint's real weight size).")
        payload = {"benchmark": "bench_hardware", "available": False, "reason": "no HF_TOKEN"}
        save_json(out_path, payload)
        return payload

    from minigpt_infer.loader import DEFAULT_CKPT_PATH, DEFAULT_REPO_ID, load_model

    model, cfg, meta = load_model(
        repo_id or DEFAULT_REPO_ID, filename or DEFAULT_CKPT_PATH,
        token=token, device=device, dtype=torch.float16,
    )
    model.eval()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"loaded checkpoint (iter={meta.get('iter')}), weights={weight_bytes / 1e6:.1f} MB fp16")

    vocab_mask = padding_vocab_mask(cfg.vocab_size, device=device)

    results = []
    for bs in [b for b in BATCH_SIZES if b <= max_bs]:
        torch.manual_seed(0)
        prompt_ids = torch.randint(0, cfg.vocab_size, (bs, PROMPT_LEN), device=device)

        timing = _time_decode_only(
            model, prompt_ids, DECODE_STEPS, vocab_mask, device, warmup, repeats,
        )
        # One decode step advances every row in the batch by exactly one new
        # token simultaneously, so median_s / DECODE_STEPS is already a
        # per-request, per-token latency -- not divided by bs again.
        per_token_latency_ms = timing.median_s / DECODE_STEPS * 1000
        throughput_tok_per_s = bs * DECODE_STEPS / timing.median_s
        row = {
            "batch_size": bs,
            "timing": timing.to_dict(),
            "per_token_latency_ms": per_token_latency_ms,
            "throughput_tok_per_s": throughput_tok_per_s,
        }
        results.append(row)
        print(
            f"bs={bs:>4}  per_token_latency={per_token_latency_ms:8.3f}ms  "
            f"throughput={throughput_tok_per_s:9.1f} tok/s"
        )

    payload = {
        "benchmark": "bench_hardware",
        "available": True,
        "prediction": "P9",
        "prediction_text": (
            "P100 beats T4 at bs=1 decode (memory-bandwidth-bound: P100 has 2.3x bandwidth); "
            "T4 beats P100 at large batch (compute-bound: T4 has fp16 tensor cores, P100 doesn't)."
        ),
        "config": {
            "checkpoint_iter": meta.get("iter"), "weight_bytes_fp16": weight_bytes,
            "gpt_config": vars(cfg), "prompt_len": PROMPT_LEN, "decode_steps": DECODE_STEPS,
            "batch_sizes": [b for b in BATCH_SIZES if b <= max_bs],
            "warmup": warmup, "repeats": repeats, "device": device,
        },
        "results": results,
    }
    save_json(out_path, payload)

    if plot_path is not None:
        _plot(results, plot_path)

    return payload


def _plot(results: list[dict], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["batch_size"] for r in results]
    latency = [r["per_token_latency_ms"] for r in results]
    throughput = [r["throughput_tok_per_s"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(bs, latency, marker="o")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("per-token decode latency (ms, median)")
    ax1.set_title("Decode latency vs batch size")
    ax1.grid(True, alpha=0.3)

    ax2.plot(bs, throughput, marker="o")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("throughput (tok/s, system-wide)")
    ax2.set_title("Decode throughput vs batch size")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--filename", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=REPO_ROOT / "docs/img/bench_hardware.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out = args.out if args.out is not None else default_results_path("bench_hardware")
    run(
        args.device, args.warmup, args.repeats, args.max_bs, out,
        None if args.no_plot else args.plot, args.repo_id, args.filename,
    )


if __name__ == "__main__":
    main()
