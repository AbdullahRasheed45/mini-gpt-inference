"""Phase 2 benchmark (docs/PLAN.md §7 Phase 2, prediction P4, §10 rules).

P4: throughput scales near-linearly with batch size up to bs~64-128, then
bends (the ridge point where the GPU goes from memory-bound to compute-bound).
This is a CPU rig here, so the *bend point* itself is not expected to land at
the same bs as the T4 prediction -- what's being checked is the qualitative
shape (linear-then-bending), with the actual knee re-measured on GPU hardware
in Phase 8's hardware study.

Usage:
    python bench/bench_batching.py
    python bench/bench_batching.py --repeats 3 --warmup 1 --max-bs 64   # faster local run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bench.common import REPO_ROOT, save_json, timeit_repeated
from minigpt_infer.config import GPTConfig
from minigpt_infer.generation import batched_greedy_generate
from minigpt_infer.model import GPT

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PROMPT_LEN = 8
MAX_NEW_TOKENS = 32


def run(
    device: str, warmup: int, repeats: int, max_bs: int, out_path: Path, plot_path: Path | None
) -> dict:
    cfg = GPTConfig()
    torch.manual_seed(0)
    model = GPT(cfg).to(device).eval()

    results = []
    for bs in [b for b in BATCH_SIZES if b <= max_bs]:
        torch.manual_seed(0)
        prompts = [
            torch.randint(0, cfg.vocab_size, (PROMPT_LEN,)).tolist() for _ in range(bs)
        ]

        with torch.no_grad():
            timing = timeit_repeated(
                lambda prompts=prompts: batched_greedy_generate(
                    model, [list(p) for p in prompts], MAX_NEW_TOKENS, pad_token_id=0,
                ),
                device=device, warmup=warmup, repeats=repeats,
            )

        total_tokens = bs * MAX_NEW_TOKENS
        throughput = total_tokens / timing.median_s
        # All requests in a static batch finish together, so per-request
        # latency is just the batch's total wall-clock time.
        per_request_latency_ms = timing.median_s * 1000
        row = {
            "batch_size": bs,
            "timing": timing.to_dict(),
            "throughput_tok_per_s": throughput,
            "per_request_latency_ms": per_request_latency_ms,
        }
        results.append(row)
        print(
            f"bs={bs:>4}  total={timing.median_s * 1000:9.1f}ms  "
            f"throughput={throughput:9.1f} tok/s  latency/request={per_request_latency_ms:9.1f}ms"
        )

    payload = {
        "benchmark": "bench_batching",
        "prediction": "P4",
        "prediction_text": (
            "throughput scales near-linearly with batch size to bs~64-128, then bends."
        ),
        "config": {
            "gpt_config": vars(cfg),
            "prompt_len": PROMPT_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "warmup": warmup,
            "repeats": repeats,
            "device": device,
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
    throughput = [r["throughput_tok_per_s"] for r in results]
    latency = [r["per_request_latency_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(bs, throughput, marker="o")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("throughput (tok/s, system-wide)")
    ax1.set_title("Throughput vs batch size")
    ax1.grid(True, alpha=0.3)

    ax2.plot(latency, throughput, marker="o")
    ax2.set_xlabel("per-request latency (ms, median)")
    ax2.set_ylabel("throughput (tok/s)")
    ax2.set_title("Throughput vs latency tradeoff")
    ax2.grid(True, alpha=0.3)
    for x, y, b in zip(latency, throughput, bs, strict=True):
        ax2.annotate(str(b), (x, y), fontsize=8, textcoords="offset points", xytext=(4, 4))

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--max-bs", type=int, default=256)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/benchmarks/bench_batching.json")
    ap.add_argument("--plot", type=Path, default=REPO_ROOT / "docs/img/bench_batching.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    plot = None if args.no_plot else args.plot
    run(args.device, args.warmup, args.repeats, args.max_bs, args.out, plot)


if __name__ == "__main__":
    main()
