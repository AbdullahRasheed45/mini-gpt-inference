"""Phase 1 benchmark (docs/PLAN.md §7 Phase 1, prediction P1, §10 rules).

P1: naive (no-cache) generate recomputes the full sequence from scratch at
every step -- O(n) forward passes each doing O(t) work, so total cost is
roughly O(n^2) in tokens generated. Cached generate does O(1) new work per
step (one cache-augmented forward), so total cost is O(n). At 512 tokens,
expect 10-40x wall-clock speedup (wall-clock gain is less than the raw FLOP
gain because naive still batches each step into one big, well-vectorized
kernel call).

Usage:
    python bench/bench_kvcache.py                          # full spec (§10): 30 repeats, 10 warmup
    python bench/bench_kvcache.py --repeats 3 --warmup 1    # fast local/CPU sanity check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bench.common import REPO_ROOT, default_results_path, save_json, timeit_repeated
from minigpt_infer.config import GPTConfig
from minigpt_infer.generation import greedy_generate_cached
from minigpt_infer.model import GPT
from minigpt_infer.reference import ReferenceGPT

N_VALUES = [16, 32, 64, 128, 256, 512]
# Kept at 1 token so N=512 (the largest bucket) still fits under
# GPTConfig().block_size=512 -- see _effective_n().
PROMPT_LEN = 1


def _effective_n(n: int, block_size: int) -> int:
    """The learned position table has exactly block_size rows; prompt_len + n
    must not exceed it. Only the N=512 bucket is actually clamped (to 511)."""
    return min(n, block_size - PROMPT_LEN)


def _build_models(seed: int, device: str, cfg: GPTConfig) -> tuple[ReferenceGPT, GPT]:
    torch.manual_seed(seed)
    ref = ReferenceGPT(cfg).to(device).eval()
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ref.state_dict())  # same weights -- isolates the cache, not the init
    return ref, model


def run(device: str, warmup: int, repeats: int, out_path: Path, plot_path: Path | None) -> dict:
    cfg = GPTConfig()
    ref, model = _build_models(seed=0, device=device, cfg=cfg)
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), device=device)

    results = []
    for n in N_VALUES:
        n_eff = _effective_n(n, cfg.block_size)

        with torch.no_grad():
            naive = timeit_repeated(
                lambda n_eff=n_eff: ref.greedy_generate(prompt.clone(), n_eff),
                device=device, warmup=warmup, repeats=repeats,
            )
            cached = timeit_repeated(
                lambda n_eff=n_eff: greedy_generate_cached(model, prompt.clone(), n_eff),
                device=device, warmup=warmup, repeats=repeats,
            )

        speedup = naive.median_s / cached.median_s
        row = {
            "n_requested": n,
            "n_effective": n_eff,
            "naive": naive.to_dict(),
            "cached": cached.to_dict(),
            "naive_ms_per_token": naive.median_s / n_eff * 1000,
            "cached_ms_per_token": cached.median_s / n_eff * 1000,
            "speedup_median": speedup,
        }
        results.append(row)
        print(
            f"N={n:>4} (eff={n_eff:>4})  naive={naive.median_s * 1000:9.1f}ms "
            f"(+-{naive.iqr_s * 1000:5.1f})  cached={cached.median_s * 1000:8.1f}ms "
            f"(+-{cached.iqr_s * 1000:5.1f})  speedup={speedup:6.2f}x"
        )

    payload = {
        "benchmark": "bench_kvcache",
        "prediction": "P1",
        "prediction_text": (
            "naive generate is O(n^2) in tokens; cached is O(n). At 512 tokens "
            "expect 10-40x wall-clock speedup."
        ),
        "config": {
            "gpt_config": vars(cfg),
            "prompt_len": PROMPT_LEN,
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

    ns = [r["n_effective"] for r in results]
    naive_ms = [r["naive"]["median_s"] * 1000 for r in results]
    cached_ms = [r["cached"]["median_s"] * 1000 for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, naive_ms, marker="o", label="naive (no cache)")
    ax.plot(ns, cached_ms, marker="o", label="cached (KV cache)")
    ax.set_xlabel("tokens generated (N)")
    ax.set_ylabel("total wall-clock latency (ms, median)")
    ax.set_title("Phase 1: naive vs KV-cached greedy decode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument(
        "--out", type=Path, default=None, help="default: bench/results/<name>_<gpu>_<ts>.json",
    )
    ap.add_argument("--plot", type=Path, default=REPO_ROOT / "docs/img/bench_kvcache.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out = args.out if args.out is not None else default_results_path("bench_kvcache")
    run(args.device, args.warmup, args.repeats, out, None if args.no_plot else args.plot)


if __name__ == "__main__":
    main()
