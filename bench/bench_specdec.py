"""Phase 6 benchmark (docs/PLAN.md §7 Phase 6, prediction P8, §10 rules).

Needs the real checkpoint and HF_TOKEN, like bench_quant.py's perplexity
section -- prompt-lookup's acceptance rate only means anything measured
against real, trained, TinyStories-shaped (repetitive) text; a random-init
model's prompt-lookup hit rate is meaningless noise.

For each drafter (prompt-lookup, self-speculative) and gamma in {1..8}:
  - acceptance rate alpha and mean accepted length per round (measured)
  - theoretical tokens/step: (1 - alpha^(gamma+1)) / (1 - alpha)
  - wall-clock speedup vs a target-only (no speculation) baseline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from bench.common import default_results_path, save_json, timeit_repeated
from minigpt_infer.engine.spec_decode import (
    PromptLookupDrafter,
    SelfSpeculativeDrafter,
    SpecDecodeStats,
    speculative_generate,
)
from minigpt_infer.generation import greedy_generate_cached

GAMMA_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_NEW_TOKENS = 96
NUM_PROMPTS = 6
PROMPT_LEN = 12
PROMPT_LOOKUP_NGRAM = 2


def theoretical_tokens_per_step(alpha: float, gamma: int) -> float:
    """(1 - alpha^(gamma+1)) / (1 - alpha) -- the expected number of tokens
    produced per verification round under the standard speculative-decoding
    model (docs/PLAN.md §7 Phase 6 benchmark spec)."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def _load_tinystories_prompts(num_prompts: int, prompt_len: int) -> list[list[int]]:
    from datasets import load_dataset

    from minigpt_infer.tokenizer import encode

    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    prompts = []
    for example in ds:
        ids = encode(example["text"])
        if len(ids) >= prompt_len:
            prompts.append(ids[:prompt_len])
        if len(prompts) >= num_prompts:
            break
    return prompts


def _aggregate(stats_list: list[SpecDecodeStats]) -> SpecDecodeStats:
    agg = SpecDecodeStats()
    for s in stats_list:
        agg.num_rounds += s.num_rounds
        agg.num_draft_tokens_proposed += s.num_draft_tokens_proposed
        agg.num_draft_tokens_accepted += s.num_draft_tokens_accepted
        agg.accepted_lengths.extend(s.accepted_lengths)
    return agg


def run_drafter_sweep(
    model, drafter_factory, prompts: list[list[int]], baseline_median_s: float,
    device: str, warmup: int, repeats: int,
) -> dict:
    results = {}
    for gamma in GAMMA_VALUES:
        stats_list = []
        for prompt in prompts:
            _tokens, stats = speculative_generate(
                model, drafter_factory(gamma), prompt, MAX_NEW_TOKENS, gamma, temperature=0.0,
            )
            stats_list.append(stats)
        agg = _aggregate(stats_list)
        alpha = agg.acceptance_rate
        theoretical = theoretical_tokens_per_step(alpha, gamma)

        timing = timeit_repeated(
            lambda gamma=gamma: speculative_generate(
                model, drafter_factory(gamma), prompts[0], MAX_NEW_TOKENS, gamma, temperature=0.0,
            ),
            device=device, warmup=warmup, repeats=repeats,
        )
        speedup = baseline_median_s / timing.median_s

        results[str(gamma)] = {
            "acceptance_rate": alpha,
            "mean_accepted_length": agg.mean_accepted_length,
            "theoretical_tokens_per_step": theoretical,
            "num_rounds": agg.num_rounds,
            "timing": timing.to_dict(),
            "speedup_vs_target_only": speedup,
        }
        print(
            f"gamma={gamma}  alpha={alpha:.3f}  mean_accepted_len={agg.mean_accepted_length:.2f}  "
            f"theoretical_tok/step={theoretical:.2f}  speedup={speedup:.2f}x"
        )
    return results


def run(device: str, warmup: int, repeats: int, out_path: Path) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set -- skipping (needs the real checkpoint for meaningful acceptance).")
        payload = {"benchmark": "bench_specdec", "available": False, "reason": "no HF_TOKEN"}
        save_json(out_path, payload)
        return payload

    from minigpt_infer.loader import load_model

    model, cfg, meta = load_model(token=token, device=device, dtype=torch.float32)
    model.eval()
    print(f"loaded checkpoint (iter={meta.get('iter')})")

    prompts = _load_tinystories_prompts(NUM_PROMPTS, PROMPT_LEN)
    print(f"using {len(prompts)} real TinyStories validation prompts")

    with torch.no_grad():
        baseline = timeit_repeated(
            lambda: greedy_generate_cached(model, torch.tensor([prompts[0]]), MAX_NEW_TOKENS),
            device=device, warmup=warmup, repeats=repeats,
        )
    print(f"target-only baseline: {baseline.median_s * 1000:.1f}ms for {MAX_NEW_TOKENS} tokens")

    print("\n--- prompt-lookup ---")
    prompt_lookup_results = run_drafter_sweep(
        model,
        lambda gamma: PromptLookupDrafter(cfg.vocab_size, ngram_size=PROMPT_LOOKUP_NGRAM),
        prompts, baseline.median_s, device, warmup, repeats,
    )

    print("\n--- self-speculative (layer skip, first half of layers) ---")
    draft_layers = max(1, cfg.n_layer // 2)
    self_spec_results = run_drafter_sweep(
        model,
        lambda gamma: SelfSpeculativeDrafter(model, draft_layers, temperature=0.0),
        prompts, baseline.median_s, device, warmup, repeats,
    )

    payload = {
        "benchmark": "bench_specdec",
        "available": True,
        "prediction": "P8",
        "prediction_text": "Prompt-lookup gets acceptance alpha~=0.3-0.6 on TinyStories.",
        "config": {
            "checkpoint_iter": meta.get("iter"), "max_new_tokens": MAX_NEW_TOKENS,
            "num_prompts": NUM_PROMPTS, "prompt_len": PROMPT_LEN,
            "self_spec_draft_layers": draft_layers, "device": device,
            "warmup": warmup, "repeats": repeats,
        },
        "baseline_target_only": baseline.to_dict(),
        "prompt_lookup": prompt_lookup_results,
        "self_speculative": self_spec_results,
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
    out = args.out if args.out is not None else default_results_path("bench_specdec")
    run(args.device, args.warmup, args.repeats, out)


if __name__ == "__main__":
    main()
