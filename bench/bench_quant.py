"""Phase 5 benchmark (docs/PLAN.md §7 Phase 5, prediction P7, §10 rules).

Three independent measurements:
  1. Memory: fp16 baseline vs int8-quantized-Linears (embeddings stay fp16).
     Pure byte counting, no GPU needed.
  2. Speed: naive dequant-then-matmul vs plain fp16 Linear (P7 predicts
     naive int8 is SLOWER -- publish that even though it's a negative
     result), and, where a GPU is available, the fused Triton dequant GEMV
     that's supposed to actually win.
  3. Quality: perplexity on a held-out TinyStories slice, fp16 vs int8, using
     the REAL trained checkpoint. Needs HF_TOKEN and network -- skipped (not
     failed) without them, same convention as tests/test_golden.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from bench.common import default_results_path, save_json, timeit_repeated
from minigpt_infer.config import GPTConfig
from minigpt_infer.model import GPT
from minigpt_infer.quant.int8 import (
    QuantizedLinear,
    quantize_model,
    quantized_state_dict_bytes,
    triton_dequant_gemv,
    triton_dequant_gemv_available,
)
from minigpt_infer.reference import ReferenceGPT

HELD_OUT_TOKEN_BUDGET = 20_000  # ~40 chunks of 512 tokens -- enough for a stable PPL estimate


def run_memory(cfg: GPTConfig) -> dict:
    torch.manual_seed(0)
    fp16_model = GPT(cfg).to(dtype=torch.float16)
    fp16_bytes = quantized_state_dict_bytes(fp16_model)

    q_model = GPT(cfg).to(dtype=torch.float16)
    q_model.load_state_dict(fp16_model.state_dict())
    quantize_model(q_model)
    q_bytes = quantized_state_dict_bytes(q_model)

    reduction_pct = (1 - q_bytes["total_bytes"] / fp16_bytes["total_bytes"]) * 100
    result = {
        "fp16_total_mb": fp16_bytes["total_bytes"] / 1e6,
        "int8_total_mb": q_bytes["total_bytes"] / 1e6,
        "embedding_mb_unchanged": fp16_bytes["embedding_bytes"] / 1e6,
        "reduction_pct": reduction_pct,
    }
    print(
        f"Memory: fp16={result['fp16_total_mb']:.1f}MB int8={result['int8_total_mb']:.1f}MB "
        f"(embedding, unquantized: {result['embedding_mb_unchanged']:.1f}MB) "
        f"reduction={reduction_pct:.1f}%"
    )
    return result


def run_speed(device: str, warmup: int, repeats: int, cfg: GPTConfig) -> dict:
    torch.manual_seed(0)
    linear = torch.nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False).to(device)
    qlinear = QuantizedLinear.from_linear(linear).to(device)
    x = torch.randn(1, cfg.n_embd, device=device)

    with torch.no_grad():
        fp16_timing = timeit_repeated(
            lambda: F.linear(x, linear.weight), device=device, warmup=warmup, repeats=repeats,
        )
        naive_int8_timing = timeit_repeated(
            lambda: qlinear(x), device=device, warmup=warmup, repeats=repeats,
        )

    speedup_naive = fp16_timing.median_s / naive_int8_timing.median_s
    print(
        f"Speed (bs=1 GEMV, {cfg.n_embd}->{3 * cfg.n_embd}): "
        f"fp16={fp16_timing.median_s * 1e6:8.1f}us "
        f"naive_int8={naive_int8_timing.median_s * 1e6:8.1f}us speedup={speedup_naive:.2f}x "
        f"({'FASTER' if speedup_naive > 1 else 'SLOWER -- P7 as predicted'})"
    )
    result = {
        "fp16": fp16_timing.to_dict(),
        "naive_int8": naive_int8_timing.to_dict(),
        "naive_int8_speedup": speedup_naive,
    }

    if triton_dequant_gemv_available():
        with torch.no_grad():
            fused_timing = timeit_repeated(
                lambda: triton_dequant_gemv(x, qlinear.weight_q, qlinear.scale),
                device=device, warmup=warmup, repeats=repeats,
            )
        speedup_fused = fp16_timing.median_s / fused_timing.median_s
        print(
            f"  fused Triton dequant GEMV={fused_timing.median_s * 1e6:8.1f}us "
            f"speedup vs fp16={speedup_fused:.2f}x"
        )
        result["triton_fused"] = fused_timing.to_dict()
        result["triton_fused_speedup"] = speedup_fused
    else:
        print("  Triton dequant GEMV unavailable on this device (needs sm70+) -- skipping.")
        result["triton_fused"] = None

    return result


def _load_tinystories_val_tokens(token_budget: int) -> list[int]:
    from datasets import load_dataset

    from minigpt_infer.tokenizer import EOT_TOKEN, encode

    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids: list[int] = []
    for example in ds:
        ids.extend(encode(example["text"]))
        ids.append(EOT_TOKEN)
        if len(ids) >= token_budget:
            break
    return ids[:token_budget]


@torch.no_grad()
def _perplexity(model, token_ids: list[int], block_size: int, device: str) -> float:
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(token_ids) - 1, block_size):
        chunk = token_ids[start:start + block_size + 1]
        if len(chunk) < 2:
            break
        idx = torch.tensor([chunk[:-1]], device=device)
        targets = torch.tensor([chunk[1:]], device=device)
        logits = model(idx)  # (1, T, vocab) -- ReferenceGPT's full-sequence forward
        loss = F.cross_entropy(logits.squeeze(0), targets.squeeze(0), reduction="sum")
        total_nll += loss.item()
        total_tokens += targets.numel()
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


def run_perplexity(device: str) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set -- skipping perplexity (needs the real checkpoint).")
        return {"available": False, "reason": "no HF_TOKEN"}

    from minigpt_infer.loader import load_model

    model, cfg, meta = load_model(token=token, device=device, dtype=torch.float16)
    print(f"loaded checkpoint (iter={meta.get('iter')}) for perplexity eval")

    fp16_ref = ReferenceGPT(cfg).to(device=device, dtype=torch.float16)
    fp16_ref.load_state_dict(model.state_dict())
    fp16_ref.eval()

    int8_ref = ReferenceGPT(cfg).to(device=device, dtype=torch.float16)
    int8_ref.load_state_dict(model.state_dict())
    quantize_model(int8_ref)
    int8_ref.eval()

    val_tokens = _load_tinystories_val_tokens(HELD_OUT_TOKEN_BUDGET)
    print(f"evaluating on {len(val_tokens)} held-out TinyStories validation tokens")

    fp16_ppl = _perplexity(fp16_ref, val_tokens, cfg.block_size, device)
    int8_ppl = _perplexity(int8_ref, val_tokens, cfg.block_size, device)
    relative_delta_pct = (int8_ppl / fp16_ppl - 1) * 100

    print(
        f"Perplexity: fp16={fp16_ppl:.4f} int8={int8_ppl:.4f} "
        f"relative_delta={relative_delta_pct:+.2f}% (target: <1%)"
    )
    return {
        "available": True,
        "checkpoint_iter": meta.get("iter"),
        "num_tokens": len(val_tokens),
        "fp16_perplexity": fp16_ppl,
        "int8_perplexity": int8_ppl,
        "relative_delta_pct": relative_delta_pct,
    }


def run(device: str, warmup: int, repeats: int, out_path: Path) -> dict:
    cfg = GPTConfig()
    payload = {
        "benchmark": "bench_quant",
        "prediction": "P7",
        "prediction_text": (
            "Weight-only int8 with a naive dequant-then-matmul is slower than fp16. "
            "Only a fused dequant GEMV kernel wins."
        ),
        "config": {"gpt_config": vars(cfg), "device": device, "warmup": warmup, "repeats": repeats},
        "memory": run_memory(cfg),
        "speed": run_speed(device, warmup, repeats, cfg),
        "perplexity": run_perplexity(device),
    }
    save_json(out_path, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument(
        "--out", type=Path, default=None, help="default: bench/results/<name>_<gpu>_<ts>.json",
    )
    args = ap.parse_args()
    out = args.out if args.out is not None else default_results_path("bench_quant")
    run(args.device, args.warmup, args.repeats, out)


if __name__ == "__main__":
    main()
