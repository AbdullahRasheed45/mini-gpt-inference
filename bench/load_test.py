"""Phase 8 load test (docs/PLAN.md §7 Phase 8, §10 rules).

A Poisson-arrival load generator against a real `LLMEngine`: request
inter-arrival times are drawn from Exp(lambda) (a Poisson arrival process —
the standard model for independent request arrivals), prompt and output
lengths from a configurable lognormal (short-tailed, roughly TinyStories-
shaped). Requests are submitted at their scheduled wall-clock arrival time,
not all at once back-to-back -- submitting everything up front would measure
batch throughput, not serving latency under a *given arrival rate*, and the
whole point of a QPS sweep is to find where that rate starts to matter.

`engine.step()` is driven in a tight loop between arrivals so already-running
and newly-arriving requests interleave exactly as continuous batching
intends (prefill-priority: a new arrival gets scheduled ahead of decode steps
for already-running requests, per docs/ARCHITECTURE.md §4).

Per request: TTFT (arrival -> first token), TPOT (mean inter-token gap
during decode, excluding the first token), E2E (arrival -> finish). Sweeps
lambda; reports p50/p95/p99 of each metric per lambda, and the highest
lambda where p95 TTFT and p95 TPOT both stay under a **stated** SLO --
docs/PLAN.md §7 Phase 8 point 2 is explicit that a "goodput" number without
a stated SLO is meaningless, so the SLO thresholds are CLI flags, not a
buried constant.

Usage:
    python bench/load_test.py                                   # full sweep
    python bench/load_test.py --lambdas 1,2 --num-requests 10    # fast sanity check
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import torch

from bench.common import REPO_ROOT, default_results_path, save_json
from minigpt_infer.config import EngineConfig, GPTConfig, SamplingParams
from minigpt_infer.engine.engine import LLMEngine
from minigpt_infer.engine.request import Request
from minigpt_infer.model import GPT
from minigpt_infer.tokenizer import padding_vocab_mask

DEFAULT_LAMBDAS = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
NUM_REQUESTS = 40
PROMPT_LEN_MU = 1.8   # lognormal params -> median prompt_len ~= exp(mu) ~= 6 tokens
PROMPT_LEN_SIGMA = 0.5
OUTPUT_LEN_MU = 2.8   # median output_len ~= exp(mu) ~= 16 tokens
OUTPUT_LEN_SIGMA = 0.6
MAX_PROMPT_LEN = 32
MAX_OUTPUT_LEN = 64


def _sample_lengths(n: int, mu: float, sigma: float, cap: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [max(1, min(cap, int(rng.lognormvariate(mu, sigma)))) for _ in range(n)]


def _make_requests(cfg: GPTConfig, num_requests: int, seed: int) -> list[tuple[list[int], int]]:
    """Returns [(prompt_token_ids, output_len), ...]."""
    torch.manual_seed(seed)
    prompt_lens = _sample_lengths(
        num_requests, PROMPT_LEN_MU, PROMPT_LEN_SIGMA, MAX_PROMPT_LEN, seed,
    )
    output_lens = _sample_lengths(
        num_requests, OUTPUT_LEN_MU, OUTPUT_LEN_SIGMA, MAX_OUTPUT_LEN, seed + 1,
    )
    requests = []
    for plen, olen in zip(prompt_lens, output_lens, strict=True):
        prompt = torch.randint(0, cfg.vocab_size, (plen,)).tolist()
        requests.append((prompt, olen))
    return requests


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan")}
    s = sorted(samples)

    def pct(p: float) -> float:
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "mean": statistics.mean(s)}


def run_one_lambda(
    model: GPT, cfg: GPTConfig, engine_cfg: EngineConfig, lam: float,
    requests: list[tuple[list[int], int]], arrival_seed: int, device: str,
) -> dict:
    num_requests = len(requests)
    rng = random.Random(arrival_seed)
    arrival_offsets: list[float] = []
    t = 0.0
    for _ in range(num_requests):
        t += rng.expovariate(lam)
        arrival_offsets.append(t)

    # Without this, a random-init model's unconstrained softmax can sample a
    # padding-region id (this cfg's vocab_size=50304 pads GPT-2's real 50257
    # ids up to a multiple of 64) that tiktoken's decoder rejects outright --
    # a real crash reproduced while writing this script, not a hypothetical.
    vocab_mask = padding_vocab_mask(cfg.vocab_size, device=device)
    engine = LLMEngine(model, engine_cfg, vocab_mask=vocab_mask)
    per_request: dict[int, dict] = {
        i: {"arrival": None, "first_token": None, "finish": None, "token_times": []}
        for i in range(num_requests)
    }

    n_submitted = 0
    n_finished = 0
    start = time.perf_counter()
    with torch.no_grad():
        while n_finished < num_requests:
            now = time.perf_counter() - start
            while n_submitted < num_requests and arrival_offsets[n_submitted] <= now:
                i = n_submitted
                prompt, olen = requests[i]
                engine.add_request(
                    Request(f"r{i}", prompt, SamplingParams(temperature=0.0, max_tokens=olen)),
                )
                per_request[i]["arrival"] = now
                n_submitted += 1

            if engine.has_unfinished_requests():
                outs = engine.step()
                now2 = time.perf_counter() - start
                for out in outs:
                    i = int(out.request_id[1:])
                    m = per_request[i]
                    if m["first_token"] is None:
                        m["first_token"] = now2
                    m["token_times"].append(now2)
                    if out.finished:
                        m["finish"] = now2
                        n_finished += 1
            elif n_submitted < num_requests:
                # Nothing to step yet -- idle until the next scheduled arrival
                # instead of busy-spinning the CPU for no reason.
                sleep_for = arrival_offsets[n_submitted] - (time.perf_counter() - start)
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.005))

    ttft_samples = []
    tpot_samples = []
    e2e_samples = []
    for m in per_request.values():
        ttft_samples.append(m["first_token"] - m["arrival"])
        e2e_samples.append(m["finish"] - m["arrival"])
        gaps = [b - a for a, b in zip(m["token_times"], m["token_times"][1:], strict=False)]
        if gaps:
            tpot_samples.append(statistics.mean(gaps))

    wall_s = time.perf_counter() - start
    achieved_qps = num_requests / wall_s
    return {
        "lambda_target": lam,
        "achieved_qps": achieved_qps,
        "wall_s": wall_s,
        "ttft_s": _percentiles(ttft_samples),
        "tpot_s": _percentiles(tpot_samples),
        "e2e_s": _percentiles(e2e_samples),
        "ttft_samples_s": ttft_samples,
        "tpot_samples_s": tpot_samples,
        "e2e_samples_s": e2e_samples,
    }


def run(
    device: str, lambdas: list[float], num_requests: int,
    slo_ttft_p95_s: float, slo_tpot_p95_s: float,
    out_path: Path, plot_path: Path | None,
) -> dict:
    cfg = GPTConfig()
    torch.manual_seed(0)
    model = GPT(cfg).to(device).eval()
    engine_cfg = EngineConfig(block_size=16, num_blocks=4096, max_batch_size=64)

    # Fixed across the whole sweep (docs/PLAN.md §10 rule 5: identical
    # workload, only lambda varies) -- otherwise a higher lambda could draw
    # a shorter-output-length sample by chance and look artificially faster,
    # confounding exactly the comparison this sweep exists to make.
    requests = _make_requests(cfg, num_requests, seed=0)

    results = []
    max_sustainable_qps = None
    for i, lam in enumerate(lambdas):
        row = run_one_lambda(
            model, cfg, engine_cfg, lam, requests, arrival_seed=100 + i, device=device,
        )
        meets_slo = (
            row["ttft_s"]["p95"] <= slo_ttft_p95_s and row["tpot_s"]["p95"] <= slo_tpot_p95_s
        )
        row["meets_slo"] = meets_slo
        if meets_slo:
            max_sustainable_qps = lam
        results.append(row)
        ttft_p50_ms = row["ttft_s"]["p50"] * 1000
        ttft_p95_ms = row["ttft_s"]["p95"] * 1000
        tpot_p50_ms = row["tpot_s"]["p50"] * 1000
        tpot_p95_ms = row["tpot_s"]["p95"] * 1000
        print(
            f"lambda={lam:6.1f} req/s  achieved={row['achieved_qps']:6.1f} req/s  "
            f"TTFT p50/p95={ttft_p50_ms:7.1f}/{ttft_p95_ms:7.1f}ms  "
            f"TPOT p50/p95={tpot_p50_ms:6.1f}/{tpot_p95_ms:6.1f}ms  "
            f"{'OK' if meets_slo else 'SLO VIOLATED'}"
        )

    payload = {
        "benchmark": "load_test",
        "slo": {
            "ttft_p95_s": slo_ttft_p95_s, "tpot_p95_s": slo_tpot_p95_s,
            "description": "max lambda where BOTH p95 TTFT and p95 TPOT stay under threshold",
        },
        "max_sustainable_qps": max_sustainable_qps,
        "config": {
            "gpt_config": vars(cfg), "num_requests_per_lambda": num_requests,
            "prompt_len_lognormal": {
                "mu": PROMPT_LEN_MU, "sigma": PROMPT_LEN_SIGMA, "cap": MAX_PROMPT_LEN,
            },
            "output_len_lognormal": {
                "mu": OUTPUT_LEN_MU, "sigma": OUTPUT_LEN_SIGMA, "cap": MAX_OUTPUT_LEN,
            },
            "engine_config": vars(engine_cfg), "device": device,
        },
        "results": results,
    }
    save_json(out_path, payload)

    if plot_path is not None:
        _plot(results, slo_ttft_p95_s, slo_tpot_p95_s, plot_path)

    return payload


def _plot(results: list[dict], slo_ttft: float, slo_tpot: float, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = [r["lambda_target"] for r in results]
    ttft_p50 = [r["ttft_s"]["p50"] * 1000 for r in results]
    ttft_p95 = [r["ttft_s"]["p95"] * 1000 for r in results]
    tpot_p50 = [r["tpot_s"]["p50"] * 1000 for r in results]
    tpot_p95 = [r["tpot_s"]["p95"] * 1000 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(lambdas, ttft_p50, marker="o", label="p50")
    ax1.plot(lambdas, ttft_p95, marker="o", label="p95")
    ax1.axhline(slo_ttft * 1000, color="red", linestyle="--", label="SLO")
    ax1.set_xlabel("arrival rate lambda (req/s)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to first token vs load")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(lambdas, tpot_p50, marker="o", label="p50")
    ax2.plot(lambdas, tpot_p95, marker="o", label="p95")
    ax2.axhline(slo_tpot * 1000, color="red", linestyle="--", label="SLO")
    ax2.set_xlabel("arrival rate lambda (req/s)")
    ax2.set_ylabel("TPOT (ms)")
    ax2.set_title("Inter-token latency vs load")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    ap.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    ap.add_argument("--slo-ttft-p95-ms", type=float, default=200.0)
    ap.add_argument("--slo-tpot-p95-ms", type=float, default=50.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=REPO_ROOT / "docs/img/load_test.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    out = args.out if args.out is not None else default_results_path("load_test")
    run(
        args.device, lambdas, args.num_requests,
        args.slo_ttft_p95_ms / 1000, args.slo_tpot_p95_ms / 1000,
        out, None if args.no_plot else args.plot,
    )


if __name__ == "__main__":
    main()
