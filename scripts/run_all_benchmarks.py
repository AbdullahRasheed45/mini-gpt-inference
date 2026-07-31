#!/usr/bin/env python3
"""Regenerates every benchmark number this project publishes (docs/PLAN.md
§7 Phase 8 acceptance: "python scripts/run_all_benchmarks.py regenerates
every published number. Every README figure traces to a committed JSON.").

Runs each bench/*.py script as a subprocess (isolates torch/RNG/device state
between them -- one script's `torch.manual_seed` or CUDA context shouldn't
leak into the next) in Phase order, then `bench.plot` to regenerate every
figure from the JSONs those scripts just wrote. Scripts that need the real
checkpoint (bench_quant's perplexity section, bench_specdec, bench_hardware)
already skip themselves gracefully when HF_TOKEN isn't set -- this script
doesn't special-case that, it just reports what ran and what didn't.

Usage:
    python scripts/run_all_benchmarks.py                  # full spec (§10): slow
    python scripts/run_all_benchmarks.py --fast            # reduced warmup/repeats
    python scripts/run_all_benchmarks.py --only bench_hardware,load_test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (module, extra CLI args for a --fast run, accepts --device). Order follows
# docs/PLAN.md's phase order, not alphabetical -- a reader scanning the
# output top-to-bottom sees the project's own history. bench_kernels is
# GPU-only and checks torch.cuda.is_available() itself; it has no --device
# or --warmup/--repeats flags at all, unlike every other script here.
BENCH_MODULES: list[tuple[str, list[str], bool]] = [
    ("bench.bench_kvcache", ["--warmup", "1", "--repeats", "3"], True),
    ("bench.bench_batching", ["--warmup", "1", "--repeats", "3", "--max-bs", "16"], True),
    ("bench.bench_paged", ["--warmup", "1", "--repeats", "3"], True),
    ("bench.bench_kernels", [], False),
    ("bench.bench_quant", ["--warmup", "1", "--repeats", "3"], True),
    ("bench.bench_specdec", ["--warmup", "1", "--repeats", "3"], True),
    ("bench.bench_hardware", ["--warmup", "1", "--repeats", "3", "--max-bs", "8"], True),
    ("bench.load_test", ["--num-requests", "8", "--lambdas", "5,20"], True),
]


def run_one(
    module: str, fast_args: list[str], accepts_device: bool, fast: bool, device: str | None,
) -> bool:
    cmd = [sys.executable, "-m", module]
    if fast:
        cmd += fast_args
    if device and accepts_device:
        cmd += ["--device", device]
    print(f"\n{'=' * 70}\n{module}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fast", action="store_true",
        help="reduced warmup/repeats/batch sizes -- for a quick local regen, not §10 spec",
    )
    ap.add_argument(
        "--device", default=None, help="forwarded to every script that accepts it",
    )
    ap.add_argument(
        "--only", default=None,
        help="comma-separated module suffixes to run, e.g. 'bench_hardware,load_test'",
    )
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    modules = BENCH_MODULES
    if args.only:
        wanted = set(args.only.split(","))
        modules = [
            (m, a, d) for m, a, d in BENCH_MODULES if m.rsplit(".", 1)[-1] in wanted
        ]

    results = {}
    for module, fast_args, accepts_device in modules:
        results[module] = run_one(module, fast_args, accepts_device, args.fast, args.device)

    if not args.skip_plot:
        print(f"\n{'=' * 70}\nbench.plot\n{'=' * 70}")
        subprocess.run([sys.executable, "-m", "bench.plot"], cwd=REPO_ROOT)

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    for module, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {module}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
