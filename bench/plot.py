"""Phase 8 plotting entry point (docs/PLAN.md §7 Phase 8 point 4).

"plot.py reads only those JSONs. No number reaches the README except
through this pipeline" -- every other bench/*.py script already writes its
own single-run plot inline (`_plot()`, called right after `save_json()`),
which already satisfies that rule for single-GPU results. This script's
distinct job is the one plot that can only be made by reading MULTIPLE
committed JSONs at once: overlaying bench_hardware.py's T4 run and its P100
run into the actual crossover chart (P9) -- something no single benchmark
run can produce by itself, since each run only ever sees its own GPU.

It also re-derives every other committed plot straight from the latest JSON
of its kind, with no benchmark re-execution, so `scripts/run_all_benchmarks.py`
can regenerate every figure the README references from disk alone.

Usage:
    python -m bench.plot                  # regenerate everything discoverable
    python -m bench.plot --only hardware   # just the T4-vs-P100 crossover
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bench.common import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "bench" / "results"
IMG_DIR = REPO_ROOT / "docs" / "img"

_ARTIFACT_RE = re.compile(r"^(?P<name>.+)_(?P<gpu>[^_]+)_(?P<timestamp>\d{8}T\d{6}Z)\.json$")


def _all_artifacts() -> list[dict]:
    """Every bench/results/*.json, parsed into {name, gpu, timestamp, path}."""
    out = []
    if not RESULTS_DIR.exists():
        return out
    for path in RESULTS_DIR.glob("*.json"):
        m = _ARTIFACT_RE.match(path.name)
        if not m:
            continue
        out.append({**m.groupdict(), "path": path})
    return out


def _latest(name: str, gpu_substring: str | None = None) -> dict | None:
    candidates = [a for a in _all_artifacts() if a["name"] == name]
    if gpu_substring is not None:
        candidates = [a for a in candidates if gpu_substring in a["gpu"].lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a["timestamp"])


def _load(artifact: dict) -> dict:
    return json.loads(artifact["path"].read_text())


def plot_hardware_crossover(out_path: Path = IMG_DIR / "hardware_crossover.png") -> bool:
    """The P9 chart: T4 and P100 bench_hardware.py runs overlaid on one
    figure. Needs both -- if only one GPU has been benchmarked so far, this
    is skipped (not a partial/misleading single-line "crossover")."""
    t4 = _latest("bench_hardware", "t4")
    p100 = _latest("bench_hardware", "p100")
    if t4 is None or p100 is None:
        missing = [n for n, a in [("t4", t4), ("p100", p100)] if a is None]
        print(f"skipping hardware crossover plot: no bench_hardware JSON for {missing}")
        return False

    t4_data, p100_data = _load(t4), _load(p100)
    if not t4_data.get("available") or not p100_data.get("available"):
        print("skipping hardware crossover plot: a run was HF_TOKEN-skipped, not real data")
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for data, label, color in [(t4_data, "T4", "tab:blue"), (p100_data, "P100", "tab:orange")]:
        bs = [r["batch_size"] for r in data["results"]]
        latency = [r["per_token_latency_ms"] for r in data["results"]]
        throughput = [r["throughput_tok_per_s"] for r in data["results"]]
        ax1.plot(bs, latency, marker="o", label=label, color=color)
        ax2.plot(bs, throughput, marker="o", label=label, color=color)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("per-token decode latency (ms, median)")
    ax1.set_title("P9: decode latency, T4 vs P100")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("throughput (tok/s, system-wide)")
    ax2.set_title("P9: decode throughput, T4 vs P100")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return True


def plot_load_test(out_path: Path = IMG_DIR / "load_test.png") -> bool:
    artifact = _latest("load_test")
    if artifact is None:
        print("skipping load_test plot: no load_test JSON found")
        return False
    data = _load(artifact)
    if not data.get("results"):
        print("skipping load_test plot: JSON has no results (empty sweep?)")
        return False

    from bench.load_test import _plot as _load_test_plot

    slo = data["slo"]
    _load_test_plot(data["results"], slo["ttft_p95_s"], slo["tpot_p95_s"], out_path)
    return True


PLOTTERS = {
    "hardware": plot_hardware_crossover,
    "load_test": plot_load_test,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(PLOTTERS), default=None)
    args = ap.parse_args()

    targets = [args.only] if args.only else list(PLOTTERS)
    for name in targets:
        PLOTTERS[name]()


if __name__ == "__main__":
    main()
