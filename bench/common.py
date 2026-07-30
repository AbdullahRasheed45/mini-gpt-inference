"""Shared benchmark harness -- implements docs/PLAN.md §10 once, reused by
every bench/*.py script so every JSON artifact under docs/ was produced under
the same rules (warmup, synchronize, median/IQR, full environment stamp).

§10 rules encoded here:
  1. warm up >=10 iters before timing
  2. torch.cuda.synchronize() immediately before/after each timed call
  3. time.perf_counter(); report median + IQR, not mean
  4. >=30 repeats for latency (caller's default; can be overridden)
  6. record git SHA, torch/CUDA version, GPU name, driver, full config, and
     all raw samples in every JSON (see env_info() / save_json())
  7. report torch.cuda.max_memory_allocated() alongside latency
Rules 5 (fixed seeds, identical prompts) and 8 (label GPU/SHA when comparing)
are the caller's responsibility -- this module only knows how to time a thunk
and stamp the result.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _nvidia_driver_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        )
        return out.strip().splitlines()[0]
    except Exception:
        return None


def env_info() -> dict[str, Any]:
    """Everything §10 rule 6 requires to make a JSON artifact self-describing."""
    info: dict[str, Any] = {
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "driver_version": None,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["driver_version"] = _nvidia_driver_version()
    return info


def sync_if_cuda(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@dataclass
class TimingResult:
    samples_s: list[float]
    median_s: float
    iqr_s: float
    min_s: float
    max_s: float
    max_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_s": self.samples_s,
            "median_s": self.median_s,
            "iqr_s": self.iqr_s,
            "min_s": self.min_s,
            "max_s": self.max_s,
            "max_memory_bytes": self.max_memory_bytes,
        }


def _iqr(sorted_samples: list[float]) -> float:
    n = len(sorted_samples)
    lower = sorted_samples[: n // 2]
    upper = sorted_samples[(n + 1) // 2:]
    q1 = statistics.median(lower) if lower else sorted_samples[0]
    q3 = statistics.median(upper) if upper else sorted_samples[-1]
    return q3 - q1


def timeit_repeated(
    fn: Callable[[], Any],
    *,
    device: str | torch.device = "cpu",
    warmup: int = 10,
    repeats: int = 30,
) -> TimingResult:
    """Time `fn()` `repeats` times after `warmup` untimed calls.

    Every call is individually bracketed by synchronize() on CUDA (rule 2) --
    not just once at the start/end -- so async kernel launches from one repeat
    can't bleed into the next repeat's timing window.
    """
    dev = torch.device(device)
    for _ in range(warmup):
        fn()
    sync_if_cuda(dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    samples: list[float] = []
    for _ in range(repeats):
        sync_if_cuda(dev)
        t0 = time.perf_counter()
        fn()
        sync_if_cuda(dev)
        t1 = time.perf_counter()
        samples.append(t1 - t0)

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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes {env: env_info(), **payload} -- callers should not duplicate env
    info themselves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {"env": env_info(), **payload}
    path.write_text(json.dumps(full, indent=2))
    print(f"wrote {path}")
