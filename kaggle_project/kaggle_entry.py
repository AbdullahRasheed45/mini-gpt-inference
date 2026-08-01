"""Script-kernel entry point for Phase 8's P100 hardware-study run
(docs/PLAN.md §7 Phase 8 point 3, prediction P9).

Pushed by hand via `kaggle kernels push` (not an automated recurring job
like mini-gpt-ddp/kaggle_project/kaggle_entry.py's training orchestrator --
this only needs to run once per code change). Clones the public
mini-gpt-inference repo fresh and runs bench_hardware.py against the real
checkpoint. Kaggle's free tier has handed out a Tesla P100 (sm60, Pascal) on
every observed API-triggered run so far (docs/PLAN.md §3) -- exactly the GPU
this benchmark exists to characterize, whatever `machine_shape` below
actually requests.

Credentials: same story as mini-gpt-ddp/kaggle_project/kaggle_entry.py --
Kaggle Secrets (kaggle_secrets.UserSecretsClient) are unreachable from a
kernel triggered via `kaggle kernels push` (confirmed empirically there), so
HF_TOKEN is injected as a literal `os.environ[...] = ...` line prepended to
this script's source immediately before pushing the rendered copy -- never
committed to git, never written to disk in this repo.

Torch reinstall required, same as Project A's DDP training kernel: Kaggle's
current base image ships torch 2.10.0+cu128, whose CUDA build has DROPPED
sm_60 (Pascal/P100) entirely -- confirmed by a real run here failing with
`CUDA error: no kernel image is available for execution on the device` on
the very first `torch.zeros(..., device="cuda")` call, not just on Triton
(docs/PLAN.md §3 only predicted Triton would fail on sm60; plain eager ops
failing too is a stricter, newer problem than the plan anticipated). Fixed
exactly as mini-gpt-ddp/kaggle_project/kaggle_entry.py already fixed it:
force-reinstall torch 2.6.0's cu118 build, whose TORCH_CUDA_ARCH_LIST still
includes 6.0.
"""

import os
import shutil
import subprocess

REPO_URL = "https://github.com/AbdullahRasheed45/mini-gpt-inference.git"
WORKDIR = "/kaggle/working/mini-gpt-inference"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    if not os.path.isdir(WORKDIR):
        run(["git", "clone", "--depth", "1", REPO_URL, WORKDIR])
    os.chdir(WORKDIR)

    run(["pip", "install", "-q", "--force-reinstall", "torch==2.6.0",
         "--index-url", "https://download.pytorch.org/whl/cu118"])
    run(["python", "-c",
         "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
         "'gpu', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0), "
         "'archs', torch.cuda.get_arch_list())"])
    run(["pip", "install", "-q", "-e", "."])
    run(["python", "-m", "bench.bench_hardware", "--no-plot"])

    # `kaggle kernels output` reliably pulls files sitting directly under
    # /kaggle/working; copy the JSON there explicitly rather than trust a
    # nested-directory pull-through that isn't documented behavior.
    results_dir = os.path.join(WORKDIR, "bench", "results")
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("bench_hardware_"):
            src = os.path.join(results_dir, fname)
            dst = os.path.join("/kaggle/working", fname)
            shutil.copy(src, dst)
            print(f"copied {src} -> {dst}")
            with open(src) as f:
                content = f.read()
            print("=== BENCH_HARDWARE_JSON_START ===")
            print(content)
            print("=== BENCH_HARDWARE_JSON_END ===")


if __name__ == "__main__":
    main()
