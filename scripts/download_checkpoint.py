#!/usr/bin/env python3
"""Download the Project A checkpoint and print a quick sanity summary.

Usage:
    HF_TOKEN=... python scripts/download_checkpoint.py
    HF_TOKEN=... python scripts/download_checkpoint.py --prompt "Once upon a time"
"""

import argparse
import os

import torch

from minigpt_infer.loader import DEFAULT_CKPT_PATH, DEFAULT_REPO_ID, load_model
from minigpt_infer.tokenizer import decode, encode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--filename", default=DEFAULT_CKPT_PATH)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, meta = load_model(args.repo_id, args.filename, token=token, device=device)
    print(f"config: {cfg}")
    print(f"checkpoint iter: {meta.get('iter')}")
    print(f"params (non-pos-emb): {model.num_params() / 1e6:.1f}M")

    idx = torch.tensor([encode(args.prompt)], device=device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=0.8, top_k=50)
    print("sample:", decode(out[0].tolist()))


if __name__ == "__main__":
    main()
