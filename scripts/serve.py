#!/usr/bin/env python3
"""Launch the OpenAI-compatible server (docs/PLAN.md §7 Phase 7) against the
real Project A checkpoint.

Usage:
    HF_TOKEN=... python scripts/serve.py
    HF_TOKEN=... python scripts/serve.py --host 0.0.0.0 --port 8000 --device cuda
"""

import argparse
import os

import torch
import uvicorn

from minigpt_infer.config import EngineConfig
from minigpt_infer.loader import DEFAULT_CKPT_PATH, DEFAULT_REPO_ID, load_model
from minigpt_infer.server.api import create_app
from minigpt_infer.tokenizer import EOT_TOKEN, encode, padding_vocab_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--filename", default=DEFAULT_CKPT_PATH)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kv-block-size", type=int, default=16)
    ap.add_argument("--num-kv-blocks", type=int, default=2048)
    ap.add_argument("--max-batch-size", type=int, default=64)
    ap.add_argument("--served-model-name", default="minigpt-infer")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    model, cfg, meta = load_model(args.repo_id, args.filename, token=token, device=args.device)
    model.eval()
    print(f"loaded checkpoint: iter={meta.get('iter')} params={model.num_params() / 1e6:.1f}M "
          f"device={args.device}")

    engine_cfg = EngineConfig(
        block_size=args.kv_block_size,
        num_blocks=args.num_kv_blocks,
        max_batch_size=args.max_batch_size,
    )
    vocab_mask = padding_vocab_mask(cfg.vocab_size, device=args.device)
    app = create_app(
        model, engine_cfg, encode, eot_token_id=EOT_TOKEN,
        vocab_mask=vocab_mask, served_model_name=args.served_model_name,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
