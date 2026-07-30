"""Load the Project A checkpoint from the Hugging Face Hub into a GPT module.

The checkpoint is a full *training* checkpoint (model + optimizer + scaler +
RNG state), not a weights-only file -- see mini-gpt-ddp/train.py. Only the
"model" key is used here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from minigpt_infer.config import GPTConfig
from minigpt_infer.model import GPT

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Abdullahrasheed45/mini-gpt-ddp"
DEFAULT_CKPT_PATH = "checkpoints/ddp_2gpu.pt"


def download_checkpoint(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_CKPT_PATH,
    token: str | None = None,
) -> Path:
    """Download (or reuse the local HF cache for) the checkpoint. Returns its path."""
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", token=token)
    return Path(local_path)


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def reconstruct_config(train_cfg: dict) -> GPTConfig:
    """Rebuild GPTConfig from a training checkpoint's saved argparse namespace.

    ckpt["config"] is `vars(argparse.Namespace)` from mini-gpt-ddp/train.py's
    get_args() -- it carries n_layer/n_head/n_embd/block_size but NOT
    vocab_size (that came from GPTConfig's own default and was never a CLI
    arg). Falls back to GPTConfig()'s defaults for anything absent, rather
    than assuming every key exists -- pulled out of load_model() so this is
    testable without a network call or the real checkpoint.
    """
    defaults = GPTConfig()
    return GPTConfig(
        vocab_size=defaults.vocab_size,
        block_size=train_cfg.get("block_size", defaults.block_size),
        n_layer=train_cfg.get("n_layer", defaults.n_layer),
        n_head=train_cfg.get("n_head", defaults.n_head),
        n_embd=train_cfg.get("n_embd", defaults.n_embd),
        dropout=0.0,   # always eval mode for inference regardless of training config
        bias=defaults.bias,
    )


def _load_raw_checkpoint(path: Path, map_location: str | torch.device) -> dict:
    """torch.load with a weights_only attempt first, falling back with a warning.

    The checkpoint holds only tensors and plain Python containers (dict/list/
    float/int -- see train.py's torch.save call), so weights_only=True *should*
    succeed. We try it first because it's the safe default; if it fails (e.g.
    the RNG-state tensors trip torch's allowlist on some version), we fall back
    to weights_only=False with a loud warning rather than silently downgrading
    safety. This checkpoint is one we trained ourselves in Project A, not an
    untrusted third-party file -- weights_only=False here is arbitrary code
    execution *in principle*, and it is an accepted, deliberate choice for a
    checkpoint we produced, not a default to reuse for arbitrary inputs.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:  # noqa: BLE001 - genuinely want to fall back on any failure
        logger.warning(
            "torch.load(weights_only=True) failed (%s: %s); falling back to "
            "weights_only=False. This is only safe for a checkpoint you trust "
            "(we trained this one ourselves in Project A).",
            type(e).__name__, e,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_model(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_CKPT_PATH,
    token: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[GPT, GPTConfig, dict]:
    """Download the checkpoint, reconstruct GPTConfig, and load weights.

    Returns (model, config, raw_checkpoint_metadata) where raw_checkpoint_metadata
    is the checkpoint dict minus "model" (so callers can inspect iter/config
    without holding the full state_dict in memory twice).
    """
    path = download_checkpoint(repo_id, filename, token)
    ckpt = _load_raw_checkpoint(path, map_location=device)

    cfg = reconstruct_config(ckpt.get("config", {}))
    model = GPT(cfg)
    state_dict = ckpt["model"]
    # DDP/torch.compile wrap state_dict keys in "module."/"_orig_mod." prefixes;
    # strip either if present so this loads regardless of how it was trained.
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "_orig_mod.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match model architecture: "
            f"missing={missing}, unexpected={unexpected}. "
            f"Reconstructed config: {cfg}"
        )

    # Weight tying must survive the load -- a state_dict load can silently
    # break tying if lm_head.weight and tok_emb.weight happen to arrive as
    # separate (but numerically-equal) tensors instead of the same object.
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), (
        "weight tying broken after loading state_dict "
        "(lm_head.weight and tok_emb.weight are no longer the same tensor)"
    )

    model = model.to(device=device, dtype=dtype)
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "model"}
    logger.info(
        "loaded checkpoint from hf://%s/%s (iter=%s, %.1fM params)",
        repo_id, filename, meta.get("iter"), model.num_params() / 1e6,
    )
    return model, cfg, meta
