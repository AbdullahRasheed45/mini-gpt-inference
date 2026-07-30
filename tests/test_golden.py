"""Phase 0 acceptance criterion: greedy generation from the real checkpoint on
the fixed prompt "Once upon a time" is recorded and must not silently drift.

Skipped whenever HF_TOKEN isn't set -- CI must never need the real (private,
614 MB) checkpoint (docs/PLAN.md §9). Run locally with:
    HF_TOKEN=... pytest tests/test_golden.py -v

To regenerate the golden file after an intentional change to model.py/loader.py:
    HF_TOKEN=... python tests/test_golden.py --record
"""

import json
import os
from pathlib import Path

import pytest
import torch

GOLDEN_PATH = Path(__file__).parent / "golden" / "once_upon_a_time_greedy.json"
PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 40

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="needs HF_TOKEN to download the real (private) checkpoint",
)


@torch.no_grad()
def _greedy(model, idx: torch.Tensor, max_new_tokens: int,
            vocab_mask: torch.Tensor) -> torch.Tensor:
    """Deterministic argmax decode against the Phase-0 (verbatim-port) GPT.

    Not a method on model.GPT: Phase 0 keeps model.py an exact port of Project
    A (see docs/PLAN.md Phase 0), and Project A's generate() never had a true
    greedy mode. Kept local to this test rather than added to model.py.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].masked_fill(vocab_mask, float("-inf"))
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def _run() -> dict:
    from minigpt_infer.loader import load_model
    from minigpt_infer.tokenizer import decode, encode, padding_vocab_mask

    model, cfg, meta = load_model(device="cpu", dtype=torch.float32)
    mask = padding_vocab_mask(cfg.vocab_size)
    idx = torch.tensor([encode(PROMPT)])
    out = _greedy(model, idx, MAX_NEW_TOKENS, mask)
    token_ids = out[0].tolist()
    return {
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "checkpoint_iter": meta.get("iter"),
        "token_ids": token_ids,
        "text": decode(token_ids),
    }


def test_golden_greedy_output_matches_recorded():
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} does not exist. Generate it with: "
        f"HF_TOKEN=... python {__file__} --record"
    )
    recorded = json.loads(GOLDEN_PATH.read_text())
    actual = _run()

    assert actual["checkpoint_iter"] == recorded["checkpoint_iter"], (
        "checkpoint on the Hub has changed (different iter) -- golden file is "
        "stale, regenerate it deliberately if this is expected"
    )
    assert actual["token_ids"] == recorded["token_ids"], (
        "greedy decode drifted from the recorded golden output -- this means "
        "model.py, loader.py, or tokenizer.py changed behavior. If intentional, "
        "regenerate with --record; if not, this is a real regression."
    )


if __name__ == "__main__":
    import sys

    if "--record" not in sys.argv:
        print(__doc__)
        sys.exit(1)
    result = _run()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {GOLDEN_PATH}")
    print("text:", result["text"])
