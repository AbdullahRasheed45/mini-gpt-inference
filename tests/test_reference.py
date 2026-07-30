"""Phase 0: the oracle (reference.ReferenceGPT) is internally consistent.

The model.GPT vs. ReferenceGPT cross-check (rung 1 of the correctness ladder,
docs/PLAN.md §8) lives in test_cache.py, not here: as of Phase 1, model.GPT's
forward() takes a ForwardBatch instead of a raw idx tensor, so a same-weights
comparison has to go through the cache-aware interface to mean anything.
ReferenceGPT itself is frozen and never changes (see reference.py's module
docstring), so its own tests stay here permanently.
"""

import torch

from minigpt_infer.reference import ReferenceGPT
from minigpt_infer.tokenizer import padding_vocab_mask
from tests.helpers import tiny_gpt_config


def test_reference_forward_shape():
    cfg = tiny_gpt_config()
    model = ReferenceGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits = model(idx)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)


def test_reference_weight_tying():
    model = ReferenceGPT(tiny_gpt_config())
    assert model.lm_head.weight is model.tok_emb.weight


def test_greedy_generate_is_deterministic():
    cfg = tiny_gpt_config()
    torch.manual_seed(0)
    model = ReferenceGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out1 = model.greedy_generate(idx.clone(), max_new_tokens=10)
    out2 = model.greedy_generate(idx.clone(), max_new_tokens=10)
    assert torch.equal(out1, out2), "greedy decoding must be exactly reproducible"


def test_greedy_generate_respects_vocab_mask():
    cfg = tiny_gpt_config()
    torch.manual_seed(0)
    model = ReferenceGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    # mask everything except token 0 -- every generated token must be 0
    mask = torch.ones(cfg.vocab_size, dtype=torch.bool)
    mask[0] = False
    out = model.greedy_generate(idx.clone(), max_new_tokens=10, vocab_mask=mask)
    assert (out[0, 4:] == 0).all()


def test_padding_vocab_mask_shape_and_content():
    mask = padding_vocab_mask(vocab_size=50304)
    assert mask.shape == (50304,)
    assert mask.sum().item() == 50304 - 50257
    assert not mask[50256]  # last real GPT-2 id (EOT) must NOT be masked
    assert mask[50257]      # first padding row must be masked
    assert mask[50303]      # last padding row must be masked
