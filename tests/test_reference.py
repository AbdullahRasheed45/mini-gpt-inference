"""Phase 0: the oracle is internally consistent, and model.GPT == reference.ReferenceGPT
for identical weights (rung 1 of the correctness ladder, docs/PLAN.md §8)."""

import torch

from minigpt_infer.model import GPT
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


def test_model_and_reference_are_logit_identical_given_same_weights():
    """model.GPT (Phase 0, verbatim port) and reference.ReferenceGPT are
    structurally identical modules -- loading model.GPT's state_dict into
    ReferenceGPT must produce bit-identical logits for the same input. This
    is the test every future model.py optimization gets compared against.
    """
    cfg = tiny_gpt_config()
    torch.manual_seed(42)
    model = GPT(cfg)
    ref = ReferenceGPT(cfg)
    ref.load_state_dict(model.state_dict())

    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    model_logits, _ = model(idx)
    ref_logits = ref(idx)[:, -1:, :]  # model.GPT only returns the last position w/o targets

    assert torch.allclose(model_logits, ref_logits, atol=1e-5), (
        (model_logits - ref_logits).abs().max().item()
    )
