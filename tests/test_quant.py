"""Phase 5: int8 weight-only quantization correctness (docs/PLAN.md §7
Phase 5 acceptance, §8 rung 9). The Triton fused-GEMV path is GPU-only and
tested separately (tests/test_quant_triton.py); everything here runs on CPU.
"""

import torch

from minigpt_infer.model import GPT
from minigpt_infer.quant.int8 import (
    QuantizedLinear,
    dequantize_per_channel,
    quantize_model,
    quantize_per_channel,
    quantized_state_dict_bytes,
)
from tests.helpers import tiny_gpt_config


def test_quantize_dequantize_roundtrip_error_bounded():
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    w_q, scale = quantize_per_channel(weight)
    w_dq = dequantize_per_channel(w_q, scale)

    assert w_q.dtype == torch.int8
    assert w_q.shape == weight.shape
    # Symmetric 8-bit quantization: per-element error is bounded by half a
    # quantization step, i.e. scale/2 for that row.
    max_err_per_row = (scale / 2).unsqueeze(1).expand_as(weight)
    assert (w_dq - weight).abs().le(max_err_per_row + 1e-6).all()


def test_quantize_per_channel_uses_full_int8_range_for_the_max_element():
    weight = torch.tensor([[1.0, -4.0, 2.0], [10.0, 0.0, -10.0]])
    w_q, scale = quantize_per_channel(weight)
    # The largest-magnitude element in each row must land exactly on +-127
    # (that's the definition of this scale choice) modulo rounding.
    assert w_q[0].abs().max().item() == 127
    assert w_q[1].abs().max().item() == 127


def test_quantize_per_channel_handles_an_all_zero_row():
    weight = torch.zeros(2, 8)
    w_q, scale = quantize_per_channel(weight)
    assert not torch.isnan(scale).any()
    assert not torch.isinf(scale).any()
    assert torch.equal(w_q, torch.zeros_like(w_q))


def test_quantized_linear_matches_reference_linear_within_quantization_error():
    torch.manual_seed(1)
    linear = torch.nn.Linear(64, 32, bias=True)
    qlinear = QuantizedLinear.from_linear(linear)

    x = torch.randn(4, 64)
    ref = linear(x)
    got = qlinear(x)

    # Not exact (that's the whole point of quantization) -- bound the error
    # via the same per-row scale used to quantize, propagated through the
    # matmul: worst case each of the 64 input dims contributes up to
    # scale/2 error, so use a generous multiplicative tolerance instead of
    # re-deriving the exact analytic bound.
    assert torch.allclose(got, ref, atol=0.5, rtol=0.05), (got - ref).abs().max().item()


def test_quantized_linear_is_a_drop_in_module():
    """Same forward signature/shape contract as nn.Linear -- this is what
    lets quantize_model() swap it in without touching model.py at all."""
    torch.manual_seed(2)
    linear = torch.nn.Linear(16, 8, bias=False)
    qlinear = QuantizedLinear.from_linear(linear)
    x = torch.randn(3, 16)
    out = qlinear(x)
    assert out.shape == (3, 8)
    assert qlinear.bias is None


def test_quantize_model_replaces_only_the_intended_linears():
    cfg = tiny_gpt_config()
    torch.manual_seed(3)
    model = GPT(cfg)

    quantize_model(model)

    for block in model.blocks:
        assert isinstance(block.attn.qkv, QuantizedLinear)
        assert isinstance(block.attn.proj, QuantizedLinear)
        assert isinstance(block.mlp.fc, QuantizedLinear)
        assert isinstance(block.mlp.proj, QuantizedLinear)

    # lm_head/tok_emb must be untouched, and weight tying must still hold.
    assert isinstance(model.tok_emb, torch.nn.Embedding)
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert model.lm_head.weight is model.tok_emb.weight


def test_quantize_model_forward_still_runs_end_to_end():
    from minigpt_infer.batch import ForwardBatch

    cfg = tiny_gpt_config()
    torch.manual_seed(4)
    model = GPT(cfg)
    quantize_model(model)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    pos = torch.arange(5).unsqueeze(0).expand(2, -1)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=True)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, cfg.vocab_size)
    assert not torch.isnan(logits).any()


def test_quantized_state_dict_bytes_shows_a_real_reduction():
    cfg = tiny_gpt_config()
    torch.manual_seed(5)
    fp_model = GPT(cfg)
    fp_bytes = quantized_state_dict_bytes(fp_model)

    q_model = GPT(cfg)
    q_model.load_state_dict(fp_model.state_dict())
    quantize_model(q_model)
    q_bytes = quantized_state_dict_bytes(q_model)

    # Embedding (untouched, still fp32) must be identical; "other" (the
    # quantized Linears) must have shrunk.
    assert q_bytes["embedding_bytes"] == fp_bytes["embedding_bytes"]
    assert q_bytes["other_bytes"] < fp_bytes["other_bytes"]
