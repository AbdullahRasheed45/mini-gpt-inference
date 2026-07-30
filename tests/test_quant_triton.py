"""Phase 5: fused Triton dequant GEMV vs the naive dequant-then-matmul path
(docs/PLAN.md §7 Phase 5). GPU-only -- skipped (not failed) without a real
CUDA device (sm70+) and triton installed, per docs/PLAN.md §9's CI plan.
"""

import pytest
import torch

from minigpt_infer.quant.int8 import (
    QuantizedLinear,
    triton_dequant_gemv,
    triton_dequant_gemv_available,
)

pytestmark = pytest.mark.gpu

requires_triton = pytest.mark.skipif(
    not triton_dequant_gemv_available(),
    reason="requires a CUDA device with compute capability >= 7.0 and triton installed",
)


@requires_triton
def test_triton_dequant_gemv_matches_naive_dequant_then_matmul():
    torch.manual_seed(0)
    linear = torch.nn.Linear(512, 1536, bias=True).to("cuda")
    qlinear = QuantizedLinear.from_linear(linear).to("cuda")
    x = torch.randn(1, 512, device="cuda")

    naive_out = qlinear(x)
    fused_out = triton_dequant_gemv(x, qlinear.weight_q, qlinear.scale, qlinear.bias)

    assert torch.allclose(naive_out, fused_out, atol=1e-2, rtol=1e-2), (
        (naive_out - fused_out).abs().max().item()
    )


@requires_triton
def test_triton_dequant_gemv_matches_naive_no_bias():
    torch.manual_seed(1)
    linear = torch.nn.Linear(256, 512, bias=False).to("cuda")
    qlinear = QuantizedLinear.from_linear(linear).to("cuda")
    x = torch.randn(4, 256, device="cuda")  # batch > 1

    naive_out = qlinear(x)
    fused_out = triton_dequant_gemv(x, qlinear.weight_q, qlinear.scale)

    assert torch.allclose(naive_out, fused_out, atol=1e-2, rtol=1e-2), (
        (naive_out - fused_out).abs().max().item()
    )


@requires_triton
def test_triton_dequant_gemv_handles_out_features_not_a_multiple_of_block():
    """out_features=100 with the default block_o=64 exercises the o_mask
    boundary (second block only has 36 real output channels)."""
    torch.manual_seed(2)
    linear = torch.nn.Linear(128, 100, bias=True).to("cuda")
    qlinear = QuantizedLinear.from_linear(linear).to("cuda")
    x = torch.randn(2, 128, device="cuda")

    naive_out = qlinear(x)
    fused_out = triton_dequant_gemv(x, qlinear.weight_q, qlinear.scale, qlinear.bias)

    assert fused_out.shape == (2, 100)
    assert torch.allclose(naive_out, fused_out, atol=1e-2, rtol=1e-2), (
        (naive_out - fused_out).abs().max().item()
    )
