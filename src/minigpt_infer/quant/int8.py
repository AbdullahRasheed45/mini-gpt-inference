"""W8A16 weight-only int8 quantization (docs/PLAN.md §7 Phase 5).

Weights are quantized to int8 per-output-channel (symmetric); activations
stay in a floating dtype (hence "W8A16", not W8A8) -- the matmul's inputs are
always float, only the WEIGHT STORAGE is int8. This halves quantized-layer
weight memory (int8 = 1 byte vs fp16 = 2 bytes) without needing activation
calibration.

Only the four big Linears per block are quantized (qkv, attn.proj, mlp.fc,
mlp.proj) -- explicitly NOT lm_head/tok_emb. Two independent reasons, either
alone would be sufficient:
  1. lm_head.weight IS tok_emb.weight (tied, model.py's __init__). Quantizing
     it means quantizing the embedding table, which is read by a gather, not
     a matmul -- a genuinely different, separate problem from quantizing a
     Linear's weight.
  2. The embedding table is already close to half of this model's total
     parameters (vocab_size=50304 x n_embd=512 ~= 25.8M of ~38M non-embedding
     + ~25.8M embedding). Quantizing everything else already captures most
     of the available win from the part that's a repeated matmul in the
     decode hot path.

Two dequant paths, deliberately both kept (docs/PLAN.md Phase 5 point 3):
  - QuantizedLinear.forward(): naive dequant-then-matmul. Predicted (and,
    per bench/bench_quant.py, measured) to be SLOWER than plain fp16 --
    materializing a full fp16 copy of the weight is strictly MORE memory
    traffic than just reading the fp16 weight directly would have been.
    Publishing this negative result is the point, not a bug to hide.
  - triton_dequant_gemv(): a fused kernel that never materializes the
    dequantized weight matrix at all, for the decode-shaped (GEMV) case.
    GPU + sm70+ only; see triton_dequant_gemv_available().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if TYPE_CHECKING:
    from minigpt_infer.model import GPT


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """weight: (out_features, in_features), any float dtype.
    Symmetric per-output-channel (per-row) int8 quantization:
        scale[o] = max(|weight[o, :]|) / 127
        W_q[o, i] = round(weight[o, i] / scale[o]), clamped to [-127, 127]
    Returns (W_q int8 (out, in), scale fp32 (out,)).
    """
    w32 = weight.float()
    amax = w32.abs().amax(dim=1)  # (out_features,)
    # A dead output channel (all-zero row) would divide by zero; its scale
    # value is then irrelevant (0 * anything = 0 either way), so clamp
    # rather than special-case it.
    scale = (amax / 127.0).clamp(min=1e-8)
    w_q = torch.round(w32 / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return w_q, scale


def dequantize_per_channel(
    w_q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return w_q.to(dtype) * scale.unsqueeze(1).to(dtype)


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear (same forward signature) with an
    int8-stored weight. See module docstring for why forward() uses the
    naive (and, deliberately, slower) dequant-then-matmul path."""

    def __init__(
        self, weight_q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight_q.shape
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinear:
        weight_q, scale = quantize_per_channel(linear.weight.data)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(weight_q, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dq = dequantize_per_channel(self.weight_q, self.scale, dtype=x.dtype)
        return F.linear(x, w_dq, self.bias)


def quantize_model(model: GPT) -> GPT:
    """In-place: replaces attn.qkv, attn.proj, mlp.fc, mlp.proj in every
    block with QuantizedLinear. Returns `model` for chaining. lm_head/
    tok_emb (tied) are never touched -- see module docstring."""
    for block in model.blocks:
        block.attn.qkv = QuantizedLinear.from_linear(block.attn.qkv)
        block.attn.proj = QuantizedLinear.from_linear(block.attn.proj)
        block.mlp.fc = QuantizedLinear.from_linear(block.mlp.fc)
        block.mlp.proj = QuantizedLinear.from_linear(block.mlp.proj)
    return model


def quantized_state_dict_bytes(model: GPT) -> dict[str, int]:
    """Byte count of every parameter/buffer currently in `model`, split into
    the tied embedding (lm_head.weight is tok_emb.weight -- counted once)
    and everything else -- for the memory-reduction acceptance check."""
    tied_ptr = model.tok_emb.weight.data_ptr()
    embedding_bytes = 0
    other_bytes = 0
    seen_ptrs: set[int] = set()
    for t in list(model.parameters()) + list(model.buffers()):
        ptr = t.data_ptr()
        if ptr in seen_ptrs:
            continue  # tied weight (lm_head.weight is tok_emb.weight): count once
        seen_ptrs.add(ptr)
        nbytes = t.numel() * t.element_size()
        if ptr == tied_ptr:
            embedding_bytes += nbytes
        else:
            other_bytes += nbytes
    return {"embedding_bytes": embedding_bytes, "other_bytes": other_bytes,
            "total_bytes": embedding_bytes + other_bytes}


def triton_dequant_gemv_available() -> bool:
    if triton is None or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 7


if triton is not None:

    @triton.jit
    def _dequant_gemv_kernel(
        x_ptr, wq_ptr, scale_ptr, bias_ptr, out_ptr,
        stride_x_b, stride_x_k,
        stride_w_o, stride_w_k,
        stride_out_b, stride_out_o,
        K, N_OUT, HAS_BIAS: tl.constexpr,
        BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        b = tl.program_id(0)
        o_block = tl.program_id(1)
        o_offsets = o_block * BLOCK_O + tl.arange(0, BLOCK_O)
        o_mask = o_offsets < N_OUT

        acc = tl.zeros([BLOCK_O], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            x_vals = tl.load(
                x_ptr + b * stride_x_b + k_offsets * stride_x_k, mask=k_mask, other=0.0,
            ).to(tl.float32)  # (BLOCK_K,)

            w_ptrs = wq_ptr + o_offsets[:, None] * stride_w_o + k_offsets[None, :] * stride_w_k
            w_mask = o_mask[:, None] & k_mask[None, :]
            # Fused dequant: the int8 weight block is converted to float and
            # immediately consumed -- the dequantized matrix is never
            # materialized in global memory, unlike QuantizedLinear.forward().
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.float32)  # (BLOCK_O, BLOCK_K)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        scale = tl.load(scale_ptr + o_offsets, mask=o_mask, other=1.0).to(tl.float32)
        result = acc * scale
        if HAS_BIAS:
            bias = tl.load(bias_ptr + o_offsets, mask=o_mask, other=0.0).to(tl.float32)
            result += bias

        tl.store(out_ptr + b * stride_out_b + o_offsets * stride_out_o, result, mask=o_mask)


def triton_dequant_gemv(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_o: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """x: (B, in_features). weight_q: (out_features, in_features) int8.
    scale: (out_features,). Returns (B, out_features) fp32.
    """
    assert triton_dequant_gemv_available(), "requires triton + CUDA sm70+"
    B, K = x.shape
    n_out, K2 = weight_q.shape
    assert K == K2, f"in_features mismatch: x has {K}, weight_q has {K2}"

    out = torch.empty(B, n_out, device=x.device, dtype=torch.float32)
    # bias_arg: a valid pointer is required even when HAS_BIAS=False (the
    # kernel just never dereferences it in that case).
    bias_arg = bias if bias is not None else scale
    grid = (B, triton.cdiv(n_out, block_o))
    _dequant_gemv_kernel[grid](
        x, weight_q, scale, bias_arg, out,
        x.stride(0), x.stride(1),
        weight_q.stride(0), weight_q.stride(1),
        out.stride(0), out.stride(1),
        K, n_out, bias is not None,
        BLOCK_O=block_o, BLOCK_K=block_k,
    )
    return out
