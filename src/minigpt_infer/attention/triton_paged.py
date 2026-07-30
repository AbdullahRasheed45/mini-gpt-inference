"""Triton paged decode attention kernel (docs/PLAN.md §7 Phase 4, point 4).

T4-only (sm75): grid (num_seqs, num_heads), one program per (sequence, head).
Each program walks that sequence's block table, accumulating with online
softmax (FlashAttention-style running max/sum, for fp16 stability) -- the
standard trick for never materializing a full score row, so this stays
correct for any context length even though decode's real motivation here is
avoiding the gather+SDPA path's Python/kernel-launch overhead, not memory.

Decode attention is a GEMV (one query vector against many keys), not a GEMM
(docs/PLAN.md Phase 4 point 4): this uses elementwise-multiply + reduction
along head_dim rather than `tl.dot`, per the plan's own guidance that this is
"simpler and likely faster" for the single-query case.

Validated against the gather+SDPA path (cache/paged.py + model.py) in
tests/test_triton.py, GPU-only. `triton_paged_attention_available()` gates
capability at import so callers degrade to gather+SDPA on hardware where
this can't run (Kaggle's P100 is sm60; Triton's `tl.dot` needs sm70+, and
while this kernel avoids `tl.dot`, the plan's stated target line is sm70+
regardless -- verify capability before trusting it, don't assume).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def triton_paged_attention_available() -> bool:
    if triton is None or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 7


if triton is not None:

    @triton.jit
    def _paged_decode_attention_kernel(
        q_ptr, k_pool_ptr, v_pool_ptr, block_table_ptr, seq_lens_ptr, out_ptr,
        stride_q_seq, stride_q_head, stride_q_dim,
        stride_kv_block, stride_kv_slot, stride_kv_head, stride_kv_dim,
        stride_bt_seq, stride_bt_block,
        stride_out_seq, stride_out_head, stride_out_dim,
        scale,
        HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        dim_offsets = tl.arange(0, HEAD_DIM)
        q_ptrs = (
            q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head
            + dim_offsets * stride_q_dim
        )
        q = tl.load(q_ptrs).to(tl.float32)  # (HEAD_DIM,)

        m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        slot_offsets = tl.arange(0, BLOCK_SIZE)

        for b in range(MAX_BLOCKS):
            block_active = b < num_blocks_needed
            block_id = tl.load(
                block_table_ptr + seq_idx * stride_bt_seq + b * stride_bt_block,
                mask=block_active, other=0,
            )

            token_pos = b * BLOCK_SIZE + slot_offsets
            slot_mask = block_active & (token_pos < seq_len)

            k_ptrs = (
                k_pool_ptr + block_id * stride_kv_block
                + slot_offsets[:, None] * stride_kv_slot
                + head_idx * stride_kv_head
                + dim_offsets[None, :] * stride_kv_dim
            )
            k_block = tl.load(k_ptrs, mask=slot_mask[:, None], other=0.0).to(tl.float32)

            # GEMV, not GEMM: elementwise multiply + reduce along HEAD_DIM
            # instead of tl.dot (docs/PLAN.md Phase 4 point 4).
            scores = tl.sum(k_block * q[None, :], axis=1) * scale  # (BLOCK_SIZE,)
            scores = tl.where(slot_mask, scores, float("-inf"))

            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)
            # A fully-inactive block (b >= num_blocks_needed) has block_max
            # == -inf; guard m_i - m_new -> -inf - -inf = NaN in that case by
            # only rescaling when m_new is finite.
            m_new_finite = tl.where(m_new == float("-inf"), 0.0, m_new)

            alpha = tl.exp(m_i - m_new_finite)
            p = tl.exp(scores - m_new_finite)  # 0 where slot_mask is False

            v_ptrs = (
                v_pool_ptr + block_id * stride_kv_block
                + slot_offsets[:, None] * stride_kv_slot
                + head_idx * stride_kv_head
                + dim_offsets[None, :] * stride_kv_dim
            )
            v_block = tl.load(v_ptrs, mask=slot_mask[:, None], other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new_finite

        out = acc / l_i
        out_ptrs = (
            out_ptr + seq_idx * stride_out_seq + head_idx * stride_out_head
            + dim_offsets * stride_out_dim
        )
        tl.store(out_ptrs, out)


def triton_paged_decode_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """q: (num_seqs, num_heads, head_dim) -- one decode-step query per seq.
    k_pool/v_pool: (num_blocks, block_size, num_heads, head_dim) -- the same
    layout PagedKVCache uses.
    block_tables: (num_seqs, max_blocks), -1 padding beyond what a row needs
    (never read: masked out via num_blocks_needed derived from seq_lens).
    seq_lens: (num_seqs,) tokens cached per seq, post-write (same convention
    as PagedKVCache.read()).
    Returns (num_seqs, num_heads, head_dim).
    """
    assert triton_paged_attention_available(), "requires triton + CUDA sm70+"
    num_seqs, num_heads, head_dim = q.shape
    max_blocks = block_tables.shape[1]
    out = torch.empty_like(q)

    block_tables_i32 = block_tables.clamp(min=0).to(torch.int32).contiguous()
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()

    grid = (num_seqs, num_heads)
    _paged_decode_attention_kernel[grid](
        q, k_pool, v_pool, block_tables_i32, seq_lens_i32, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_tables_i32.stride(0), block_tables_i32.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        HEAD_DIM=head_dim, BLOCK_SIZE=block_size, MAX_BLOCKS=max_blocks,
    )
    return out
