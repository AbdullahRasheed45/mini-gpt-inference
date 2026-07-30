"""Phase 4: Triton paged decode attention vs gather+SDPA (docs/PLAN.md §7
Phase 4 acceptance, §8 rung 5). GPU-only -- Triton requires a real CUDA
device (sm70+) to compile and run at all, so every test here is skipped
(not failed) when unavailable, per docs/PLAN.md §9's CI plan.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from minigpt_infer.attention.triton_paged import (
    triton_paged_attention_available,
    triton_paged_decode_attention,
)
from minigpt_infer.cache.paged import PagedKVCache

pytestmark = pytest.mark.gpu

requires_triton = pytest.mark.skipif(
    not triton_paged_attention_available(),
    reason="requires a CUDA device with compute capability >= 7.0 and triton installed",
)


def _reference_gather_sdpa(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
    block_tables: torch.Tensor, seq_lens: torch.Tensor, block_size: int, scale: float,
) -> torch.Tensor:
    """The Phase 3 gather path (PagedKVCache.read()) + plain SDPA -- the
    known-correct baseline this kernel must match."""
    num_seqs, num_heads, head_dim = q.shape
    cache = PagedKVCache.__new__(PagedKVCache)
    cache.block_size = block_size
    cache.num_blocks = k_pool.shape[0]
    cache.k_pool = [k_pool]
    cache.v_pool = [v_pool]

    class _Batch:
        pass

    batch = _Batch()
    batch.block_tables = block_tables
    batch.seq_lens = seq_lens
    k, v = cache.read(0, batch)  # (num_seqs, num_heads, max_len, head_dim)

    max_len = k.shape[2]
    col = torch.arange(max_len, device=q.device).unsqueeze(0)
    valid = col < seq_lens.unsqueeze(1)  # (num_seqs, max_len)
    neg_inf = torch.finfo(q.dtype).min
    bias = torch.where(valid, torch.tensor(0.0, device=q.device, dtype=q.dtype), neg_inf)
    attn_mask = bias.unsqueeze(1).unsqueeze(1)  # (num_seqs, 1, 1, max_len)

    q4 = q.unsqueeze(2)  # (num_seqs, num_heads, 1, head_dim)
    out = F.scaled_dot_product_attention(q4, k, v, attn_mask=attn_mask, scale=scale)
    return out.squeeze(2)  # (num_seqs, num_heads, head_dim)


def _build_random_paged_scenario(
    num_seqs: int, num_heads: int, head_dim: int, block_size: int,
    num_blocks: int, seq_lens: list[int], device: str = "cuda", dtype=torch.float16,
):
    torch.manual_seed(0)
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    k_pool = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_pool = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)

    max_blocks = max((n + block_size - 1) // block_size for n in seq_lens)
    block_tables = torch.full((num_seqs, max_blocks), -1, dtype=torch.long, device=device)
    next_block = 0
    for i, n in enumerate(seq_lens):
        nb = (n + block_size - 1) // block_size
        for j in range(nb):
            block_tables[i, j] = next_block
            next_block += 1
    assert next_block <= num_blocks

    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)
    return q, k_pool, v_pool, block_tables, seq_lens_t


@requires_triton
def test_triton_matches_gather_sdpa_uniform_lengths():
    device = "cuda"
    num_seqs, num_heads, head_dim, block_size = 4, 4, 64, 8
    seq_lens = [16, 16, 16, 16]
    scale = 1.0 / math.sqrt(head_dim)

    q, k_pool, v_pool, block_tables, seq_lens_t = _build_random_paged_scenario(
        num_seqs, num_heads, head_dim, block_size, num_blocks=16, seq_lens=seq_lens, device=device,
    )

    triton_out = triton_paged_decode_attention(
        q, k_pool, v_pool, block_tables, seq_lens_t, block_size, scale,
    )
    ref_out = _reference_gather_sdpa(
        q, k_pool, v_pool, block_tables, seq_lens_t, block_size, scale,
    )

    assert torch.allclose(triton_out, ref_out, atol=2e-2, rtol=2e-2), (
        (triton_out - ref_out).abs().max().item()
    )


@requires_triton
def test_triton_matches_gather_sdpa_ragged_lengths():
    """Different seq_lens per row -- exercises the num_blocks_needed masking
    (a row shorter than the batch's longest must not attend past its own
    real length, including a partially-filled last block)."""
    device = "cuda"
    num_seqs, num_heads, head_dim, block_size = 5, 4, 64, 8
    seq_lens = [3, 8, 9, 17, 24]  # spans sub-block, exact-block, and multi-block cases
    scale = 1.0 / math.sqrt(head_dim)

    q, k_pool, v_pool, block_tables, seq_lens_t = _build_random_paged_scenario(
        num_seqs, num_heads, head_dim, block_size, num_blocks=32, seq_lens=seq_lens, device=device,
    )

    triton_out = triton_paged_decode_attention(
        q, k_pool, v_pool, block_tables, seq_lens_t, block_size, scale,
    )
    ref_out = _reference_gather_sdpa(
        q, k_pool, v_pool, block_tables, seq_lens_t, block_size, scale,
    )

    assert torch.allclose(triton_out, ref_out, atol=2e-2, rtol=2e-2), (
        (triton_out - ref_out).abs().max().item()
    )


@requires_triton
def test_triton_matches_gather_sdpa_single_token_sequences():
    """seq_len=1 -- a single block, entirely one real slot. Edge case for
    the online-softmax accumulator's initial -inf handling."""
    device = "cuda"
    num_seqs, num_heads, head_dim, block_size = 2, 2, 64, 8
    seq_lens = [1, 1]
    scale = 1.0 / math.sqrt(head_dim)

    q, k_pool, v_pool, block_tables, seq_lens_t = _build_random_paged_scenario(
        num_seqs, num_heads, head_dim, block_size, num_blocks=4, seq_lens=seq_lens, device=device,
    )

    triton_out = triton_paged_decode_attention(
        q, k_pool, v_pool, block_tables, seq_lens_t, block_size, scale,
    )
    ref_out = _reference_gather_sdpa(
        q, k_pool, v_pool, block_tables, seq_lens_t, block_size, scale,
    )

    assert torch.allclose(triton_out, ref_out, atol=2e-2, rtol=2e-2), (
        (triton_out - ref_out).abs().max().item()
    )
