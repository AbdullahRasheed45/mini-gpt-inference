"""ForwardBatch: everything the model needs for one forward pass.

Introduced in full here (Phase 1) even though the paged fields stay unused
until Phase 3, per docs/PLAN.md §5 -- the model's forward signature must not
change again after this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minigpt_infer.cache.base import KVCacheBase


@dataclass
class ForwardBatch:
    input_ids: torch.Tensor          # prefill: (B, T);  decode: (B, 1)
    position_ids: torch.Tensor       # same shape as input_ids. ABSOLUTE positions.
    is_prefill: bool
    cache: KVCacheBase | None = None
    attn_mask: torch.Tensor | None = None    # (B, 1, Tq, Tk) additive float mask, or None

    # --- paged fields, unused until Phase 3 ---
    block_tables: torch.Tensor | None = None  # (B, max_blocks) int32, -1 = unallocated
    slot_mapping: torch.Tensor | None = None  # (B*T,) int32, flat write index into pool
    seq_lens: torch.Tensor | None = None      # (B,) int32, tokens cached per seq

    def __post_init__(self) -> None:
        assert self.input_ids.shape == self.position_ids.shape, (
            f"input_ids {tuple(self.input_ids.shape)} and position_ids "
            f"{tuple(self.position_ids.shape)} must have the same shape"
        )
        if self.is_prefill:
            assert self.input_ids.shape[1] >= 1
        else:
            assert self.input_ids.shape[1] == 1, "decode batches must have T=1"


def build_left_padded_batch(
    prompts: list[list[int]],
    pad_token_id: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad a ragged batch of prompts and build the prefill attn_mask
    (docs/PLAN.md Phase 2 points 1-3).

    Left padding (not right) is used so every row's real tokens end at the
    same column -- decode then appends at one shared cache offset for the
    whole batch, satisfying StaticKVCache's uniform-current-length
    requirement without any per-row bookkeeping.

    Returns:
      input_ids, position_ids: (B, max_len)
      attn_mask: (B, 1, max_len, max_len) additive causal+padding bias
      pad_mask: (B, max_len) bool, True at pad columns -- kept by the caller
        to build the decode-time mask via extend_decode_mask(), since those
        columns stay masked for the rest of generation.
    """
    B = len(prompts)
    max_len = max(len(p) for p in prompts)
    assert max_len > 0, "every prompt must have at least one token"

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
    real_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):
        assert len(p) > 0, f"prompt {i} is empty"
        input_ids[i, max_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        real_mask[i, max_len - len(p):] = True
    pad_mask = ~real_mask

    # Pad slots get position 0 -- junk, but masked out of attention anyway;
    # clamp(min=0) only exists so we never index position -1.
    position_ids = (real_mask.cumsum(dim=-1) - 1).clamp(min=0)

    attn_mask = _causal_plus_padding_bias(pad_mask, num_queries=max_len, dtype=dtype)

    # docs/PLAN.md Phase 2 pitfall: "a fully-masked row produces NaN after
    # softmax" -- true if masking used literal -inf. It does NOT apply here:
    # every bias value is finfo.min (finite, never -inf, by the clamp in
    # _causal_plus_padding_bias), and softmax over a row that's uniformly
    # finfo.min is a harmless uniform distribution, not NaN -- that's exactly
    # why the plan says to use finfo.min instead of float("-inf") in the
    # first place. A query row *entirely* inside the left-pad prefix (never
    # read -- see model.py, only the last position's logits are used, and the
    # last column is always real) can legitimately hit this. The real,
    # unconditional invariant is just: never actually NaN or +/-inf.
    assert not torch.isnan(attn_mask).any() and not torch.isinf(attn_mask).any(), (
        "additive attention mask must stay finite (NaN/inf would poison softmax)"
    )

    return input_ids, position_ids, attn_mask, pad_mask


def _causal_plus_padding_bias(
    pad_mask: torch.Tensor, num_queries: int, dtype: torch.dtype
) -> torch.Tensor:
    device = pad_mask.device
    neg_inf = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((num_queries, pad_mask.shape[1]), neg_inf, device=device, dtype=dtype),
        diagonal=1,
    )
    key_pad_bias = torch.where(
        pad_mask,
        torch.tensor(neg_inf, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
    )
    # Summing two finfo.min biases (a causal-masked AND padding key) can
    # overflow to literal -inf in fp32/fp64 -- clamp back to the floor so the
    # additive-mask invariant (finite, never actual -inf/NaN) holds exactly,
    # per docs/PLAN.md Phase 2 point 3.
    bias = (causal.unsqueeze(0) + key_pad_bias.unsqueeze(1)).clamp(min=neg_inf)
    return bias.unsqueeze(1)  # (B, 1, Tq, Tk)


def extend_decode_mask(
    pad_mask: torch.Tensor, new_kv_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """Decode-time attn_mask: the fixed left-padding bias from the prompt,
    extended with zero bias for every token appended after the prompt.

    Only the initial left-pad prefix is ever padding -- every token appended
    during decode is real for every row -- so the "extra" columns need no
    causal component either (a single decode query always attends to the
    entire valid past). Shape: (B, 1, 1, new_kv_len).
    """
    B, prompt_len = pad_mask.shape
    device = pad_mask.device
    neg_inf = torch.finfo(dtype).min
    prompt_bias = torch.where(
        pad_mask,
        torch.tensor(neg_inf, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
    )
    extra = new_kv_len - prompt_len
    assert extra >= 0, f"new_kv_len={new_kv_len} < prompt_len={prompt_len}"
    if extra > 0:
        tail = torch.zeros((B, extra), device=device, dtype=dtype)
        bias = torch.cat([prompt_bias, tail], dim=-1)
    else:
        bias = prompt_bias
    return bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, new_kv_len)
