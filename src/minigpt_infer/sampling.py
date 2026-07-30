"""Batched sampling (docs/PLAN.md §5 SamplingParams, §7 Phase 2).

Everything here operates on a whole `(B, vocab)` logits tensor at once. Top-k
and top-p accept either a scalar (same cutoff for the whole batch) or a
per-row `(B,)` tensor -- both paths are fully vectorized (sort + gather/scan),
never a Python loop over rows, per docs/PLAN.md Phase 2 point 4.

Repetition penalty is the one exception: it takes a per-row *boolean* seen-mask
rather than a ragged list of already-generated ids, specifically so it stays
vectorized too (a ragged list would force a Python loop or an awkward padded
gather with a sentinel index). Building/maintaining that mask is the caller's
job (see `update_seen_mask`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_row_tensor(
    value: torch.Tensor | float | int | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float | int,
) -> torch.Tensor:
    if value is None:
        value = default
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(batch_size)
        assert t.shape == (batch_size,), f"expected shape ({batch_size},), got {tuple(t.shape)}"
        return t
    return torch.full((batch_size,), value, device=device, dtype=dtype)


def apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    """Per-row top-k filtering. `top_k`: (B,) int64, clamped to [1, vocab].

    Finds each row's k-th largest logit via sort+gather (no torch.topk loop --
    a single sort already gives every rank we need) and masks anything below
    it. Ties at the threshold can let a few extra tokens through; that matches
    the standard (e.g. HF transformers) top-k behavior.
    """
    B, V = logits.shape
    k = top_k.clamp(min=1, max=V)
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    threshold = sorted_logits.gather(1, (k - 1).unsqueeze(1))  # (B, 1)
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(logits < threshold, neg_inf)


def apply_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Per-row nucleus filtering. `top_p`: (B,) float in (0, 1].

    Sorts each row once, keeps the smallest prefix whose cumulative
    probability mass reaches `top_p`, and always keeps at least the single
    highest-probability token (so a row can never end up fully masked).
    """
    neg_inf = torch.finfo(logits.dtype).min
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)

    # Drop a (sorted-order) token if the mass *strictly before* it already
    # exceeds top_p -- i.e. keep the first token that crosses the threshold,
    # drop everything after.
    sorted_remove = (cumprobs - probs) > top_p.unsqueeze(1)
    sorted_remove[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_remove, neg_inf)
    out = torch.full_like(logits, neg_inf)
    out.scatter_(1, sorted_idx, sorted_logits)
    return out


def update_seen_mask(seen_mask: torch.Tensor, new_ids: torch.Tensor) -> None:
    """In-place: mark `new_ids` (B, 1) as seen in `seen_mask` (B, vocab) bool."""
    seen_mask.scatter_(1, new_ids, True)


def apply_repetition_penalty(
    logits: torch.Tensor, seen_mask: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Standard repetition penalty (Keskar et al. / HF transformers formula):
    positive logits of already-seen tokens are divided by `penalty`, negative
    ones multiplied -- both push seen tokens' probability down for penalty > 1.
    """
    if penalty == 1.0:
        return logits
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen_mask, penalized, logits)


def sample_batch(
    logits: torch.Tensor,
    temperature: torch.Tensor | float = 1.0,
    top_k: torch.Tensor | int | None = None,
    top_p: torch.Tensor | float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one next token per row of `logits` (B, vocab) -> (B, 1).

    `temperature`, `top_k`, `top_p` each accept a scalar (applied to the whole
    batch) or a `(B,)` tensor (per-row values) -- see docs/PLAN.md Phase 2
    point 4: "group by params, or apply per-row and mask." This implements
    the per-row path directly since it costs no more than the scalar path
    once vectorized.

    `temperature <= 0` for a row means greedy (argmax) for that row,
    regardless of top_k/top_p -- mixing greedy and stochastic rows in one
    batched call is exactly what per-row temperature is for.
    """
    B, V = logits.shape
    device, dtype = logits.device, logits.dtype

    temp = _as_row_tensor(temperature, B, device, dtype, default=1.0)
    greedy_rows = temp <= 0
    safe_temp = temp.clamp(min=1e-6).unsqueeze(-1)
    scaled = logits / safe_temp

    if top_k is not None:
        k = _as_row_tensor(top_k, B, device, torch.long, default=V)
        scaled = apply_top_k(scaled, k)
    if top_p is not None:
        p = _as_row_tensor(top_p, B, device, dtype, default=1.0)
        scaled = apply_top_p(scaled, p)

    probs = F.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)

    if greedy_rows.any():
        greedy_ids = torch.argmax(logits, dim=-1, keepdim=True)
        sampled = torch.where(greedy_rows.unsqueeze(-1), greedy_ids, sampled)
    return sampled
