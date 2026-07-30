"""Minimal single-sequence(-batch) generation driver for Phase 1/2.

This is NOT the serving engine -- Phase 3 (engine/engine.py) introduces a real
continuous-batching scheduler with admission, preemption, and per-request
lifecycle. This module exists so Phase 1/2's benchmarks and tests have
something to actually call: allocate a cache, prefill once, then decode token
by token, synchronously, for a batch where every row advances together.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from minigpt_infer.batch import ForwardBatch, build_left_padded_batch, extend_decode_mask
from minigpt_infer.cache.static import StaticKVCache
from minigpt_infer.model import GPT
from minigpt_infer.sampling import sample_batch


def _make_cache(model: GPT, batch_size: int, max_seq_len: int) -> StaticKVCache:
    param = next(model.parameters())
    return StaticKVCache(
        n_layer=model.cfg.n_layer,
        n_head=model.cfg.n_head,
        head_dim=model.head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=param.device,
        dtype=param.dtype,
    )


@torch.no_grad()
def _run_cached(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    next_token_fn: Callable[[torch.Tensor], torch.Tensor],
    cache: StaticKVCache | None = None,
) -> torch.Tensor:
    """Shared prefill+decode loop. next_token_fn(logits: (B, vocab)) -> (B, 1) ids."""
    model.eval()
    B, T0 = idx.shape
    device = idx.device
    if cache is None:
        cache = _make_cache(model, B, model.cfg.block_size)

    pos = torch.arange(T0, device=device).unsqueeze(0).expand(B, -1)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=True, cache=cache)
    logits = model(batch)
    cache.advance(T0, B)

    generated = idx
    for _ in range(max_new_tokens):
        next_id = next_token_fn(logits)
        generated = torch.cat([generated, next_id], dim=1)
        cur_len = int(cache.seq_lens[0].item())
        pos = torch.full((B, 1), cur_len, device=device, dtype=torch.long)
        batch = ForwardBatch(input_ids=next_id, position_ids=pos, is_prefill=False, cache=cache)
        logits = model(batch)
        cache.advance(1, B)
    return generated


def greedy_generate_cached(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    vocab_mask: torch.Tensor | None = None,
    cache: StaticKVCache | None = None,
) -> torch.Tensor:
    """Deterministic argmax decode -- what Phase 1's exact-match tests compare
    against reference.ReferenceGPT.greedy_generate()."""

    def next_token(logits: torch.Tensor) -> torch.Tensor:
        if vocab_mask is not None:
            logits = logits.masked_fill(vocab_mask, float("-inf"))
        return torch.argmax(logits, dim=-1, keepdim=True)

    return _run_cached(model, idx, max_new_tokens, next_token, cache)


@torch.no_grad()
def _run_batched(
    model: GPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    pad_token_id: int,
    eot_token_id: int | None,
    next_token_fn: Callable[[torch.Tensor], torch.Tensor],
) -> list[list[int]]:
    """Phase 2: many ragged-length prompts, one shared cache (docs/PLAN.md §7
    Phase 2). Left-pads prompts to a common length so decode appends at one
    shared cache offset for every row (StaticKVCache's uniform-length
    requirement), then runs prefill + decode exactly like _run_cached but
    with an explicit causal+padding attn_mask throughout.

    Finished rows (hit eot_token_id) still participate in every forward pass
    -- static batching can't skip them, that's what Phase 3's continuous
    batching is for -- but their sampled token is overwritten with EOT and
    excluded from the returned sequence, per docs/PLAN.md Phase 2 pitfalls
    ("track a per-sequence finished flag and ignore them").
    """
    model.eval()
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    B = len(prompts)

    input_ids, position_ids, attn_mask, pad_mask = build_left_padded_batch(
        prompts, pad_token_id, device=device, dtype=dtype,
    )
    prompt_len = input_ids.shape[1]
    cache = _make_cache(model, B, prompt_len + max_new_tokens)
    # Per-row left-pad count -- the cache's seq_lens is uniform across the
    # batch (total slots filled, padding included), but each row's *true*
    # absolute position must not count its own left-pad prefix. Without this
    # offset, decode position_ids would be wrong by pad_lens[i] for every row
    # that isn't the longest prompt in the batch.
    pad_lens = pad_mask.sum(dim=-1)  # (B,)

    batch = ForwardBatch(
        input_ids=input_ids, position_ids=position_ids,
        is_prefill=True, cache=cache, attn_mask=attn_mask,
    )
    logits = model(batch)
    cache.advance(prompt_len, B)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated: list[list[int]] = [list(p) for p in prompts]

    for _ in range(max_new_tokens):
        next_id = next_token_fn(logits)  # (B, 1)
        if eot_token_id is not None:
            next_id = torch.where(
                finished.unsqueeze(-1), torch.full_like(next_id, eot_token_id), next_id
            )
        for i in range(B):
            if not finished[i]:
                generated[i].append(int(next_id[i, 0]))
        if eot_token_id is not None:
            finished = finished | (next_id.squeeze(-1) == eot_token_id)
        if bool(finished.all()):
            break

        cur_len = int(cache.seq_lens[0].item())
        pos = (cur_len - pad_lens).unsqueeze(-1).to(torch.long)  # per-row, (B, 1)
        dec_mask = extend_decode_mask(pad_mask, new_kv_len=cur_len + 1, dtype=dtype)
        batch = ForwardBatch(
            input_ids=next_id, position_ids=pos,
            is_prefill=False, cache=cache, attn_mask=dec_mask,
        )
        logits = model(batch)
        cache.advance(1, B)

    return generated


def batched_greedy_generate(
    model: GPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    pad_token_id: int,
    eot_token_id: int | None = None,
    vocab_mask: torch.Tensor | None = None,
) -> list[list[int]]:
    """Greedy decode over a batch of ragged-length prompts. Each returned
    sequence must exactly match calling greedy_generate_cached on that prompt
    alone -- see tests/test_batching.py's batch-invariance test."""

    def next_token(logits: torch.Tensor) -> torch.Tensor:
        if vocab_mask is not None:
            logits = logits.masked_fill(vocab_mask, float("-inf"))
        return torch.argmax(logits, dim=-1, keepdim=True)

    return _run_batched(model, prompts, max_new_tokens, pad_token_id, eot_token_id, next_token)


def batched_sample_generate(
    model: GPT,
    prompts: list[list[int]],
    max_new_tokens: int,
    pad_token_id: int,
    eot_token_id: int | None = None,
    temperature: torch.Tensor | float = 1.0,
    top_k: torch.Tensor | int | None = None,
    top_p: torch.Tensor | float | None = None,
    vocab_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Stochastic decode over a batch of ragged-length prompts, with optional
    per-row temperature/top_k/top_p (see sampling.sample_batch)."""

    def next_token(logits: torch.Tensor) -> torch.Tensor:
        if vocab_mask is not None:
            logits = logits.masked_fill(vocab_mask, float("-inf"))
        return sample_batch(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator,
        )

    return _run_batched(model, prompts, max_new_tokens, pad_token_id, eot_token_id, next_token)


def sample_generate_cached(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    vocab_mask: torch.Tensor | None = None,
    cache: StaticKVCache | None = None,
) -> torch.Tensor:
    """Stochastic sampling decode -- same semantics as Project A's original
    generate()/reference.ReferenceGPT.generate(), but O(n) via the cache."""

    def next_token(logits: torch.Tensor) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        if vocab_mask is not None:
            logits = logits.masked_fill(vocab_mask, float("-inf"))
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    return _run_cached(model, idx, max_new_tokens, next_token, cache)
