"""Speculative decoding (docs/PLAN.md §7 Phase 6).

Implements the exact rejection-sampling verification algorithm (Leviathan
et al. 2023 / Chen et al. 2023):
  1. A draft proposes gamma tokens x_1..x_gamma with probabilities q(x_i).
  2. The target runs ONE forward over the gamma proposed tokens (continuing
     its existing KV cache), yielding p at gamma+1 positions: the position
     right after the cache's current end (already known from the previous
     step -- see `pending_logits` below) plus one new distribution per draft
     token consumed.
  3. For i in 1..gamma: accept x_i with probability min(1, p(x_i)/q(x_i)).
     On first rejection, sample the correction from normalize(max(0, p-q))
     and stop.
  4. If all gamma accepted, sample a bonus token from p at position gamma+1.

This is provably distribution-preserving: the accepted token at each step is
distributed EXACTLY as if sampled from the target alone, not merely "close."
tests/test_spec_decode.py verifies this empirically (chi-square) rather than
just implementing the algorithm and hoping -- most projects claim the
property and never check it (docs/PLAN.md's framing, and the actual point
of this phase).

Greedy exactness (temperature=0) is a special case of the SAME algorithm,
not a separate code path: representing the target's distribution as a
one-hot vector at its argmax makes `min(1, p(x_i)/q(x_i))` accept iff x_i is
the target's argmax, and the rejection residual `max(0, p-q)` collapses to a
one-hot vector at the target's argmax too -- so verify_and_accept()
reproduces plain greedy decoding exactly whenever `temperature<=0`, with no
special-casing needed. See tests/test_spec_decode.py's greedy-exactness test.

Two draft proposers (docs/PLAN.md Phase 6 points 1-2):
  - PromptLookupDrafter: n-gram search over existing context, no model at
    all. A deterministic proposal's implicit draft distribution q is a
    one-hot vector at the proposed token (probability 1) -- the correct,
    degenerate special case of the general algorithm, not an approximation
    of it.
  - SelfSpeculativeDrafter: runs only the first `draft_layers` of the target
    model's own blocks (same weights) as a cheap approximate forward. Uses
    no cache (a fresh, uncached forward over the growing draft context each
    step, mirroring reference.py's oracle) -- simple and unambiguously
    correct, which matters more here than draft-side throughput; K<N layers
    is still cheaper per token than the full model regardless.

A trained draft model (docs/PLAN.md Phase 6 point 3, explicitly a stretch
goal) is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F

from minigpt_infer.batch import ForwardBatch, build_speculative_verify_mask
from minigpt_infer.cache.static import StaticKVCache
from minigpt_infer.model import GPT


class DraftProposer(Protocol):
    def propose(
        self, context: list[int], gamma: int, vocab_mask: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor] | None:
        """Return (draft_token_ids, draft_probs) where draft_probs has shape
        (len(draft_token_ids), vocab_size), or None if this drafter has
        nothing to propose right now (caller falls back to a plain
        target-only decode step for this round)."""
        ...


class PromptLookupDrafter:
    """n-gram search over the existing context (docs/PLAN.md Phase 6 point
    1). Searches backward for the most recent earlier occurrence of the last
    `ngram_size` tokens and proposes whatever followed it. No model, no
    training -- TinyStories is repetitive, so the hit rate should be real
    (prediction P8, docs/BENCHMARKS.md)."""

    def __init__(self, vocab_size: int, ngram_size: int = 3) -> None:
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size

    def propose(
        self, context: list[int], gamma: int, vocab_mask: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor] | None:
        n = self.ngram_size
        if len(context) < n:
            return None
        needle = context[-n:]
        match_end = None
        # Search strictly BEFORE the needle's own occurrence (range end
        # excludes len(context)-n, the needle's own start) so a match always
        # refers to genuine repetition, not the needle finding itself.
        for start in range(len(context) - n - 1, -1, -1):
            if context[start:start + n] == needle:
                candidate_end = start + n
                if candidate_end < len(context):  # there's at least one real follower
                    match_end = candidate_end
                    break
        if match_end is None:
            return None

        draft_ids = context[match_end:match_end + gamma]
        if not draft_ids:
            return None

        draft_probs = torch.zeros(len(draft_ids), self.vocab_size)
        for i, tok in enumerate(draft_ids):
            draft_probs[i, tok] = 1.0  # deterministic proposal: one-hot q
        return draft_ids, draft_probs


def _forward_first_k_layers(model: GPT, batch: ForwardBatch, num_layers: int) -> torch.Tensor:
    """Embedding + first `num_layers` blocks + final ln_f/lm_head -- the
    layer-skip draft forward. Reuses the target model's own submodules and
    attention mechanics directly (Block.forward is already a clean,
    reusable per-layer unit); num_layers < model.cfg.n_layer is the whole
    point (fewer layers, cheaper, approximate)."""
    x = model.drop(model.tok_emb(batch.input_ids) + model.pos_emb(batch.position_ids))
    for layer_idx in range(num_layers):
        x = model.blocks[layer_idx](x, layer_idx, batch)
    x = model.ln_f(x)
    return model.lm_head(x if batch.return_all_logits else x[:, -1, :])


class SelfSpeculativeDrafter:
    """Layer-skip draft (docs/PLAN.md Phase 6 point 2): run only the first
    `draft_layers` of the target model's blocks, same weights, as a cheap
    approximate forward. Free -- no extra parameters, no training."""

    def __init__(self, model: GPT, draft_layers: int, temperature: float = 1.0) -> None:
        assert 0 < draft_layers < model.cfg.n_layer, (
            f"draft_layers={draft_layers} must be strictly between 0 and "
            f"n_layer={model.cfg.n_layer} to be a genuine (cheaper) approximation"
        )
        self.model = model
        self.draft_layers = draft_layers
        self.temperature = temperature

    @torch.no_grad()
    def propose(
        self, context: list[int], gamma: int, vocab_mask: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor] | None:
        device = next(self.model.parameters()).device
        block_size = self.model.cfg.block_size
        draft = list(context)
        draft_ids: list[int] = []
        probs_rows: list[torch.Tensor] = []

        for _ in range(gamma):
            window = draft[-block_size:]
            idx = torch.tensor([window], device=device)
            pos = torch.arange(len(window), device=device).unsqueeze(0)
            batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=True, cache=None)
            logits = _forward_first_k_layers(self.model, batch, self.draft_layers).squeeze(0)
            probs = _logits_to_probs(logits, self.temperature, vocab_mask)
            next_tok = _sample_from_probs(probs)
            draft.append(next_tok)
            draft_ids.append(next_tok)
            probs_rows.append(probs)

        if not draft_ids:
            return None
        return draft_ids, torch.stack(probs_rows)


def _logits_to_probs(
    logits: torch.Tensor, temperature: float, vocab_mask: torch.Tensor | None,
) -> torch.Tensor:
    """temperature<=0 -> one-hot at argmax. This is what makes greedy
    exactness fall out of the general verify_and_accept() algorithm with no
    special-casing -- see module docstring."""
    if vocab_mask is not None:
        logits = logits.masked_fill(vocab_mask, float("-inf"))
    if temperature <= 0:
        idx = torch.argmax(logits, dim=-1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return one_hot
    return F.softmax(logits / temperature, dim=-1)


def _sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())


def verify_and_accept(
    target_p: torch.Tensor,
    draft_ids: list[int],
    draft_q: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """The exact algorithm (module docstring steps 3-4).

    target_p: (gamma+1, vocab) -- target's distribution at each of the
      gamma+1 positions (index 0 predicts draft_ids[0]'s slot, ..., index
      gamma-1 predicts draft_ids[gamma-1]'s slot, index gamma is the "bonus"
      distribution for the position after the last draft token).
    draft_q: (gamma, vocab) -- draft's distribution at each proposed position
      (a one-hot row for a deterministic proposer).

    Returns (accepted_token_ids, num_draft_tokens_accepted). accepted_token_ids
    always has length num_draft_tokens_accepted + 1: either every draft token
    plus a bonus token (all accepted), or the accepted prefix plus one
    resampled correction token (first rejection).
    """
    gamma = len(draft_ids)
    accepted: list[int] = []
    for i in range(gamma):
        x_i = draft_ids[i]
        p_i = target_p[i, x_i].item()
        q_i = draft_q[i, x_i].item()
        accept_prob = min(1.0, p_i / q_i) if q_i > 0 else 0.0
        r = torch.rand(1, generator=generator).item()
        if r < accept_prob:
            accepted.append(x_i)
            continue

        residual = (target_p[i] - draft_q[i]).clamp(min=0)
        residual_sum = residual.sum()
        if residual_sum <= 0:
            # Degenerate (e.g. target_p and draft_q coincide exactly at this
            # position): fall back to sampling target_p directly rather than
            # dividing by ~0.
            residual, residual_sum = target_p[i], target_p[i].sum()
        resample_dist = residual / residual_sum
        correction = _sample_from_probs(resample_dist, generator)
        accepted.append(correction)
        return accepted, i

    bonus = _sample_from_probs(target_p[gamma], generator)
    accepted.append(bonus)
    return accepted, gamma


@dataclass
class SpecDecodeStats:
    num_rounds: int = 0
    num_draft_tokens_proposed: int = 0
    num_draft_tokens_accepted: int = 0
    accepted_lengths: list[int] = field(default_factory=list)  # tokens produced per round

    @property
    def acceptance_rate(self) -> float:
        if self.num_draft_tokens_proposed == 0:
            return 0.0
        return self.num_draft_tokens_accepted / self.num_draft_tokens_proposed

    @property
    def mean_accepted_length(self) -> float:
        if not self.accepted_lengths:
            return 0.0
        return sum(self.accepted_lengths) / len(self.accepted_lengths)


def _advance_one_step(
    model: GPT, cache: StaticKVCache, token_id: int, position: int, device: torch.device,
) -> torch.Tensor:
    idx = torch.tensor([[token_id]], device=device)
    pos = torch.tensor([[position]], device=device)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=False, cache=cache)
    logits = model(batch).squeeze(0)
    cache.advance(1, 1)
    return logits


@torch.no_grad()
def speculative_generate(
    target_model: GPT,
    drafter: DraftProposer,
    prompt_ids: list[int],
    max_new_tokens: int,
    gamma: int,
    temperature: float = 1.0,
    vocab_mask: torch.Tensor | None = None,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[list[int], SpecDecodeStats]:
    """Returns (new_token_ids, stats). `new_token_ids` excludes the prompt."""
    device = next(target_model.parameters()).device
    dtype = next(target_model.parameters()).dtype
    cfg = target_model.cfg
    cache = StaticKVCache(
        cfg.n_layer, cfg.n_head, target_model.head_dim,
        max_batch_size=1, max_seq_len=cfg.block_size, device=device, dtype=dtype,
    )

    prompt_len = len(prompt_ids)
    idx = torch.tensor([prompt_ids], device=device)
    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=True, cache=cache)
    pending_logits = target_model(batch).squeeze(0)
    cache.advance(prompt_len, 1)

    generated = list(prompt_ids)
    stats = SpecDecodeStats()
    finished = False

    while len(generated) - prompt_len < max_new_tokens and not finished:
        cache_len = int(cache.seq_lens[0].item())
        assert cache_len == len(generated), "next_position == num_computed_tokens invariant broken"

        remaining = max_new_tokens - (len(generated) - prompt_len)
        g = min(gamma, max(0, remaining - 1))  # leave room for >=1 real/bonus token
        proposal = drafter.propose(generated, g, vocab_mask) if g > 0 else None

        if proposal is None:
            next_tok = _sample_from_probs(
                _logits_to_probs(pending_logits, temperature, vocab_mask), generator,
            )
            generated.append(next_tok)
            stats.num_rounds += 1
            stats.accepted_lengths.append(1)
            if eot_token_id is not None and next_tok == eot_token_id:
                break
            pending_logits = _advance_one_step(target_model, cache, next_tok, cache_len, device)
            continue

        draft_ids, draft_q = proposal
        g = len(draft_ids)

        didx = torch.tensor([draft_ids], device=device)
        dpos = torch.arange(cache_len, cache_len + g, device=device).unsqueeze(0)
        vmask = build_speculative_verify_mask(cache_len, g, dtype=dtype, device=device)
        vbatch = ForwardBatch(
            input_ids=didx, position_ids=dpos, is_prefill=False,
            cache=cache, attn_mask=vmask, return_all_logits=True,
        )
        o_logits = target_model(vbatch).squeeze(0)  # (g, vocab)

        raw_target_p = torch.cat([pending_logits.unsqueeze(0), o_logits], dim=0)  # (g+1, vocab)
        target_p = torch.stack([
            _logits_to_probs(raw_target_p[i], temperature, vocab_mask) for i in range(g + 1)
        ])

        accepted_ids, num_draft_accepted = verify_and_accept(
            target_p, draft_ids, draft_q, generator,
        )
        stats.num_rounds += 1
        stats.num_draft_tokens_proposed += g
        stats.num_draft_tokens_accepted += num_draft_accepted
        stats.accepted_lengths.append(len(accepted_ids))

        cache.advance(num_draft_accepted, 1)
        generated.extend(accepted_ids)

        if eot_token_id is not None and eot_token_id in accepted_ids:
            eot_offset = accepted_ids.index(eot_token_id)
            generated = generated[: len(generated) - len(accepted_ids) + eot_offset + 1]
            finished = True
            break

        # The extra confirmed token (resample correction, or bonus if every
        # draft token was accepted) was never written into the cache by the
        # verification forward above -- only the g draft tokens were. One
        # more real decode step writes it and produces next round's
        # pending_logits.
        extra_token = accepted_ids[-1]
        new_cache_len = int(cache.seq_lens[0].item())
        pending_logits = _advance_one_step(target_model, cache, extra_token, new_cache_len, device)

    new_tokens = generated[prompt_len:prompt_len + max_new_tokens]
    return new_tokens, stats
