"""Phase 1: KV cache correctness (docs/PLAN.md §7 Phase 1, §8 rungs 1-2).

The two things that matter most here:
  1. Cached generation must produce EXACT (not just close) greedy token ids
     vs. the naive reference, for many prompts.
  2. The is_causal decode pitfall (docs/PLAN.md Phase 1 pitfalls) must be
     caught by a test that would fail if someone "fixed" is_causal back to
     True for decode -- not just implicitly relied upon.
"""

from unittest import mock

import torch
import torch.nn.functional as F

from minigpt_infer.batch import ForwardBatch
from minigpt_infer.cache.static import StaticKVCache
from minigpt_infer.generation import greedy_generate_cached
from minigpt_infer.model import GPT
from minigpt_infer.reference import ReferenceGPT
from tests.helpers import tiny_gpt_config


def _load_matching_reference(model: GPT, cfg) -> ReferenceGPT:
    ref = ReferenceGPT(cfg)
    ref.load_state_dict(model.state_dict())
    return ref


def test_single_step_logit_parity_prefill():
    """Rung 1: a single prefill forward through the cached model must match
    the oracle's full forward at the last position, bit-close (fp32)."""
    cfg = tiny_gpt_config()
    torch.manual_seed(0)
    model = GPT(cfg)
    ref = _load_matching_reference(model, cfg)

    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    cache = StaticKVCache(cfg.n_layer, cfg.n_head, model.head_dim,
                           max_batch_size=2, max_seq_len=cfg.block_size)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=True, cache=cache)

    cached_logits = model(batch)
    ref_logits = ref(idx)[:, -1, :]

    assert torch.allclose(cached_logits, ref_logits, atol=1e-5), (
        (cached_logits - ref_logits).abs().max().item()
    )


def test_exact_greedy_match_across_many_prompts():
    """Rung 2 -- the acceptance criterion from docs/PLAN.md Phase 1: exact
    (not approximate) greedy token-id match, >=20 prompts."""
    cfg = tiny_gpt_config()
    torch.manual_seed(1)
    model = GPT(cfg)
    ref = _load_matching_reference(model, cfg)

    num_prompts = 20
    prompt_len = 4
    max_new_tokens = 12
    mismatches = []

    for p in range(num_prompts):
        torch.manual_seed(1000 + p)
        idx = torch.randint(0, cfg.vocab_size, (1, prompt_len))

        cached_out = greedy_generate_cached(model, idx.clone(), max_new_tokens)
        ref_out = ref.greedy_generate(idx.clone(), max_new_tokens)

        if not torch.equal(cached_out, ref_out):
            mismatches.append((p, cached_out.tolist(), ref_out.tolist()))

    assert not mismatches, f"{len(mismatches)}/{num_prompts} prompts mismatched: {mismatches[:3]}"


def test_exact_greedy_match_with_batch_size_greater_than_one():
    """Same test, but with a batch of prompts sharing one cache -- Phase 1's
    StaticKVCache must handle B>1 as long as all rows share the same length."""
    cfg = tiny_gpt_config()
    torch.manual_seed(2)
    model = GPT(cfg)
    ref = _load_matching_reference(model, cfg)

    B, prompt_len, max_new_tokens = 4, 5, 10
    idx = torch.randint(0, cfg.vocab_size, (B, prompt_len))

    cached_out = greedy_generate_cached(model, idx.clone(), max_new_tokens)
    for b in range(B):
        ref_out = ref.greedy_generate(idx[b:b + 1].clone(), max_new_tokens)
        assert torch.equal(cached_out[b:b + 1], ref_out), f"row {b} mismatched"


def test_decode_calls_sdpa_with_is_causal_false():
    """White-box test of the exact pitfall documented in model.py's module
    docstring and docs/PLAN.md Phase 1: spy on the real argument passed to
    SDPA during a decode step, rather than inferring it indirectly from
    output behavior.

    (An earlier version of this test tried to infer the bug from whether
    changing cached content changed the final argmax -- that's fragile
    against an untrained random-init model, where one vocab index can
    dominate argmax regardless of input, even though the logits genuinely do
    differ underneath. Spying on the actual kwarg is unambiguous.)
    """
    cfg = tiny_gpt_config()
    torch.manual_seed(0)
    model = GPT(cfg)
    cache = StaticKVCache(cfg.n_layer, cfg.n_head, model.head_dim,
                           max_batch_size=1, max_seq_len=cfg.block_size)

    calls = []
    real_sdpa = F.scaled_dot_product_attention

    def spy_sdpa(q, k, v, *args, **kwargs):
        calls.append({"is_causal": kwargs.get("is_causal"),
                       "q_len": q.shape[2], "kv_len": k.shape[2]})
        return real_sdpa(q, k, v, *args, **kwargs)

    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    pos = torch.arange(4).unsqueeze(0)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=True, cache=cache)
    with mock.patch("minigpt_infer.model.F.scaled_dot_product_attention", side_effect=spy_sdpa):
        model(batch)
    cache.advance(4, 1)

    for call in calls:  # the prefill calls: q_len == kv_len, is_causal must be True
        assert call["is_causal"] is True
        assert call["q_len"] == call["kv_len"]
    calls.clear()

    next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
    pos2 = torch.full((1, 1), 4, dtype=torch.long)
    batch2 = ForwardBatch(input_ids=next_tok, position_ids=pos2, is_prefill=False, cache=cache)
    with mock.patch("minigpt_infer.model.F.scaled_dot_product_attention", side_effect=spy_sdpa):
        model(batch2)

    assert len(calls) == cfg.n_layer, "expected one SDPA call per layer"
    for call in calls:
        assert call["is_causal"] is False, f"decode step called SDPA with is_causal=True: {call}"
        assert call["kv_len"] > call["q_len"], (
            "decode should attend over more cached keys than the current query length"
        )


def test_position_ids_beyond_block_size_asserts():
    cfg = tiny_gpt_config(block_size=8)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 1))
    pos = torch.tensor([[8]])  # == block_size, must be rejected (valid range is 0..block_size-1)
    batch = ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=False,
                          cache=StaticKVCache(cfg.n_layer, cfg.n_head, model.head_dim, 1, 8))
    try:
        model(batch)
        raise AssertionError("expected an assertion error for position_ids >= block_size")
    except AssertionError as e:
        assert "block_size" in str(e)


def test_cache_write_then_read_roundtrip():
    cache = StaticKVCache(n_layer=1, n_head=2, head_dim=4, max_batch_size=1, max_seq_len=8)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    batch = ForwardBatch(
        input_ids=torch.zeros(1, 3, dtype=torch.long),
        position_ids=torch.arange(3).unsqueeze(0),
        is_prefill=True,
    )
    cache.write(0, k, v, batch)
    read_k, read_v = cache.read(0, batch)
    assert torch.equal(read_k, k)
    assert torch.equal(read_v, v)


def test_cache_rejects_non_uniform_seq_lens():
    cache = StaticKVCache(n_layer=1, n_head=2, head_dim=4, max_batch_size=2, max_seq_len=8)
    cache.seq_lens[0] = 3
    cache.seq_lens[1] = 5  # rows disagree -- must be caught, not silently corrupt the cache
    k = torch.randn(2, 2, 1, 4)
    v = torch.randn(2, 2, 1, 4)
    batch = ForwardBatch(
        input_ids=torch.zeros(2, 1, dtype=torch.long),
        position_ids=torch.zeros(2, 1, dtype=torch.long),
        is_prefill=False,
    )
    try:
        cache.write(0, k, v, batch)
        raise AssertionError("expected an assertion error for non-uniform seq_lens")
    except AssertionError as e:
        assert "share the same current length" in str(e)
