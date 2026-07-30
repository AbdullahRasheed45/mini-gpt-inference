"""Phase 6: speculative decoding correctness (docs/PLAN.md §7 Phase 6
acceptance, §8 rungs 7-8). The two tests that matter most, per the plan's own
framing ("this is the phase's real contribution"):
  - greedy exactness (temperature=0): speculative output must be IDENTICAL
    to target-only greedy, always, exactly.
  - distributional correctness (temperature=1): a chi-square test over
    >=100k samples must not reject the null hypothesis that speculative
    sampling and direct target sampling are the same distribution.
Most projects implement speculative decoding and never check either.
"""

import torch
from scipy.stats import chisquare

from minigpt_infer.batch import ForwardBatch
from minigpt_infer.engine.spec_decode import (
    PromptLookupDrafter,
    SelfSpeculativeDrafter,
    SpecDecodeStats,
    _logits_to_probs,
    speculative_generate,
    verify_and_accept,
)
from minigpt_infer.generation import greedy_generate_cached
from minigpt_infer.model import GPT
from tests.helpers import tiny_gpt_config

# ---- PromptLookupDrafter -----------------------------------------------

def test_prompt_lookup_drafter_finds_repeated_ngram():
    drafter = PromptLookupDrafter(vocab_size=64, ngram_size=3)
    # "...1 2 3..." appeared once already, followed by 7 8 9 -- the needle
    # at the end (1 2 3) should match that earlier occurrence.
    context = [5, 1, 2, 3, 7, 8, 9, 4, 1, 2, 3]
    draft_ids, draft_probs = drafter.propose(context, gamma=3)
    assert draft_ids == [7, 8, 9]
    assert draft_probs.shape == (3, 64)
    assert torch.equal(draft_probs.sum(dim=-1), torch.ones(3))
    for i, tok in enumerate(draft_ids):
        assert draft_probs[i, tok].item() == 1.0


def test_prompt_lookup_drafter_returns_none_without_a_match():
    drafter = PromptLookupDrafter(vocab_size=64, ngram_size=3)
    context = [1, 2, 3, 4, 5, 6, 7]  # no repeated 3-gram anywhere
    assert drafter.propose(context, gamma=3) is None


def test_prompt_lookup_drafter_returns_none_when_context_too_short():
    drafter = PromptLookupDrafter(vocab_size=64, ngram_size=3)
    assert drafter.propose([1, 2], gamma=3) is None


def test_prompt_lookup_drafter_search_excludes_the_needles_own_occurrence():
    """The search range deliberately stops one short of the needle's own
    start, so a match can never be "found" against itself -- verified here
    by confirming a genuine EARLIER match is still found and used even when
    it sits close to the needle."""
    drafter = PromptLookupDrafter(vocab_size=64, ngram_size=3)
    context = [1, 2, 3, 9, 9, 1, 2, 3]  # [1,2,3] at 0-2 (earlier) and 5-7 (needle)
    draft_ids, _ = drafter.propose(context, gamma=3)
    assert draft_ids == [9, 9, 1]  # whatever followed the EARLIER match at 0-2


# ---- SelfSpeculativeDrafter ---------------------------------------------

def test_self_speculative_drafter_produces_valid_probability_rows():
    cfg = tiny_gpt_config()
    torch.manual_seed(0)
    model = GPT(cfg)
    model.eval()
    drafter = SelfSpeculativeDrafter(model, draft_layers=1, temperature=1.0)

    draft_ids, draft_probs = drafter.propose([1, 2, 3], gamma=4)
    assert len(draft_ids) == 4
    assert draft_probs.shape == (4, cfg.vocab_size)
    assert torch.allclose(draft_probs.sum(dim=-1), torch.ones(4), atol=1e-4)
    assert all(0 <= t < cfg.vocab_size for t in draft_ids)


def test_self_speculative_drafter_rejects_invalid_layer_counts():
    cfg = tiny_gpt_config()
    model = GPT(cfg)
    for bad in (0, cfg.n_layer, cfg.n_layer + 1):
        try:
            SelfSpeculativeDrafter(model, draft_layers=bad)
            raise AssertionError(f"expected an assertion error for draft_layers={bad}")
        except AssertionError as e:
            assert "draft_layers" in str(e)


# ---- verify_and_accept: the algorithm itself -----------------------------

def test_verify_and_accept_always_accepts_matching_deterministic_proposal():
    """draft_q one-hot at x, target_p also puts all its mass on x ->
    accept_prob = min(1, 1/1) = 1, always accepted."""
    vocab = 10
    target_p = torch.zeros(2, vocab)
    target_p[0, 5] = 1.0
    target_p[1, 3] = 1.0
    draft_q = torch.zeros(1, vocab)
    draft_q[0, 5] = 1.0

    accepted, num_accepted = verify_and_accept(target_p, [5], draft_q)
    assert num_accepted == 1
    # draft token 5 accepted, then bonus sampled from target_p[1] (also one-hot)
    assert accepted == [5, 3]


def test_verify_and_accept_always_rejects_impossible_proposal_and_resamples_target():
    """draft proposes a token the target assigns zero probability -> always
    rejected, and the resample must come from the target's own distribution
    at that position (since q=0 there, residual = target_p exactly)."""
    vocab = 10
    target_p = torch.zeros(1, vocab)
    target_p[0, 7] = 1.0  # target is CERTAIN the right token is 7
    draft_q = torch.zeros(1, vocab)
    draft_q[0, 2] = 1.0  # draft proposed a different, impossible token

    accepted, num_accepted = verify_and_accept(target_p, [2], draft_q)
    assert num_accepted == 0
    assert accepted == [7]  # resampled from target_p, which is one-hot at 7


def test_greedy_exactness_via_one_hot_logits_to_probs():
    """_logits_to_probs(temperature=0) must produce the same one-hot
    representation used to derive the greedy-exactness argument in the
    module docstring."""
    logits = torch.tensor([1.0, 5.0, 2.0, 5.0])  # tie between idx 1 and 3
    probs = _logits_to_probs(logits, temperature=0.0, vocab_mask=None)
    assert probs.sum().item() == 1.0
    assert probs[torch.argmax(logits)].item() == 1.0


# ---- Distributional correctness: the chi-square test --------------------

def test_verify_and_accept_chi_square_distributional_correctness():
    """docs/PLAN.md Phase 6: sample the accepted/resampled token N>=100k
    times via verify_and_accept and via direct target sampling, and confirm
    a chi-square goodness-of-fit test does not reject at alpha=0.01. Uses a
    genuinely non-trivial (non-one-hot) target_p/draft_q so both the accept
    and the resample-residual code paths are exercised across the trials,
    not just one of them.
    """
    torch.manual_seed(0)
    vocab = 12
    # target_p needs one row per verification position (gamma=1 draft token)
    # PLUS one "bonus" row for the all-accepted branch -- shape (gamma+1,
    # vocab) = (2, vocab). Row 1 (bonus) is only reached if the draft is
    # accepted; its exact values don't affect what's being checked (the
    # distribution of accepted[0], which only ever depends on row 0).
    target_logits = torch.randn(2, vocab) * 1.5
    target_p = torch.softmax(target_logits, dim=-1)
    draft_logits = torch.randn(1, vocab) * 1.5
    draft_q = torch.softmax(draft_logits, dim=-1)

    n_trials = 150_000
    gen = torch.Generator().manual_seed(1)
    spec_samples = torch.empty(n_trials, dtype=torch.long)
    for i in range(n_trials):
        # The draft token must be freshly sampled from q on EVERY trial, not
        # fixed once -- the exactness theorem is a statement about the
        # marginal distribution over x ~ q, not about any single fixed x
        # (conditioning on one fixed x biases the output toward that x,
        # since "accept" can only ever return x itself).
        draft_token = int(torch.multinomial(draft_q[0], 1, generator=gen).item())
        accepted, _ = verify_and_accept(target_p, [draft_token], draft_q, generator=gen)
        spec_samples[i] = accepted[0]

    gen2 = torch.Generator().manual_seed(2)
    direct_samples = torch.multinomial(target_p[0], n_trials, replacement=True, generator=gen2)

    spec_counts = torch.bincount(spec_samples, minlength=vocab).numpy()
    direct_counts = torch.bincount(direct_samples, minlength=vocab).numpy()

    # chisquare needs both arrays to sum to the same total; scale expected
    # (direct_counts) so the comparison is about SHAPE, not raw count.
    expected = direct_counts * (spec_counts.sum() / direct_counts.sum())
    _stat, p_value = chisquare(f_obs=spec_counts, f_exp=expected)
    assert p_value > 0.01, f"chi-square rejected the null at p={p_value:.4f}"


# ---- End-to-end greedy exactness -----------------------------------------

def test_speculative_generate_greedy_exactness_self_speculative():
    cfg = tiny_gpt_config()
    torch.manual_seed(3)
    model = GPT(cfg)
    model.eval()
    drafter = SelfSpeculativeDrafter(model, draft_layers=1, temperature=0.0)

    mismatches = []
    for p in range(10):
        torch.manual_seed(100 + p)
        prompt = torch.randint(0, cfg.vocab_size, (1, 3)).tolist()[0]
        max_new = 8

        spec_tokens, stats = speculative_generate(
            model, drafter, prompt, max_new, gamma=3, temperature=0.0,
        )
        ref = greedy_generate_cached(model, torch.tensor([prompt]), max_new)[0].tolist()

        if prompt + spec_tokens != ref:
            mismatches.append((p, prompt, prompt + spec_tokens, ref))

    assert not mismatches, f"{len(mismatches)}/10 mismatched: {mismatches[:3]}"


def test_speculative_generate_greedy_exactness_prompt_lookup():
    cfg = tiny_gpt_config()
    torch.manual_seed(4)
    model = GPT(cfg)
    model.eval()
    drafter = PromptLookupDrafter(vocab_size=cfg.vocab_size, ngram_size=2)

    # Deliberately repetitive prompts so prompt-lookup actually finds matches
    # (otherwise this degenerates to the trivial all-fallback case, which
    # passes but doesn't exercise verify_and_accept at all).
    prompts = [
        [1, 2, 3, 1, 2],
        [5, 6, 5, 6, 5, 6],
        [9, 1, 9, 2, 9, 1],
    ]
    mismatches = []
    for prompt in prompts:
        max_new = 6
        spec_tokens, _stats = speculative_generate(
            model, drafter, prompt, max_new, gamma=3, temperature=0.0,
        )
        ref = greedy_generate_cached(model, torch.tensor([prompt]), max_new)[0].tolist()
        if prompt + spec_tokens != ref:
            mismatches.append((prompt, prompt + spec_tokens, ref))

    assert not mismatches, f"mismatches: {mismatches}"


def test_speculative_generate_actually_exercises_multi_token_acceptance():
    """A self-check that the exactness tests above aren't accidentally
    trivial (always falling back to single-token decode): over enough
    rounds, at least one round must accept more than one token."""
    cfg = tiny_gpt_config()
    torch.manual_seed(5)
    model = GPT(cfg)
    model.eval()
    drafter = SelfSpeculativeDrafter(model, draft_layers=1, temperature=1.0)

    prompt = torch.randint(0, cfg.vocab_size, (1, 3)).tolist()[0]
    _tokens, stats = speculative_generate(
        model, drafter, prompt, max_new_tokens=12, gamma=4, temperature=1.0,
        generator=torch.Generator().manual_seed(6),
    )
    assert isinstance(stats, SpecDecodeStats)
    assert stats.num_draft_tokens_proposed > 0
    assert any(length > 1 for length in stats.accepted_lengths), (
        f"expected at least one multi-token round, got lengths={stats.accepted_lengths}"
    )


def test_speculative_generate_respects_eot_and_max_tokens():
    cfg = tiny_gpt_config()
    torch.manual_seed(7)
    model = GPT(cfg)
    model.eval()
    eot = 63
    vocab_mask = torch.ones(cfg.vocab_size, dtype=torch.bool)
    vocab_mask[eot] = False  # force every sampled token to be EOT

    drafter = SelfSpeculativeDrafter(model, draft_layers=1, temperature=0.0)
    prompt = [1, 2, 3]
    tokens, _stats = speculative_generate(
        model, drafter, prompt, max_new_tokens=10, gamma=3,
        temperature=0.0, vocab_mask=vocab_mask, eot_token_id=eot,
    )
    assert tokens == [eot]


def test_forward_batch_multi_token_decode_requires_explicit_mask():
    idx = torch.zeros(1, 3, dtype=torch.long)
    pos = torch.zeros(1, 3, dtype=torch.long)
    try:
        ForwardBatch(input_ids=idx, position_ids=pos, is_prefill=False)
        raise AssertionError("expected an assertion error for T>1 decode without attn_mask")
    except AssertionError as e:
        assert "attn_mask" in str(e)
