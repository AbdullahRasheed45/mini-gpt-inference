"""Phase 2: batched sampling correctness (docs/PLAN.md §5 SamplingParams, §7
Phase 2 point 4 -- top-k/top-p must be vectorized per-row, not a Python loop).
"""

import torch

from minigpt_infer.sampling import (
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_batch,
    update_seen_mask,
)


def test_apply_top_k_keeps_exactly_k_finite_per_row_with_distinct_values():
    torch.manual_seed(0)
    logits = torch.randn(3, 20)  # continuous values -> no ties
    k = torch.tensor([1, 5, 20])
    out = apply_top_k(logits, k)
    finite_counts = (out > torch.finfo(out.dtype).min).sum(dim=-1)
    assert finite_counts.tolist() == [1, 5, 20]


def test_apply_top_k_keeps_the_actual_highest_values():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = apply_top_k(logits, torch.tensor([2]))
    kept = (out > torch.finfo(out.dtype).min).squeeze(0)
    # the two largest values are 5.0 (idx 1) and 4.0 (idx 4)
    assert kept.tolist() == [False, True, False, False, True]


def test_apply_top_p_always_keeps_the_top_token():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    out = apply_top_p(logits, torch.tensor([1e-6]))  # near-zero mass allowed
    kept = (out > torch.finfo(out.dtype).min).squeeze(0)
    assert kept[0].item() is True or kept.tolist()[0]
    assert kept.sum().item() == 1


def test_apply_top_p_keeps_more_tokens_as_p_grows():
    torch.manual_seed(1)
    logits = torch.randn(1, 50)
    small = (apply_top_p(logits, torch.tensor([0.1])) > torch.finfo(logits.dtype).min).sum()
    large = (apply_top_p(logits, torch.tensor([0.99])) > torch.finfo(logits.dtype).min).sum()
    assert large >= small


def test_sample_batch_temperature_zero_is_greedy():
    torch.manual_seed(0)
    logits = torch.randn(4, 30)
    out = sample_batch(logits, temperature=0.0)
    assert torch.equal(out, torch.argmax(logits, dim=-1, keepdim=True))


def test_sample_batch_mixed_greedy_and_stochastic_rows():
    """Per-row temperature: row 0 greedy, row 1 stochastic -- one batched call
    must handle both without corrupting the greedy row."""
    torch.manual_seed(0)
    logits = torch.randn(2, 30)
    temps = torch.tensor([0.0, 1.0])
    out = sample_batch(logits, temperature=temps, generator=torch.Generator().manual_seed(0))
    assert out[0, 0].item() == torch.argmax(logits[0]).item()


def test_sample_batch_is_deterministic_with_a_seeded_generator():
    torch.manual_seed(0)
    logits = torch.randn(4, 30)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    out1 = sample_batch(logits, temperature=1.0, top_k=10, generator=gen1)
    out2 = sample_batch(logits, temperature=1.0, top_k=10, generator=gen2)
    assert torch.equal(out1, out2)


def test_apply_repetition_penalty_reduces_seen_token_score():
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    seen = torch.tensor([[True, True, False]])
    out = apply_repetition_penalty(logits, seen, penalty=2.0)
    assert out[0, 0].item() == 1.0    # positive logit divided by penalty
    assert out[0, 1].item() == -4.0   # negative logit multiplied by penalty
    assert out[0, 2].item() == 0.5    # unseen token untouched


def test_apply_repetition_penalty_noop_at_penalty_one():
    logits = torch.randn(2, 10)
    seen = torch.rand(2, 10) > 0.5
    out = apply_repetition_penalty(logits, seen, penalty=1.0)
    assert torch.equal(out, logits)


def test_update_seen_mask_marks_new_ids():
    seen = torch.zeros(2, 5, dtype=torch.bool)
    update_seen_mask(seen, torch.tensor([[1], [3]]))
    assert seen.tolist() == [[False, True, False, False, False], [False, False, False, True, False]]
