"""Phase 2: static batching correctness (docs/PLAN.md §7 Phase 2, §8 rung 3).

The single most important test here is batch invariance: running N prompts
together (left-padded, ragged lengths, shared cache) must produce byte-for-
byte the same greedy output as running each prompt alone. This is the test
that catches every padding/position/mask bug -- docs/PLAN.md says explicitly
"do not skip it."
"""

import torch

from minigpt_infer.batch import build_left_padded_batch, extend_decode_mask
from minigpt_infer.generation import batched_greedy_generate, greedy_generate_cached
from minigpt_infer.model import GPT
from tests.helpers import tiny_gpt_config

PAD_TOKEN_ID = 0


def test_batch_invariance_ragged_prompts():
    cfg = tiny_gpt_config(vocab_size=64)
    torch.manual_seed(3)
    model = GPT(cfg)

    prompts = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [10],
        [11, 12, 3, 4, 5],
        [13, 14],
        [15, 16, 17],
        [18, 19, 20, 21],
    ]
    max_new_tokens = 6

    batched_out = batched_greedy_generate(
        model, [list(p) for p in prompts], max_new_tokens, pad_token_id=PAD_TOKEN_ID,
    )

    for i, p in enumerate(prompts):
        idx = torch.tensor([p], dtype=torch.long)
        solo_out = greedy_generate_cached(model, idx, max_new_tokens)[0].tolist()
        assert batched_out[i] == solo_out, (
            f"row {i} (prompt={p}) diverged from solo generation: "
            f"batched={batched_out[i]} solo={solo_out}"
        )


def test_batch_invariance_single_row_batch_matches_solo():
    """A "batch" of exactly one prompt must equal running it directly through
    the (non-batched) cached path -- a degenerate but real edge case of the
    left-padding code path (zero padding needed)."""
    cfg = tiny_gpt_config(vocab_size=64)
    torch.manual_seed(4)
    model = GPT(cfg)
    prompt = [5, 6, 7, 8]

    batched_out = batched_greedy_generate(
        model, [prompt], max_new_tokens=5, pad_token_id=PAD_TOKEN_ID,
    )
    solo_out = greedy_generate_cached(model, torch.tensor([prompt]), max_new_tokens=5)[0].tolist()
    assert batched_out[0] == solo_out


def test_left_padded_batch_position_ids():
    prompts = [[1, 2, 3], [4, 5]]
    input_ids, position_ids, _attn_mask, pad_mask = build_left_padded_batch(prompts, PAD_TOKEN_ID)

    assert input_ids.tolist() == [[1, 2, 3], [0, 4, 5]]
    # row 0: fully real -> positions 0,1,2. row 1: one pad slot (position
    # clamped to 0, unused since it's masked out) then real positions 0,1.
    assert position_ids.tolist() == [[0, 1, 2], [0, 0, 1]]
    assert pad_mask.tolist() == [[False, False, False], [True, False, False]]


def test_left_padded_batch_mask_stays_finite_even_for_all_pad_query_rows():
    """docs/PLAN.md Phase 2 pitfall: a fully-masked row produces NaN after
    softmax -- true for literal -inf masking. Using finfo.min instead (as
    this code does) makes even a query row entirely inside the left-pad
    prefix safe: softmax over an all-finfo.min row is a harmless uniform
    distribution, not NaN. That row is never actually read (only the last,
    always-real position's logits are used), but it must not corrupt the
    tensor with NaN/inf regardless."""
    prompts = [[1], [2, 3, 4, 5, 6]]  # row 0 has 4 left-pad slots
    _input_ids, _position_ids, attn_mask, _pad_mask = build_left_padded_batch(
        prompts, PAD_TOKEN_ID, dtype=torch.float32,
    )
    assert not torch.isnan(attn_mask).any()
    assert not torch.isinf(attn_mask).any(), "additive mask must stay finite (finfo.min, not -inf)"
    # the actually-used row (last query position) must be well-formed for
    # every batch row: real columns unmasked, pad columns masked.
    last_row_bias = attn_mask[:, 0, -1, :]
    assert (last_row_bias[0] == torch.tensor([torch.finfo(torch.float32).min] * 4 + [0.0])).all()
    assert (last_row_bias[1] == 0.0).all()


def test_extend_decode_mask_grows_by_one_and_keeps_padding_masked():
    pad_mask = torch.tensor([[True, False, False], [False, False, False]])
    mask0 = extend_decode_mask(pad_mask, new_kv_len=3, dtype=torch.float32)
    assert mask0.shape == (2, 1, 1, 3)
    assert mask0[0, 0, 0, 0].item() == torch.finfo(torch.float32).min  # padded key stays masked
    assert mask0[0, 0, 0, 1].item() == 0.0
    assert mask0[1, 0, 0, 0].item() == 0.0  # row 1 has no padding at all

    mask1 = extend_decode_mask(pad_mask, new_kv_len=4, dtype=torch.float32)
    assert mask1.shape == (2, 1, 1, 4)
    assert mask1[0, 0, 0, 0].item() == torch.finfo(torch.float32).min  # still masked
    assert mask1[0, 0, 0, 3].item() == 0.0  # newly appended real token: unmasked


def test_batched_generate_stops_at_eot_and_does_not_grow_past_it():
    cfg = tiny_gpt_config(vocab_size=64)
    torch.manual_seed(5)
    model = GPT(cfg)
    eot = 63

    # Force the model to immediately emit EOT for every row by masking every
    # other vocab id, isolating the finished-flag bookkeeping from real model
    # behavior (rather than hoping a random-init model happens to emit EOT).
    vocab_mask = torch.ones(cfg.vocab_size, dtype=torch.bool)
    vocab_mask[eot] = False

    prompts = [[1, 2], [3, 4, 5]]
    out = batched_greedy_generate(
        model, [list(p) for p in prompts], max_new_tokens=10,
        pad_token_id=PAD_TOKEN_ID, eot_token_id=eot, vocab_mask=vocab_mask,
    )
    for i, p in enumerate(prompts):
        assert out[i] == [*p, eot], (
            f"row {i}: expected generation to stop right after EOT, got {out[i]}"
        )
