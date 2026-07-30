"""Phase 3: paged KV cache + engine correctness (docs/PLAN.md §7 Phase 3,
§8 rung 4).

The two exact-match tests here are the phase's real acceptance bar: paged
generation, whether one sequence at a time or several sharing one engine with
staggered arrivals, must produce byte-identical greedy output to Phase 1's
StaticKVCache-based reference. Two real bugs were caught and fixed by these
tests during development (see engine.py's docstrings):
  1. `_sample_and_advance` double-incrementing num_computed_tokens on top of
     `_prefill_one`'s explicit T-token jump (corrupted every sequence's very
     first decode step).
  2. `_run_decode` iterating `self.scheduler.running` directly while
     `scheduler.finish()` mutates that same list mid-iteration (classic
     "remove while iterating" bug -- silently skipped whichever sequence
     came right after one that finished in the same batch).
Both were invisible in a single-request-at-a-time test and only surfaced
once >=2 real concurrent sequences were exercised together.
"""

import torch

from minigpt_infer.batch import ForwardBatch
from minigpt_infer.cache.paged import BlockManager, PagedKVCache
from minigpt_infer.config import EngineConfig, SamplingParams
from minigpt_infer.engine.engine import LLMEngine
from minigpt_infer.engine.request import Request
from minigpt_infer.generation import greedy_generate_cached
from minigpt_infer.model import GPT
from tests.helpers import tiny_gpt_config


def _run_to_completion(
    engine: LLMEngine, request_ids: list[str], max_steps: int = 200
) -> dict[str, list[int]]:
    got = {rid: [] for rid in request_ids}
    steps = 0
    while engine.has_unfinished_requests() and steps < max_steps:
        for out in engine.step():
            got[out.request_id].extend(out.new_token_ids)
        steps += 1
    assert steps < max_steps, "engine did not converge -- likely a scheduler livelock"
    return got


def test_paged_engine_matches_static_reference_single_request_many_prompts():
    """Rung 4's core claim: >=20 prompts, one request per engine at a time."""
    cfg = tiny_gpt_config(vocab_size=64, block_size=32)
    torch.manual_seed(11)
    model = GPT(cfg)
    model.eval()
    max_new_tokens = 8

    mismatches = []
    for p in range(20):
        torch.manual_seed(1000 + p)
        prompt_len = 3 + (p % 5)
        prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len)).tolist()[0]

        ref = greedy_generate_cached(model, torch.tensor([prompt]), max_new_tokens)[0].tolist()

        engine = LLMEngine(model, EngineConfig(block_size=4, num_blocks=256, max_batch_size=4))
        engine.add_request(
            Request(f"r{p}", prompt, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        )
        got = _run_to_completion(engine, [f"r{p}"])
        full = prompt + got[f"r{p}"]

        if full != ref:
            mismatches.append((p, prompt, full, ref))

    assert not mismatches, f"{len(mismatches)}/20 prompts mismatched: {mismatches[:3]}"


def test_paged_engine_matches_static_reference_concurrent_ragged_requests():
    """The actual continuous-batching scenario: several ragged-length
    requests sharing one engine, admitted in staggered waves (max_batch_size
    forces some to wait), decoded together once running. This is what caught
    both real bugs documented in this module's docstring."""
    cfg = tiny_gpt_config(vocab_size=64, block_size=32)
    torch.manual_seed(7)
    model = GPT(cfg)
    model.eval()
    max_new_tokens = 6

    prompts, refs = {}, {}
    for p in range(6):
        torch.manual_seed(2000 + p)
        prompt_len = 2 + p
        prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len)).tolist()[0]
        prompts[f"r{p}"] = prompt
        ref = greedy_generate_cached(model, torch.tensor([prompt]), max_new_tokens)
        refs[f"r{p}"] = ref[0].tolist()

    engine = LLMEngine(model, EngineConfig(block_size=4, num_blocks=256, max_batch_size=3))
    for rid, prompt in prompts.items():
        engine.add_request(
            Request(rid, prompt, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        )

    got = _run_to_completion(engine, list(prompts))
    mismatches = [rid for rid in prompts if prompts[rid] + got[rid] != refs[rid]]

    assert not mismatches, f"mismatched requests: {mismatches}"
    assert engine.block_manager.num_free_blocks == 256, "blocks leaked after all requests finished"


def test_preemption_recovers_correct_output_under_a_tiny_pool():
    """A pool too small for both requests to grow at once forces a real
    preemption (not just the scheduler-level unit test in
    tests/test_scheduler.py) -- and the preempted request must still finish
    with the exact correct output after being recomputed from scratch."""
    cfg = tiny_gpt_config(vocab_size=64, block_size=32)
    torch.manual_seed(3)
    model = GPT(cfg)
    model.eval()
    max_new_tokens = 4

    prompt_a = [1, 2]
    prompt_b = [4, 5]
    ref_a = greedy_generate_cached(model, torch.tensor([prompt_a]), max_new_tokens)[0].tolist()
    ref_b = greedy_generate_cached(model, torch.tensor([prompt_b]), max_new_tokens)[0].tolist()

    # block_size=2, prompt_len=2 -> each starts at 1 block. Both admit
    # together (2 blocks used, 2 free). Both decode in lockstep (identical
    # prompt length, same speed) and both need a 3rd block at the SAME step
    # (position 4 needs ceil(5/2)=3 blocks) -- but only 4 blocks exist total,
    # and each sequence needs up to 3 at completion (2+4=6 tokens ->
    # ceil(6/2)=3), so both peaking at once (6 > 4) is infeasible while both
    # completing separately (3 <= 4) is not -- forces exactly one preemption,
    # recoverable once the other finishes and frees its blocks.
    engine = LLMEngine(model, EngineConfig(block_size=2, num_blocks=4, max_batch_size=2))
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    engine.add_request(Request("a", prompt_a, sp))
    engine.add_request(Request("b", prompt_b, sp))

    got = _run_to_completion(engine, ["a", "b"])

    assert prompt_a + got["a"] == ref_a
    assert prompt_b + got["b"] == ref_b
    assert engine.scheduler.num_preemptions > 0, (
        "test setup should have forced at least one preemption"
    )
    assert engine.block_manager.num_free_blocks == 4


def test_paged_kvcache_write_then_read_roundtrip():
    cache = PagedKVCache(n_layer=1, n_head=2, head_dim=4, block_size=2, num_blocks=4)
    bm = BlockManager(num_blocks=4, block_size=2)

    from minigpt_infer.engine.request import Request as _Req
    from minigpt_infer.engine.request import SequenceState as _Seq
    seq = _Seq(request=_Req("x", [0, 0, 0], SamplingParams()))
    bm.allocate(seq, 3)  # 2 blocks

    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    slot_mapping = torch.tensor([
        seq.block_table[0] * 2 + 0, seq.block_table[0] * 2 + 1, seq.block_table[1] * 2 + 0,
    ])
    block_tables = torch.tensor([seq.block_table + [-1] * 2])
    seq_lens = torch.tensor([3])

    batch = ForwardBatch(
        input_ids=torch.zeros(1, 3, dtype=torch.long), position_ids=torch.arange(3).unsqueeze(0),
        is_prefill=True, block_tables=block_tables, slot_mapping=slot_mapping, seq_lens=seq_lens,
    )
    cache.write(0, k, v, batch)
    read_k, read_v = cache.read(0, batch)
    assert torch.allclose(read_k, k)
    assert torch.allclose(read_v, v)
