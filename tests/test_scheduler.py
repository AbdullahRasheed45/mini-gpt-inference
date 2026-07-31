"""Phase 3: Scheduler + BlockManager (docs/PLAN.md §7 Phase 3, §8 rung 6).

Pure Python -- no torch, no model -- since Request/SequenceState/BlockManager/
Scheduler are all plain bookkeeping. Deliberately kept separate from
test_paged.py (which needs a real model to exercise the actual attention
math) so these run instantly and pin down the allocator/scheduler contracts
in isolation.
"""

import random

from minigpt_infer.cache.paged import BlockManager
from minigpt_infer.config import SamplingParams
from minigpt_infer.engine.request import Request, SequenceState, SequenceStatus
from minigpt_infer.engine.scheduler import Scheduler


def _seq(request_id: str, prompt_len: int, max_tokens: int = 10) -> SequenceState:
    req = Request(request_id, list(range(prompt_len)), SamplingParams(max_tokens=max_tokens))
    return SequenceState(request=req)


# ---- BlockManager ----------------------------------------------------------

def test_block_manager_allocate_tracks_free_count():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq("a", prompt_len=9)  # ceil(9/4) = 3 blocks
    bm.allocate(seq, 9)
    assert len(seq.block_table) == 3
    assert bm.num_free_blocks == 7


def test_block_manager_allocate_is_idempotent_for_already_covered_length():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq("a", prompt_len=9)
    bm.allocate(seq, 9)
    bm.allocate(seq, 9)  # same length again -- must not grow further
    assert len(seq.block_table) == 3
    assert bm.num_free_blocks == 7


def test_block_manager_append_slot_only_grows_at_block_boundary():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq("a", prompt_len=3)
    bm.allocate(seq, 3)  # 1 block, 1 slot free within it
    assert len(seq.block_table) == 1

    seq.num_computed_tokens = 3
    assert bm.append_slot(seq) is True  # position 3 still fits in block 0
    assert len(seq.block_table) == 1

    seq.num_computed_tokens = 4
    assert bm.append_slot(seq) is True  # position 4 needs a 2nd block
    assert len(seq.block_table) == 2


def test_block_manager_append_slot_returns_false_when_pool_exhausted():
    bm = BlockManager(num_blocks=1, block_size=4)
    seq = _seq("a", prompt_len=4)
    bm.allocate(seq, 4)  # uses the pool's only block
    assert bm.num_free_blocks == 0

    seq.num_computed_tokens = 4
    assert bm.append_slot(seq) is False  # needs a 2nd block; none free
    assert len(seq.block_table) == 1  # must not have mutated on failure


def test_block_manager_free_returns_blocks_to_pool():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq("a", prompt_len=9)
    bm.allocate(seq, 9)
    bm.free(seq)
    assert bm.num_free_blocks == 10
    assert seq.block_table == []


def test_block_manager_can_allocate_respects_free_count():
    bm = BlockManager(num_blocks=2, block_size=4)
    assert bm.can_allocate(8) is True   # exactly 2 blocks
    assert bm.can_allocate(9) is False  # needs 3


def test_block_manager_1000_randomized_alloc_free_cycles_leak_no_blocks():
    """docs/PLAN.md Phase 3 acceptance: no leaked blocks after 1000 randomized
    alloc/free cycles -- the free-list size must return to its initial value."""
    rng = random.Random(0)
    bm = BlockManager(num_blocks=64, block_size=4)
    live: list[SequenceState] = []

    for i in range(1000):
        action = rng.choice(["alloc", "free"]) if live else "alloc"
        if action == "alloc" and bm.num_free_blocks > 0:
            seq = _seq(f"s{i}", prompt_len=rng.randint(1, min(16, bm.num_free_blocks * 4)))
            if bm.can_allocate(len(seq.request.prompt_token_ids)):
                bm.allocate(seq, len(seq.request.prompt_token_ids))
                live.append(seq)
        elif live:
            seq = live.pop(rng.randrange(len(live)))
            bm.free(seq)

    for seq in live:
        bm.free(seq)

    assert bm.num_free_blocks == 64, f"blocks leaked: {bm.num_free_blocks}/64 free after cleanup"


def test_fragmentation_stats_reports_internal_fragmentation():
    bm = BlockManager(num_blocks=10, block_size=4)
    seq = _seq("a", prompt_len=5)  # 2 blocks (8 slots), 5 used -> 3 wasted
    bm.allocate(seq, 5)
    seq.num_computed_tokens = 5
    seq.status = SequenceStatus.RUNNING
    stats = bm.fragmentation_stats([seq])
    assert stats["internal_fragmentation_slots"] == 3
    assert stats["allocated_blocks"] == 2
    assert stats["free_blocks"] == 8


# ---- Scheduler --------------------------------------------------------------

def test_schedule_admits_up_to_max_batch_size():
    bm = BlockManager(num_blocks=100, block_size=4)
    sched = Scheduler(bm, max_batch_size=2)
    for i in range(3):
        sched.add_request(_seq(f"r{i}", prompt_len=4))

    batch, is_prefill = sched.schedule()
    assert is_prefill is True
    assert len(batch) == 2
    assert len(sched.waiting) == 1
    assert len(sched.running) == 2


def test_schedule_returns_decode_when_nothing_new_to_admit():
    bm = BlockManager(num_blocks=100, block_size=4)
    sched = Scheduler(bm, max_batch_size=2)
    sched.add_request(_seq("r0", prompt_len=4))
    sched.schedule()  # admits r0 (prefill)

    batch, is_prefill = sched.schedule()
    assert is_prefill is False
    assert [s.request.request_id for s in batch] == ["r0"]


def test_schedule_prefill_priority_over_decoding_running_set():
    """docs/PLAN.md Phase 3 point 5.2: if anything was admitted this step,
    prefill it -- don't decode the running set in the same step."""
    bm = BlockManager(num_blocks=100, block_size=4)
    sched = Scheduler(bm, max_batch_size=2)
    sched.add_request(_seq("r0", prompt_len=4))
    sched.schedule()  # r0 now running

    sched.add_request(_seq("r1", prompt_len=4))
    batch, is_prefill = sched.schedule()
    assert is_prefill is True
    assert [s.request.request_id for s in batch] == ["r1"]


def test_schedule_stops_admitting_when_head_of_line_does_not_fit():
    bm = BlockManager(num_blocks=1, block_size=4)  # room for exactly 1 short seq
    sched = Scheduler(bm, max_batch_size=10)
    sched.add_request(_seq("big", prompt_len=8))    # needs 2 blocks -- never fits
    sched.add_request(_seq("small", prompt_len=2))  # would fit, but is behind "big"

    batch, is_prefill = sched.schedule()
    assert is_prefill is False  # nothing admitted
    assert batch == []
    assert [s.request.request_id for s in sched.waiting] == ["big", "small"]


def test_finish_frees_blocks_and_removes_from_running():
    bm = BlockManager(num_blocks=10, block_size=4)
    sched = Scheduler(bm, max_batch_size=10)
    sched.add_request(_seq("r0", prompt_len=4))
    sched.schedule()
    seq = sched.running[0]

    sched.finish(seq, "length")
    assert seq.status == SequenceStatus.FINISHED
    assert seq.finish_reason == "length"
    assert seq not in sched.running
    assert bm.num_free_blocks == 10


def test_ensure_capacity_preempts_newest_running_when_pool_exhausted():
    """docs/PLAN.md Phase 3 point 5.5: preempt-by-recompute evicts the NEWEST
    running sequence, frees its blocks, and returns it to the front of
    `waiting` -- exercised here with a pool too small for everyone to grow."""
    bm = BlockManager(num_blocks=2, block_size=4)  # exactly 2 blocks total
    sched = Scheduler(bm, max_batch_size=10)

    sched.add_request(_seq("old", prompt_len=4))   # admitted first (older)
    sched.schedule()
    sched.add_request(_seq("new", prompt_len=4))
    sched.schedule()
    assert bm.num_free_blocks == 0
    assert [s.request.request_id for s in sched.running] == ["old", "new"]

    for s in sched.running:
        s.num_computed_tokens = 4  # both about to need a 2nd block

    sched.ensure_capacity_for_decode_step()

    assert [s.request.request_id for s in sched.running] == ["old"]
    assert [s.request.request_id for s in sched.waiting] == ["new"]
    assert sched.num_preemptions == 1
    assert len(sched.running[0].block_table) == 2  # "old" got the freed block


def test_cancel_by_id_frees_blocks_for_a_running_request():
    bm = BlockManager(num_blocks=10, block_size=4)
    sched = Scheduler(bm, max_batch_size=10)
    sched.add_request(_seq("r0", prompt_len=4))
    sched.schedule()
    seq = sched.running[0]
    assert bm.num_free_blocks == 9

    found = sched.cancel_by_id("r0")
    assert found is True
    assert bm.num_free_blocks == 10
    assert sched.running == []
    assert seq.status is SequenceStatus.FINISHED
    assert seq.finish_reason == "cancelled"


def test_cancel_by_id_removes_a_still_waiting_request():
    bm = BlockManager(num_blocks=10, block_size=4)
    sched = Scheduler(bm, max_batch_size=1)  # only 1 admitted, "r1" stays waiting
    sched.add_request(_seq("r0", prompt_len=4))
    sched.add_request(_seq("r1", prompt_len=4))
    sched.schedule()
    assert [s.request.request_id for s in sched.waiting] == ["r1"]

    found = sched.cancel_by_id("r1")
    assert found is True
    assert list(sched.waiting) == []
    assert bm.num_free_blocks == 9  # r0's allocation untouched


def test_cancel_by_id_returns_false_for_unknown_request():
    bm = BlockManager(num_blocks=10, block_size=4)
    sched = Scheduler(bm, max_batch_size=10)
    assert sched.cancel_by_id("does-not-exist") is False
