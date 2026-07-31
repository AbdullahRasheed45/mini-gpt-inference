"""Continuous batching scheduler (docs/PLAN.md §7 Phase 3, point 5).

One step() = one of:
  - a PREFILL step: admit as many waiting requests as there's room for
    (blocks free AND max_batch_size not exceeded), run them.
  - a DECODE step: no requests were admitted this cycle, so advance every
    currently-running sequence by one token instead.

This "prefill-priority" policy (admit-then-prefill beats decode every step
that has anything to admit) is deliberately simple -- no mixing prefill and
decode in the same forward call, no chunked prefill (docs/PLAN.md §12 lists
chunked prefill as an explicit non-goal).
"""

from __future__ import annotations

from collections import deque

from minigpt_infer.cache.paged import BlockManager
from minigpt_infer.engine.request import SequenceState, SequenceStatus


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_batch_size: int) -> None:
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.waiting: deque[SequenceState] = deque()
        self.running: list[SequenceState] = []
        self.num_preemptions = 0

    def add_request(self, seq: SequenceState) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def schedule(self) -> tuple[list[SequenceState], bool]:
        """Returns (batch, is_prefill)."""
        admitted: list[SequenceState] = []
        while self.waiting and len(self.running) + len(admitted) < self.max_batch_size:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq.num_tokens):
                # Head-of-line request doesn't fit right now. Don't skip ahead
                # to a smaller one behind it -- that's starvation-prone and
                # out of scope here; just stop admitting this step.
                break
            self.block_manager.allocate(seq, seq.num_tokens)
            seq.status = SequenceStatus.RUNNING
            admitted.append(seq)
            self.waiting.popleft()

        if admitted:
            self.running.extend(admitted)
            return admitted, True

        return list(self.running), False

    def ensure_capacity_for_decode_step(self) -> None:
        """Every running sequence is about to write one more token. Any whose
        current tail block is already full needs a fresh one. If none are
        free, preempt-by-recompute (docs/PLAN.md Phase 3 point 5.5): evict the
        NEWEST running sequence (least sunk cost), free its blocks, return it
        to the front of `waiting`, and retry.
        """
        for seq in list(self.running):
            if seq.status != SequenceStatus.RUNNING:
                continue  # already preempted earlier in this same pass
            while not self.block_manager.append_slot(seq):
                victim = self._newest_running_other_than(seq)
                if victim is None:
                    # `seq` itself is the only running sequence left and it
                    # still doesn't fit (a single sequence longer than the
                    # entire pool) -- nothing left to preempt in its favor.
                    self._preempt(seq)
                    break
                self._preempt(victim)

    def _newest_running_other_than(self, seq: SequenceState) -> SequenceState | None:
        for candidate in reversed(self.running):
            if candidate is not seq and candidate.status == SequenceStatus.RUNNING:
                return candidate
        return None

    def _preempt(self, seq: SequenceState) -> None:
        self.block_manager.free(seq)
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def finish(self, seq: SequenceState, reason: str) -> None:
        """Free blocks in the same step the sequence finishes (docs/PLAN.md
        Phase 3 pitfalls: freeing late is the #1 cause of phantom OOM)."""
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.block_manager.free(seq)
        if seq in self.running:
            self.running.remove(seq)

    def cancel_by_id(self, request_id: str) -> bool:
        """docs/PLAN.md Phase 7 acceptance: a client disconnecting mid-stream
        must free the sequence's KV blocks immediately, not leak them until
        some later timeout -- the #1 phantom-OOM cause (see finish()) applies
        just as much to a cancellation as to a normal completion. Returns
        True if a matching request was found (running or still waiting)."""
        for seq in self.running:
            if seq.request.request_id == request_id:
                self.finish(seq, "cancelled")
                return True
        for seq in list(self.waiting):
            if seq.request.request_id == request_id:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "cancelled"
                return True
        return False
