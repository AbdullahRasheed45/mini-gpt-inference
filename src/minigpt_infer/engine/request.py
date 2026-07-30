"""Request/sequence state (docs/PLAN.md §7 Phase 3).

A `Request` is the user-facing, immutable ask. A `SequenceState` is the
engine's mutable bookkeeping for running it: how many tokens are actually in
the KV cache right now, which blocks back it, what's been generated so far.

Kept deliberately free of any tensor/torch import -- this module is pure
Python state, testable (and tested, tests/test_scheduler.py) without a model
or even torch being loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from minigpt_infer.config import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    eot_token_id: int | None = None


@dataclass
class SequenceState:
    request: Request
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    # Tokens already written into the KV cache -- the single invariant that
    # matters most in this phase (docs/PLAN.md Phase 3 pitfalls):
    # next_position == num_computed_tokens, always. If this and the token the
    # engine is about to feed the model ever disagree, generation is silently
    # wrong (duplicate or skipped position).
    num_computed_tokens: int = 0
    finish_reason: str | None = None
    detokenized_text: str = ""

    @property
    def prompt_len(self) -> int:
        return len(self.request.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total logical sequence length: prompt + generated so far."""
        return self.prompt_len + len(self.output_token_ids)

    @property
    def last_token_id(self) -> int:
        """The token to feed the model on the next decode step."""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.request.prompt_token_ids[-1]

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    text_delta: str
    finished: bool
    finish_reason: str | None = None
