"""LLMEngine (docs/PLAN.md §7 Phase 3): ties Scheduler + PagedKVCache + the
model together into one synchronous step() call.

Deliberate scope decisions (documented, not accidental gaps):
  - Prefill runs ONE sequence at a time (a single-row forward per newly
    admitted request), even when several are admitted in the same step.
    Batched/packed ragged prefill into a shared paged cache is real
    complexity (either padding into over-allocated blocks, defeating paged
    memory efficiency, or a varlen/packed attention path) that isn't needed
    to demonstrate the phase's actual point: continuous batching across
    heterogeneous-length RUNNING sequences during decode, which IS fully
    batched below. Chunked prefill is an explicit non-goal (docs/PLAN.md §12).
  - `SamplingParams.n` (multiple completions per request) and `.seed`
    (per-request reproducible sampling) are not implemented: batched
    multinomial sampling shares one RNG stream across the whole decode batch,
    so per-row-seeded sampling would need a Python-level loop, defeating the
    vectorization sampling.py exists for. `n=1` is asserted in add_request().
  - Stop-string detection checks the full accumulated decoded text every step
    (not per-token, so a stop string split across token boundaries is still
    caught -- docs/PLAN.md Phase 3 pitfalls). It truncates the delta in the
    step where the stop is found. It does NOT retroactively un-emit text from
    an *earlier* step if the stop string's opening characters were already
    flushed before its closing characters arrived -- a real streaming server
    needs a lookback buffer to do that properly; that's Phase 7's job, not
    this synchronous engine's.
"""

from __future__ import annotations

import torch

from minigpt_infer.batch import ForwardBatch, build_paged_decode_mask
from minigpt_infer.cache.paged import BlockManager, PagedKVCache
from minigpt_infer.config import EngineConfig
from minigpt_infer.engine.request import Request, RequestOutput, SequenceState
from minigpt_infer.engine.scheduler import Scheduler
from minigpt_infer.model import GPT
from minigpt_infer.sampling import sample_batch
from minigpt_infer.tokenizer import IncrementalDetokenizer


class LLMEngine:
    def __init__(
        self,
        model: GPT,
        engine_cfg: EngineConfig,
        vocab_mask: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.engine_cfg = engine_cfg
        param = next(model.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.vocab_mask = vocab_mask

        self.cache = PagedKVCache(
            n_layer=model.cfg.n_layer,
            n_head=model.cfg.n_head,
            head_dim=model.head_dim,
            block_size=engine_cfg.block_size,
            num_blocks=engine_cfg.num_blocks,
            device=self.device,
            dtype=self.dtype,
        )
        self.block_manager = BlockManager(engine_cfg.num_blocks, engine_cfg.block_size)
        self.scheduler = Scheduler(self.block_manager, engine_cfg.max_batch_size)
        self.max_blocks_per_seq = self.block_manager.blocks_needed(model.cfg.block_size)
        self._detokenizers: dict[str, IncrementalDetokenizer] = {}

    def add_request(self, request: Request) -> None:
        assert request.sampling_params.n == 1, (
            "n>1 completions/request not supported (see module docstring)"
        )
        prompt_len = len(request.prompt_token_ids)
        assert prompt_len <= self.model.cfg.block_size, (
            f"prompt has {prompt_len} tokens > block_size={self.model.cfg.block_size}"
        )
        self._detokenizers[request.request_id] = IncrementalDetokenizer()
        self.scheduler.add_request(SequenceState(request=request))

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[RequestOutput]:
        batch_seqs, is_prefill = self.scheduler.schedule()
        if not batch_seqs:
            return []
        if is_prefill:
            return [self._prefill_one(seq) for seq in batch_seqs]
        self.scheduler.ensure_capacity_for_decode_step()
        # Re-read self.scheduler.running (not batch_seqs): preemption above
        # may have shrunk it. Pass a COPY, not the live list: _sample_and_advance
        # below iterates this batch and calls scheduler.finish() for anything
        # that just finished, which mutates self.scheduler.running in place
        # (list.remove()) -- iterating that same list object while it's being
        # mutated mid-loop silently skips whichever sequence comes right after
        # the one that just finished (classic "remove while iterating" bug,
        # caught by tests/test_paged.py's multi-request exact-match test).
        if not self.scheduler.running:
            return []
        return self._run_decode(list(self.scheduler.running))

    def _pad_block_table(self, block_table: list[int]) -> list[int]:
        return block_table + [-1] * (self.max_blocks_per_seq - len(block_table))

    def _slot_for(self, block_table: list[int], token_index: int) -> int:
        bs = self.engine_cfg.block_size
        return block_table[token_index // bs] * bs + (token_index % bs)

    def _sample_and_advance(
        self, seqs: list[SequenceState], logits: torch.Tensor
    ) -> list[RequestOutput]:
        """Shared tail of prefill/decode: sample, update state, detokenize,
        check finish, free blocks for anything that just finished.

        Does NOT touch num_computed_tokens -- callers already wrote a
        specific number of tokens into the cache (T for prefill, 1 for
        decode) before calling this, and must advance the counter by exactly
        that amount themselves. The token sampled *here* is next step's
        input, not yet cached, so this function must never bump the counter
        on its account (a real bug caught by tests/test_paged.py's exact
        paged-vs-static comparison: bumping it here on top of prefill's
        T-token jump silently wrote/read one position ahead of ground truth).
        """
        temps = torch.tensor(
            [s.request.sampling_params.temperature for s in seqs], device=self.device,
        )
        top_ks = [s.request.sampling_params.top_k for s in seqs]
        top_ps = [s.request.sampling_params.top_p for s in seqs]
        top_k = None if all(k is None for k in top_ks) else torch.tensor(
            [k if k is not None else logits.shape[-1] for k in top_ks], device=self.device
        )
        top_p = None if all(p is None for p in top_ps) else torch.tensor(
            [p if p is not None else 1.0 for p in top_ps], device=self.device
        )

        masked_logits = logits
        if self.vocab_mask is not None:
            masked_logits = logits.masked_fill(self.vocab_mask, float("-inf"))
        next_ids = sample_batch(
            masked_logits, temperature=temps, top_k=top_k, top_p=top_p,
        )  # (B, 1)

        outputs = []
        for i, seq in enumerate(seqs):
            token_id = int(next_ids[i, 0].item())
            seq.output_token_ids.append(token_id)

            delta = self._detokenizers[seq.request.request_id].add_token(token_id)
            seq.detokenized_text += delta

            finish_reason = self._check_finish(seq, token_id)
            if finish_reason is not None:
                tail = self._detokenizers[seq.request.request_id].finalize()
                seq.detokenized_text += tail
                delta += tail
                delta, stop_str = self._truncate_at_stop(seq, delta)
                self.scheduler.finish(seq, finish_reason if stop_str is None else "stop")

            outputs.append(RequestOutput(
                request_id=seq.request.request_id,
                new_token_ids=[token_id],
                text_delta=delta,
                finished=seq.is_finished(),
                finish_reason=seq.finish_reason,
            ))
        return outputs

    def _check_finish(self, seq: SequenceState, token_id: int) -> str | None:
        if seq.request.eot_token_id is not None and token_id == seq.request.eot_token_id:
            return "eot"
        if len(seq.output_token_ids) >= seq.request.sampling_params.max_tokens:
            return "length"
        for stop in seq.request.sampling_params.stop:
            if stop in seq.detokenized_text:
                return "stop"
        return None

    def _truncate_at_stop(self, seq: SequenceState, delta: str) -> tuple[str, str | None]:
        """If a stop string appears in the full accumulated text, trim this
        step's delta so the stop string itself isn't included -- best effort,
        see module docstring for the streaming caveat."""
        for stop in seq.request.sampling_params.stop:
            idx = seq.detokenized_text.find(stop)
            if idx == -1:
                continue
            # How much of `delta` falls at or after the stop string's start.
            emitted_before_delta = len(seq.detokenized_text) - len(delta)
            cut = max(0, idx - emitted_before_delta)
            return delta[:cut], stop
        return delta, None

    def _prefill_one(self, seq: SequenceState) -> RequestOutput:
        # prompt + anything already generated -- NOT just the original prompt.
        # For a fresh request output_token_ids is empty, so this is the
        # prompt alone. For a request being RE-admitted after preemption
        # (docs/PLAN.md Phase 3 point 5.5, "preempt by recompute"),
        # output_token_ids already holds tokens this sequence committed to
        # before eviction; recomputation must rebuild the cache over all of
        # them too, or the resumed sequence silently forgets its own
        # progress and Scheduler.schedule()'s block-sizing (which already
        # correctly uses seq.num_tokens = prompt_len + len(output_token_ids))
        # would no longer match what's actually written here.
        full_context = seq.request.prompt_token_ids + seq.output_token_ids
        T = len(full_context)
        input_ids = torch.tensor([full_context], device=self.device, dtype=torch.long)
        position_ids = torch.arange(T, device=self.device, dtype=torch.long).unsqueeze(0)
        slot_mapping = torch.tensor(
            [self._slot_for(seq.block_table, i) for i in range(T)],
            device=self.device, dtype=torch.long,
        )
        block_tables = torch.tensor(
            [self._pad_block_table(seq.block_table)], device=self.device, dtype=torch.long,
        )
        seq_lens = torch.tensor([T], device=self.device, dtype=torch.long)

        batch = ForwardBatch(
            input_ids=input_ids, position_ids=position_ids, is_prefill=True,
            cache=self.cache, block_tables=block_tables,
            slot_mapping=slot_mapping, seq_lens=seq_lens,
        )
        # q_len == kv_len == T here (fresh sequence, empty cache) -> plain
        # causal attention, no explicit mask needed (model.py's is_causal
        # branch). is_prefill=True is exactly this invariant.
        logits = self.model(batch)  # (1, vocab)
        assert seq.num_computed_tokens == 0
        seq.num_computed_tokens = T  # invariant: next_position == num_computed_tokens
        return self._sample_and_advance([seq], logits)[0]

    def _run_decode(self, seqs: list[SequenceState]) -> list[RequestOutput]:
        B = len(seqs)
        input_ids = torch.tensor(
            [[s.last_token_id] for s in seqs], device=self.device, dtype=torch.long,
        )
        position_ids = torch.tensor(
            [[s.num_computed_tokens] for s in seqs], device=self.device, dtype=torch.long,
        )
        for s, pos_row in zip(seqs, position_ids.tolist(), strict=True):
            assert pos_row[0] == s.num_computed_tokens, (
                "next_position == num_computed_tokens invariant broken"
            )

        seq_lens_after = torch.tensor(
            [s.num_computed_tokens + 1 for s in seqs], device=self.device, dtype=torch.long,
        )
        max_len = int(seq_lens_after.max().item())
        block_tables = torch.tensor(
            [self._pad_block_table(s.block_table) for s in seqs],
            device=self.device, dtype=torch.long,
        )
        slot_mapping = torch.tensor(
            [self._slot_for(s.block_table, s.num_computed_tokens) for s in seqs],
            device=self.device, dtype=torch.long,
        )
        attn_mask = build_paged_decode_mask(seq_lens_after, max_len, dtype=self.dtype)

        batch = ForwardBatch(
            input_ids=input_ids, position_ids=position_ids, is_prefill=False,
            cache=self.cache, attn_mask=attn_mask, block_tables=block_tables,
            slot_mapping=slot_mapping, seq_lens=seq_lens_after,
        )
        logits = self.model(batch)  # (B, vocab)
        assert logits.shape[0] == B
        for s in seqs:
            s.num_computed_tokens += 1  # this step wrote exactly 1 token per row
        return self._sample_and_advance(seqs, logits)
