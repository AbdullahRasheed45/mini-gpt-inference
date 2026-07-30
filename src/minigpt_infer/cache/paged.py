"""Paged KV cache (docs/PLAN.md §7 Phase 3 -- the vLLM core mechanism).

Two separate concerns, deliberately not merged into one class:

  BlockManager -- pure bookkeeping. Which block ids belong to which sequence.
    Never touches a tensor. This is what Phase 3's leak/fragmentation tests
    exercise directly, without needing a model or even a GPU.

  PagedKVCache -- the actual (num_blocks, block_size, n_head, head_dim)
    storage pool per layer, plus write() (scatter new k/v by slot_mapping) and
    read() (gather a dense (B, n_head, max_len, head_dim) tensor from
    block_tables -- docs/PLAN.md Phase 3 point 4(a), "gather + SDPA"). This is
    the KVCacheBase implementation model.py's attention layer actually calls;
    it has no idea which sequence owns which block, it just indexes.

The engine (engine/engine.py) is what ties them together: it asks
BlockManager for block ids, builds slot_mapping/block_tables tensors from
those ids, and passes them to PagedKVCache via ForwardBatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minigpt_infer.batch import ForwardBatch
    from minigpt_infer.engine.request import SequenceState


class BlockManager:
    """Free-list allocator over `num_blocks` block ids. Owns no tensors."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self.num_preemptions = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= len(self._free)

    def allocate(self, seq: SequenceState, num_tokens: int) -> None:
        """Grow `seq.block_table` to cover `num_tokens` total tokens. Only
        allocates the *additional* blocks needed beyond what it already has,
        so this is safe to call again later (e.g. after append_slot grew it)."""
        needed = self.blocks_needed(num_tokens) - len(seq.block_table)
        if needed <= 0:
            return
        assert needed <= len(self._free), (
            f"cannot allocate {needed} blocks for seq {seq.request.request_id}: "
            f"only {len(self._free)} free"
        )
        for _ in range(needed):
            seq.block_table.append(self._free.pop())

    def append_slot(self, seq: SequenceState) -> bool:
        """Ensure room exists for one more token (seq.num_computed_tokens + 1).
        Grows block_table by one block only when the current tail block is
        exactly full. Returns False (no mutation) if a new block is needed but
        none are free -- the caller (Scheduler) must preempt something first.
        """
        needed_blocks = self.blocks_needed(seq.num_computed_tokens + 1)
        if needed_blocks <= len(seq.block_table):
            return True
        if not self._free:
            return False
        seq.block_table.append(self._free.pop())
        return True

    def free(self, seq: SequenceState) -> None:
        self._free.extend(seq.block_table)
        seq.block_table = []

    def fragmentation_stats(self, running: list[SequenceState]) -> dict[str, int]:
        """Internal fragmentation: allocated-but-unused slots in each running
        sequence's tail block. There is no external fragmentation by
        construction (any free block can back any sequence -- unlike malloc,
        blocks are uniform size)."""
        allocated_slots = sum(len(s.block_table) * self.block_size for s in running)
        used_slots = sum(s.num_computed_tokens for s in running)
        return {
            "free_blocks": len(self._free),
            "allocated_blocks": self.num_blocks - len(self._free),
            "internal_fragmentation_slots": allocated_slots - used_slots,
        }


class PagedKVCache:
    """KVCacheBase implementation backed by a fixed-size block pool.

    write() scatters new tokens' k/v into the pool by absolute slot index
    (ForwardBatch.slot_mapping). read() gathers a dense, per-row-padded
    (B, n_head, max_len, head_dim) tensor from ForwardBatch.block_tables --
    positions beyond a row's real length are garbage (possibly another
    sequence's leftover data from a freed block) and MUST be masked by the
    caller via ForwardBatch.attn_mask; this class has no notion of per-row
    valid length, only the engine does.
    """

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (num_blocks, block_size, n_head, head_dim)
        self.k_pool = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v_pool = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, batch: ForwardBatch) -> None:
        B, H, T, D = k.shape
        assert batch.slot_mapping is not None, "PagedKVCache requires ForwardBatch.slot_mapping"
        assert batch.slot_mapping.numel() == B * T, (
            f"slot_mapping has {batch.slot_mapping.numel()} entries, expected B*T={B * T}"
        )
        # (B, H, T, D) -> (B*T, H, D), matching the pool's flattened (slot, H, D) layout.
        k_flat = k.transpose(1, 2).reshape(B * T, H, D)
        v_flat = v.transpose(1, 2).reshape(B * T, H, D)
        pool_k = self.k_pool[layer_idx].view(-1, H, D)
        pool_v = self.v_pool[layer_idx].view(-1, H, D)
        pool_k[batch.slot_mapping] = k_flat
        pool_v[batch.slot_mapping] = v_flat

    def read(self, layer_idx: int, batch: ForwardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch.block_tables is not None, "PagedKVCache requires ForwardBatch.block_tables"
        assert batch.seq_lens is not None, "PagedKVCache requires ForwardBatch.seq_lens"
        block_tables = batch.block_tables  # (B, max_blocks), -1 = unallocated
        B = block_tables.shape[0]
        max_len = int(batch.seq_lens.max().item())
        num_blocks_needed = self.blocks_needed(max_len)

        # -1 (unallocated) would index the pool's last block if used directly;
        # clamp to 0 -- always a real, allocated block for THIS row up to
        # num_blocks_needed, since every row has at least ceil(seq_lens[i]/bs)
        # blocks and num_blocks_needed only exceeds that for shorter rows,
        # whose extra gathered columns get masked out by seq_lens anyway.
        bt = block_tables[:, :num_blocks_needed].clamp(min=0)  # (B, num_blocks_needed)

        pool_k = self.k_pool[layer_idx]  # (num_blocks, block_size, H, D)
        pool_v = self.v_pool[layer_idx]
        H, D = pool_k.shape[2], pool_k.shape[3]

        gathered_k = pool_k[bt]  # (B, num_blocks_needed, block_size, H, D)
        gathered_v = pool_v[bt]
        gathered_k = gathered_k.reshape(B, num_blocks_needed * self.block_size, H, D)[:, :max_len]
        gathered_v = gathered_v.reshape(B, num_blocks_needed * self.block_size, H, D)[:, :max_len]

        k = gathered_k.permute(0, 2, 1, 3)  # (B, H, max_len, D)
        v = gathered_v.permute(0, 2, 1, 3)
        return k, v

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size
