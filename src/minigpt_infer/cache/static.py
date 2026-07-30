"""StaticKVCache: contiguous, pre-allocated per-sequence KV cache.

One (B, n_head, max_seq_len, head_dim) tensor per layer, per k/v. Used for
Phase 1 (single/uniform-length decode) and Phase 2 (padded batching). Replaced
by cache/paged.py's block-table cache in Phase 3, which implements the same
KVCacheBase protocol.

Phase 1/2 constraint (relaxed by the paged cache, not by this class): every
row in the batch is assumed to have the SAME current length at the start of
each forward() call. This holds for synchronous batched generation (every
request in the batch advances one token per step together) but not for
continuous batching, where requests join and leave mid-stream -- that's
exactly why Phase 3 needs a different cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minigpt_infer.batch import ForwardBatch


class StaticKVCache:
    def __init__(
        self,
        n_layer: int,
        n_head: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (max_batch_size, n_head, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, batch: ForwardBatch) -> None:
        B, _H, T, _D = k.shape
        starts = self.seq_lens[:B]
        # All rows must share the same current length -- Phase 2's padding
        # masks handle *content* differences between rows, but the cache
        # write offset itself is still synchronous across the batch here.
        # A silent violation would write to the wrong offset with no crash,
        # so this is asserted rather than assumed.
        assert torch.equal(starts, starts[0].expand_as(starts)), (
            "StaticKVCache requires every row in the batch to share the same "
            f"current length before this write; got {starts.tolist()}"
        )
        start = int(starts[0].item())
        assert start + T <= self.max_seq_len, (
            f"sequence length {start + T} exceeds max_seq_len={self.max_seq_len}"
        )
        self.k[layer_idx][:B, :, start:start + T, :] = k
        self.v[layer_idx][:B, :, start:start + T, :] = v

    def read(self, layer_idx: int, batch: ForwardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = batch.input_ids.shape
        start = int(self.seq_lens[0].item())
        end = start + T
        return self.k[layer_idx][:B, :, :end, :], self.v[layer_idx][:B, :, :end, :]

    def advance(self, num_new_tokens: int, batch_size: int) -> None:
        """Call once per forward() (not once per layer) after every layer has
        written -- all layers write the same `T` new tokens at the same
        offset, so the length only advances once per step, not once per
        layer."""
        self.seq_lens[:batch_size] += num_new_tokens

    def reset(self) -> None:
        self.seq_lens.zero_()
