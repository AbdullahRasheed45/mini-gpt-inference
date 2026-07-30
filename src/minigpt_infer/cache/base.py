"""KVCacheBase: the interface every cache implementation (static, paged) shares.

model.py's attention layer only ever calls write() then read() -- it never
knows whether it's talking to a contiguous StaticKVCache (Phase 1-2) or a
block-table-indexed PagedKVCache (Phase 3+).
"""

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from minigpt_infer.batch import ForwardBatch


class KVCacheBase(Protocol):
    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
              batch: "ForwardBatch") -> None:
        """Write this layer's new k/v (B, n_head, T, head_dim) into the cache."""
        ...

    def read(self, layer_idx: int, batch: "ForwardBatch") -> tuple[torch.Tensor, torch.Tensor]:
        """Return this layer's full cached k/v up to the current length,
        shape (B, n_head, total_len, head_dim), including what write() just added."""
        ...
