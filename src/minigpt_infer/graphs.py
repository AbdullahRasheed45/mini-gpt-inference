"""CUDA graph capture/replay for the decode step (docs/PLAN.md §7 Phase 4).

Demonstrates and measures the overhead-reduction technique in isolation:
captures ONE decode step at a fixed KV length, then replays it many times
with fresh input data copied into static buffers between replays -- exactly
the workload docs/PLAN.md's own benchmark framing describes ("a CUDA graph...
measures the replay() call"), not a full varying-length generation loop.

Why this doesn't reuse StaticKVCache: its write()/read() call `.item()` to
turn a length tensor into a Python int for slicing -- a host-side
synchronization that CUDA graph capture cannot safely record (a graph is a
fixed sequence of GPU kernels; a capturing stream is not allowed to
synchronize). `_FixedPositionCache` below sidesteps this entirely: the
position being decoded from is baked in as a plain Python int at
construction time (valid here -- this benchmark's whole point is many
replays of the *same* shape/position, not an advancing generation loop), so
neither write() nor read() ever reads a tensor's value back to the host.

Integrating graphs into the full paged, continuous-batching engine (variable
per-request length, one graph per shape bucket, re-capture on bucket
change, index_copy_-style graph-safe cache updates) is real additional
engineering that vLLM/TensorRT-LLM do -- out of scope here; this phase's job
is to demonstrate and measure the technique, not productionize it (see
docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from minigpt_infer.batch import ForwardBatch

if TYPE_CHECKING:
    from minigpt_infer.model import GPT


def graphs_supported() -> bool:
    return torch.cuda.is_available()


class _FixedPositionCache:
    """KVCacheBase, graph-capture-safe: every write/read targets the same
    baked-in Python-int position `kv_length` -- no `.item()`, no host sync."""

    def __init__(
        self, n_layer: int, n_head: int, head_dim: int, batch_size: int,
        kv_length: int, device: str, dtype: torch.dtype,
    ) -> None:
        self.kv_length = kv_length
        shape = (batch_size, n_head, kv_length + 1, head_dim)
        self.k = [torch.randn(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.randn(shape, device=device, dtype=dtype) for _ in range(n_layer)]

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, batch: ForwardBatch) -> None:
        end = self.kv_length + 1
        self.k[layer_idx][:, :, self.kv_length:end, :] = k
        self.v[layer_idx][:, :, self.kv_length:end, :] = v

    def read(self, layer_idx: int, batch: ForwardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.kv_length + 1
        return self.k[layer_idx][:, :, :end, :], self.v[layer_idx][:, :, :end, :]


class CUDAGraphRunner:
    """One graph per (batch_size, kv_length) shape bucket. Capture once,
    replay many times with fresh token ids copied into a static buffer."""

    def __init__(self, model: GPT, batch_size: int, kv_length: int, device: str = "cuda") -> None:
        assert graphs_supported(), "CUDA graphs require a CUDA device"
        self.model = model
        self.batch_size = batch_size
        self.kv_length = kv_length
        self.device = device
        dtype = next(model.parameters()).dtype

        self.cache = _FixedPositionCache(
            model.cfg.n_layer, model.cfg.n_head, model.head_dim,
            batch_size, kv_length, device, dtype,
        )
        self.static_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.static_position_ids = torch.full(
            (batch_size, 1), kv_length, dtype=torch.long, device=device,
        )
        self.static_output: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def _forward_once(self) -> torch.Tensor:
        batch = ForwardBatch(
            input_ids=self.static_input_ids, position_ids=self.static_position_ids,
            is_prefill=False, cache=self.cache,
        )
        return self.model(batch)

    def capture(self, warmup_iters: int = 5) -> None:
        # Warm up on a side stream first: cuBLAS/cuDNN lazily allocate
        # workspaces on first use, and capturing that allocation poisons the
        # graph (docs/PLAN.md Phase 4 pitfalls).
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(warmup_iters):
                self._forward_once()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self._forward_once()

    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.graph is not None, "call capture() first"
        self.static_input_ids.copy_(input_ids)
        self.graph.replay()
        return self.static_output
