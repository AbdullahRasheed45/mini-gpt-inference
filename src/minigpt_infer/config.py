"""Config dataclasses: model architecture, engine tuning, and sampling params.

GPTConfig is copied verbatim from Project A (mini-gpt-ddp/model.py) -- the
checkpoint's state_dict shapes depend on it exactly, so it must not drift.
"""

from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    vocab_size: int = 50304          # GPT-2 vocab (50257) padded to /64 for tensor-core efficiency
    block_size: int = 512            # context length; TinyStories rarely needs more
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0             # keep 0 for pretraining; data is not repeated much
    bias: bool = False               # cleaner and slightly faster without biases


@dataclass
class EngineConfig:
    """Tuning knobs for the serving engine (paged cache, scheduler, batching)."""

    block_size: int = 16             # tokens per KV cache block (Phase 3)
    num_blocks: int = 2048           # fixed pool size; deterministic benchmarks
                                      # need a fixed budget, not "all free memory"
    max_batch_size: int = 256        # max concurrent running sequences
    max_num_seqs_per_step: int = 256


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0         # 0.0 => greedy
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    n: int = 1
