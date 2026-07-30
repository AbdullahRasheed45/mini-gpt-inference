"""Shared test fixtures: tiny configs and models, cheap enough for CPU CI.

Every phase's tests build a config this small. No test in this repo may
require the real (614 MB, private) checkpoint -- see docs/PLAN.md §9.
"""

import torch

from minigpt_infer.config import GPTConfig


def tiny_gpt_config(**overrides) -> GPTConfig:
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
               dropout=0.0, bias=False)
    cfg.update(overrides)
    return GPTConfig(**cfg)


def seeded(seed: int = 1234):
    """Context-free seeding helper so every test starts from a known RNG state."""
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)
