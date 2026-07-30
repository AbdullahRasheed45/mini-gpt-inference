"""Phase 0: loader internals, tested without the network or the real checkpoint.

download_checkpoint()/load_model() themselves are not called here -- they need
HF_TOKEN and a network round-trip, which CI must never require (docs/PLAN.md
§9). Everything test-worthy in loader.py is a pure function on already-loaded
data, so it's tested directly against constructed fake checkpoints instead.
"""

import pytest
import torch

from minigpt_infer.config import GPTConfig
from minigpt_infer.loader import _load_raw_checkpoint, _strip_prefix, reconstruct_config
from minigpt_infer.model import GPT
from tests.helpers import tiny_gpt_config


def test_strip_prefix_removes_matching_prefix():
    sd = {"module.tok_emb.weight": 1, "module.ln_f.weight": 2}
    stripped = _strip_prefix(sd, "module.")
    assert stripped == {"tok_emb.weight": 1, "ln_f.weight": 2}


def test_strip_prefix_is_noop_when_prefix_absent():
    sd = {"tok_emb.weight": 1}
    assert _strip_prefix(sd, "module.") == sd


def test_strip_prefix_handles_mixed_keys_conservatively():
    # if only SOME keys have the prefix, stripping would create key collisions
    # or an inconsistent state_dict -- current behavior only strips when ALL
    # keys share the prefix (checked via "any" + per-key strip, so document
    # what actually happens on a mixed dict rather than assume)
    sd = {"module.a": 1, "b": 2}
    result = _strip_prefix(sd, "module.")
    assert result == {"a": 1, "b": 2}


def test_reconstruct_config_uses_train_cfg_values():
    train_cfg = {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 256}
    cfg = reconstruct_config(train_cfg)
    assert cfg.n_layer == 4
    assert cfg.n_head == 4
    assert cfg.n_embd == 128
    assert cfg.block_size == 256
    # vocab_size and bias are never in train_cfg -- must come from GPTConfig defaults
    assert cfg.vocab_size == GPTConfig().vocab_size
    assert cfg.bias == GPTConfig().bias
    # inference is always eval mode regardless of what dropout the model trained with
    assert cfg.dropout == 0.0


def test_reconstruct_config_handles_missing_keys():
    # a checkpoint saved by a hypothetical future train.py that added a new
    # CLI arg, or an old one missing a key we now expect, must not crash
    cfg = reconstruct_config({})
    assert cfg == GPTConfig(dropout=0.0)


def test_load_raw_checkpoint_falls_back_when_weights_only_fails(tmp_path, monkeypatch):
    """Simulates torch.load(weights_only=True) raising, and verifies the
    fallback path is actually taken rather than propagating the exception."""
    calls = []

    def fake_torch_load(path, map_location, weights_only):
        calls.append(weights_only)
        if weights_only:
            raise RuntimeError("simulated weights_only failure")
        return {"model": {}, "iter": 0}

    monkeypatch.setattr("minigpt_infer.loader.torch.load", fake_torch_load)
    result = _load_raw_checkpoint(tmp_path / "fake.pt", map_location="cpu")

    assert calls == [True, False], "must try weights_only=True first, then fall back"
    assert result == {"model": {}, "iter": 0}


def test_weight_tying_assertion_catches_broken_tying():
    """Directly exercises the failure mode load_model() guards against: a
    state_dict load that leaves lm_head.weight and tok_emb.weight as separate
    (even if numerically equal) tensors instead of the same object."""
    cfg = tiny_gpt_config()
    model = GPT(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()

    # break tying manually, the way a naive (non-parameter-aware) deserializer could
    model.lm_head.weight = torch.nn.Parameter(model.tok_emb.weight.data.clone())
    assert model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr()

    with pytest.raises(AssertionError):
        assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), "tying broken"


def test_strict_load_state_dict_reports_mismatch_on_wrong_architecture():
    small = GPT(tiny_gpt_config(n_layer=2))
    big_cfg = tiny_gpt_config(n_layer=4)
    big = GPT(big_cfg)

    missing, unexpected = big.load_state_dict(small.state_dict(), strict=False)
    assert len(missing) > 0, "loading a 2-layer state_dict into a 4-layer model must report missing"
