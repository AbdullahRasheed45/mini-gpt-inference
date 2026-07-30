"""GPT model, ported from Project A (mini-gpt-ddp/model.py).

Phase 0 committed this as a verbatim port. This (Phase 1) is the cache-aware
rewrite: forward() now takes a ForwardBatch instead of a raw idx tensor, and
supports prefill (T>1, into an empty cache) and decode (T=1, attending over
previously-cached KV) as two distinct attention paths. See docs/PLAN.md §5 and
§7 (Phase 1) for the exact interface this was designed against.

The single most important pitfall this file's attention forward encodes
(docs/PLAN.md, Phase 1 pitfalls): SDPA's `is_causal=True` is only correct when
q_len == kv_len (a fresh prefill into an empty cache). PyTorch aligns the
causal mask to the *top-left* when q_len != kv_len, so a naive
`is_causal=True` decode step (q_len=1, kv_len=L) masks out every cached key
except position 0 -- silently, with no error, no crash, no NaN. It just
produces attention over one token instead of all of them. That specific bug
does not exist in this file; see the `is_causal = (kv_len == q_len)` line
below and its accompanying assert.

Design notes inherited from Project A (T4-specific):
- Attention uses torch.nn.functional.scaled_dot_product_attention (SDPA).
  FlashAttention-2 requires Ampere+; T4 is Turing. SDPA picks the best
  available kernel (memory-efficient attention works on T4) automatically.
- No bf16 anywhere: T4 supports fp16 but not bf16.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt_infer.batch import ForwardBatch
from minigpt_infer.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # single fused projection for q, k, v
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, layer_idx: int, batch: ForwardBatch) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if batch.cache is not None:
            batch.cache.write(layer_idx, k, v, batch)
            k, v = batch.cache.read(layer_idx, batch)
            # k, v: (B, n_head, kv_len, head_dim) where kv_len = tokens cached
            # so far INCLUDING the ones just written above.

        dropout_p = self.dropout if self.training else 0.0

        if batch.attn_mask is not None:
            # Phase 2+: an explicit combined causal+padding mask was built by
            # the caller. is_causal and attn_mask are mutually exclusive in
            # SDPA -- pass only the mask.
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=batch.attn_mask, dropout_p=dropout_p, is_causal=False,
            )
        else:
            kv_len, q_len = k.shape[2], q.shape[2]
            is_causal = kv_len == q_len
            if not is_causal:
                # kv_len > q_len means this is a decode step against a
                # non-empty cache. Every cached position (0..kv_len-1) is
                # causally at-or-before the new query token by construction
                # (nothing has been written for a position that doesn't yet
                # exist), so the query may attend to all of them -- no mask
                # needed. What it must NOT do is pass is_causal=True here;
                # see the module docstring.
                assert q_len == 1, (
                    f"non-causal attention without an explicit mask requires q_len==1 "
                    f"(a single decode step); got q_len={q_len}, kv_len={kv_len}. "
                    "Chunked prefill into a non-empty cache needs an explicit mask "
                    "and is out of scope (docs/PLAN.md §12)."
                )
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, layer_idx: int, batch: ForwardBatch) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), layer_idx, batch)   # pre-norm residual, GPT-2 style
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.n_embd // cfg.n_head
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: embedding and unembedding share parameters
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper, sec 2.3)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, batch: ForwardBatch) -> torch.Tensor:
        """Returns logits at the LAST position only, shape (B, vocab) --
        whether this call is a prefill or a decode step. An inference engine
        never needs logits at any position except the one it's about to
        sample from next; Project A's training-time full-sequence logits (for
        computing loss against targets) has no equivalent here, since this
        project never trains anything.
        """
        # Skipped while a CUDA graph is capturing: evaluating this assert's
        # tensor comparison forces a device-to-host sync (`.item()`-like,
        # via Tensor.__bool__), which CUDA graph capture forbids outright
        # ("operation not permitted when stream is capturing" -- confirmed
        # on real hardware, not a hypothetical). Graph capture in graphs.py
        # always runs warmup iterations in plain eager mode first, which
        # already exercises this exact check on the exact same static
        # position_ids buffer, so skipping it only during the capturing
        # window itself does not skip the check -- it just skips re-checking
        # something already verified moments earlier on the same tensor.
        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            assert batch.position_ids.max() < self.cfg.block_size, (
                f"position {batch.position_ids.max().item()} >= block_size="
                f"{self.cfg.block_size} (the learned position embedding table "
                f"only has block_size rows; there is nothing to index past it)"
            )
        x = self.drop(self.tok_emb(batch.input_ids) + self.pos_emb(batch.position_ids))
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, layer_idx, batch)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1, :])
