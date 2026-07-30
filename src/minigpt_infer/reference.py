"""Correctness oracle. NEVER OPTIMIZE THIS FILE.

A frozen, self-contained copy of Project A's original (pre-caching) GPT
forward/generate: full re-forward of the whole sequence on every step, no KV
cache, `idx[:, -block_size:]` cropping. Every optimization introduced in
model.py -- KV cache, batching, paging, quantization, speculative decoding --
must be validated against this file's output. If model.py and this file ever
disagree on greedy token ids for the same weights and prompt, model.py has a
bug, full stop.

Deliberately duplicates ReferenceGPT's module structure (not just imports
GPT from model.py) so that this file keeps working unchanged no matter how
model.py's forward signature evolves through the phases. The submodule names
(tok_emb, pos_emb, blocks, ln_f, lm_head, ...) are identical to model.GPT, so
the same state_dict loads into either -- that's what makes an apples-to-apples
comparison possible.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt_infer.config import GPTConfig


class _ReferenceCausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class _ReferenceMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class _ReferenceBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = _ReferenceCausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = _ReferenceMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ReferenceGPT(nn.Module):
    """Naive, unoptimized GPT. Full re-forward every decode step. Never touch."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(_ReferenceBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Full forward, whole sequence. Returns logits at every position."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        vocab_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Byte-for-byte the original Project A sampling loop.

        vocab_mask: optional (vocab,) bool tensor, True where a token id is
        never valid output (e.g. the 47 padding rows above GPT-2's real 50257
        vocab -- see tokenizer.py). Project A never needed this since it never
        sampled at inference time in production; the inference engine does, so
        the oracle must support it too, or a "cached == reference" comparison
        would only be valid until the naive path got unlucky and sampled a
        padding id.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if vocab_mask is not None:
                logits = logits.masked_fill(vocab_mask, float("-inf"))
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    @torch.no_grad()
    def greedy_generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        vocab_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic argmax decoding -- what exact-match tests compare against.

        Sampling-based generate() can't be compared token-for-token across two
        implementations even with the same seed, because torch.multinomial's
        RNG consumption depends on control flow that can differ subtly between
        naive and cached paths (e.g. differing tensor shapes touch the RNG
        stream differently on some backends). Greedy decoding has no such
        ambiguity: same logits in exact fp32 arithmetic -> same argmax, always.
        This is what Phase 1's "exact greedy match" tests actually use.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            if vocab_mask is not None:
                logits = logits.masked_fill(vocab_mask, float("-inf"))
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
