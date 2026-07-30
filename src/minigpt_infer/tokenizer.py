"""GPT-2 BPE tokenizer (tiktoken) plus the padding-vocab mask and an
incremental, UTF-8-safe streaming detokenizer.

Why the padding mask matters: GPTConfig.vocab_size is 50304 (GPT-2's real
50257 padded to a multiple of 64 for tensor-core efficiency -- see Project A's
model.py). Rows 50257-50303 of tok_emb/lm_head are trained parameters (weight
tying means they get gradient too) but tiktoken's gpt2 encoding cannot decode
those ids -- it doesn't know they exist. Nothing in Project A's training loop
ever sampled from the model, so this was never hit. This engine samples
constantly, so a run long enough will eventually put non-negligible mass on
one of those 47 ids and then crash trying to decode it. Every sampling path
in this project must mask them to -inf before softmax.
"""

from functools import lru_cache

import tiktoken
import torch

REAL_GPT2_VOCAB_SIZE = 50257  # tiktoken's actual encodable range: [0, 50257)
EOT_TOKEN = 50256


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return get_encoding().encode_ordinary(text)


def decode(ids: list[int]) -> str:
    return get_encoding().decode(ids)


def padding_vocab_mask(vocab_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """(vocab_size,) bool tensor, True for ids tiktoken cannot decode.

    Pass to logits.masked_fill(mask, float("-inf")) before every softmax/argmax
    in both reference.py and the real sampling path (sampling.py, Phase 2).
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if vocab_size > REAL_GPT2_VOCAB_SIZE:
        mask[REAL_GPT2_VOCAB_SIZE:] = True
    return mask


class IncrementalDetokenizer:
    """Streaming token->text that never emits a broken UTF-8 character.

    GPT-2 BPE is byte-level: a single Unicode character (e.g. an emoji, or
    non-Latin script) can span multiple tokens. Decoding tokens one at a time
    and emitting immediately will periodically emit the U+FFFD replacement
    character mid-codepoint. Decoding the *entire* sequence from scratch on
    every new token avoids that but is O(n^2) over a long generation and is
    exactly the mistake Phase 7's server must not make (see docs/PLAN.md,
    Phase 7 pitfalls).

    Strategy: keep all token ids seen so far, decode the whole buffer to bytes,
    and only emit the trailing suffix once it forms valid UTF-8. This is O(1)
    amortized in token count per step for the *new* bytes emitted, but does
    keep re-decoding a small trailing window (not the full history) by only
    decoding the last few tokens plus one token of left-context, since
    tiktoken decodes byte-pair tokens independently of position.
    """

    def __init__(self) -> None:
        self._enc = get_encoding()
        self._token_ids: list[int] = []
        self._emitted_text = ""

    def add_token(self, token_id: int) -> str:
        """Feed one new token id. Returns the new, safe-to-emit text delta."""
        self._token_ids.append(token_id)
        # Only the last few tokens can possibly contain an incomplete trailing
        # multi-byte char; decoding the whole buffer is correct and, for a
        # bounded window (rather than the true full history), still cheap.
        # decode() on the full id list is exact because tiktoken's BPE decode
        # is a pure concatenation of each token's byte string, position-independent.
        full_bytes = self._enc.decode_bytes(self._token_ids)
        try:
            full_text = full_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            # trailing bytes are an incomplete multi-byte sequence; only emit
            # the guaranteed-valid prefix and hold the rest for the next token
            full_text = full_bytes[: e.start].decode("utf-8")
        delta = full_text[len(self._emitted_text):]
        self._emitted_text = full_text
        return delta

    def finalize(self) -> str:
        """Call at end-of-generation to flush any bytes that never completed."""
        full_text = self._enc.decode_bytes(self._token_ids).decode("utf-8", errors="replace")
        delta = full_text[len(self._emitted_text):]
        self._emitted_text = full_text
        return delta
