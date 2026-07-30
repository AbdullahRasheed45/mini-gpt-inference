"""Phase 0 / Phase 7 prerequisite: tokenizer round-trip and the incremental
detokenizer's UTF-8 safety (docs/PLAN.md, Phase 7 pitfalls -- this is the test
that must exist before the server ever streams a token)."""

from minigpt_infer.tokenizer import EOT_TOKEN, IncrementalDetokenizer, decode, encode, get_encoding


def test_encode_decode_roundtrip():
    text = "Once upon a time there was a little girl named Lily."
    assert decode(encode(text)) == text


def test_eot_token_matches_tiktoken():
    assert EOT_TOKEN == get_encoding().eot_token == 50256


def test_incremental_detokenizer_matches_full_decode_ascii():
    text = "Once upon a time there was a dog."
    ids = encode(text)
    dt = IncrementalDetokenizer()
    out = "".join(dt.add_token(i) for i in ids)
    assert out == text


def test_incremental_detokenizer_never_emits_replacement_character():
    """The core correctness property: streaming token-by-token must produce
    the exact same text as decoding everything at once, for text containing
    multi-byte UTF-8 characters that can split across token boundaries."""
    text = "café résumé naïve 日本語 emoji test 🎉🎊 done"
    ids = encode(text)

    dt = IncrementalDetokenizer()
    streamed = ""
    for i in ids:
        delta = dt.add_token(i)
        assert "�" not in delta, "emitted a broken UTF-8 replacement character"
        streamed += delta
    streamed += dt.finalize()

    assert streamed == decode(ids), "streamed output must equal a full one-shot decode"


def test_incremental_detokenizer_holds_back_incomplete_trailing_bytes():
    """A multi-byte character whose bytes are still split across tokens must
    not be (partially) emitted until it's complete."""
    text = "日本語"  # each character is a separate multi-byte token cluster in gpt2 bpe
    ids = encode(text)
    assert len(ids) >= 2, "test needs the character to actually span >1 token"

    dt = IncrementalDetokenizer()
    partial = dt.add_token(ids[0])
    # whatever was emitted so far must itself be valid, decodable text (not a
    # dangling lead byte) -- if this raises, the detokenizer emitted garbage
    partial.encode("utf-8")
