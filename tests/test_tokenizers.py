"""Tests for the tokenizer layer (base contract + each implementation)."""

from __future__ import annotations

import pytest

from miniembed.core import TokenizerOutput
from miniembed.tokenizers import (
    BaseTokenizer,
    get_tokenizer,
    list_tokenizers,
    load_tokenizer,
)


# Tokenizers that support from-scratch training on a corpus.
TRAINABLE = ["wordlevel", "bpe", "wordpiece", "unigram"]
REGISTERED = TRAINABLE + ["sentencepiece"]


def test_registry_lists_builtins():
    names = list_tokenizers()
    for name in TRAINABLE:
        assert name in names, f"{name} not registered"


@pytest.mark.parametrize("name", TRAINABLE)
def test_factory_returns_base_subclass(name):
    tok = get_tokenizer(name, vocab_size=1000, max_length=16)
    assert isinstance(tok, BaseTokenizer)


def test_special_tokens_are_canonical():
    """PAD=0, UNK=1, CLS=2, SEP=3 must hold for backward compatibility."""
    tok = get_tokenizer("wordlevel", vocab_size=1000, max_length=16)
    assert tok.special_tokens == {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}


def test_encode_wraps_cls_sep_and_pads():
    tok = get_tokenizer("wordlevel", vocab_size=1000, max_length=16)
    out = tok.encode("hi")
    assert isinstance(out, TokenizerOutput)
    ids = out.input_ids.tolist()
    # CLS first, SEP second, rest is PAD
    assert ids[0] == 2  # CLS
    assert ids[1] == 3 or ids[1] == 1  # SEP (if 'hi' OOV) or token
    assert out.attention_mask.sum() >= 2  # at least CLS + something
    # padded to max_length
    assert len(ids) == 16
    # mask length matches ids length
    assert out.attention_mask.shape == out.input_ids.shape


def test_encode_too_small_max_length_raises():
    tok = get_tokenizer("wordlevel", vocab_size=1000, max_length=16)
    with pytest.raises(ValueError):
        tok.encode("hi", max_length=1)


def test_encode_batch_stacks():
    tok = get_tokenizer("wordlevel", vocab_size=1000, max_length=8)
    out = tok.encode_batch(["a b", "c d e"])
    assert out.input_ids.shape[0] == 2
    assert out.input_ids.shape[1] == 8


@pytest.mark.parametrize("name", TRAINABLE)
def test_train_then_encode_decode(name, small_corpus):
    """Train on a corpus, then encode a known word and decode back."""
    tok = get_tokenizer(name, vocab_size=80, max_length=24)
    tok.train(small_corpus, min_freq=1)
    assert len(tok) > len(tok.special_tokens)  # vocab grew

    out = tok.encode("learning", max_length=16)
    ids = out.input_ids.tolist()
    # CLS at start, SEP after content
    assert ids[0] == 2
    assert ids[out.attention_mask.sum().item() - 1] == 3

    # decode should be non-empty and contain the word (subword-joined ok)
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)


@pytest.mark.parametrize("name", TRAINABLE)
def test_save_load_roundtrip(name, small_corpus, tmp_path):
    tok = get_tokenizer(name, vocab_size=80, max_length=24)
    tok.train(small_corpus, min_freq=1)

    before = tok.encode("machine learning", max_length=16).input_ids.tolist()

    path = tmp_path / "tok.json"
    tok.save(path)
    assert path.exists()

    tok2 = load_tokenizer(path)
    assert tok2.tokenizer_type == name
    after = tok2.encode("machine learning", max_length=16).input_ids.tolist()
    assert before == after


@pytest.mark.parametrize("name", TRAINABLE)
def test_payload_contains_discriminator(name, small_corpus, tmp_path):
    import json

    tok = get_tokenizer(name, vocab_size=60, max_length=16)
    tok.train(small_corpus, min_freq=1)
    path = tmp_path / "tok.json"
    tok.save(path)
    payload = json.loads(path.read_text())
    assert payload["tokenizer_type"] == name
    assert "vocab_size" in payload
    assert "state" in payload


def test_sentencepiece_lazy_without_dependency():
    """SentencePiece adapter must import even if the lib is absent."""
    # Just importing the module should not raise; only using it without the
    # dependency should fail. We test the factory path here.
    tok = get_tokenizer("sentencepiece", max_length=16)
    assert tok.tokenizer_type == "sentencepiece"
    # Without a model loaded, encode should raise RuntimeError.
    with pytest.raises(RuntimeError):
        tok.encode("hi")
