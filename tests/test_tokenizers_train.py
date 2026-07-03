"""Dedicated tests for the from-scratch training algorithms."""

from __future__ import annotations

import pytest

from miniembed.tokenizers import get_tokenizer


@pytest.mark.parametrize("name", ["bpe", "wordpiece", "unigram"])
def test_training_produces_nontrivial_vocab(name, small_corpus):
    tok = get_tokenizer(name, vocab_size=100, max_length=24)
    tok.train(small_corpus, min_freq=1)
    # Vocab must be larger than just the 4 special tokens.
    assert len(tok) > 4
    # And must not exceed requested size.
    assert len(tok) <= 100


@pytest.mark.parametrize("name", ["bpe", "wordpiece", "unigram"])
def test_training_is_deterministic(name, small_corpus):
    """Same corpus + same seed inputs => identical vocabulary order."""
    a = get_tokenizer(name, vocab_size=60, max_length=16)
    b = get_tokenizer(name, vocab_size=60, max_length=16)
    a.train(small_corpus, min_freq=1)
    b.train(small_corpus, min_freq=1)
    assert len(a) == len(b)
    # Encode the same text -> same ids.
    ids_a = a.encode("machine", max_length=12).input_ids.tolist()
    ids_b = b.encode("machine", max_length=12).input_ids.tolist()
    assert ids_a == ids_b


def test_bpe_merges_grow_vocab(small_corpus):
    tok = get_tokenizer("bpe", vocab_size=100, max_length=16)
    tok.train(small_corpus, min_freq=1)
    # After training there must be at least one merge.
    assert len(tok.merges) >= 1


def test_unigram_viterbi_handles_single_char():
    """A single unknown-ish char present in alphabet must segment."""
    tok = get_tokenizer("unigram", vocab_size=50, max_length=16)
    tok.train(["cat", "car", "card", "cart"], min_freq=1)
    ids = tok.encode("cat", max_length=8).input_ids.tolist()
    # CLS ... SEP -> no UNK in the middle
    assert 1 not in ids[1:-1]  # no UNK token in content region


def test_wordpiece_continuation_prefix(small_corpus):
    tok = get_tokenizer("wordpiece", vocab_size=80, max_length=16)
    tok.train(small_corpus, min_freq=1)
    # At least some continuation tokens (##) should exist after training.
    has_continuation = any(t.startswith("##") for t in tok.token_to_id)
    assert has_continuation
