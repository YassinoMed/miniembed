"""
Tokenizer package: pluggable tokenizers behind a common interface.

Use :func:`get_tokenizer` to instantiate a tokenizer by name (the same names
used in :class:`~miniembed.config.TokenizerConfig`):

>>> from miniembed.tokenizers import get_tokenizer
>>> tok = get_tokenizer("wordlevel", vocab_size=8000)
>>> tok.encode("hello world")

Importing this package registers every built-in tokenizer. To add a custom
tokenizer, decorate its class with ``@register("tokenizer", "my_name")`` and
import the module before calling :func:`get_tokenizer`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..registry import available, get, register
from .base import BaseTokenizer
from .bpe import BPETokenizer
from .sentencepiece import SentencePieceTokenizer
from .unigram import UnigramTokenizer
from .wordlevel import WordLevelTokenizer
from .wordpiece import WordPieceTokenizer


def get_tokenizer(name: str, **kwargs: Any) -> BaseTokenizer:
    """Instantiate a registered tokenizer by name.

    Parameters
    ----------
    name : str
        Registered tokenizer name (``"wordlevel"``, ``"bpe"``,
        ``"wordpiece"``, ``"unigram"``, ``"sentencepiece"``).
    **kwargs
        Forwarded to the tokenizer constructor.

    Raises
    ------
    KeyError
        If ``name`` is not a registered tokenizer.
    """
    cls = get("tokenizer", name)
    return cls(**kwargs)


def list_tokenizers() -> List[str]:
    """Return the names of all registered tokenizers."""
    return available("tokenizer")


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    """Load a tokenizer from a JSON file written by :meth:`BaseTokenizer.save`.

    The file's ``tokenizer_type`` field selects the subclass; the remainder of
    the payload restores its state.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    name = payload["tokenizer_type"]
    tok = get_tokenizer(name)
    tok.load_state(payload)
    return tok


__all__ = [
    "BaseTokenizer",
    "WordLevelTokenizer",
    "BPETokenizer",
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "SentencePieceTokenizer",
    "get_tokenizer",
    "list_tokenizers",
    "load_tokenizer",
    "register",
]
