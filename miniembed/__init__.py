"""
MiniEmbed v2 — tiny, powerful, configurable text embedding toolkit.

This package is the evolving "platform" layer of the MiniEmbed project. It is
built alongside (not on top of) the original ``src`` package, which remains
unchanged for strict backward compatibility.

Phase 1 (this release) ships a configurable Transformer Bi-Encoder and a
family of pluggable tokenizers, all implemented from scratch in PyTorch:

>>> from miniembed.models import EmbeddingModel, ModelConfig
>>> from miniembed.tokenizers import get_tokenizer
>>> model = EmbeddingModel(ModelConfig())
>>> tok = get_tokenizer("wordlevel")

Quickly explore the available components:

>>> from miniembed.tokenizers import list_tokenizers
>>> from miniembed.registry import available
>>> list_tokenizers()                 # ['bpe', 'sentencepiece', ...]
>>> available("ffn"), available("pooling")

See ``miniembed/README.md`` for the full Phase 1 guide.
"""

from __future__ import annotations

# Importing config + registry first is harmless; importing models/tokenizers
# registers all built-in components as a side effect.
from . import config as config  # noqa: F401
from .config import (
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)
from .core import TokenizerOutput
from .registry import available, get, register

__version__ = "2.0.0a1"

# Re-export the most common entry points at package level for convenience.
# Submodules are imported lazily via __getattr__ to keep import time low and to
# avoid pulling torch at ``import miniembed`` time in environments that only
# need config helpers.
_SUBMODULES = {
    "EmbeddingModel": "miniembed.models",
    "get_tokenizer": "miniembed.tokenizers",
    "BaseTokenizer": "miniembed.tokenizers",
}


def __getattr__(name: str):  # PEP 562
    if name in _SUBMODULES:
        import importlib

        mod = importlib.import_module(_SUBMODULES[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'miniembed' has no attribute {name!r}")


__all__ = [
    "__version__",
    "ModelConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "TokenizerOutput",
    "register",
    "get",
    "available",
    "EmbeddingModel",
    "BaseTokenizer",
    "get_tokenizer",
]
