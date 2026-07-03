"""
Model package: configurable Transformer Bi-Encoder and its components.

Public surface
--------------
* :class:`EmbeddingModel` — the assembled encoder (config-driven).
* :class:`ModelConfig` — re-exported from :mod:`miniembed.config`.
* Component factories: :func:`build_attention`, :func:`build_ffn`,
  :func:`build_pooling`.

Importing this package registers every built-in attention/FFN/pooling variant
so they can be selected by name from a config.
"""

from __future__ import annotations

from ..config import ModelConfig
from .attention import (
    MultiHeadAttention,
    SDPAAttention,
    build_attention,
)
from .embedding_model import EmbeddingModel
from .encoder import TransformerEncoderLayer
from .feedforward import GEGLUFFN, GELUFFN, SwiGLUFFN, build_ffn
from .pooling import (
    AttentionPooling,
    CLSPooling,
    GeMPooling,
    MaxPooling,
    MeanPooling,
    WeightedMeanPooling,
    build_pooling,
)
from .positional import (
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    apply_rotary,
    build_positional,
)

__all__ = [
    "EmbeddingModel",
    "ModelConfig",
    # components
    "MultiHeadAttention",
    "SDPAAttention",
    "build_attention",
    "GELUFFN",
    "SwiGLUFFN",
    "GEGLUFFN",
    "build_ffn",
    "MeanPooling",
    "CLSPooling",
    "MaxPooling",
    "AttentionPooling",
    "GeMPooling",
    "WeightedMeanPooling",
    "build_pooling",
    "SinusoidalPositionalEncoding",
    "RotaryEmbedding",
    "apply_rotary",
    "build_positional",
    "TransformerEncoderLayer",
]
