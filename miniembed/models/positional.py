"""
Positional encodings: sinusoidal (legacy) and Rotary (RoPE).

The sinusoidal variant is a direct port of the original ``src/model.py``
implementation, byte-for-byte compatible so the pre-trained ``models/mini``
weights load unchanged.

RoPE is provided as the modern alternative. Because RoPE rotates Q and K
*inside* the attention layer (rather than adding a signal to the embedding),
it is implemented as a helper that returns the rotation cos/sin tables and a
:func:`apply_rotary` function; the attention module calls it. The model picks
between the two via ``ModelConfig.position_type``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding ("Attention Is All You Need").

    Adds position information to token embeddings using sin/cos functions at
    geometric frequencies. Registered as a buffer (not a parameter).

    Parameters
    ----------
    d_model : int
        Embedding dimensionality.
    max_seq_len : int
        Maximum sequence length to precompute.
    dropout : float
        Dropout applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape ``[B, L, d_model]``."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Computes the cos/sin tables for rotating query/key pairs by their
    position-dependent angle. Rotation is applied in :func:`apply_rotary`,
    which the attention layer must call on Q and K.

    The implementation uses the "GPT-NeoX" half-rotation layout (rotate the two
    halves of the head dimension), which is the most widely deployed form.

    Parameters
    ----------
    head_dim : int
        Per-head dimensionality (``d_model // num_heads``). Must be even.
    max_seq_len : int
        Maximum sequence length to precompute.
    base : float
        Frequency base (10000 by default, as in the original RoPE paper).
    """

    def __init__(
        self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
        self.head_dim = head_dim
        self.base = base

        # Inverse frequencies: [head_dim // 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [L, head_dim/2]
        # Duplicate to full head_dim: [L, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tables of shape ``[seq_len, head_dim]``."""
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dim: ``[-x2, x1]``.

    Used by :func:`apply_rotary` to implement the GPT-NeoX RoPE layout.
    """
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to ``x``.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``[B, num_heads, L, head_dim]``.
    cos, sin : torch.Tensor
        Tables of shape ``[L, head_dim]`` (will be sliced/reshaped).

    Returns
    -------
    torch.Tensor
        Rotated tensor, same shape as ``x``.
    """
    seq_len = x.size(-2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def build_positional(config, d_model: int, max_seq_len: int, dropout: float):
    """Factory selecting the positional module from ``config.position_type``.

    Returns ``None`` for ``rope`` because RoPE does not have a standalone
    module in the embedding stem — it is handled inside attention. The model
    builder interprets ``None`` as "skip adding positional info in the stem".
    """
    from ..config import POSITION_TYPES  # noqa: F401 (documented reference)

    if config.position_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
    if config.position_type == "rope":
        return None
    raise ValueError(f"Unknown position_type {config.position_type!r}")


__all__ = [
    "SinusoidalPositionalEncoding",
    "RotaryEmbedding",
    "rotate_half",
    "apply_rotary",
    "build_positional",
]
