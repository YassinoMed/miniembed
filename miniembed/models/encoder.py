"""
Transformer encoder layer (Pre-LayerNorm), modular.

A single layer combining attention + FFN with residual connections. Pre-LN
ordering (normalize before the sublayer) matches the legacy ``src/model.py``
implementation, preserving ``models/mini`` compatibility.

The layer is agnostic to *which* attention / FFN it uses — it just calls the
injected modules — so swapping backends (MHA vs SDPA, GELU vs SwiGLU) requires
no change here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm encoder layer with attention + FFN residuals.

    Parameters
    ----------
    attention : nn.Module
        Attention submodule produced by :func:`build_attention`.
    feed_forward : nn.Module
        FFN submodule produced by :func:`build_ffn`.
    d_model : int
        Model width (for LayerNorm).
    dropout : float
        Residual-path dropout.
    """

    def __init__(
        self,
        attention: nn.Module,
        feed_forward: nn.Module,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention block
        normed = self.norm1(x)
        x = x + self.dropout(self.attention(normed, attention_mask))

        # Pre-norm feed-forward block
        normed = self.norm2(x)
        x = x + self.dropout(self.feed_forward(normed))
        return x


__all__ = ["TransformerEncoderLayer"]
