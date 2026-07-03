"""
Multi-Head Attention with two interchangeable backends.

* :class:`MultiHeadAttention` — the explicit, from-scratch implementation
  (ported from ``src/model.py``). Computes QK^T, softmax, AV manually. This is
  the backend that reproduces ``models/mini`` exactly.

* :class:`SDPAAttention` — uses ``torch.nn.functional.scaled_dot_product_attention``
  (PyTorch >= 2.0). On supported hardware this dispatches to the fused
  ``flash``/``memory-efficient`` kernel, giving large speedups and lower
  memory with no external dependency. If the SDPA path is unavailable (old
  PyTorch), :func:`build_attention` silently falls back to the explicit MHA.

RoPE
----
When the model uses rotary embeddings, the attention layer expects a
``rotary`` module and applies it to Q and K before computing attention. With
sinusoidal positions, ``rotary`` is ``None`` and the inputs are used as-is.

Backend selection is logged once at construction via the ``logging`` module so
operators know which kernel is active.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register
from .positional import RotaryEmbedding, apply_rotary

logger = logging.getLogger(__name__)


@register("attention", "mha")
class MultiHeadAttention(nn.Module):
    """Explicit from-scratch Multi-Head Self-Attention.

    Identical numerics to the legacy implementation, so ``models/mini``
    weights load with no discrepancy.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rotary: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        self.rotary = rotary

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.size()
        return x.view(b, l, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, _ = x.size()

        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        if self.rotary is not None:
            cos, sin = self.rotary(l)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [b,h,l,l]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [b,1,1,l]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [b,h,l,d_k]
        context = (
            context.transpose(1, 2).contiguous().view(b, l, self.d_model)
        )
        return self.W_o(context)


@register("attention", "sdpa")
class SDPAAttention(nn.Module):
    """Multi-Head Attention backed by ``scaled_dot_product_attention``.

    On PyTorch >= 2.0 this dispatches to fused kernels (Flash on capable GPUs,
    memory-efficient otherwise). Falls back gracefully if unavailable.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rotary: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "SDPAAttention requires PyTorch >= 2.0 "
                "(scaled_dot_product_attention not found)."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout_p = dropout
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.rotary = rotary

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.size()
        return x.view(b, l, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, _ = x.size()

        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        if self.rotary is not None:
            cos, sin = self.rotary(l)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        # Build the additive attention bias from the padding mask.
        attn_mask = None
        if attention_mask is not None:
            # [b, l] -> [b, 1, 1, l] additive, 0 keep / -inf mask
            attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * (
                torch.finfo(q.dtype).min
            )

        # training=False disables the dropout inside SDPA even when self.training
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.W_o(out)


def build_attention(
    name: str,
    d_model: int,
    num_heads: int,
    dropout: float = 0.1,
    rotary: Optional[RotaryEmbedding] = None,
    max_seq_len: int = 128,
) -> nn.Module:
    """Factory with auto-fallback: SDPA -> MHA if SDPA unavailable."""
    from ..registry import get

    if name == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
        logger.info(
            "SDPA backend unavailable (PyTorch < 2.0); falling back to MHA."
        )
        name = "mha"
    if name == "sdpa":
        logger.info(
            "Using SDPA attention backend (Flash/mem-efficient on capable HW)."
        )
    else:
        logger.info("Using explicit Multi-Head Attention backend.")
    cls = get("attention", name)
    return cls(d_model=d_model, num_heads=num_heads, dropout=dropout, rotary=rotary)


__all__ = [
    "MultiHeadAttention",
    "SDPAAttention",
    "build_attention",
]
