"""
Pooling strategies: collapse a sequence of token vectors into one vector.

Six implementations, selectable via ``ModelConfig.pooling_type``. Every
strategy shares the signature::

    forward(token_embeddings, attention_mask) -> pooled [B, d_model]

The mean pooling variant reproduces the legacy ``models/mini`` behavior
exactly (mean over non-padded tokens), so weights load with no discrepancy.

L2 normalization is intentionally *not* applied here: it lives in the model's
``encode()`` method so pooling and normalization stay decoupled and testable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


def _mask3d(attention_mask: torch.Tensor) -> torch.Tensor:
    """Expand ``[B, L]`` mask to ``[B, L, 1]`` float for broadcasting."""
    return attention_mask.unsqueeze(-1).float()


@register("pooling", "mean")
class MeanPooling(nn.Module):
    """Mean over non-padded tokens (legacy ``models/mini`` behavior)."""

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = _mask3d(attention_mask)
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


@register("pooling", "cls")
class CLSPooling(nn.Module):
    """Take the first token (``[CLS]``) representation."""

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return token_embeddings[:, 0]


@register("pooling", "max")
class MaxPooling(nn.Module):
    """Max over non-padded tokens (padded positions set to -inf)."""

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = _mask3d(attention_mask)
        neg_inf = torch.finfo(token_embeddings.dtype).min
        masked = token_embeddings.masked_fill(mask == 0, neg_inf)
        return torch.max(masked, dim=1).values


@register("pooling", "attention")
class AttentionPooling(nn.Module):
    """Learned attention weights over tokens.

    A small projection scores each token; softmax (over valid tokens) yields
    the mixing weights.
    """

    def __init__(self, d_model: int, **_kwargs):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        scores = self.scorer(token_embeddings).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        return torch.sum(token_embeddings * weights, dim=1)


@register("pooling", "gem")
class GeMPooling(nn.Module):
    """Generalized Mean Pooling (Radenovic et al.).

    ``gem(x) = (mean(x^p))^(1/p)`` with a learnable ``p``. Smaller ``p``
    approaches min-pooling; ``p=1`` is mean-pooling. Clamp keeps gradients
    stable for tiny values.
    """

    def __init__(self, d_model: int, p: float = 3.0, eps: float = 1e-6, **_kwargs):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = token_embeddings.clamp(min=self.eps).pow(self.p)
        mask = _mask3d(attention_mask)
        summed = torch.sum(x * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (summed / counts).pow(1.0 / self.p)


@register("pooling", "weighted_mean")
class WeightedMeanPooling(nn.Module):
    """Position-weighted mean: a learnable scalar weight per position index.

    Early tokens get their own weight; this can model recency/primacy biases.
    A small embedding table indexed by position provides the weights.
    """

    def __init__(self, d_model: int, max_seq_len: int = 128, **_kwargs):
        super().__init__()
        self.max_seq_len = max_seq_len
        # One scalar weight per position (parameter, not embedding lookup).
        self.position_weights = nn.Parameter(torch.ones(max_seq_len))

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        b, l, _ = token_embeddings.size()
        w = self.position_weights[:l].unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        w = w * attention_mask.unsqueeze(-1).float()
        weighted = token_embeddings * w
        denom = torch.clamp(w.sum(dim=1), min=1e-9)
        return weighted.sum(dim=1) / denom


def build_pooling(
    name: str, d_model: int, max_seq_len: int = 128, pooling_p: float = 3.0
) -> nn.Module:
    """Factory returning the pooling module registered under ``name``."""
    from ..registry import get

    cls = get("pooling", name)
    if name == "gem":
        return cls(d_model=d_model, p=pooling_p)
    if name in ("attention", "weighted_mean"):
        return cls(d_model=d_model, max_seq_len=max_seq_len)
    return cls()


    # The ``cls()`` calls for parameter-free poolings (mean/cls/max) ignore
    # extra kwargs, hence the branches above for parametrized variants.


__all__ = [
    "MeanPooling",
    "CLSPooling",
    "MaxPooling",
    "AttentionPooling",
    "GeMPooling",
    "WeightedMeanPooling",
    "build_pooling",
]
