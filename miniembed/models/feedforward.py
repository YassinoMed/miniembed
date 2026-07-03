"""
Position-wise Feed-Forward Network variants.

Three implementations, selectable via ``ModelConfig.ffn_type``:

* ``gelu``   : the classic 2-layer FFN with GELU (legacy, used by
  ``models/mini``). Output dim = ``d_model``.
* ``swiglu`` : gated FFN using SiLU. Projects to ``d_ff`` for the gate and the
  value, multiplies, then projects back. ``d_ff`` is the *intermediate* size.
* ``geglu``  : same gated structure with GELU as the gate activation.

Gated variants (SwiGLU / GEGLU) follow the formulation of
*Shazeer (2020) "GLU Variants Improve Transformer"*.

For gated FFNs the effective hidden width is ``d_ff`` (the user-facing
parameter); internally the up/gate projections output ``d_ff`` each. This keeps
``ModelConfig.d_ff`` meaning consistent across variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


@register("ffn", "gelu")
class GELUFFN(nn.Module):
    """Classic 2-layer FFN: ``Linear -> GELU -> Dropout -> Linear``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class _GatedFFN(nn.Module):
    """Base for GLU-style gated FFNs. Subclasses pick the gate activation."""

    gate_act: staticmethod  # set by subclasses

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # gate and up projections both map d_model -> d_ff.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._gate(self.w_gate(x)) * self.w_up(x)
        h = self.dropout(h)
        return self.w_down(h)


@register("ffn", "swiglu")
class SwiGLUFFN(_GatedFFN):
    """SwiGLU gated FFN (gate activation = SiLU/Swish)."""

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


@register("ffn", "geglu")
class GEGLUFFN(_GatedFFN):
    """GEGLU gated FFN (gate activation = GELU)."""

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


def build_ffn(name: str, d_model: int, d_ff: int, dropout: float = 0.1) -> nn.Module:
    """Factory returning the FFN module registered under ``name``."""
    from ..registry import get

    cls = get("ffn", name)
    return cls(d_model=d_model, d_ff=d_ff, dropout=dropout)


__all__ = ["GELUFFN", "SwiGLUFFN", "GEGLUFFN", "build_ffn"]
