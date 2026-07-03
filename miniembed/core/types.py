"""
Core type definitions for MiniEmbed v2.

These types are shared across tokenizers and models so that the two layers
compose cleanly. The :class:`TokenizerOutput` mirrors the contract already
consumed by the legacy ``src`` model (``input_ids`` + ``attention_mask``),
keeping strict backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenizerOutput:
    """Output of a tokenizer ``encode`` call.

    Parameters
    ----------
    input_ids : torch.Tensor
        Token ids of shape ``[seq_len]`` (single sequence) or
        ``[batch, seq_len]``. dtype ``torch.long``.
    attention_mask : torch.Tensor
        Binary mask matching ``input_ids`` shape, ``1`` for real tokens and
        ``0`` for padding.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: str | torch.device) -> TokenizerOutput:
        """Move both tensors to ``device`` and return a new instance."""
        return TokenizerOutput(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


__all__ = ["TokenizerOutput"]
