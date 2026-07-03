"""
Tokenizer interface (abstract base).

All MiniEmbed tokenizers implement :class:`BaseTokenizer`. The contract is
deliberately close to the legacy ``SimpleTokenizer`` so that any tokenizer can
be plugged into the same model forward pass.

Serialization contract
----------------------
Every concrete tokenizer serializes to a JSON dict containing at least::

    {"tokenizer_type": "<registered name>", ...}

so that :meth:`BaseTokenizer.load` can dispatch to the right subclass. The
canonical special-token ids are fixed (``PAD=0, UNK=1, CLS=2, SEP=3``) to keep
the pre-trained ``models/mini`` checkpoint loadable.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from ..core.types import TokenizerOutput


class BaseTokenizer(ABC):
    """Abstract base for all MiniEmbed tokenizers.

    Subclasses must implement :meth:`_encode_ids`, :meth:`decode`,
    :meth:`_save_state`, :meth:`_load_state`, :meth:`train` (optional) and
    :meth:`__len__`. The base class handles padding, special-token wrapping
    (``[CLS] ... [SEP]``), attention-mask creation and JSON I/O.
    """

    #: Registered name (set by subclasses). Used as the ``tokenizer_type``
    #: discriminator when saving/loading.
    tokenizer_type: str = "base"

    def __init__(
        self,
        vocab_size: int = 30000,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        cls_token_id: int = 2,
        sep_token_id: int = 3,
        max_length: int = 128,
        lowercase: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.lowercase = lowercase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def special_tokens(self) -> Dict[str, int]:
        return {
            "[PAD]": self.pad_token_id,
            "[UNK]": self.unk_token_id,
            "[CLS]": self.cls_token_id,
            "[SEP]": self.sep_token_id,
        }

    def encode(self, text: str, max_length: int | None = None) -> TokenizerOutput:
        """Encode ``text`` to padded ``input_ids`` + ``attention_mask``.

        Wraps the token sequence with ``[CLS]`` / ``[SEP]`` and right-pads to
        ``max_length`` (defaults to ``self.max_length``).
        """
        max_len = max_length if max_length is not None else self.max_length
        if max_len < 2:
            raise ValueError(
                f"max_length must be >= 2 to fit [CLS] and [SEP], got {max_len}."
            )

        ids = self._encode_ids(text)
        # Reserve 2 slots for [CLS] and [SEP].
        ids = ids[: max_len - 2]
        full = [self.cls_token_id, *ids, self.sep_token_id]

        attention_mask = [1] * len(full)
        pad_len = max_len - len(full)
        if pad_len > 0:
            full.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        return TokenizerOutput(
            input_ids=torch.tensor(full, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    def encode_batch(
        self, texts: List[str], max_length: int | None = None
    ) -> TokenizerOutput:
        """Encode a list of texts and stack them into a ``[B, L]`` batch."""
        outputs = [self.encode(t, max_length) for t in texts]
        return TokenizerOutput(
            input_ids=torch.stack([o.input_ids for o in outputs]),
            attention_mask=torch.stack([o.attention_mask for o in outputs]),
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _encode_ids(self, text: str) -> List[int]:
        """Tokenize ``text`` and map to ids (WITHOUT special tokens / padding)."""

    @abstractmethod
    def decode(self, token_ids: Iterable[int]) -> str:
        """Map ids back to a string (special tokens dropped)."""

    @abstractmethod
    def _save_state(self) -> Dict[str, Any]:
        """Subclass-specific state to merge into the JSON payload."""

    @abstractmethod
    def _load_state(self, state: Dict[str, Any]) -> None:
        """Restore subclass-specific state from a JSON payload."""

    def train(self, corpus: Iterable[str], vocab_size: int | None = None) -> None:
        """Train/build the vocabulary on ``corpus``.

        Not every tokenizer needs training (e.g. word-level can be built by
        frequency), so the default implementation raises ``NotImplementedError``
        to make the contract explicit.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement train()."
        )

    @abstractmethod
    def __len__(self) -> int:
        """Current vocabulary size (number of entries, incl. special tokens)."""

    # ------------------------------------------------------------------
    # Serialization (uniform JSON format with discriminator)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload: Dict[str, Any] = {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "max_length": self.max_length,
            "lowercase": self.lowercase,
            "state": self._save_state(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def save_state(self) -> Dict[str, Any]:
        """Return the full serializable payload (used by the model manager)."""
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "max_length": self.max_length,
            "lowercase": self.lowercase,
            "state": self._save_state(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        """Restore in-place from a payload produced by :meth:`save_state`."""
        self.vocab_size = payload["vocab_size"]
        self.pad_token_id = payload["pad_token_id"]
        self.unk_token_id = payload["unk_token_id"]
        self.cls_token_id = payload["cls_token_id"]
        self.sep_token_id = payload["sep_token_id"]
        self.max_length = payload.get("max_length", self.max_length)
        self.lowercase = payload.get("lowercase", self.lowercase)
        self._load_state(payload.get("state", {}))


__all__ = ["BaseTokenizer"]
