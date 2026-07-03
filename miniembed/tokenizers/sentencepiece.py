"""
SentencePiece tokenizer adapter (optional, lazy).

SentencePiece is a heavy native dependency. To keep MiniEmbed's core free of
mandatory external libs (per the project's "from scratch + fallback" policy),
the ``sentencepiece`` import is performed lazily: the module imports fine even
when the package is absent. An informative :class:`ImportError` is raised only
when the tokenizer is actually instantiated or used.

This adapter wraps a trained SentencePiece model file and exposes the
:class:`BaseTokenizer` contract, so it can be selected from config just like
the from-scratch tokenizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..registry import register
from .base import BaseTokenizer


def _require_sentencepiece():
    """Import ``sentencepiece`` lazily with a helpful error if missing."""
    try:
        import sentencepiece as spm  # type: ignore

        return spm
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "The 'sentencepiece' tokenizer requires the optional "
            "'sentencepiece' package. Install it with: pip install sentencepiece"
        ) from exc


@register("tokenizer", "sentencepiece")
class SentencePieceTokenizer(BaseTokenizer):
    """Adapter around a trained SentencePiece model.

    Notes
    -----
    SentencePiece manages its own special tokens (``<pad>``, ``<unk>``,
    ``<s>``, ``</s>``). This adapter maps them onto MiniEmbed's canonical ids
    (PAD=0, UNK=1, CLS=2, SEP=3) so the rest of the pipeline is agnostic to
    the tokenizer kind.
    """

    tokenizer_type = "sentencepiece"

    def __init__(self, model_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_path = model_path
        self._processor = None
        if model_path is not None:
            self._load_processor(model_path)

    def _load_processor(self, path: str) -> None:
        spm = _require_sentencepiece()
        self._processor = spm.SentencePieceProcessor()
        self._processor.Load(str(path))
        self.model_path = str(path)

    # ------------------------------------------------------------------
    def _encode_ids(self, text: str) -> List[int]:
        if self._processor is None:
            raise RuntimeError(
                "SentencePieceTokenizer has no loaded model. Pass "
                "model_path=... or call train()."
            )
        return list(self._processor.EncodeAsIds(text))

    def decode(self, token_ids: Iterable[int]) -> str:
        if self._processor is None:
            raise RuntimeError("SentencePieceTokenizer has no loaded model.")
        return self._processor.DecodeIds(list(token_ids))

    def train(self, corpus: Iterable[str], vocab_size: int | None = None, **kw: Any) -> None:
        """Train a SentencePiece model on ``corpus``.

        This writes the corpus to a temp file and invokes ``sentencepiece``.
        Provided for API completeness; production training is usually done
        offline with the ``spm_train`` CLI.
        """
        import tempfile

        spm = _require_sentencepiece()
        if vocab_size is not None:
            self.vocab_size = vocab_size

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for line in corpus:
                f.write(line + "\n")
            corpus_path = f.name
        model_prefix = tempfile.mktemp(prefix="miniembed_sp_")
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=kw.get("model_type", "unigram"),
        )
        self._load_processor(model_prefix + ".model")

    # ------------------------------------------------------------------
    def _save_state(self) -> Dict[str, Any]:
        # We do not embed the binary .model in JSON. The user keeps the model
        # file and points back to it via model_path.
        return {"model_path": self.model_path}

    def _load_state(self, state: Dict[str, Any]) -> None:
        path = state.get("model_path")
        if path:
            self._load_processor(path)

    def __len__(self) -> int:
        if self._processor is None:
            return len(self.special_tokens)
        return self._processor.GetPieceSize()


__all__ = ["SentencePieceTokenizer"]
