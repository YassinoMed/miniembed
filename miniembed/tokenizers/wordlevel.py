"""
Word-level tokenizer (port of the legacy ``SimpleTokenizer``).

This is the tokenizer used to train the shipped ``models/mini`` checkpoint.
Keeping it byte-compatible (same special ids, same regex split, same vocab
ordering) is what lets the new architecture load the old weights.

Vocabulary is stored as ``word -> id``. The training step counts words by
frequency and keeps the top ``vocab_size - <special>`` most frequent tokens
(with ``min_freq`` floor), exactly like the original implementation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from ..registry import register
from .base import BaseTokenizer

#: Word/punctuation tokenization pattern, identical to the legacy tokenizer.
_WORD_RE = re.compile(r"\b\w+\b|[^\w\s]")


@register("tokenizer", "wordlevel")
class WordLevelTokenizer(BaseTokenizer):
    """Word-level tokenizer compatible with ``models/mini``."""

    tokenizer_type = "wordlevel"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.word_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_word: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

    # ------------------------------------------------------------------
    def _preprocess(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        text = text.strip()
        return _WORD_RE.findall(text)

    def _encode_ids(self, text: str) -> List[int]:
        return [
            self.word_to_id.get(tok, self.unk_token_id)
            for tok in self._preprocess(text)
        ]

    def decode(self, token_ids: Iterable[int]) -> str:
        skip = {self.pad_token_id, self.cls_token_id, self.sep_token_id}
        return " ".join(
            self.id_to_word.get(i, "[UNK]") for i in token_ids if i not in skip
        )

    # ------------------------------------------------------------------
    def train(
        self,
        corpus: Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 2,
    ) -> None:
        if vocab_size is not None:
            self.vocab_size = vocab_size

        counts: Counter[str] = Counter()
        for text in tqdm(corpus, desc="Building word-level vocabulary"):
            counts.update(self._preprocess(text))

        max_words = self.vocab_size - len(self.special_tokens)
        # Stable order: frequency desc, then alphabetical for determinism.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        # Reset to special tokens only, then re-add frequent words.
        self.word_to_id = dict(self.special_tokens)
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}

        for word, freq in ranked[:max_words]:
            if freq < min_freq:
                continue
            if word in self.word_to_id:
                continue
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    # ------------------------------------------------------------------
    def _save_state(self) -> Dict[str, Any]:
        # id_to_word rebuilt from word_to_id on load; store str ids for JSON.
        return {"word_to_id": self.word_to_id}

    def _load_state(self, state: Dict[str, Any]) -> None:
        self.word_to_id = {str(k): int(v) for k, v in state["word_to_id"].items()}
        self.id_to_word = {int(v): k for k, v in self.word_to_id.items()}

    def __len__(self) -> int:
        return len(self.word_to_id)


__all__ = ["WordLevelTokenizer"]
