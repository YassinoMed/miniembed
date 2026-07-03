"""
Byte-Pair-Encoding tokenizer, implemented from scratch.

Training algorithm
------------------
1. Pre-tokenize the corpus with a simple regex (words + punctuation), each
   pre-token becomes a tuple of its initial characters plus an end-of-word
   marker ``</w>`` so word boundaries survive merging.
2. Repeatedly count adjacent symbol pairs across the whole corpus and merge
   the single most frequent pair, until ``vocab_size`` symbols are reached.

Encoding
--------
At inference time we apply the learned merges in priority order (longest
trained first wins): for a pre-token we greedily merge the highest-priority
adjacent pair still present, repeating until no merge applies. This is the
standard BPE greedy procedure.

Decoding
--------
Concatenate the subword strings of the ids and drop ``</w>`` markers.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from ..registry import register
from .base import BaseTokenizer

#: Same pre-tokenization heuristic as the word-level tokenizer, so corpora are
#: comparable across tokenizer kinds.
_PRETOK_RE = re.compile(r"\b\w+\b|[^\w\s]")
_END = "</w>"


def _word_to_symbols(word: str) -> Tuple[str, ...]:
    """Split a pre-token into initial symbols, with an end-of-word marker."""
    return (*tuple(word), _END)


@register("tokenizer", "bpe")
class BPETokenizer(BaseTokenizer):
    """Byte-Pair-Encoding tokenizer trained from scratch."""

    tokenizer_type = "bpe"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # id -> symbol string (vocab). Initialized with special tokens only.
        self.id_to_token: Dict[int, str] = {
            self.pad_token_id: "[PAD]",
            self.unk_token_id: "[UNK]",
            self.cls_token_id: "[CLS]",
            self.sep_token_id: "[SEP]",
        }
        self.token_to_id: Dict[str, int] = {v: k for k, v in self.id_to_token.items()}
        # Ordered list of merges: (a, b). Priority = index in list.
        self.merges: List[Tuple[str, str]] = []
        # Initial alphabet (single chars) is added during training; keep track
        # so encoding of an unseen char falls back to [UNK] cleanly.
        self._alphabet: set[str] = set()

    # ------------------------------------------------------------------
    def _preprocess(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return _PRETOK_RE.findall(text)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        corpus: Iterable[str],
        vocab_size: int | None = None,
        min_freq: int = 2,
    ) -> None:
        if vocab_size is not None:
            self.vocab_size = vocab_size

        # 1. Build word frequency table over the corpus.
        word_freq: Counter[str] = Counter()
        for text in tqdm(corpus, desc="BPE: counting words"):
            word_freq.update(self._preprocess(text))

        # 2. Represent each unique word as a tuple of symbols + </w>.
        #    words[word_symbols] = frequency
        words: Dict[Tuple[str, ...], int] = {
            _word_to_symbols(w): f for w, f in word_freq.items() if f >= min_freq
        }

        # 3. Seed the alphabet: every character that appears.
        self._alphabet = {ch for syms in words for ch in syms if ch != _END}
        self._alphabet.add(_END)

        # Reset vocab to specials + alphabet.
        self.id_to_token = {
            self.pad_token_id: "[PAD]",
            self.unk_token_id: "[UNK]",
            self.cls_token_id: "[CLS]",
            self.sep_token_id: "[SEP]",
        }
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        for sym in sorted(self._alphabet):
            if sym not in self.token_to_id:
                self._add_token(sym)
        self.merges = []

        # 4. Iteratively merge the most frequent adjacent pair.
        target_size = self.vocab_size
        pbar = tqdm(desc="BPE: merging", total=None)
        while len(self.token_to_id) < target_size:
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for syms, freq in words.items():
                for a, b in zip(syms, syms[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            # Tie-break deterministically: highest freq, then lexicographic.
            best, best_freq = max(
                pair_counts.items(), key=lambda kv: (kv[1], kv[0])
            )
            if best_freq < 1:
                break
            self._merge_pair(best, words)
            self.merges.append(best)
            self._add_token(best[0] + best[1])
            pbar.update(1)
        pbar.close()

    def _add_token(self, symbol: str) -> None:
        if symbol in self.token_to_id:
            return
        # Next id = max(existing) + 1, so specials keep their fixed ids.
        idx = (max(self.id_to_token) + 1) if self.id_to_token else 0
        self.id_to_token[idx] = symbol
        self.token_to_id[symbol] = idx

    @staticmethod
    def _merge_pair(
        pair: Tuple[str, str], words: Dict[Tuple[str, ...], int]
    ) -> None:
        """Apply ``pair`` merge in place across all word symbol tuples."""
        a, b = pair
        merged = a + b
        new_words: Dict[Tuple[str, ...], int] = {}
        for syms, freq in words.items():
            if len(syms) < 2:
                new_words[syms] = new_words.get(syms, 0) + freq
                continue
            out: List[str] = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            key = tuple(out)
            new_words[key] = new_words.get(key, 0) + freq
        words.clear()
        words.update(new_words)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_word(self, word: str) -> List[int]:
        syms = list(_word_to_symbols(word))
        # Map unknown chars to a sentinel that will resolve to UNK.
        if any(s not in self.token_to_id for s in syms):
            # Replace any char not in the alphabet with UNK directly.
            syms = [s if s in self.token_to_id else "[UNK]" for s in syms]

        # Apply merges in priority order until stable.
        while len(syms) > 1:
            # Find the pair with the smallest merge index present in syms.
            best_idx = None
            best_rank = None
            for i in range(len(syms) - 1):
                pair = (syms[i], syms[i + 1])
                if pair in self._merge_rank:
                    rank = self._merge_rank[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
            if best_idx is None:
                break
            a, b = syms[best_idx], syms[best_idx + 1]
            syms[best_idx : best_idx + 2] = [a + b]

        return [
            self.token_to_id.get(s, self.unk_token_id) if s != "[UNK]" else self.unk_token_id
            for s in syms
        ]

    @property
    def _merge_rank(self) -> Dict[Tuple[str, str], int]:
        # Lazily computed mapping (pair -> priority). Cached on the instance.
        cache = getattr(self, "_merge_rank_cache", None)
        if cache is None or len(cache) != len(self.merges):
            cache = {pair: i for i, pair in enumerate(self.merges)}
            self._merge_rank_cache = cache  # type: ignore[attr-defined]
        return cache

    def _encode_ids(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in self._preprocess(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        skip = {self.pad_token_id, self.cls_token_id, self.sep_token_id}
        out: List[str] = []
        for i in token_ids:
            if i in skip:
                continue
            sym = self.id_to_token.get(i, "[UNK]")
            if sym == _END:
                out.append(" ")
            elif sym.endswith(_END):
                out.append(sym[: -len(_END)] + " ")
            else:
                out.append(sym)
        text = "".join(out).strip()
        return re.sub(r"\s+", " ", text)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _save_state(self) -> Dict[str, Any]:
        # Store merges as list of [a, b]; vocab derivable from merges + alphabet.
        return {
            "merges": [list(m) for m in self.merges],
            # Store the full vocab too (id -> token) for robust loading even
            # if a future training change alters id assignment.
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }

    def _load_state(self, state: Dict[str, Any]) -> None:
        self.id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.merges = [tuple(m) for m in state.get("merges", [])]
        self._alphabet = {
            ch for sym in self.token_to_id if len(sym) == 1 for ch in sym
        }
        # Invalidate merge-rank cache.
        if hasattr(self, "_merge_rank_cache"):
            del self._merge_rank_cache  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.token_to_id)


__all__ = ["BPETokenizer"]
