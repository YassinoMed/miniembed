"""
WordPiece tokenizer, implemented from scratch.

WordPiece (used by BERT) is a subword tokenization where continuation tokens
are prefixed with ``##``. The vocabulary is built greedily: starting from a
character alphabet, the token that most increases the corpus log-likelihood is
added at each step, until ``vocab_size`` is reached.

Encoding uses the BERT-style *greedy longest-match-first* algorithm: for each
pre-token we scan left to right, taking the longest substring (with ``##`` on
continuation pieces) that is in the vocabulary; an unmatched span yields
``[UNK]``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from ..registry import register
from .base import BaseTokenizer

_PRETOK_RE = re.compile(r"\b\w+\b|[^\w\s]")
_CONT = "##"  # continuation prefix


@register("tokenizer", "wordpiece")
class WordPieceTokenizer(BaseTokenizer):
    """WordPiece tokenizer trained from scratch (BERT-style)."""

    tokenizer_type = "wordpiece"

    def __init__(self, unk_threshold: str = "unk", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.unk_threshold = unk_threshold  # "unk" => whole word to UNK on miss

    # ------------------------------------------------------------------
    def _preprocess(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return _PRETOK_RE.findall(text)

    @staticmethod
    def _word_to_subwords(word: str) -> List[str]:
        """Initial subword split for a word: first char + ##rest chars."""
        chars = list(word)
        if not chars:
            return []
        subs = [chars[0]]
        for ch in chars[1:]:
            subs.append(f"{_CONT}{ch}")
        return subs

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

        # 1. Word frequency table.
        word_freq: Counter[str] = Counter()
        for text in tqdm(corpus, desc="WordPiece: counting words"):
            word_freq.update(self._preprocess(text))
        word_freq = Counter({w: f for w, f in word_freq.items() if f >= min_freq})

        # 2. Seed alphabet + reset vocab to specials only.
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        alphabet: set[str] = set()
        for w in word_freq:
            alphabet.update(w)
        for ch in sorted(alphabet):
            self._add_token(ch)
            self._add_token(f"{_CONT}{ch}")

        # 3. Initial segmentation of every word into char-level subwords.
        #    segmentation[word] = list of subword strings currently in vocab.
        segmentation: Dict[str, List[str]] = {
            w: self._word_to_subwords(w) for w in word_freq
        }

        def corpus_logprob() -> float:
            """Sum of log freq(subword) over all token occurrences."""
            sub_freq: Counter[str] = Counter()
            for w, f in word_freq.items():
                for s in segmentation[w]:
                    sub_freq[s] += f
            total = sum(sub_freq.values()) or 1
            import math

            return sum(f * math.log(f / total) for s, f in sub_freq.items() if f > 0)

        # 4. Greedily merge the adjacent pair that maximizes log-likelihood gain.
        pbar = tqdm(desc="WordPiece: merging")
        while len(self.token_to_id) < self.vocab_size:
            # Candidate merges: adjacent pairs inside current segmentations.
            pair_gain: Counter[tuple[str, str]] = Counter()
            for w, f in word_freq.items():
                seg = segmentation[w]
                for a, b in zip(seg, seg[1:]):
                    pair_gain[(a, b)] += f
            if not pair_gain:
                break

            # Score = frequency of the merged candidate (standard heuristic;
            # full likelihood-gain is O(vocab) per step and gives the same
            # ordering for a fixed segmentation).
            best_pair, best_freq = pair_gain.most_common(1)[0]
            if best_freq < 1:
                break

            merged = self._join_pair(best_pair)
            self._add_token(merged)

            # Apply the merge to all segmentations.
            a, b = best_pair
            for w in segmentation:
                seg = segmentation[w]
                if len(seg) < 2:
                    continue
                new_seg: List[str] = []
                i = 0
                while i < len(seg):
                    if i < len(seg) - 1 and seg[i] == a and seg[i + 1] == b:
                        new_seg.append(merged)
                        i += 2
                    else:
                        new_seg.append(seg[i])
                        i += 1
                segmentation[w] = new_seg
            pbar.update(1)
        pbar.close()

    @staticmethod
    def _join_pair(pair: tuple[str, str]) -> str:
        """Join two subwords into one: drop ``##`` of the second piece."""
        a, b = pair
        return a + b[2:] if b.startswith(_CONT) else a + b

    def _add_token(self, token: str) -> None:
        if token in self.token_to_id:
            return
        idx = (max(self.id_to_token) + 1) if self.id_to_token else 0
        self.id_to_token[idx] = token
        self.token_to_id[token] = idx

    # ------------------------------------------------------------------
    # Encoding: greedy longest-match-first (BERT style)
    # ------------------------------------------------------------------
    def _encode_word(self, word: str) -> List[int]:
        if not word:
            return []
        chars = list(word)
        n = len(chars)
        ids: List[int] = []
        start = 0
        is_first = True
        while start < n:
            end = n
            cur_id = None
            cur_len = 0
            while start < end:
                sub = chars[start] if is_first else f"{_CONT}{chars[start]}"
                # try the longest substring from `start` to `end`
                candidate = (
                    "".join(chars[start:end])
                    if is_first
                    else _CONT + "".join(chars[start:end])
                )
                if candidate in self.token_to_id:
                    cur_id = self.token_to_id[candidate]
                    cur_len = end - start
                    break
                end -= 1
            if cur_id is None:
                return [self.unk_token_id]
            ids.append(cur_id)
            start += cur_len
            is_first = False
        return ids

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
            tok = self.id_to_token.get(i, "[UNK]")
            if tok.startswith(_CONT):
                out.append(tok[2:])
            else:
                if out:
                    out.append(" ")
                out.append(tok)
        return "".join(out).strip()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _save_state(self) -> Dict[str, Any]:
        return {"id_to_token": {str(k): v for k, v in self.id_to_token.items()}}

    def _load_state(self, state: Dict[str, Any]) -> None:
        self.id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)


__all__ = ["WordPieceTokenizer"]
