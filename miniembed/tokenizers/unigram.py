"""
Unigram tokenizer (SentencePiece-style), implemented from scratch.

Training algorithm (EM with pruning)
------------------------------------
1. Build a seed vocabulary of candidate subwords from the corpus (all
   substrings of each pre-token, weighted by frequency), restricted to a
   reasonable maximum length.
2. Run a few EM iterations to estimate the log-probability of each candidate
   subword (unigram model: maximize corpus likelihood).
3. Keep the top ``vocab_size`` subwords by likelihood loss; repeat until the
   target size is reached.

Decoding (Viterbi)
------------------
Given a word, find the segmentation maximizing the product of subword
probabilities. Implemented as a standard dynamic-programming Viterbi pass.
Unseen characters fall back to ``[UNK]``.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from ..registry import register
from .base import BaseTokenizer

_PRETOK_RE = re.compile(r"\b\w+\b|[^\w\s]")
_NEG_INF = float("-inf")


@register("tokenizer", "unigram")
class UnigramTokenizer(BaseTokenizer):
    """Unigram tokenizer (SentencePiece-style) trained from scratch."""

    tokenizer_type = "unigram"

    def __init__(
        self,
        max_piece_len: int = 16,
        em_iterations: int = 2,
        prune_factor: float = 0.75,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_piece_len = max_piece_len
        self.em_iterations = em_iterations
        self.prune_factor = prune_factor
        # piece -> log probability
        self.piece_logprob: Dict[str, float] = {}
        self.id_to_token: Dict[int, str] = dict(
            (v, k) for k, v in self.special_tokens.items()
        )
        self.token_to_id: Dict[str, int] = dict(self.special_tokens)

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
        seed_factor: int = 1_000_000,
    ) -> None:
        if vocab_size is not None:
            self.vocab_size = vocab_size

        # 1. Word frequency.
        word_freq: Counter[str] = Counter()
        for text in tqdm(corpus, desc="Unigram: counting words"):
            word_freq.update(self._preprocess(text))
        word_freq = Counter({w: f for w, f in word_freq.items() if f >= min_freq})

        # 2. Seed candidate vocabulary: all substrings of length <= max_piece_len.
        cand_freq: Counter[str] = Counter()
        for w, f in word_freq.items():
            n = len(w)
            for i in range(n):
                for j in range(i + 1, min(n, i + self.max_piece_len) + 1):
                    cand_freq[w[i:j]] += f
        if not cand_freq:
            return

        # Keep a generous seed (seed_factor * vocab_size), capped to available.
        seed_size = min(len(cand_freq), max(self.vocab_size * 50, 1000))
        seed = [p for p, _ in cand_freq.most_common(seed_size)]
        # Ensure the alphabet (single chars) is always present.
        alphabet = sorted({p for p in cand_freq if len(p) == 1})
        for ch in alphabet:
            if ch not in seed:
                seed.append(ch)

        # 3. Initialize probabilities from candidate frequencies (normalized).
        total = sum(cand_freq[p] for p in seed) or 1
        pieces: Dict[str, float] = {p: cand_freq[p] / total for p in seed}

        # 4. EM iterations.
        words = list(word_freq.keys())
        freqs = [word_freq[w] for w in words]
        for _ in range(self.em_iterations):
            pieces = self._em_step(pieces, words, freqs)

        # 5. Prune to vocab_size: keep pieces with highest "loss" contribution.
        #    Loss ~ -log(prob) * frequency of best use; we approximate by prob.
        ranked = sorted(pieces.items(), key=lambda kv: kv[1], reverse=True)
        # Always keep the alphabet so any word is segmentable.
        kept = set(alphabet)
        for piece, _ in ranked:
            if len(self.token_to_id) >= self.vocab_size:
                break
            kept.add(piece)
            self._register(piece, pieces[piece])
        # Make sure alphabet is registered even if ranked short.
        for ch in alphabet:
            if ch not in self.token_to_id:
                self._register(ch, pieces.get(ch, math.exp(-20)))

    def _register(self, piece: str, prob: float) -> None:
        if piece in self.token_to_id:
            return
        idx = (max(self.id_to_token) + 1) if self.id_to_token else 0
        self.id_to_token[idx] = piece
        self.token_to_id[piece] = idx
        # Floor prob so log is finite.
        self.piece_logprob[piece] = math.log(max(prob, 1e-12))

    def _em_step(
        self,
        pieces: Dict[str, float],
        words: List[str],
        freqs: List[int],
    ) -> Dict[str, float]:
        """One EM iteration: re-estimate piece probabilities via Viterbi."""
        logprob = {p: math.log(max(v, 1e-12)) for p, v in pieces.items()}
        expected: Counter[str] = Counter()
        for w, f in zip(words, freqs):
            seg = self._viterbi(w, logprob)
            if seg is None:
                continue
            for piece in seg:
                expected[piece] += f
        total = sum(expected.values()) or 1
        new_pieces = {p: expected[p] / total for p in pieces if expected[p] > 0}
        # Keep alphabet even if it had zero expected count.
        for p in pieces:
            if len(p) == 1 and p not in new_pieces:
                new_pieces[p] = max(pieces[p], 1e-6)
        return new_pieces

    # ------------------------------------------------------------------
    # Viterbi segmentation
    # ------------------------------------------------------------------
    def _viterbi(
        self, word: str, logprob: Dict[str, float]
    ) -> Optional[List[str]]:
        """Best segmentation of ``word`` maximizing summed log-prob."""
        n = len(word)
        if n == 0:
            return []
        # best_score[i] = best score for word[:i]
        best_score = [_NEG_INF] * (n + 1)
        best_back: List[Optional[Tuple[int, str]]] = [None] * (n + 1)
        best_score[0] = 0.0
        for end in range(1, n + 1):
            start_lo = max(0, end - self.max_piece_len)
            for start in range(start_lo, end):
                piece = word[start:end]
                if piece not in logprob:
                    continue
                score = best_score[start] + logprob[piece]
                if score > best_score[end]:
                    best_score[end] = score
                    best_back[end] = (start, piece)
        if best_score[n] == _NEG_INF:
            return None  # unsegmentable (char not in alphabet)
        # Reconstruct.
        seg: List[str] = []
        cur = n
        while cur > 0:
            back = best_back[cur]
            if back is None:
                return None
            start, piece = back
            seg.append(piece)
            cur = start
        seg.reverse()
        return seg

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def _encode_ids(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in self._preprocess(text):
            seg = self._viterbi(word, self.piece_logprob)
            if seg is None:
                ids.append(self.unk_token_id)
            else:
                ids.extend(self.token_to_id.get(p, self.unk_token_id) for p in seg)
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        skip = {self.pad_token_id, self.cls_token_id, self.sep_token_id}
        return " ".join(
            self.id_to_token.get(i, "[UNK]") for i in token_ids if i not in skip
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _save_state(self) -> Dict[str, Any]:
        return {
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "piece_logprob": self.piece_logprob,
        }

    def _load_state(self, state: Dict[str, Any]) -> None:
        self.id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        # Keep only pieces that are in the vocab.
        self.piece_logprob = {
            p: float(lp)
            for p, lp in state["piece_logprob"].items()
            if p in self.token_to_id
        }

    def __len__(self) -> int:
        return len(self.token_to_id)


__all__ = ["UnigramTokenizer"]
