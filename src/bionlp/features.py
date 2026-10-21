"""Feature extraction for sentences *in the context of their abstract*.

A sentence alone is ambiguous ("Patients were randomized ..." can open METHODS or
summarise BACKGROUND). Two cheap signals resolve most of it:

* **position** - where the sentence sits in the abstract (abstracts follow a strong
  BACKGROUND -> OBJECTIVE -> METHODS -> RESULTS -> CONCLUSIONS order);
* **context** - the words of the neighbouring sentences.

Every sample is a :class:`SentenceContext`, built from an ordered list of sentences.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple, Self

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


class SentenceContext(NamedTuple):
    text: str
    position: int  # 0-based index within the abstract
    n_sentences: int
    prev_text: str = ""
    next_text: str = ""


def build_contexts(sentences: Sequence[str]) -> list[SentenceContext]:
    """Turn the ordered sentences of one abstract into model inputs."""
    n = len(sentences)
    return [
        SentenceContext(
            text=s,
            position=i,
            n_sentences=n,
            prev_text=sentences[i - 1] if i > 0 else "",
            next_text=sentences[i + 1] if i + 1 < n else "",
        )
        for i, s in enumerate(sentences)
    ]


def normalize_text(text: str) -> str:
    """Lowercase and collapse numbers to one token ("p < 0.05", "125 patients" -> "0").

    Numbers are a strong RESULTS/METHODS cue, but their exact values are noise.
    """
    return _NUMBER_RE.sub(" 0 ", text.lower())


def position_matrix(
    contexts: Sequence[SentenceContext], *, max_position: int = 15, n_bins: int = 10
) -> sp.csr_matrix:
    """One-hot position features (linear models need the non-linearity spelled out).

    Columns: absolute index (clipped) | relative-position bin | abstract length (clipped)
    | is_first | is_last | relative position (continuous).
    """
    n = len(contexts)
    pos = np.fromiter((c.position for c in contexts), dtype=np.int64, count=n)
    total = np.fromiter((max(c.n_sentences, 1) for c in contexts), dtype=np.int64, count=n)
    rel = np.where(total > 1, pos / np.maximum(total - 1, 1), 0.0)

    abs_col = np.minimum(pos, max_position)
    bin_col = (max_position + 1) + np.minimum((rel * n_bins).astype(np.int64), n_bins - 1)
    len_col = (max_position + 1 + n_bins) + np.minimum(total, 2 * max_position)
    offset = max_position + 1 + n_bins + 2 * max_position + 1
    rows = np.arange(n)
    one_hot_rows = np.concatenate([rows, rows, rows])
    one_hot_cols = np.concatenate([abs_col, bin_col, len_col])
    data = np.ones(3 * n)
    first, last = pos == 0, pos == total - 1
    extra_rows = np.concatenate([rows[first], rows[last], rows])
    extra_cols = np.concatenate(
        [np.full(first.sum(), offset), np.full(last.sum(), offset + 1), np.full(n, offset + 2)]
    )
    extra_data = np.concatenate([np.ones(first.sum() + last.sum()), rel])
    return sp.csr_matrix(
        (
            np.concatenate([data, extra_data]),
            (
                np.concatenate([one_hot_rows, extra_rows]),
                np.concatenate([one_hot_cols, extra_cols]),
            ),
        ),
        shape=(n, offset + 3),
    )


class AbstractFeaturizer(TransformerMixin, BaseEstimator):
    """TF-IDF of the sentence, optionally + TF-IDF of its neighbours + position features.

    The neighbour blocks reuse the sentence vocabulary but get their own columns, so the
    classifier learns separate weights for "word in this sentence" vs "word next door".
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        normalize_numbers: bool = True,
        use_position: bool = True,
        use_context: bool = True,
        context_weight: float = 0.5,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.normalize_numbers = normalize_numbers
        self.use_position = use_position
        self.use_context = use_context
        self.context_weight = context_weight

    def fit(self, X: Sequence[SentenceContext], y: object = None) -> Self:
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            preprocessor=normalize_text if self.normalize_numbers else None,
            dtype=np.float32,
        )
        self.vectorizer_.fit([c.text for c in X])
        return self

    def transform(self, X: Sequence[SentenceContext]) -> sp.csr_matrix:
        vec = self.vectorizer_
        blocks = [vec.transform([c.text for c in X])]
        if self.use_context:
            blocks.append(vec.transform([c.prev_text for c in X]) * self.context_weight)
            blocks.append(vec.transform([c.next_text for c in X]) * self.context_weight)
        if self.use_position:
            blocks.append(position_matrix(X).astype(np.float32))
        return sp.hstack(blocks, format="csr")
