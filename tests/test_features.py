from __future__ import annotations

import numpy as np

from bionlp.features import AbstractFeaturizer, build_contexts, normalize_text, position_matrix

SENTENCES = ["Asthma is common .", "We tested drug X .", "It worked in 85 % of cases ."]


def test_build_contexts() -> None:
    contexts = build_contexts(SENTENCES)
    assert [c.position for c in contexts] == [0, 1, 2]
    assert {c.n_sentences for c in contexts} == {3}
    assert contexts[0].prev_text == ""
    assert contexts[1].prev_text == SENTENCES[0]
    assert contexts[1].next_text == SENTENCES[2]
    assert contexts[2].next_text == ""


def test_normalize_text_collapses_numbers() -> None:
    assert normalize_text("P < 0.05 in 1,250 Patients").split() == [
        "p",
        "<",
        "0",
        "in",
        "0",
        "patients",
    ]


def test_position_matrix_one_hot() -> None:
    X = position_matrix(build_contexts(SENTENCES)).toarray()
    assert X.shape[0] == 3
    # absolute position one-hot occupies the first columns
    assert X[0, 0] == X[1, 1] == X[2, 2] == 1
    is_first, is_last, relative = X[:, -3], X[:, -2], X[:, -1]
    assert is_first.tolist() == [1, 0, 0]
    assert is_last.tolist() == [0, 0, 1]
    np.testing.assert_allclose(relative, [0.0, 0.5, 1.0])
    # each row: abs, bin, length one-hots + is_first/last flags + relative
    assert X[:, :-1].sum(axis=1).tolist() == [4, 3, 4]


def test_position_matrix_single_sentence() -> None:
    X = position_matrix(build_contexts(["Only one ."])).toarray()
    assert X[0, -3] == X[0, -2] == 1  # first and last
    assert X[0, -1] == 0


def test_featurizer_blocks() -> None:
    contexts = build_contexts(SENTENCES) * 2
    base = AbstractFeaturizer(min_df=1, use_position=False, use_context=False).fit(contexts)
    full = AbstractFeaturizer(min_df=1).fit(contexts)
    n_vocab = len(base.vectorizer_.vocabulary_)
    assert base.transform(contexts).shape == (6, n_vocab)
    Xf = full.transform(contexts)
    assert Xf.shape[1] == 3 * n_vocab + position_matrix(contexts).shape[1]
    # first sentence has no previous sentence -> empty "prev" block
    assert Xf[0, n_vocab : 2 * n_vocab].nnz == 0
    assert Xf[1, n_vocab : 2 * n_vocab].nnz > 0
