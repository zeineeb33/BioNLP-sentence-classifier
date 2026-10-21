"""Model registry: every trainable model is a named, fully specified sklearn pipeline.

The four entries form an ablation, from the original notebook baseline to the served model.
Regularisation strengths were picked on the dev split (``bionlp tune``); test is never used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from bionlp.features import AbstractFeaturizer

SEED = 13


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    build: Callable[[], Pipeline]


def _logreg(alpha: float = 1e-6) -> SGDClassifier:
    # One-vs-rest logistic regression trained with SGD: ~10x faster than lbfgs on
    # 180k x 900k sparse features for about the same dev accuracy, which keeps the
    # in-Docker training step short. log_loss gives usable confidence scores.
    return SGDClassifier(loss="log_loss", alpha=alpha, max_iter=20, tol=None, random_state=SEED)


def _nb_tfidf() -> Pipeline:
    featurizer = AbstractFeaturizer(
        ngram_range=(1, 1),
        min_df=1,
        sublinear_tf=False,
        normalize_numbers=False,
        use_position=False,
        use_context=False,
    )
    return Pipeline([("features", featurizer), ("clf", MultinomialNB(alpha=0.1))])


def _lr_tfidf() -> Pipeline:
    featurizer = AbstractFeaturizer(use_position=False, use_context=False)
    return Pipeline([("features", featurizer), ("clf", _logreg())])


def _lr_tfidf_position() -> Pipeline:
    featurizer = AbstractFeaturizer(use_position=True, use_context=False)
    return Pipeline([("features", featurizer), ("clf", _logreg())])


def _lr_tfidf_position_context() -> Pipeline:
    featurizer = AbstractFeaturizer(use_position=True, use_context=True, context_weight=0.5)
    return Pipeline([("features", featurizer), ("clf", _logreg())])


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "nb_tfidf",
            "TF-IDF unigrams + Multinomial Naive Bayes (original notebook baseline)",
            _nb_tfidf,
        ),
        ModelSpec(
            "lr_tfidf",
            "TF-IDF 1-2 grams + logistic regression, sentence text only",
            _lr_tfidf,
        ),
        ModelSpec(
            "lr_tfidf_position",
            "TF-IDF 1-2 grams + sentence position features + logistic regression",
            _lr_tfidf_position,
        ),
        ModelSpec(
            "lr_tfidf_position_context",
            "TF-IDF 1-2 grams + position features + neighbouring-sentence TF-IDF "
            "+ logistic regression (served model)",
            _lr_tfidf_position_context,
        ),
    )
}

DEFAULT_MODEL = "lr_tfidf_position_context"


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        known = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"unknown model {name!r}; expected one of: {known}") from None
