from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bionlp.data import load_split
from bionlp.evaluation import comparison_markdown, evaluate_pipeline, render_model_card
from bionlp.features import SentenceContext
from bionlp.training import load_model


class AlwaysMethods:
    def predict(self, X: Sequence[SentenceContext]) -> np.ndarray:
        return np.array(["METHODS"] * len(X))


def test_metrics_of_constant_predictor(data_dir: Path) -> None:
    abstracts = load_split(data_dir, "test")
    report = evaluate_pipeline(AlwaysMethods(), abstracts, split="test")
    n_methods = sum(a.labels.count("METHODS") for a in abstracts)
    assert report.n_sentences == sum(len(a.sentences) for a in abstracts)
    assert report.accuracy == round(n_methods / report.n_sentences, 4)
    assert report.per_class["METHODS"].recall == 1.0
    assert report.per_class["RESULTS"].f1 == 0.0
    cm = np.array(report.confusion_matrix)
    assert cm.shape == (5, 5)
    assert cm.sum() == report.n_sentences
    methods = report.labels.index("METHODS")
    assert cm[:, methods].sum() == report.n_sentences  # every prediction in one column


def test_model_card_and_table(data_dir: Path, model_dir: Path) -> None:
    model = load_model(model_dir)
    report = evaluate_pipeline(model.pipeline, load_split(data_dir, "test"), split="test")
    card = render_model_card(model.metadata, report)
    assert "# Model card: `lr_tfidf_position_context`" in card
    assert f"{report.accuracy:.4f}" in card
    assert "| **BACKGROUND** |" in card
    table = comparison_markdown([("a", report), ("b", report)])
    assert table.count("\n") == 3
    assert report.sentences_per_second > 0
