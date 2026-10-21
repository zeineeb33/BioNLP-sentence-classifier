"""Evaluation: metrics report, metrics JSON and a short model card."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from bionlp.data import LABELS, Abstract
from bionlp.features import SentenceContext, build_contexts

if TYPE_CHECKING:
    from bionlp.training import ModelMetadata


class Predictor(Protocol):
    def predict(self, X: Sequence[SentenceContext]) -> np.ndarray: ...


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationReport(BaseModel):
    split: str
    n_abstracts: int
    n_sentences: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, ClassMetrics]
    labels: list[str]
    confusion_matrix: list[list[int]]  # rows = gold, columns = predicted, order = labels
    predict_seconds: float

    @property
    def sentences_per_second(self) -> float:
        return self.n_sentences / self.predict_seconds if self.predict_seconds else float("inf")


def evaluate_pipeline(
    pipeline: Predictor, abstracts: Sequence[Abstract], *, split: str = "dev"
) -> EvaluationReport:
    contexts: list[SentenceContext] = []
    gold: list[str] = []
    for abstract in abstracts:
        contexts.extend(build_contexts(abstract.sentences))
        gold.extend(abstract.labels)
    start = time.perf_counter()
    predicted = pipeline.predict(contexts)
    elapsed = time.perf_counter() - start

    labels = list(LABELS)
    report = classification_report(
        gold, predicted, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        label: ClassMetrics(
            precision=round(report[label]["precision"], 4),
            recall=round(report[label]["recall"], 4),
            f1=round(report[label]["f1-score"], 4),
            support=int(report[label]["support"]),
        )
        for label in labels
    }
    return EvaluationReport(
        split=split,
        n_abstracts=len(abstracts),
        n_sentences=len(gold),
        accuracy=round(float(accuracy_score(gold, predicted)), 4),
        macro_f1=round(float(f1_score(gold, predicted, labels=labels, average="macro")), 4),
        weighted_f1=round(float(f1_score(gold, predicted, labels=labels, average="weighted")), 4),
        per_class=per_class,
        labels=labels,
        confusion_matrix=confusion_matrix(gold, predicted, labels=labels).tolist(),
        predict_seconds=round(elapsed, 3),
    )


def confusion_matrix_markdown(report: EvaluationReport) -> str:
    header = "| gold \\ predicted | " + " | ".join(report.labels) + " |"
    sep = "|---" * (len(report.labels) + 1) + "|"
    rows = [
        f"| **{label}** | " + " | ".join(str(v) for v in row) + " |"
        for label, row in zip(report.labels, report.confusion_matrix, strict=True)
    ]
    return "\n".join([header, sep, *rows])


def per_class_markdown(report: EvaluationReport) -> str:
    lines = ["| label | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for label, m in report.per_class.items():
        lines.append(f"| {label} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.support} |")
    return "\n".join(lines)


def comparison_markdown(rows: Sequence[tuple[str, EvaluationReport]]) -> str:
    lines = [
        "| model | accuracy | macro-F1 | weighted-F1 | sentences/s |",
        "|---|---|---|---|---|",
    ]
    for name, r in rows:
        lines.append(
            f"| `{name}` | {r.accuracy:.4f} | {r.macro_f1:.4f} | {r.weighted_f1:.4f} "
            f"| {r.sentences_per_second:,.0f} |"
        )
    return "\n".join(lines)


def render_model_card(metadata: ModelMetadata, report: EvaluationReport) -> str:
    return f"""# Model card: `{metadata.name}`

{metadata.description}.

## Intended use
Label each sentence of an English biomedical abstract (randomised controlled trials) with its
rhetorical section: {", ".join(metadata.labels)}. Assistive structuring and search only:
not validated for clinical decision making.

## Training
- Data: {metadata.dataset}, train split ({metadata.n_train_sentences:,} sentences)
- Trained: {metadata.trained_at} in {metadata.train_seconds:.0f}s (CPU), scikit-learn \
{metadata.sklearn_version}, Python {metadata.python_version}
- Dev split: accuracy {_fmt(metadata.dev_accuracy)}, macro-F1 {_fmt(metadata.dev_macro_f1)}

## Evaluation ({report.split} split: {report.n_abstracts:,} abstracts, \
{report.n_sentences:,} sentences)
- Accuracy **{report.accuracy:.4f}**, macro-F1 **{report.macro_f1:.4f}**, weighted-F1 \
{report.weighted_f1:.4f}
- Throughput: {report.sentences_per_second:,.0f} sentences/s (batch, single process)

{per_class_markdown(report)}

Confusion matrix:

{confusion_matrix_markdown(report)}

## Limitations
- Trained on RCT abstracts only; other study types (reviews, case reports) are out of domain.
- BACKGROUND vs OBJECTIVE is the main confusion: both open the abstract.
- Served on raw user text, sentence boundaries come from a rule-based splitter, while the
  test set uses gold sentence boundaries.
"""


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
