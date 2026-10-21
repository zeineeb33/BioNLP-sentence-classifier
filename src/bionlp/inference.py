"""Inference: raw abstract -> sentences -> section labels -> structured abstract."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from bionlp.data import LABELS
from bionlp.features import SentenceContext, build_contexts
from bionlp.segmentation import split_sentences
from bionlp.training import ModelMetadata, TrainedModel, load_model

logger = logging.getLogger(__name__)


class SentencePrediction(BaseModel):
    index: int
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class Section(BaseModel):
    """A run of consecutive sentences sharing a label."""

    label: str
    text: str
    sentence_indices: list[int]


class StructuredAbstract(BaseModel):
    sentences: list[SentencePrediction]
    sections: list[Section]


class AbstractStructurer:
    def __init__(self, model: TrainedModel) -> None:
        self.pipeline = model.pipeline
        self.metadata: ModelMetadata = model.metadata
        self.classes: list[str] = [str(c) for c in self.pipeline.classes_]
        # Report probabilities in canonical section order, not sklearn's alphabetical order.
        self._order = sorted(range(len(self.classes)), key=lambda i: _rank(self.classes[i]))

    @classmethod
    def from_dir(cls, model_dir: Path) -> AbstractStructurer:
        logger.info("loading model from %s", model_dir)
        return cls(load_model(model_dir))

    def structure_text(self, text: str) -> StructuredAbstract:
        return self.structure_many([split_sentences(text)])[0]

    def structure_sentences(self, sentences: Sequence[str]) -> StructuredAbstract:
        return self.structure_many([sentences])[0]

    def structure_many(self, abstracts: Sequence[Sequence[str]]) -> list[StructuredAbstract]:
        """Label several pre-split abstracts with a single vectorised model call."""
        contexts: list[SentenceContext] = []
        for sentences in abstracts:
            contexts.extend(build_contexts([s.strip() for s in sentences]))
        probas = self.pipeline.predict_proba(contexts) if contexts else np.empty((0, 0))

        results: list[StructuredAbstract] = []
        offset = 0
        for sentences in abstracts:
            rows = probas[offset : offset + len(sentences)]
            offset += len(sentences)
            predictions = [
                self._prediction(i, text, row)
                for i, (text, row) in enumerate(zip(sentences, rows, strict=True))
            ]
            results.append(StructuredAbstract(sentences=predictions, sections=_group(predictions)))
        return results

    def _prediction(self, index: int, text: str, row: np.ndarray) -> SentencePrediction:
        best = int(row.argmax())
        return SentencePrediction(
            index=index,
            text=text.strip(),
            label=self.classes[best],
            confidence=round(float(row[best]), 4),
            probabilities={self.classes[i]: round(float(row[i]), 4) for i in self._order},
        )


def _rank(label: str) -> int:
    return LABELS.index(label) if label in LABELS else len(LABELS)


def _group(predictions: Sequence[SentencePrediction]) -> list[Section]:
    sections: list[Section] = []
    for p in predictions:
        if sections and sections[-1].label == p.label:
            sections[-1].text += " " + p.text
            sections[-1].sentence_indices.append(p.index)
        else:
            sections.append(Section(label=p.label, text=p.text, sentence_indices=[p.index]))
    return sections
