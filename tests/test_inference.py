from __future__ import annotations

import pytest

from bionlp.data import LABELS
from bionlp.inference import AbstractStructurer

ABSTRACT = (
    "Asthma is a leading cause of morbidity worldwide. "
    "We aimed to evaluate the efficacy of metformin in asthma. "
    "Patients were randomly assigned to metformin or placebo for 6 weeks. "
    "Symptoms improved significantly in the metformin group (p < 0.05). "
    "Metformin is effective and safe for patients with asthma."
)


def test_structure_text(structurer: AbstractStructurer) -> None:
    result = structurer.structure_text(ABSTRACT)
    assert [s.label for s in result.sentences] == list(LABELS)
    assert [s.index for s in result.sentences] == [0, 1, 2, 3, 4]
    for sentence in result.sentences:
        assert list(sentence.probabilities) == list(LABELS)  # canonical order
        assert sum(sentence.probabilities.values()) == pytest.approx(1, abs=1e-3)
        assert sentence.confidence == max(sentence.probabilities.values())
    assert [s.label for s in result.sections] == list(LABELS)


def test_sections_merge_consecutive_sentences(structurer: AbstractStructurer) -> None:
    result = structurer.structure_sentences(
        [
            "Patients were randomly assigned to metformin or placebo for 6 weeks .",
            "The primary outcome was measured at baseline and after 6 weeks .",
        ]
    )
    labels = [s.label for s in result.sentences]
    assert len(result.sections) == len(set(labels)) if labels[0] == labels[1] else 2
    assert sum(len(s.sentence_indices) for s in result.sections) == 2


def test_structure_many_matches_single_calls(structurer: AbstractStructurer) -> None:
    a = ["Asthma is common .", "We tested metformin ."]
    b = ["Patients were enrolled .", "It worked ( p < 0.05 ) .", "Metformin is safe ."]
    batch = structurer.structure_many([a, b])
    assert batch == [structurer.structure_sentences(a), structurer.structure_sentences(b)]
    assert structurer.structure_many([]) == []
