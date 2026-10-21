"""Shared fixtures: a tiny synthetic PubMed-RCT-formatted corpus and a model trained on it.

No test touches the network or the real dataset.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bionlp.api.app import create_app
from bionlp.config import Settings
from bionlp.data import Abstract, load_split
from bionlp.inference import AbstractStructurer
from bionlp.training import load_model, save_model, train_model

TEMPLATES: dict[str, list[str]] = {
    "BACKGROUND": [
        "{d} is a leading cause of morbidity worldwide .",
        "Little is known about the long-term burden of {d} in adults .",
        "Current treatments for {d} remain unsatisfactory .",
    ],
    "OBJECTIVE": [
        "We aimed to evaluate the efficacy of {t} in {d} .",
        "The aim of this study was to assess whether {t} reduces symptoms of {d} .",
    ],
    "METHODS": [
        "Patients were randomly assigned to {t} or placebo for {n} weeks .",
        "The primary outcome was measured at baseline and after {n} weeks .",
        "A total of {n} participants with {d} were enrolled .",
    ],
    "RESULTS": [
        "Symptoms improved significantly in the {t} group ( p < 0.0{n} ) .",
        "The mean difference between groups was {n} points ( 95 % CI ) .",
        "Adverse events occurred in {n} % of patients receiving {t} .",
    ],
    "CONCLUSIONS": [
        "{t} is effective and safe for patients with {d} .",
        "These findings support the use of {t} in clinical practice .",
    ],
}
DISEASES = ["asthma", "diabetes", "knee osteoarthritis", "depression", "hypertension"]
TREATMENTS = ["prednisolone", "metformin", "exercise therapy", "acupuncture", "vitamin D"]


def make_abstracts(n: int, seed: int) -> list[Abstract]:
    rng = random.Random(seed)
    abstracts = []
    for i in range(n):
        d, t = rng.choice(DISEASES), rng.choice(TREATMENTS)
        sentences, labels = [], []
        for label, templates in TEMPLATES.items():
            for template in rng.sample(templates, k=rng.randint(1, 2)):
                sentences.append(template.format(d=d, t=t, n=rng.randint(2, 9)))
                labels.append(label)
        abstracts.append(Abstract(f"{seed}{i:05d}", tuple(sentences), tuple(labels)))
    return abstracts


def to_pubmed_format(abstracts: list[Abstract]) -> str:
    blocks = []
    for a in abstracts:
        lines = [f"###{a.abstract_id}"] + [
            f"{label}\t{sentence}" for label, sentence in zip(a.labels, a.sentences, strict=True)
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    for split, (n, seed) in {"train": (120, 1), "dev": (30, 2), "test": (30, 3)}.items():
        (path / f"{split}.txt").write_text(to_pubmed_format(make_abstracts(n, seed)))
    return path


@pytest.fixture(scope="session")
def model_dir(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    trained, _ = train_model(
        "lr_tfidf_position_context", load_split(data_dir, "train"), load_split(data_dir, "dev")
    )
    return save_model(trained, tmp_path_factory.mktemp("models") / "lr_tfidf_position_context")


@pytest.fixture(scope="session")
def structurer(model_dir: Path) -> AbstractStructurer:
    return AbstractStructurer(load_model(model_dir))


@pytest.fixture
def settings(model_dir: Path, data_dir: Path) -> Settings:
    return Settings(model_dir=model_dir, data_dir=data_dir, max_text_chars=2_000, max_batch_size=3)


@pytest.fixture
def client(settings: Settings, structurer: AbstractStructurer) -> Iterator[TestClient]:
    with TestClient(create_app(settings, structurer)) as test_client:
        yield test_client
