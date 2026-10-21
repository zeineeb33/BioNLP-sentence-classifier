from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bionlp.data import LABELS, Abstract, load_split
from bionlp.models import MODEL_REGISTRY, get_spec
from bionlp.training import flatten, load_model, save_model, train_model


def test_flatten(data_dir: Path) -> None:
    abstracts = load_split(data_dir, "dev")
    X, y = flatten(abstracts)
    assert len(X) == len(y) == sum(len(a.sentences) for a in abstracts)
    assert set(y) == set(LABELS)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_every_registry_model_trains(name: str, data_dir: Path) -> None:
    model, report = train_model(name, load_split(data_dir, "train"), load_split(data_dir, "dev"))
    assert report is not None
    assert report.accuracy > 0.7  # synthetic data is easy; this catches broken pipelines
    assert model.metadata.name == name
    assert model.metadata.dev_accuracy == report.accuracy


def test_position_features_help_on_ambiguous_text() -> None:
    """Identical sentences in different positions can only be told apart with position."""
    train = [
        Abstract(
            str(i), ("Results were examined .", "Results were examined ."), ("METHODS", "RESULTS")
        )
        for i in range(30)
    ]
    _, plain = train_model("lr_tfidf", train, train)
    _, with_position = train_model("lr_tfidf_position", train, train)
    assert plain is not None
    assert with_position is not None
    assert plain.accuracy == 0.5
    assert with_position.accuracy == 1.0


def test_params_override(data_dir: Path) -> None:
    model, report = train_model(
        "nb_tfidf", load_split(data_dir, "train"), None, params={"clf__alpha": 0.5}
    )
    assert report is None
    assert model.pipeline.get_params()["clf__alpha"] == 0.5
    assert model.metadata.params == {"clf__alpha": "0.5"}
    assert model.metadata.dev_accuracy is None


def test_save_load_roundtrip(data_dir: Path, tmp_path: Path) -> None:
    model, _ = train_model("lr_tfidf_position_context", load_split(data_dir, "train"))
    save_model(model, tmp_path / "m")
    loaded = load_model(tmp_path / "m")
    X, _ = flatten(load_split(data_dir, "test"))
    np.testing.assert_array_equal(model.pipeline.predict(X), loaded.pipeline.predict(X))
    assert loaded.metadata == model.metadata


def test_load_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bionlp train"):
        load_model(tmp_path)


def test_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_spec("bert")
