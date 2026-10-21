from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bionlp.cli import app

runner = CliRunner()


def test_train_evaluate_predict(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "models"
    result = runner.invoke(
        app, ["train", "-m", "nb_tfidf", "-m", "lr_tfidf", "--data-dir", str(data_dir),
              "--output-dir", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "nb_tfidf" / "model.joblib").exists()

    result = runner.invoke(
        app, ["evaluate", str(out / "nb_tfidf"), str(out / "lr_tfidf"), "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "| `lr_tfidf` |" in result.output
    metrics = json.loads((out / "lr_tfidf" / "metrics_test.json").read_text())
    assert metrics["split"] == "test"
    assert (out / "lr_tfidf" / "MODEL_CARD.md").exists()

    result = runner.invoke(
        app,
        ["predict", "Asthma is common. We tested metformin.", "--model-dir", str(out / "lr_tfidf")],
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["sentences"]) == 2

    result = runner.invoke(app, ["predict", "-", "--model-dir", str(out / "lr_tfidf")], input="Hi.")
    assert json.loads(result.stdout)["sentences"][0]["text"] == "Hi."


def test_train_all(data_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["train", "-m", "all", "--data-dir", str(data_dir), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("*/model.joblib"))) == 4


def test_evaluate_rejects_unknown_split(model_dir: Path) -> None:
    result = runner.invoke(app, ["evaluate", str(model_dir), "--split", "train"])
    assert result.exit_code != 0


def test_tune(data_dir: Path) -> None:
    result = runner.invoke(
        app, ["tune", "-m", "nb_tfidf", "--param", "clf__alpha", "--values", "0.1,1.0",
              "--data-dir", str(data_dir)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "(best)" in result.output
    assert result.output.count("\n| ") == 3


def test_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        "bionlp.data.download_dataset",
        lambda data_dir, force: calls.append(data_dir) or {"train": data_dir / "train.txt"},
    )
    result = runner.invoke(app, ["download", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls == [tmp_path]


def test_serve_reads_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "uvicorn.run", lambda target, **kwargs: captured.update(kwargs, target=target)
    )
    monkeypatch.setenv("PORT", "10000")
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 10000
    assert captured["target"] == "bionlp.api.app:create_app"
    assert captured["factory"] is True
