from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bionlp.api.app import create_app
from bionlp.config import Settings
from tests.test_inference import ABSTRACT


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "lr_tfidf_position_context"


def test_index_serves_ui(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/v1/structure" in response.text


def test_model_info(client: TestClient, settings: Settings) -> None:
    info = client.get("/v1/model").json()
    assert info["labels"][0] == "BACKGROUND"
    assert info["test_accuracy"] is None
    (settings.model_dir / "metrics_test.json").write_text(
        json.dumps({"accuracy": 0.9, "macro_f1": 0.8})
    )
    try:
        info = client.get("/v1/model").json()
    finally:
        (settings.model_dir / "metrics_test.json").unlink()
    assert (info["test_accuracy"], info["test_macro_f1"]) == (0.9, 0.8)


def test_structure_text(client: TestClient) -> None:
    response = client.post("/v1/structure", json={"text": ABSTRACT})
    assert response.status_code == 200
    body = response.json()
    assert [s["label"] for s in body["sentences"]] == [
        "BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS",
    ]  # fmt: skip
    assert body["model"] == "lr_tfidf_position_context"
    assert "X-Process-Time-Ms" in response.headers


def test_structure_presplit_sentences(client: TestClient) -> None:
    response = client.post(
        "/v1/structure", json={"sentences": ["Asthma is common .", "It worked ."]}
    )
    assert response.status_code == 200
    assert [s["text"] for s in response.json()["sentences"]] == [
        "Asthma is common .",
        "It worked .",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ABSTRACT, "sentences": ["x"]},
        {"text": ""},
        {"sentences": []},
        {"sentences": ["ok", "  "]},
        {"text": "x" * 2_001},  # settings.max_text_chars = 2000
        {"text": "... ?!"},  # no sentence left after splitting
    ],
)
def test_structure_rejects_invalid_input(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/v1/structure", json=payload).status_code == 422


def test_structure_rejects_too_many_sentences(client: TestClient, settings: Settings) -> None:
    sentences = ["A sentence ."] * (settings.max_sentences + 1)
    response = client.post("/v1/structure", json={"sentences": sentences})
    assert response.status_code == 422
    assert "too many sentences" in response.json()["detail"]


def test_batch(client: TestClient) -> None:
    response = client.post(
        "/v1/structure/batch", json={"abstracts": [ABSTRACT, "Asthma is common."]}
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert len(results[0]["sentences"]) == 5
    assert len(results[1]["sentences"]) == 1


def test_batch_limits(client: TestClient) -> None:
    assert client.post("/v1/structure/batch", json={"abstracts": []}).status_code == 422
    too_many = {"abstracts": ["Asthma is common."] * 4}  # settings.max_batch_size = 3
    assert client.post("/v1/structure/batch", json=too_many).status_code == 422


def test_openapi_documents_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/v1/model", "/v1/structure", "/v1/structure/batch"} <= set(paths)


def test_model_loaded_at_startup(model_dir: Path) -> None:
    with TestClient(create_app(Settings(model_dir=model_dir))) as client:
        assert client.get("/health").status_code == 200


def test_startup_fails_fast_without_model(tmp_path: Path) -> None:
    with (
        pytest.raises(FileNotFoundError),
        TestClient(create_app(Settings(model_dir=tmp_path))),
    ):
        pass  # pragma: no cover
