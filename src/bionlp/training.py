"""Training pipeline and model artifacts.

An artifact is a directory::

    models/<name>/
      model.joblib    # the fitted sklearn pipeline
      metadata.json   # what was trained, on what, how well it did on dev
"""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from bionlp import __version__
from bionlp.data import DATASET_COMMIT, LABELS, Abstract
from bionlp.evaluation import EvaluationReport, evaluate_pipeline
from bionlp.features import SentenceContext, build_contexts
from bionlp.models import get_spec

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ModelMetadata(BaseModel):
    name: str
    description: str
    labels: list[str]
    package_version: str
    sklearn_version: str
    python_version: str
    dataset: str
    trained_at: str
    train_seconds: float
    n_train_sentences: int
    dev_accuracy: float | None = None
    dev_macro_f1: float | None = None
    params: dict[str, str]


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metadata: ModelMetadata


def flatten(abstracts: Sequence[Abstract]) -> tuple[list[SentenceContext], np.ndarray]:
    """Abstracts -> one sample per sentence (with its abstract context) and its label."""
    contexts: list[SentenceContext] = []
    labels: list[str] = []
    for abstract in abstracts:
        contexts.extend(build_contexts(abstract.sentences))
        labels.extend(abstract.labels)
    return contexts, np.asarray(labels)


def train_model(
    name: str,
    train: Sequence[Abstract],
    dev: Sequence[Abstract] | None = None,
    *,
    params: dict[str, object] | None = None,
) -> tuple[TrainedModel, EvaluationReport | None]:
    """Fit registry model ``name`` on ``train``; report dev metrics when ``dev`` is given."""
    spec = get_spec(name)
    pipeline = spec.build()
    if params:
        pipeline.set_params(**params)
    X, y = flatten(train)
    logger.info("training %s on %d sentences", name, len(y))
    start = time.perf_counter()
    pipeline.fit(X, y)
    elapsed = time.perf_counter() - start
    logger.info("trained %s in %.1fs", name, elapsed)

    report = evaluate_pipeline(pipeline, dev) if dev else None
    if report is not None:
        logger.info("dev: accuracy=%.4f macro_f1=%.4f", report.accuracy, report.macro_f1)
    metadata = ModelMetadata(
        name=name,
        description=spec.description,
        labels=list(LABELS),
        package_version=__version__,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        dataset=f"PubMed 20k RCT @ {DATASET_COMMIT[:10]}",
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        train_seconds=round(elapsed, 2),
        n_train_sentences=len(y),
        dev_accuracy=report.accuracy if report else None,
        dev_macro_f1=report.macro_f1 if report else None,
        params={k: repr(v) for k, v in (params or {}).items()},
    )
    return TrainedModel(pipeline, metadata), report


def save_model(model: TrainedModel, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, output_dir / MODEL_FILE, compress=3)
    (output_dir / METADATA_FILE).write_text(model.metadata.model_dump_json(indent=2) + "\n")
    logger.info("saved model to %s", output_dir)
    return output_dir


def load_model(model_dir: Path) -> TrainedModel:
    """Load an artifact directory. joblib unpickles code: only load artifacts you built."""
    model_path = model_dir / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found; run `bionlp train` first")
    metadata = ModelMetadata.model_validate(json.loads((model_dir / METADATA_FILE).read_text()))
    pipeline = joblib.load(model_path)
    return TrainedModel(pipeline, metadata)
