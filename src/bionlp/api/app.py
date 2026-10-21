"""FastAPI app: JSON API + a single-page web UI, served by one process."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse

from bionlp import __version__
from bionlp.api.schemas import (
    BatchStructureRequest,
    BatchStructureResponse,
    HealthResponse,
    ModelInfo,
    StructureRequest,
    StructureResponse,
)
from bionlp.config import Settings, get_settings
from bionlp.inference import AbstractStructurer
from bionlp.logging_utils import configure_logging
from bionlp.segmentation import split_sentences

logger = logging.getLogger("bionlp.api")


def create_app(
    settings: Settings | None = None, structurer: AbstractStructurer | None = None
) -> FastAPI:
    """App factory. ``structurer`` can be injected (tests); otherwise it is loaded at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if structurer is None:
            start = time.perf_counter()
            app.state.structurer = AbstractStructurer.from_dir(settings.model_dir)
            logger.info("model ready in %.1fs", time.perf_counter() - start)
        else:
            app.state.structurer = structurer
        yield

    app = FastAPI(
        title="BioNLP Abstract Structurer",
        version=__version__,
        description=(
            "Split a biomedical abstract into sentences and label each one as BACKGROUND, "
            "OBJECTIVE, METHODS, RESULTS or CONCLUSIONS. Model trained on PubMed 20k RCT."
        ),
        lifespan=lifespan,
    )
    index_html = files("bionlp").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if request.url.path != "/health":
            logger.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    def get_structurer(request: Request) -> AbstractStructurer:
        return request.app.state.structurer

    def check_sentences(sentences: list[str]) -> list[str]:
        if not sentences:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "no sentence found in input")
        if len(sentences) > settings.max_sentences:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"too many sentences ({len(sentences)} > {settings.max_sentences})",
            )
        return sentences

    def check_text(text: str) -> list[str]:
        if len(text) > settings.max_text_chars:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"text too long ({len(text)} > {settings.max_text_chars} characters)",
            )
        return check_sentences(split_sentences(text))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        """Liveness/readiness probe (used by Render's health check)."""
        return HealthResponse(model=get_structurer(request).metadata.name, version=__version__)

    @app.get("/v1/model", response_model=ModelInfo, tags=["model"])
    def model_info(request: Request) -> ModelInfo:
        """Metadata and evaluation metrics of the served model."""
        meta = get_structurer(request).metadata
        info = ModelInfo(**meta.model_dump())
        metrics_path = settings.model_dir / "metrics_test.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            info.test_accuracy, info.test_macro_f1 = metrics["accuracy"], metrics["macro_f1"]
        return info

    @app.post("/v1/structure", response_model=StructureResponse, tags=["structure"])
    def structure(payload: StructureRequest, request: Request) -> StructureResponse:
        """Structure one abstract: split into sentences (if raw text) and label each one."""
        structurer = get_structurer(request)
        if payload.text is not None:
            sentences = check_text(payload.text)
        else:
            sentences = check_sentences(payload.sentences or [])
        result = structurer.structure_sentences(sentences)
        return StructureResponse(**result.model_dump(), model=structurer.metadata.name)

    @app.post("/v1/structure/batch", response_model=BatchStructureResponse, tags=["structure"])
    def structure_batch(payload: BatchStructureRequest, request: Request) -> BatchStructureResponse:
        """Structure several raw abstracts in one vectorised model call."""
        if len(payload.abstracts) > settings.max_batch_size:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"too many abstracts ({len(payload.abstracts)} > {settings.max_batch_size})",
            )
        structurer = get_structurer(request)
        name = structurer.metadata.name
        results = structurer.structure_many([check_text(text) for text in payload.abstracts])
        return BatchStructureResponse(
            results=[StructureResponse(**r.model_dump(), model=name) for r in results],
            model=name,
        )

    return app
