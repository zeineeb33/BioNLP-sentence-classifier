# syntax=docker/dockerfile:1
# Multi-stage build: deps -> train model on PubMed 20k RCT -> slim non-root runtime.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Training is CPU-only and takes ~1-2 min, so the image is reproducible from the repo alone.
FROM builder AS trainer
ENV OMP_NUM_THREADS=1
RUN /app/.venv/bin/bionlp download --data-dir /data \
 && /app/.venv/bin/bionlp train -m lr_tfidf_position_context --data-dir /data --output-dir /models \
 && /app/.venv/bin/bionlp evaluate /models/lr_tfidf_position_context --split test --data-dir /data

FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /app/.venv /app/.venv
COPY --from=trainer /models/lr_tfidf_position_context /app/model
ENV PATH=/app/.venv/bin:$PATH \
    BIONLP_MODEL_DIR=/app/model \
    PORT=8000
USER app
EXPOSE 8000
# Render injects $PORT; `bionlp serve` reads it.
CMD ["bionlp", "serve"]
