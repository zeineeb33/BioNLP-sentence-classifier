"""BioNLP abstract structurer: download data, train, tune, evaluate, predict, serve."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from bionlp.config import get_settings
from bionlp.logging_utils import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
logger = logging.getLogger("bionlp")

DataDir = Annotated[Path | None, typer.Option(help="Dataset directory [default: settings]")]


@app.callback()
def _main(
    log_level: Annotated[str | None, typer.Option(help="Logging level")] = None,
) -> None:
    configure_logging(log_level or get_settings().log_level)


@app.command()
def download(data_dir: DataDir = None, force: bool = False) -> None:
    """Download PubMed 20k RCT (pinned commit, SHA-256 verified, cached)."""
    from bionlp.data import download_dataset

    paths = download_dataset(data_dir or get_settings().data_dir, force=force)
    for split, path in paths.items():
        logger.info("%s -> %s", split, path)


@app.command()
def train(
    model: Annotated[
        list[str], typer.Option("--model", "-m", help="Registry name(s), or 'all'")
    ] = ["lr_tfidf_position_context"],  # noqa: B006 - typer needs a literal default
    data_dir: DataDir = None,
    output_dir: Annotated[Path | None, typer.Option(help="Where model dirs go")] = None,
) -> None:
    """Train registry model(s) on the train split; report dev metrics; save artifacts."""
    from bionlp.data import load_split
    from bionlp.models import MODEL_REGISTRY
    from bionlp.training import save_model, train_model

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    output_dir = output_dir or settings.models_dir
    names = list(MODEL_REGISTRY) if model == ["all"] else model
    train_split, dev_split = load_split(data_dir, "train"), load_split(data_dir, "dev")
    for name in names:
        trained, _ = train_model(name, train_split, dev_split)
        save_model(trained, output_dir / name)


@app.command()
def tune(
    model: Annotated[str, typer.Option("--model", "-m")],
    param: Annotated[str, typer.Option(help="Pipeline parameter, e.g. clf__alpha")],
    values: Annotated[str, typer.Option(help="Comma-separated values (parsed as JSON)")],
    data_dir: DataDir = None,
) -> None:
    """Grid-search one hyperparameter: fit on train, score on dev (test is never used)."""
    from bionlp.data import load_split
    from bionlp.training import train_model

    data_dir = data_dir or get_settings().data_dir
    train_split, dev_split = load_split(data_dir, "train"), load_split(data_dir, "dev")
    results = []
    for raw in values.split(","):
        value = json.loads(raw)
        _, report = train_model(model, train_split, dev_split, params={param: value})
        if report is not None:
            results.append((value, report))
    best = max(results, key=lambda r: r[1].macro_f1)
    typer.echo(f"| {param} | dev accuracy | dev macro-F1 |\n|---|---|---|")
    for value, report in results:
        marker = " (best)" if value == best[0] else ""
        typer.echo(f"| {value}{marker} | {report.accuracy:.4f} | {report.macro_f1:.4f} |")


@app.command()
def evaluate(
    model_dirs: Annotated[list[Path], typer.Argument(help="Model artifact directories")],
    split: Annotated[str, typer.Option(help="dev or test")] = "test",
    data_dir: DataDir = None,
) -> None:
    """Evaluate artifacts; write metrics_<split>.json + MODEL_CARD.md next to each model."""
    from bionlp.data import load_split
    from bionlp.evaluation import comparison_markdown, evaluate_pipeline, render_model_card
    from bionlp.training import load_model

    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be 'dev' or 'test'")
    abstracts = load_split(data_dir or get_settings().data_dir, split)  # type: ignore[arg-type]
    rows = []
    for model_dir in model_dirs:
        model = load_model(model_dir)
        report = evaluate_pipeline(model.pipeline, abstracts, split=split)
        (model_dir / f"metrics_{split}.json").write_text(report.model_dump_json(indent=2) + "\n")
        (model_dir / "MODEL_CARD.md").write_text(render_model_card(model.metadata, report))
        logger.info(
            "%s %s: accuracy=%.4f macro_f1=%.4f",
            model.metadata.name,
            split,
            report.accuracy,
            report.macro_f1,
        )
        rows.append((model.metadata.name, report))
    typer.echo(comparison_markdown(rows))


@app.command()
def predict(
    text: Annotated[str, typer.Argument(help="Abstract text, or '-' to read stdin")],
    model_dir: Annotated[Path | None, typer.Option(help="Model directory")] = None,
) -> None:
    """Structure one abstract and print the JSON result."""
    import sys

    from bionlp.inference import AbstractStructurer

    structurer = AbstractStructurer.from_dir(model_dir or get_settings().model_dir)
    raw = sys.stdin.read() if text == "-" else text
    typer.echo(json.dumps(structurer.structure_text(raw).model_dump(), indent=2))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Run the API + web UI with uvicorn (port defaults to $PORT)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "bionlp.api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
