"""Runtime configuration, read from environment variables (prefix ``BIONLP_``) or ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIONLP_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data/pubmed_20k_rct")
    models_dir: Path = Path("models")
    # Directory of the model served by the API (contains model.joblib).
    model_dir: Path = Path("models/lr_tfidf_position_context")

    host: str = "0.0.0.0"  # noqa: S104 - container deployment
    # Render (and most PaaS) inject PORT; BIONLP_PORT works too.
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "BIONLP_PORT"))
    log_level: str = "INFO"

    max_text_chars: int = 20_000
    max_sentences: int = 200
    max_batch_size: int = 32


def get_settings() -> Settings:
    return Settings()
