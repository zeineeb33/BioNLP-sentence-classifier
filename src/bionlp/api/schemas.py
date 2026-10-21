"""Request/response models of the HTTP API (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from bionlp.inference import StructuredAbstract

EXAMPLE_ABSTRACT = (
    "Emotional eating is associated with overeating and the development of obesity. "
    "The aim of this study was to test if attention bias for food moderates the effect of "
    "self-reported emotional eating during sad mood on actual food intake. "
    "Participants (N = 85) were randomly assigned to a sad or a neutral mood induction. "
    "Food intake was significantly higher after the sad mood induction (p < 0.05). "
    "Attention bias for food may help explain why some people overeat when they feel sad."
)


class StructureRequest(BaseModel):
    """Either a raw abstract (`text`, split server-side) or pre-split `sentences`."""

    text: str | None = Field(default=None, min_length=1, examples=[EXAMPLE_ABSTRACT])
    sentences: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.text is None) == (self.sentences is None):
            raise ValueError("provide exactly one of 'text' or 'sentences'")
        if self.sentences is not None and any(not s.strip() for s in self.sentences):
            raise ValueError("sentences must be non-empty strings")
        return self


class StructureResponse(StructuredAbstract):
    model: str


class BatchStructureRequest(BaseModel):
    abstracts: list[str] = Field(min_length=1, examples=[[EXAMPLE_ABSTRACT]])


class BatchStructureResponse(BaseModel):
    results: list[StructureResponse]
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    version: str


class ModelInfo(BaseModel):
    name: str
    description: str
    labels: list[str]
    trained_at: str
    dataset: str
    dev_accuracy: float | None
    dev_macro_f1: float | None
    test_accuracy: float | None = None
    test_macro_f1: float | None = None
