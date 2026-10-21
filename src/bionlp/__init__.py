"""Biomedical abstract structuring: sentence-level section classification on PubMed RCT."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bionlp-sentence-classifier")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"
