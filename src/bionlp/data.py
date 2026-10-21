"""PubMed 20k RCT dataset: download (cached, checksum-verified) and parsing.

File format (one file per split)::

    ###24293578
    OBJECTIVE<TAB>To investigate the efficacy of ...
    METHODS<TAB>A total of 125 patients ...
    <blank line between abstracts>
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

LABELS: tuple[str, ...] = ("BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS")
Split = Literal["train", "dev", "test"]
SPLITS: tuple[Split, ...] = ("train", "dev", "test")

# Pinned to an immutable commit of https://github.com/Franck-Dernoncourt/pubmed-rct so that
# every training run (locally, in CI, in the Docker build) sees byte-identical data.
DATASET_COMMIT = "17ed2cb0590decfca0266add0c76f254f67232b4"
DATASET_BASE_URL = f"https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/{DATASET_COMMIT}/PubMed_20k_RCT"
DATASET_SHA256: dict[Split, str] = {
    "train": "8fda243202edd207fb61214e21f5629bba545522b921816d20d516e64d602d5b",
    "dev": "8561119a162708a87e3138103470d10781c6b94ba390183d7d6df6297a738a25",
    "test": "8511b832c1ee66b3df52df23c16ad81a69711ddd5b3a485421da3f8e9d5b44f4",
}


@dataclass(frozen=True, slots=True)
class Abstract:
    """One abstract: ordered sentences and their gold section labels."""

    abstract_id: str
    sentences: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.sentences) != len(self.labels):
            raise ValueError(f"abstract {self.abstract_id}: sentences/labels length mismatch")


def parse_pubmed_rct(lines: Iterable[str]) -> Iterator[Abstract]:
    """Parse PubMed RCT formatted lines into :class:`Abstract` objects."""
    abstract_id: str | None = None
    sentences: list[str] = []
    labels: list[str] = []

    def flush() -> Abstract | None:
        if abstract_id is not None and sentences:
            return Abstract(abstract_id, tuple(sentences), tuple(labels))
        return None

    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n").rstrip("\r")
        if line.startswith("###"):
            if (abstract := flush()) is not None:
                yield abstract
            abstract_id, sentences, labels = line[3:].strip(), [], []
        elif not line.strip():
            continue
        else:
            label, sep, sentence = line.partition("\t")
            if not sep or label not in LABELS:
                raise ValueError(f"line {lineno}: expected 'LABEL<TAB>sentence', got {line[:80]!r}")
            if abstract_id is None:
                raise ValueError(f"line {lineno}: sentence before any '###<id>' header")
            sentences.append(sentence.strip())
            labels.append(label)
    if (abstract := flush()) is not None:
        yield abstract


def load_split(data_dir: Path, split: Split) -> list[Abstract]:
    path = data_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bionlp download` first")
    with path.open(encoding="utf-8") as fh:
        abstracts = list(parse_pubmed_rct(fh))
    n_sentences = sum(len(a.sentences) for a in abstracts)
    logger.info("loaded %s: %d abstracts, %d sentences", split, len(abstracts), n_sentences)
    return abstracts


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    data_dir: Path,
    *,
    base_url: str = DATASET_BASE_URL,
    checksums: Mapping[Split, str] = DATASET_SHA256,
    splits: Iterable[Split] = SPLITS,
    force: bool = False,
    timeout: float = 60.0,
) -> dict[Split, Path]:
    """Download the dataset splits into ``data_dir``.

    Files already present with the expected checksum are reused (cache hit). Downloads go to a
    temporary file and are moved into place only once the checksum matches, so an interrupted
    or corrupted download never poisons the cache.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[Split, Path] = {}
    for split in splits:
        target = data_dir / f"{split}.txt"
        expected = checksums[split]
        if not force and target.exists() and sha256_of(target) == expected:
            logger.info("cache hit: %s", target)
            paths[split] = target
            continue
        url = f"{base_url.rstrip('/')}/{split}.txt"
        logger.info("downloading %s", url)
        with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
                    shutil.copyfileobj(response, tmp)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        actual = sha256_of(tmp_path)
        if actual != expected:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {url}: expected {expected}, got {actual}")
        tmp_path.replace(target)
        paths[split] = target
    return paths
