from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pytest

from bionlp.data import Abstract, download_dataset, load_split, parse_pubmed_rct

SAMPLE = """###111
OBJECTIVE\tTo test X .
METHODS\tWe did Y .

###222
BACKGROUND\tZ is common .
RESULTS\tIt worked ( p < 0.05 ) .
CONCLUSIONS\tX is good .
"""


def test_parse_pubmed_rct() -> None:
    abstracts = list(parse_pubmed_rct(SAMPLE.splitlines(keepends=True)))
    assert [a.abstract_id for a in abstracts] == ["111", "222"]
    assert abstracts[0].labels == ("OBJECTIVE", "METHODS")
    assert abstracts[1].sentences[1] == "It worked ( p < 0.05 ) ."


@pytest.mark.parametrize(
    ("text", "message"),
    [("###1\nFOO\tbar\n", "expected 'LABEL<TAB>sentence'"), ("METHODS\tx\n", "before any")],
)
def test_parse_rejects_malformed_lines(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(parse_pubmed_rct(text.splitlines()))


def test_abstract_validates_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        Abstract("1", ("a", "b"), ("METHODS",))


def test_load_split(data_dir: Path) -> None:
    abstracts = load_split(data_dir, "dev")
    assert len(abstracts) == 30
    with pytest.raises(FileNotFoundError, match="bionlp download"):
        load_split(data_dir / "missing", "train")


@pytest.fixture
def source(tmp_path: Path) -> tuple[str, dict[str, str]]:
    src = tmp_path / "remote"
    src.mkdir()
    checksums = {}
    for split in ("train", "dev", "test"):
        content = SAMPLE.replace("###", f"###{split}").encode()
        (src / f"{split}.txt").write_bytes(content)
        checksums[split] = hashlib.sha256(content).hexdigest()
    return src.as_uri(), checksums


def test_download_and_cache(
    source: tuple[str, dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, checksums = source
    target = tmp_path / "data"
    paths = download_dataset(target, base_url=base_url, checksums=checksums)  # type: ignore[arg-type]
    assert sorted(paths) == ["dev", "test", "train"]
    assert len(load_split(target, "train")) == 2
    assert not list(target.glob("*.part"))

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("cache miss: tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    download_dataset(target, base_url=base_url, checksums=checksums)  # type: ignore[arg-type]


def test_download_rejects_bad_checksum(source: tuple[str, dict[str, str]], tmp_path: Path) -> None:
    base_url, checksums = source
    bad = {**checksums, "train": "0" * 64}
    target = tmp_path / "data"
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_dataset(target, base_url=base_url, checksums=bad, splits=["train"])  # type: ignore[arg-type]
    assert not list(target.iterdir())  # nothing half-written left in the cache


def test_download_cleans_up_on_network_error(tmp_path: Path) -> None:
    target = tmp_path / "data"
    with pytest.raises(OSError):  # noqa: PT011 - URLError subclasses OSError
        download_dataset(
            target,
            base_url=(tmp_path / "nowhere").as_uri(),
            checksums={"train": "0" * 64},  # type: ignore[dict-item]
            splits=["train"],
        )
    assert not list(target.iterdir())
