"""Rule-based sentence splitter for biomedical abstracts.

Dependency-free on purpose (no NLTK/spaCy models to download into a 512 MB container).
It handles what actually shows up in abstracts: decimals ("p < 0.05"), common
abbreviations ("vs.", "e.g.", "et al.") and pasted section headings ("METHODS:").
"""

from __future__ import annotations

import re

# Lower-cased tokens that end with a period without ending the sentence. Deliberately short:
# units ("ms."), "etc." and single letters ("vitamin D.") often *do* end a sentence.
ABBREVIATIONS = frozenset(
    {"al", "approx", "cf", "e.g", "eg", "fig", "figs", "i.e", "ie", "no", "nos", "resp", "vs"}
)

# Headings of structured abstracts ("BACKGROUND:", "Methods and Results:") are removed so the
# model has to infer the section from the sentence itself.
_HEADING_RE = re.compile(
    r"""^\s*(?:background|introduction|context|objectives?|aims?|purpose|goals?|
    methods?|materials?\s+and\s+methods|design|setting|participants|patients|
    interventions?|measurements|main\s+outcome\s+measures?|methods\s+and\s+results|
    results?|findings|conclusions?|interpretation|discussion|significance|
    trial\s+registration|clinical\s+trial\s+registration)\s*:\s*""",
    re.IGNORECASE | re.VERBOSE,
)
# Candidate boundary: terminal punctuation, optional closing quote/bracket, whitespace,
# then something that can start a sentence.
_BOUNDARY_RE = re.compile(r"""([.!?])["')\]]*\s+(?=["'(\[]?[A-Z0-9])""")
_WHITESPACE_RE = re.compile(r"\s+")
_LAST_TOKEN_RE = re.compile(r"(\S+)$")


def _is_abbreviation(text_before: str) -> bool:
    match = _LAST_TOKEN_RE.search(text_before)
    if not match:
        return False
    return match.group(1).lstrip("([\"'").lower() in ABBREVIATIONS


def split_sentences(text: str, *, strip_headings: bool = True) -> list[str]:
    """Split a raw abstract into sentences, preserving order."""
    sentences: list[str] = []
    # Blank lines are hard boundaries (paragraphs / structured sections).
    for paragraph in re.split(r"\n\s*\n", text):
        block = _WHITESPACE_RE.sub(" ", paragraph).strip()
        if not block:
            continue
        start = 0
        for match in _BOUNDARY_RE.finditer(block):
            end = match.start(1)
            if match.group(1) == "." and _is_abbreviation(block[start:end]):
                continue
            sentences.append(block[start : match.end()].strip())
            start = match.end()
        sentences.append(block[start:].strip())
    if strip_headings:
        sentences = [_HEADING_RE.sub("", s, count=1) for s in sentences]
    return [s for s in sentences if any(ch.isalnum() for ch in s)]
