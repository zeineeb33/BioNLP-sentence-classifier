from __future__ import annotations

import pytest

from bionlp.segmentation import split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("   \n ", []),
        ("One sentence without a period", ["One sentence without a period"]),
        (
            "Pain decreased (p < 0.05). Function improved by 4.5 points.",
            ["Pain decreased (p < 0.05).", "Function improved by 4.5 points."],
        ),
        (
            "Drug A vs. Placebo was tested. Smith et al. Reported similar data.",
            ["Drug A vs. Placebo was tested.", "Smith et al. Reported similar data."],
        ),
        (
            "Levels of vitamin D. Patients were enrolled.",
            ["Levels of vitamin D.", "Patients were enrolled."],
        ),
        ("Is it safe? Yes! 12 patients agreed.", ["Is it safe?", "Yes!", "12 patients agreed."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        (
            "e.g. Crohn disease (see Fig. 2) was common.",
            ["e.g. Crohn disease (see Fig. 2) was common."],
        ),
        ("Lines\nare  joined\twith spaces.", ["Lines are joined with spaces."]),
        ("First paragraph\n\nSecond paragraph", ["First paragraph", "Second paragraph"]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_strips_structured_abstract_headings() -> None:
    text = (
        "BACKGROUND: Asthma is common.\n"
        "METHODS AND RESULTS: We randomized 40 adults. Results: It worked."
    )
    assert split_sentences(text) == ["Asthma is common.", "We randomized 40 adults.", "It worked."]
    assert split_sentences(text, strip_headings=False)[0] == "BACKGROUND: Asthma is common."


def test_drops_punctuation_only_fragments() -> None:
    assert split_sentences("Done.\n\n. . .\n\nMore.") == ["Done.", "More."]
