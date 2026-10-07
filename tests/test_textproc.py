"""Tests for stopwords, tokenizer, and the in-house Porter stemmer."""

from __future__ import annotations

from pathlib import Path

from recrawl.porter import stem
from recrawl.stopwords import is_stopword
from recrawl.textproc import analyze, tokenize


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    terms = tokenize("The Guardian reports on AI AND politics, widely!")
    assert "the" not in terms
    assert "and" not in terms
    assert "wide" not in terms
    assert "reports" in terms
    assert "politics" in terms


def test_tokenize_handles_unicode_and_apostrophes() -> None:
    terms = tokenize("L'équipe won't l'état")
    assert "won't" in terms


def test_stem_classic_cases() -> None:
    # Expected stems verified against Martin Porter's reference C implementation.
    cases = {
        "caresses": "caress",
        "ponies": "poni",
        "ties": "ti",
        "caress": "caress",
        "cats": "cat",
        "feed": "feed",
        "agreed": "agre",
        "plastered": "plaster",
        "bled": "bled",
        "motoring": "motor",
        "sing": "sing",
        "conflated": "conflat",
        "troubled": "troubl",
        "sized": "size",
        "hopping": "hop",
        "tanned": "tan",
        "falling": "fall",
        "hissing": "hiss",
        "fizzed": "fizz",
        "failing": "fail",
        "filing": "file",
        "happy": "happi",
        "relational": "relat",
        "conditional": "condit",
    }
    for word, expected in cases.items():
        assert stem(word) == expected, f"{word} -> {stem(word)} != {expected}"


def test_stem_matches_reference_vocabulary() -> None:
    """All 23,531 words in the author's test battery stem identically (zero mismatches)."""
    data_dir = Path(__file__).parent / "data" / "porter"
    words = (data_dir / "voc.txt").read_text(encoding="utf-8").splitlines()
    expected = (data_dir / "out.txt").read_text(encoding="utf-8").splitlines()
    assert len(words) == len(expected)
    mismatches = [
        (word, want, stem(word))
        for word, want in zip(words, expected, strict=True)
        if stem(word) != want
    ]
    assert mismatches == [], f"{len(mismatches)} mismatches: {mismatches[:5]}"


def test_analyze_stems_each_term() -> None:
    assert analyze(["caresses", "ponies"]) == ["caress", "poni"]


def test_is_stopword_basic() -> None:
    assert is_stopword("the")
    assert is_stopword("and")
    assert not is_stopword("news")
