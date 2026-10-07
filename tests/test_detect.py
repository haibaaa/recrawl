"""Tests for shingle-based change detection."""

from recrawl.detect import (
    content_sha,
    detect_change,
    jaccard,
    shingles,
    strip_volatile,
    tokenize_words,
    truth_changed,
)

ARTICLE = (
    "The minister said the new policy would take effect next month. "
    "Officials confirmed that funding for the programme remains unchanged. "
    "Opponents argued the plan was rushed and lacked public consultation."
)

ARTICLE_EDITED = (
    "The minister said the new policy would take effect next month. "
    "Officials confirmed that funding for the programme has been doubled. "
    "Opponents argued the plan was rushed and lacked public consultation."
)


def test_tokenize_words_lowercases_and_splits() -> None:
    assert tokenize_words("Hello, World! 2026") == ["hello", "world", "2026"]


def test_shingles_word_level() -> None:
    assert shingles("one two three four", k=2) == {"one two", "two three", "three four"}


def test_shingles_short_text_single_shingle() -> None:
    assert shingles("a b", k=5) == {"a b"}


def test_shingles_empty_text() -> None:
    assert shingles("", k=5) == set()


def test_jaccard_identical() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_counts_identical() -> None:
    assert jaccard(set(), set()) == 1.0


def test_jaccard_one_empty() -> None:
    assert jaccard({"a"}, set()) == 0.0


def test_strip_volatile_removes_times_and_dates() -> None:
    text = "Updated 3:45 PM on 2026-10-06 , published October 6, 2026 two hours ago"
    stripped = strip_volatile(text)
    for token in ("3:45", "2026-10-06", "October 6, 2026", "two hours ago", "ago"):
        assert token not in stripped


def test_detect_change_identical_text_is_unchanged() -> None:
    changed, similarity = detect_change(ARTICLE, ARTICLE)
    assert not changed
    assert similarity == 1.0


def test_detect_change_detects_edit() -> None:
    changed, similarity = detect_change(ARTICLE, ARTICLE_EDITED)
    assert similarity < 1.0
    assert changed is (similarity < 0.85)


def test_truth_ignores_timestamp_churn() -> None:
    version_a = "Body text of the story. Updated at 3:45 PM"
    version_b = "Body text of the story. Updated at 5:10 PM"
    assert not truth_changed(version_a, version_b)


def test_truth_detects_real_edit() -> None:
    assert truth_changed(ARTICLE, ARTICLE_EDITED)


def test_content_sha_is_stable() -> None:
    assert content_sha("abc") == content_sha("abc")
    assert content_sha("abc") != content_sha("abd")
