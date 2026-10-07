"""Change detection between two versions of a page.

Two signals are computed on extracted main text:

- shingle similarity (word-level k-shingles, Jaccard) used by the detector;
- a "volatile-stripped" exact hash used as reference truth, where dates,
  clock times and relative-time phrases are removed so that timestamp churn
  alone does not count as a content change.
"""

from __future__ import annotations

import hashlib
import re

VOLATILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}(T[\d:+.]*)?\b"),
    re.compile(
        r"\b\d{1,2}(st|nd|rd|th)?\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago\b", re.IGNORECASE),
    re.compile(r"\b(last|yesterday|today|tomorrow|just now|moments ago)\b", re.IGNORECASE),
    re.compile(r"\b(updated|published|posted|on)\s*:?\s*", re.IGNORECASE),
    re.compile(r"\bago\b", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens of *text*, ignoring punctuation and markup."""
    return _WORD_RE.findall(text.lower())


def shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingle set of *text* (a single shingle if shorter than k)."""
    words = tokenize_words(text)
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two shingle sets; two empty sets count as identical."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def strip_volatile(text: str) -> str:
    """Remove date/time/relative-time phrases so timestamp churn alone cannot
    register as a content change."""
    for pattern in VOLATILE_PATTERNS:
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def content_sha(text: str) -> str:
    """SHA-256 of *text* as a hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_change(
    prev_text: str, new_text: str, tau: float = 0.85, k: int = 5
) -> tuple[bool, float]:
    """Predict whether page content changed between two extracted texts.

    Returns ``(predicted_changed, similarity)`` where the prediction is
    ``similarity < tau``.
    """
    similarity = jaccard(shingles(prev_text, k), shingles(new_text, k))
    return similarity < tau, similarity


def truth_changed(prev_text: str, new_text: str) -> bool:
    """Reference truth: volatile-stripped content differs between versions."""
    return content_sha(strip_volatile(prev_text)) != content_sha(strip_volatile(new_text))
