"""Text processing: tokenization, stopword filtering, Porter stemming."""

from __future__ import annotations

import re

from recrawl.porter import stem
from recrawl.stopwords import is_stopword

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase *text* and split into alphabetic tokens (≥2 chars), skipping stopwords."""
    terms: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        term = match.group(0)
        if len(term) < 2 or is_stopword(term):
            continue
        terms.append(term)
    return terms


def analyze(terms: list[str]) -> list[str]:
    """Stem each *term* in place order (terms should already be filtered)."""
    return [stem(term) for term in terms]
