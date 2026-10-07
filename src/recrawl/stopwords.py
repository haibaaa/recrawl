"""Small English stopword list for news text (kept short: news keywords are meaningful)."""

from __future__ import annotations

STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by for from had has have he her his i if in is it its
    of on or our she so that the their them then there they this to was we were what
    when where which who will with you your not no nor than too very just can dont
    do does did about after against also am among been before being between both
    because could during each else even every from further here how into may might
    more most much must myself off once one only other others out over same should s
    such t than those through until up upon us wasn were what what's when where
    whether while who whom why would
    """.split()
)


def is_stopword(term: str) -> bool:
    """True for a stopword (after the caller has lowercased *term*)."""
    return term in STOPWORDS
