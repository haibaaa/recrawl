"""Inverted index built from the crawl store with title-zone weighting.

Indexed fields (zones): body (weight 1.0) and title (weight 2.0). Term
frequencies are recorded per document as the zone-weighted count; each posting
flag *in_title* is kept so `title:` queries can restrict matches.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

from recrawl.normalize import host_of
from recrawl.store import Store
from recrawl.textproc import analyze, tokenize

TITLE_ZONE_WEIGHT = 2.0


@dataclass
class Posting:
    """One document occurrence of a term, with body/title frequency split."""

    doc_id: int
    tf_body: float
    tf_title: float

    @property
    def in_title(self) -> bool:
        return self.tf_title > 0.0


@dataclass
class DocMeta:
    """Document-level metadata needed for scoring and presentation."""

    url: str
    title: str
    host: str
    importance: float
    last_change_ts: float | None
    text_len: int


@dataclass
class InvertedIndex:
    """Positional-independent inverted index with per-title-weight doc norms."""

    docs: list[DocMeta]
    postings: dict[str, list[Posting]]
    doc_norms: dict[float, list[float]]

    @property
    def doc_count(self) -> int:
        return len(self.docs)


_TRACKER_PREFIXES = (
    "at_campaign=",
    "at_mid=",
    "at_ptrs=",
    "maca=",
    "traffic_source=",
    "acquisitionData=",
    "INTCMP=",
    "REFPVID=",
)


def _canonical_url(url: str) -> str:
    """Drop fragment and common news-tracker query params so duplicates collapse."""
    url = url.split("#", 1)[0]
    base, _, query = url.partition("?")
    if query == "":
        return url
    leading = query.split("&", 1)[0]
    if leading.startswith(_TRACKER_PREFIXES) or query.startswith("at_campaign"):
        return base
    return url


def build_index(store: Store) -> InvertedIndex:
    """Index every tracked page that has extracted text (canonical URLs)."""
    docs: list[DocMeta] = []
    postings: dict[str, list[Posting]] = {}
    seen_urls: set[str] = set()
    for page in store.all_pages():
        if not page.text:
            continue
        canonical = _canonical_url(page.url)
        if canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        title_terms = analyze(tokenize(page.title))
        body_terms = analyze(tokenize(page.text))
        if not title_terms and not body_terms:
            continue
        doc_id = len(docs)
        body_counts: dict[str, float] = {}
        title_counts: dict[str, float] = {}
        for term in title_terms:
            title_counts[term] = title_counts.get(term, 0.0) + 1.0
        for term in body_terms:
            body_counts[term] = body_counts.get(term, 0.0) + 1.0
        docs.append(
            DocMeta(
                url=canonical,
                title=page.title,
                host=host_of(canonical),
                importance=page.importance,
                last_change_ts=page.last_change_ts,
                text_len=len(page.text),
            )
        )
        for term in set(body_counts) | set(title_counts):
            postings.setdefault(term, []).append(
                Posting(
                    doc_id=doc_id,
                    tf_body=body_counts.get(term, 0.0),
                    tf_title=title_counts.get(term, 0.0),
                )
            )
    doc_norms: dict[float, list[float]] = {}
    for title_weight in (1.0, TITLE_ZONE_WEIGHT):
        norms = [0.0] * len(docs)
        for _term, term_postings in postings.items():
            for posting in term_postings:
                weight = 1.0 + math.log(posting.tf_body + title_weight * posting.tf_title)
                norms[posting.doc_id] += weight * weight
        doc_norms[title_weight] = [math.sqrt(value) for value in norms]
    return InvertedIndex(docs=docs, postings=postings, doc_norms=doc_norms)


def save_index(index: InvertedIndex, path: Path) -> None:
    """Write *index* to *path* as pickle (local research tool)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(index, handle)


def load_index(path: Path) -> InvertedIndex:
    """Read an index previously written by :func:`save_index`."""
    with path.open("rb") as handle:
        return pickle.load(handle)  # type: ignore[no-any-return]
