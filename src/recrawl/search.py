"""Query parsing and ranking: lnc.ltc cosine with title zones + quality g(d).

Ranking variants for the ablation study:
  tf-idf (lnc.ltc cosine)          - base
  + title zones                    - title: terms + zone-weighted tfs
  + g(d) (importance + freshness)  - net = cos + alpha*g_imp + beta*g_fresh

A Robertson BM25 score is computed alongside as an independent baseline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from recrawl.indexer import TITLE_ZONE_WEIGHT, InvertedIndex
from recrawl.normalize import host_of
from recrawl.textproc import analyze, tokenize

_BM25_K1 = 1.5
_BM25_B = 0.75

DEFAULT_ALPHA = 0.01
DEFAULT_BETA = 0.01


@dataclass
class ParsedQuery:
    """Zone-restricted query terms ready for scoring."""

    terms: list[str] = field(default_factory=list)
    title_only: set[str] = field(default_factory=set)
    site: str | None = None


@dataclass
class SearchHit:
    """One ranked document with its scoring variants."""

    url: str
    title: str
    host: str
    cosine: float
    net: float
    bm25: float


def parse_query(raw: str) -> ParsedQuery:
    """Parse *raw*: bare terms plus optional `title:` and `site:` restrictions."""
    query = ParsedQuery()
    for token in raw.split():
        lower = token.lower()
        if lower.startswith("title:"):
            value = lower[len("title:") :]
            query.title_only.update(analyze(tokenize(value)))
        elif lower.startswith("site:"):
            value = lower[len("site:") :]
            query.site = host_of(value if "://" in value else f"https://{value}")
        else:
            query.terms.extend(analyze(tokenize(token)))
    return query


def _idf(index: InvertedIndex, term: str) -> float:
    df = len(index.postings.get(term, ()))
    return math.log(1.0 + index.doc_count / (df + 0.5)) if df else 0.0


def _query_weights(
    index: InvertedIndex, query: ParsedQuery, title_weight: float = TITLE_ZONE_WEIGHT
) -> dict[str, float]:
    """Query vector weights: (1 + ln tf) * idf, scaled by title-zone weight."""
    counts: dict[str, float] = {}
    for term in query.terms:
        counts[term] = counts.get(term, 0.0) + 1.0
    for term in query.title_only:
        counts[term] = counts.get(term, 0.0) + title_weight
    weights: dict[str, float] = {}
    for term, tf in counts.items():
        if term not in index.postings:
            continue
        weights[term] = (1.0 + math.log(tf)) * _idf(index, term)
    return weights


def search_lnc_ltc(
    index: InvertedIndex,
    query: ParsedQuery,
    k: int,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    now: float | None = None,
    title_weight: float = TITLE_ZONE_WEIGHT,
) -> list[SearchHit]:
    """Rank by lnc.ltc cosine plus g(d) = alpha*importance + beta*freshness.

    *title_weight* selects the doc norms to use (e.g. 2.0 = zones on,
    1.0 = zones degenerate to a single body zone for the no-zone ablation).
    """
    weights = _query_weights(index, query, title_weight)
    if not weights:
        return []
    norm_q = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm_q == 0.0:
        return []
    norms = index.doc_norms[title_weight]
    now = now if now is not None else time.time()
    accumulators: dict[int, float] = {}
    for term, weight_q in weights.items():
        for posting in index.postings.get(term, ()):
            doc = index.docs[posting.doc_id]
            if query.site is not None and doc.host != query.site:
                continue
            if term in query.title_only and not posting.in_title:
                continue
            if norms[posting.doc_id] == 0.0:
                continue
            weight_d = 1.0 + math.log(posting.tf_body + title_weight * posting.tf_title)
            accumulators[posting.doc_id] = (
                accumulators.get(posting.doc_id, 0.0) + weight_d * weight_q
            )
    hits = [
        _hit_from(
            index,
            doc_id,
            accumulators[doc_id] / norms[doc_id] / norm_q,
            alpha,
            beta,
            now,
        )
        for doc_id in accumulators
    ]
    hits.sort(key=lambda hit: hit.net, reverse=True)
    return hits[:k]


def _hit_from(
    index: InvertedIndex,
    doc_id: int,
    cosine: float,
    alpha: float,
    beta: float,
    now: float,
) -> SearchHit:
    doc = index.docs[doc_id]
    freshness = _freshness(doc.last_change_ts, now)
    net = cosine + alpha * doc.importance + beta * freshness
    return SearchHit(
        url=doc.url,
        title=doc.title,
        host=doc.host,
        cosine=cosine,
        net=net,
        bm25=0.0,
    )


def _freshness(last_change_ts: float | None, now: float) -> float:
    """Freshness in [0, 1]: recently changed pages score higher."""
    if last_change_ts is None:
        return 0.0
    days = max(0.0, (now - last_change_ts) / 86400.0)
    return 1.0 / (1.0 + days)


def search_bm25(index: InvertedIndex, query: ParsedQuery, k: int) -> list[SearchHit]:
    """Robertson BM25 baseline (body+title weighted tf, no zones, no g(d))."""
    if not query.terms and not query.title_only:
        return []
    avgdl = sum(doc.text_len for doc in index.docs) / max(index.doc_count, 1)

    def robertson_idf(term: str) -> float:
        df = len(index.postings.get(term, ()))
        return math.log((index.doc_count - df + 0.5) / (df + 0.5)) if df else 0.0

    accumulators: dict[int, float] = {}
    terms = set(query.terms) | set(query.title_only)
    for term in terms:
        idf = robertson_idf(term)
        if idf <= 0.0:
            continue
        for posting in index.postings.get(term, ()):
            doc = index.docs[posting.doc_id]
            if query.site is not None and doc.host != query.site:
                continue
            if term in query.title_only and not posting.in_title:
                continue
            dl = max(doc.text_len, 1)
            tf = posting.tf_body + posting.tf_title
            saturation = (tf * (_BM25_K1 + 1.0)) / (
                tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / avgdl)
            )
            accumulators[posting.doc_id] = accumulators.get(posting.doc_id, 0.0) + idf * saturation
    hits = [
        SearchHit(
            url=index.docs[doc_id].url,
            title=index.docs[doc_id].title,
            host=index.docs[doc_id].host,
            cosine=0.0,
            net=0.0,
            bm25=score,
        )
        for doc_id, score in accumulators.items()
    ]
    hits.sort(key=lambda hit: hit.bm25, reverse=True)
    return hits[:k]
