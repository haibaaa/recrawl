"""Tests for the inverted index, query parsing, and ranking."""

from __future__ import annotations

from pathlib import Path

from recrawl.indexer import TITLE_ZONE_WEIGHT, build_index
from recrawl.search import (
    ParsedQuery,
    parse_query,
    search_bm25,
    search_lnc_ltc,
)
from recrawl.store import Store


def _record(store: Store, url: str, title: str, text: str, i: int) -> None:
    store.add_page(url, "www.example.com", ts=float(i + 1))
    store.record_fetch(
        ts=float(i + 1),
        text=text,
        truth_changed=False,
        pred_changed=False,
        sim=1.0,
        sha_raw=f"r{i}",
        sha_stripped=f"s{i}",
        url=url,
        host="www.example.com",
        status=200,
        nbytes=len(text),
        title=title,
        error=None,
    )


def _doc_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "test.db")
    for i, (title, text) in enumerate(
        [
            ("Gaza ceasefire talks resume", "talks intensify today as mediators push both sides"),
            ("Markets rebound on rate news", "investors cheered as the central bank cut rates"),
            ("AI model wins coding contest", "the model beat human programmers at a contest"),
        ]
    ):
        _record(store, f"https://www.example.com/{i}", title, text, i)
    return store


def test_build_index_counts_zones_and_skips_empty(tmp_path: Path) -> None:
    store = _doc_store(tmp_path)
    _record(store, "https://www.example.com/empty", "", "", 99)
    index = build_index(store)
    store.close()
    assert index.doc_count == 3
    assert "ceasefir" in index.postings
    assert any(p.in_title for p in index.postings["talk"])
    assert len(index.postings["talk"]) == 1
    title_posting = index.postings["talk"][0]
    assert title_posting.tf_body == 1.0
    assert title_posting.tf_title == 1.0
    assert index.doc_norms[TITLE_ZONE_WEIGHT][title_posting.doc_id] > 0.0
    assert index.doc_norms[1.0][title_posting.doc_id] > 0.0


def test_parse_query_zones() -> None:
    parsed = parse_query("ceasefire title:talks site:www.example.com")
    assert parsed.site == "www.example.com"
    assert "talk" in parsed.title_only
    assert "ceasefir" in parsed.terms


def _hits(texts_by_title: dict[str, str], raw_query: str, tmp_path: Path) -> dict[str, float]:
    store = Store(tmp_path / "q.db")
    for i, (title, text) in enumerate(texts_by_title.items()):
        store.add_page(f"https://h.example/{i}", "h.example", ts=float(i + 1))
        store.record_fetch(
            ts=float(i + 1),
            text=text,
            truth_changed=False,
            pred_changed=False,
            sim=1.0,
            sha_raw=f"r{i}",
            sha_stripped=f"s{i}",
            url=f"https://h.example/{i}",
            host="h.example",
            status=200,
            nbytes=len(text),
            title=title,
            error=None,
        )
    index = build_index(store)
    store.close()
    hits = search_lnc_ltc(index, parse_query(raw_query), k=10, alpha=0.0, beta=0.0)
    return {hit.url: hit.cosine for hit in hits}


def test_cosine_prefers_title_matches(tmp_path: Path) -> None:
    scores = _hits(
        {
            "california": "news roundup today",
            "news": "california roundup today report",
        },
        "california",
        tmp_path,
    )
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    assert best == "https://h.example/0"


def test_title_zone_restricts_matches(tmp_path: Path) -> None:
    store = _doc_store(tmp_path)
    index = build_index(store)
    store.close()
    hits = search_lnc_ltc(index, parse_query("title:talks"), k=10, alpha=0.0, beta=0.0)
    assert hits
    assert hits[0].url == "https://www.example.com/0"


def test_net_mixing_orders_by_importance_on_ties(tmp_path: Path) -> None:
    store = Store(tmp_path / "tie.db")
    for i in (0, 1):
        store.add_page(f"https://t.example/{i}", "t.example", ts=float(i + 1))
        store.record_fetch(
            ts=float(i + 1),
            text="market report mixed day",
            truth_changed=False,
            pred_changed=False,
            sim=1.0,
            sha_raw=f"r{i}",
            sha_stripped=f"s{i}",
            url=f"https://t.example/{i}",
            host="t.example",
            status=200,
            nbytes=22,
            title="market report",
            error=None,
        )
    store.conn.execute("UPDATE pages SET importance = 1.0 WHERE url = 'https://t.example/1'")
    store.conn.execute("UPDATE pages SET importance = 0.0 WHERE url = 'https://t.example/0'")
    store.conn.commit()
    index = build_index(store)
    store.close()
    query = parse_query("market report")
    bare = search_lnc_ltc(index, query, k=10, alpha=0.0, beta=0.0)
    mixed = search_lnc_ltc(index, query, k=10, alpha=0.3, beta=0.0)
    # identical content -> identical cosine; importance breaks the tie.
    assert bare[0].url == "https://t.example/0"
    assert mixed[0].url == "https://t.example/1"


def test_bm25_ranks_and_filters_site(tmp_path: Path) -> None:
    store = _doc_store(tmp_path)
    index = build_index(store)
    store.close()
    hits = search_bm25(index, parse_query("talks"), k=10)
    assert hits and all(hit.bm25 > 0 for hit in hits)
    narrowed = search_bm25(
        index, ParsedQuery(terms=["talks"], title_only=set(), site="www.example.com"), k=10
    )
    assert all(hit.host == "www.example.com" for hit in narrowed)
