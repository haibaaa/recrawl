"""Tests for the SQLite store."""

from pathlib import Path

import pytest

from recrawl.store import Store


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


def test_add_page_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/x", "a.example", ts=1.0)
    store.add_page("https://a.example/x", "a.example", ts=2.0)
    assert store.page_count() == 1
    store.close()


def test_record_fetch_first_fetch_sets_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/x", "a.example", ts=1.0)
    changed = store.record_fetch(
        url="https://a.example/x",
        host="a.example",
        ts=10.0,
        status=200,
        nbytes=100,
        title="T",
        text="body",
        sha_raw="raw1",
        sha_stripped="strip1",
        truth_changed=None,
        pred_changed=None,
        sim=None,
        error=None,
    )
    assert changed is False
    page = store.get_page("https://a.example/x")
    assert page is not None
    assert page.fetch_count == 1
    assert page.change_count == 0
    assert page.last_fetch_ts == 10.0
    assert page.first_fetch_ts == 10.0
    assert page.text == "body"
    store.close()


def test_record_fetch_counts_change_and_stores_version(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/x", "a.example", ts=1.0)
    kwargs = {
        "url": "https://a.example/x",
        "host": "a.example",
        "status": 200,
        "nbytes": 100,
        "title": "T",
        "sha_raw": "r",
        "sha_stripped": "s",
        "error": None,
    }
    store.record_fetch(
        ts=10.0,
        text="version one",
        truth_changed=None,
        pred_changed=None,
        sim=None,
        **kwargs,
    )
    changed = store.record_fetch(
        ts=20.0,
        text="version two",
        truth_changed=True,
        pred_changed=True,
        sim=0.5,
        **{**kwargs, "sha_raw": "r2", "sha_stripped": "s2"},
    )
    assert changed is True
    page = store.get_page("https://a.example/x")
    assert page is not None
    assert page.change_count == 1
    assert page.last_change_ts == 20.0
    assert store.version_text("https://a.example/x", at_ts=15.0) == "version one"
    assert store.version_text("https://a.example/x", at_ts=25.0) == "version two"
    store.close()


def test_fetch_history_is_time_ordered(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/x", "a.example", ts=1.0)
    for ts in (30.0, 10.0, 20.0):
        store.record_fetch(
            url="https://a.example/x",
            host="a.example",
            ts=ts,
            status=200,
            nbytes=1,
            title="",
            text="t",
            sha_raw="r",
            sha_stripped="s",
            truth_changed=False,
            pred_changed=False,
            sim=1.0,
            error=None,
        )
    history = store.fetch_history()
    assert [row.ts for row in history] == [10.0, 20.0, 30.0]
    store.close()


def test_recompute_importance_normalises_to_unit_range(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/1", "a.example", ts=1.0)
    store.add_page("https://b.example/2", "b.example", ts=1.0)
    store.add_backlink("https://a.example/1", "https://b.example/2")
    store.add_backlink("https://a.example/1", "https://b.example/2")  # deduplicated
    store.recompute_importance()
    pages = {page.url: page.importance for page in store.all_pages()}
    assert pages["https://b.example/2"] > pages["https://a.example/1"]
    assert max(pages.values()) == pytest.approx(1.0)
    store.close()


def test_pass_times_recorded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_pass(ts=100.0, fetched=5, changed=1, errors=0)
    store.record_pass(ts=200.0, fetched=6, changed=2, errors=1)
    assert store.pass_times() == [100.0, 200.0]
    store.close()


def test_meta_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set_meta("k", "v")
    assert store.get_meta("k") == "v"
    assert store.get_meta("missing") is None
    store.close()
