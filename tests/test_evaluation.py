"""Tests for the evaluation harness (detector scoring, simulation input, CSV)."""

from __future__ import annotations

from pathlib import Path

from recrawl.evaluation import build_simulation_input, live_detector_scores, write_csv
from recrawl.store import Store


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


def test_live_detector_scores_byte_hash_vs_shingle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/x", "a.example", ts=1.0)
    base = {
        "url": "https://a.example/x",
        "host": "a.example",
        "status": 200,
        "nbytes": 100,
        "title": "T",
        "error": None,
    }
    # fetch 1: baseline, no truth yet.
    store.record_fetch(
        ts=10.0,
        text="v1",
        truth_changed=None,
        pred_changed=None,
        sim=None,
        sha_raw="r1",
        sha_stripped="s1",
        **base,
    )
    # fetch 2: real content change; both methods predict it.
    store.record_fetch(
        ts=20.0,
        text="v2",
        truth_changed=True,
        pred_changed=True,
        sim=0.5,
        sha_raw="r2",
        sha_stripped="s2",
        **base,
    )
    # fetch 3: volatile-only change (ad script) — truth unchanged; shingle says
    # unchanged (stripped extraction ignores it), raw byte hash says changed.
    store.record_fetch(
        ts=30.0,
        text="v2",
        truth_changed=False,
        pred_changed=False,
        sim=0.99,
        sha_raw="r3",
        sha_stripped="s2",
        **base,
    )
    scores = live_detector_scores(store)
    assert scores["shingle"].n == 2
    assert scores["shingle"].f1 == 1.0
    # byte-hash: correct on fetch 2, false positive on fetch 3.
    assert scores["byte-hash"].n == 2
    assert scores["byte-hash"].fp == 1
    assert scores["byte-hash"].f1 == 2 / 3
    store.close()


def test_build_simulation_input_collects_truth_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_page("https://a.example/x", "a.example", ts=1.0)
    store.conn.execute("UPDATE pages SET importance = 0.7 WHERE url = ?", ("https://a.example/x",))
    store.conn.commit()
    store.record_pass(1000.0, fetched=1, changed=0, errors=0)
    store.record_pass(2000.0, fetched=1, changed=1, errors=0)
    store.record_fetch(
        ts=10.0,
        text="v1",
        truth_changed=None,
        pred_changed=None,
        sim=None,
        sha_raw="r1",
        sha_stripped="s1",
        url="https://a.example/x",
        host="a.example",
        status=200,
        nbytes=1,
        title="T",
        error=None,
    )
    store.record_fetch(
        ts=1500.0,
        text="v2",
        truth_changed=True,
        pred_changed=True,
        sim=0.4,
        sha_raw="r2",
        sha_stripped="s2",
        url="https://a.example/x",
        host="a.example",
        status=200,
        nbytes=1,
        title="T",
        error=None,
    )
    data = build_simulation_input(store)
    assert data.pass_times == [1000.0, 2000.0]
    assert data.truth_events["https://a.example/x"] == [1500.0]
    assert data.available_from["https://a.example/x"] == 10.0
    assert data.importance["https://a.example/x"] == 0.7
    store.close()


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "report.csv"
    rows: list[dict[str, object]] = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    write_csv(rows, path)
    assert path.read_text(encoding="utf-8") == "a,b\n1,x\n2,y\n"


def test_write_csv_empty_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    write_csv([], path)
    assert not path.exists()
