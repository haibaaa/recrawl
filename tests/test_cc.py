"""Tests for Common Crawl index parsing and label helpers."""

from __future__ import annotations

from recrawl.cc import SnapshotPair, durable_candidates, parse_index_lines


def test_parse_index_lines_keeps_only_200_records() -> None:
    text = "\n".join(
        [
            '{"url": "https://a.example/x", "timestamp": "20260601", "digest": "AAA",'
            ' "filename": "warc/00001.warc.gz", "offset": "100", "length": "50",'
            ' "status": "200"}',
            '{"url": "https://a.example/x", "timestamp": "20260701", "digest": "BBB",'
            ' "status": "404"}',
            "not json",
            "",
        ]
    )
    captures = parse_index_lines(text, "CC-MAIN-2026-39")
    assert len(captures) == 1
    capture = captures[0]
    assert capture.crawl == "CC-MAIN-2026-39"
    assert capture.digest == "AAA"
    assert capture.offset == 100
    assert capture.length == 50
    assert capture.status == 200


def test_parse_index_lines_empty_body() -> None:
    assert parse_index_lines("No Captures found for: x\n", "CC-MAIN-2026-39") == []


def test_durable_candidates_ranks_shallow_paths_first() -> None:
    urls = [
        "https://www.bbc.com/news/world/europe/very/deep/article-12345",
        "https://www.bbc.com/news/world",
        "https://www.bbc.com/news",
    ]
    ranked = durable_candidates(urls, limit=2)
    assert ranked[0] == "https://www.bbc.com/news"
    assert ranked[1] == "https://www.bbc.com/news/world"


def test_durable_candidates_prefers_cc_captured_hosts() -> None:
    urls = [
        "https://www.bbc.com/news",
        "https://www.aljazeera.com/news/2026/oct/06/some/story",
        "https://www.aljazeera.com/opinions/",
    ]
    ranked = durable_candidates(urls, limit=3)
    assert "https://www.aljazeera.com/opinions/" == ranked[0]
    assert "https://www.aljazeera.com/news/2026/oct/06/some/story" == ranked[1]
    assert "https://www.bbc.com/news" == ranked[2]


def test_durable_candidates_dedupes() -> None:
    assert durable_candidates(["https://a.example/", "https://a.example/"], limit=10) == [
        "https://a.example/"
    ]


def test_summary_counts_payload_only_changes() -> None:
    pair = SnapshotPair(
        url="https://a.example/x",
        crawl_a="CC-A",
        crawl_b="CC-B",
        timestamp_a="20260601",
        timestamp_b="20260701",
        digest_changed=True,
        sim=0.95,
        pred_changed=False,
        stripped_changed=False,
    )
    unchanged = SnapshotPair(
        url="https://a.example/y",
        crawl_a="CC-A",
        crawl_b="CC-B",
        timestamp_a="20260601",
        timestamp_b="20260701",
        digest_changed=False,
        sim=1.0,
        pred_changed=False,
        stripped_changed=False,
    )
    from recrawl.evaluation import cc_summary

    summary = cc_summary([pair, unchanged])
    assert summary.total_pairs == 2
    assert summary.digest_changed == 1
    assert summary.payload_only_changes == 1
    shingle_row = next(row for row in summary.rows if row["method"] == "shingle")
    assert shingle_row["n"] == 2
    assert shingle_row["recall"] == 0.0
