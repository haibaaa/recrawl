"""Evaluation harness: detector P/R, freshness-vs-budget curves, search P@k."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from recrawl.cc import SnapshotPair
from recrawl.indexer import TITLE_ZONE_WEIGHT, InvertedIndex, _canonical_url
from recrawl.metrics import DetectorScores
from recrawl.scheduler import POLICIES, SimulationInput, simulate
from recrawl.search import parse_query, search_bm25, search_lnc_ltc
from recrawl.store import Store


@dataclass
class CcSummary:
    """Per-method scores against the CC payload-digest oracle, plus raw counts."""

    rows: list[dict[str, object]]
    total_pairs: int
    digest_changed: int
    payload_only_changes: int


def live_detector_scores(store: Store) -> dict[str, DetectorScores]:
    """Detector P/R on live history against the volatile-stripped truth.

    Competing methods: shingle Jaccard (ours) vs raw-text byte hash (baseline).
    """
    scores = {"shingle": DetectorScores(), "byte-hash": DetectorScores()}
    previous_sha: dict[str, str] = {}
    for row in store.fetch_history():
        current_sha = _raw_sha(store, row.url, row.ts)
        prior = previous_sha.get(row.url)
        if row.truth_changed is not None:
            truth = bool(row.truth_changed)
            if row.pred_changed is not None:
                scores["shingle"].add(predicted=bool(row.pred_changed), truth=truth)
            if prior is not None and current_sha is not None:
                scores["byte-hash"].add(predicted=prior != current_sha, truth=truth)
        if current_sha is not None:
            previous_sha[row.url] = current_sha
    return scores


def _raw_sha(store: Store, url: str, ts: float) -> str | None:
    rows = store.conn.execute(
        "SELECT sha_raw FROM fetch_log WHERE url = ? AND ts = ? AND sha_raw != ''",
        (url, ts),
    ).fetchone()
    return rows["sha_raw"] if rows else None


def build_simulation_input(store: Store) -> SimulationInput:
    """Convert the live history into the counterfactually replayable input."""
    pass_times = store.pass_times()
    truth_events: dict[str, list[float]] = {}
    available_from: dict[str, float] = {}
    for row in store.fetch_history():
        if row.truth_changed == 1:
            truth_events.setdefault(row.url, []).append(row.ts)
        available_from[row.url] = min(available_from.get(row.url, row.ts), row.ts)
    importance = {page.url: page.importance for page in store.all_pages()}
    return SimulationInput(
        pass_times=pass_times,
        truth_events=truth_events,
        available_from=available_from,
        importance=importance,
    )


def freshness_curves(store: Store, budgets: list[int]) -> list[dict[str, float | str]]:
    """Simulate every policy at every per-pass budget; returns metric rows.

    ``budget`` is fetches per pass (mirroring ``loop --budget``); total fetch
    volume for each run is ``budget x passes``.
    """
    data = build_simulation_input(store)
    rows: list[dict[str, float | str]] = []
    for budget in budgets:
        for policy in POLICIES:
            result = simulate(data, policy=policy, budget=budget)
            rows.append(
                {
                    "policy": policy,
                    "budget": budget,
                    "fresh_fraction": result.fresh_fraction,
                    "missed_changes": result.missed_changes,
                    "stale_change_hours": result.stale_change_hours,
                    "fetches": result.fetches,
                }
            )
    return rows


def cc_summary(pairs: list[SnapshotPair]) -> CcSummary:
    """Score shingle and stripped-hash predictions against CC payload digests."""
    scores = {"shingle": DetectorScores(), "stripped-hash": DetectorScores()}
    digest_changed_count = sum(1 for pair in pairs if pair.digest_changed)
    payload_only = 0
    for pair in pairs:
        if pair.pred_changed is None or pair.stripped_changed is None:
            continue
        scores["shingle"].add(predicted=pair.pred_changed, truth=pair.digest_changed)
        scores["stripped-hash"].add(predicted=pair.stripped_changed, truth=pair.digest_changed)
        if pair.digest_changed and not pair.stripped_changed:
            payload_only += 1
    rows = [
        {
            "method": method,
            "n": detector.tp + detector.fp + detector.tn + detector.fn,
            "precision": detector.precision,
            "recall": detector.recall,
            "f1": detector.f1,
            "accuracy": detector.accuracy,
        }
        for method, detector in scores.items()
    ]
    return CcSummary(
        rows=rows,
        total_pairs=len(pairs),
        digest_changed=digest_changed_count,
        payload_only_changes=payload_only,
    )


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write *rows* (uniform keys) to *path* as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class SearchEval:
    """Per-method mean P@k over the judged queries."""

    summary: list[dict[str, float | str]]
    per_query: list[dict[str, object]]


@dataclass
class _Ranked:
    method: str
    urls: list[str]


def _canonical(url: str) -> str:
    return _canonical_url(url.rstrip("/"))


def _rank_with(index: InvertedIndex, query: str, method: str, k: int) -> _Ranked:
    parsed = parse_query(query)
    if method == "tfidf":
        hits = search_lnc_ltc(index, parsed, k, alpha=0.0, beta=0.0, title_weight=1.0)
    elif method == "tfidf+zones":
        hits = search_lnc_ltc(index, parsed, k, alpha=0.0, beta=0.0, title_weight=TITLE_ZONE_WEIGHT)
    elif method == "+g(d)":
        hits = search_lnc_ltc(index, parsed, k, title_weight=TITLE_ZONE_WEIGHT)
    else:
        hits = search_bm25(index, parsed, k)
    return _Ranked(method=method, urls=[_canonical(hit.url) for hit in hits])


def search_eval(index: InvertedIndex, queryset: Path, k: int = 5) -> SearchEval:
    """P@k for each ranking method against a judged JSONL query file.

    Format per line: {"query": "...", "relevant": ["https://...", ...]}
    """
    methods = ("tfidf", "tfidf+zones", "+g(d)", "bm25")
    per_query: list[dict[str, object]] = []
    for line in queryset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        query = record["query"]
        relevant = set(_canonical(url) for url in record["relevant"])
        ranked = {method: _rank_with(index, query, method, k).urls for method in methods}
        per_query.append(
            {
                "query": query,
                "n_relevant": len(relevant),
                **{
                    f"{method}_p@{rank}": _precision(ranked[method][:rank], relevant, rank)
                    for method in methods
                    for rank in range(1, k + 1)
                },
            }
        )
    summary = [
        {
            "method": method,
            "mean_p@1": _mean(per_query, f"{method}_p@1"),
            "mean_p@3": _mean(per_query, f"{method}_p@3"),
            "mean_p@5": _mean(per_query, f"{method}_p@5"),
        }
        for method in methods
    ]
    return SearchEval(summary=summary, per_query=per_query)


def _precision(urls: list[str], relevant: set[str], rank: int) -> float:
    return sum(1 for url in urls if url in relevant) / rank


def _mean(rows: list[dict[str, object]], key: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    return round(sum(float(row[key]) for row in rows) / n, 3)  # type: ignore[arg-type]
