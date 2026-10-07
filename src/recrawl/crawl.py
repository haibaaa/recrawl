"""Crawl pass execution: frontier selection, polite fetching, change logging."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from recrawl.config import (
    DEFAULT_BUDGET_PER_PASS,
    FETCH_TIMEOUT_SECONDS,
    HOST_DELAY_SECONDS,
    JACCARD_TAU,
    SHINGLE_K,
    USER_AGENT,
)
from recrawl.detect import content_sha, detect_change, strip_volatile, truth_changed
from recrawl.extract import extract, extract_links
from recrawl.fetcher import fetch
from recrawl.frontier import Candidate, Frontier, sleep_until
from recrawl.normalize import host_of, normalize_url
from recrawl.robots import RobotsCache
from recrawl.scheduler import UrlState, score
from recrawl.seeds import is_allowed_host
from recrawl.store import Store

_MAX_ROBOTS_DELAY = 10.0


@dataclass
class PassStats:
    """Counters for one completed crawl pass."""

    started_ts: float
    duration_s: float
    fetched: int
    changed: int
    errors: int
    robots_blocked: int
    discovered: int


@dataclass
class FetchOutcome:
    """Per-URL result counters for one fetch inside a pass."""

    changed: bool
    error: str | None
    html: str


def run_pass(
    store: Store,
    session: requests.Session,
    robots: RobotsCache,
    budget: int = DEFAULT_BUDGET_PER_PASS,
    policy: str = "proposed",
    page_cap: int = 500,
) -> PassStats:
    """Execute one crawl pass: fetch every due tracked URL in policy order,
    respecting robots.txt and per-host politeness, logging change detections.

    Raises:
        ValueError: if *policy* is not a live-crawl policy.
    """
    if policy in ("random", "oracle"):
        raise ValueError(f"Policy {policy!r} is only usable in replay simulation")
    started = time.time()
    frontier = Frontier(host_delay=HOST_DELAY_SECONDS)
    _add_candidates(store, frontier, started, policy)

    fetched = changed = errors = robots_blocked = discovered = 0
    while frontier.remaining() > 0 and fetched < budget:
        now = time.time()
        candidate = frontier.next_ready(now)
        if candidate is None:
            wait = frontier.wait_seconds(now)
            if wait <= 0:
                break
            sleep_until(now + wait)
            continue
        if not robots.can_fetch(candidate.url):
            robots_blocked += 1
            frontier.mark_skipped(candidate)
            continue

        delay = robots.crawl_delay(candidate.url)
        if delay is not None:
            frontier.host_delay_override[candidate.host] = min(delay, _MAX_ROBOTS_DELAY)

        outcome = _fetch_one(store, session, candidate, time.time())
        fetched += 1
        if outcome.changed:
            changed += 1
        if outcome.error:
            errors += 1
        if outcome.html and store.page_count() < page_cap:
            discovered += _enqueue_links(store, candidate.url, outcome.html, page_cap)
        frontier.mark_fetched(candidate, time.time())

    store.recompute_importance()
    finished = time.time()
    stats = PassStats(
        started_ts=started,
        duration_s=finished - started,
        fetched=fetched,
        changed=changed,
        errors=errors,
        robots_blocked=robots_blocked,
        discovered=discovered,
    )
    store.record_pass(started, fetched, changed, errors)
    return stats


def run_loop(
    store: Store,
    budget: int,
    policy: str,
    page_cap: int,
    interval_s: float,
    passes: int | None,
) -> list[PassStats]:
    """Run *passes* crawl passes (None = forever), sleeping *interval_s* between them."""
    results: list[PassStats] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        robots = RobotsCache(session)
        count = 0
        while passes is None or count < passes:
            stats = run_pass(store, session, robots, budget, policy, page_cap)
            results.append(stats)
            count += 1
            if passes is not None and count >= passes:
                break
            time.sleep(interval_s)
    return results


def _add_candidates(store: Store, frontier: Frontier, now: float, policy: str) -> None:
    states = [
        UrlState(
            url=page.url,
            importance=page.importance,
            last_fetch_ts=page.last_fetch_ts,
            available_ts=page.discovered_ts,
            fetch_count=page.fetch_count,
            change_count=page.change_count,
        )
        for page in store.all_pages()
    ]
    priorities = {state.url: score(state, now, policy) for state in states}
    for state in states:
        frontier.add(
            Candidate(
                url=state.url,
                host=host_of(state.url),
                priority=priorities[state.url],
                last_fetch_ts=state.last_fetch_ts,
            )
        )


def _fetch_one(
    store: Store,
    session: requests.Session,
    candidate: Candidate,
    now: float,
) -> FetchOutcome:
    """Fetch, extract, detect change for one URL; returns per-fetch outcome counters."""
    previous = store.get_page(candidate.url)
    try:
        result = fetch(
            candidate.url,
            session,
            timeout=FETCH_TIMEOUT_SECONDS,
        )
    except (requests.exceptions.ConnectionError, TimeoutError, OSError) as exc:
        store.record_fetch(
            url=candidate.url,
            host=candidate.host,
            ts=now,
            status=None,
            nbytes=0,
            title="",
            text="",
            sha_raw="",
            sha_stripped="",
            truth_changed=None,
            pred_changed=None,
            sim=None,
            error=str(exc),
        )
        return FetchOutcome(changed=False, error=str(exc), html="")

    if result.status != 200 or not result.html:
        error = None if result.status == 200 else f"HTTP {result.status}"
        store.record_fetch(
            url=candidate.url,
            host=candidate.host,
            ts=now,
            status=result.status,
            nbytes=result.nbytes,
            title="",
            text="",
            sha_raw="",
            sha_stripped="",
            truth_changed=None,
            pred_changed=None,
            sim=None,
            error=error,
        )
        return FetchOutcome(changed=False, error=error, html="")

    extracted = extract(result.html, result.final_url)
    prev_text = previous.text if previous else ""
    if prev_text:
        pred_changed, sim = detect_change(prev_text, extracted.text, JACCARD_TAU, SHINGLE_K)
        changed = truth_changed(prev_text, extracted.text)
        truth_flag: bool | None = changed
        pred_flag: bool | None = pred_changed
    else:
        sim = None
        truth_flag = None
        pred_flag = None
        changed = False

    store.record_fetch(
        url=candidate.url,
        host=candidate.host,
        ts=now,
        status=result.status,
        nbytes=result.nbytes,
        title=extracted.title,
        text=extracted.text,
        sha_raw=content_sha(extracted.text),
        sha_stripped=content_sha(strip_volatile(extracted.text)),
        truth_changed=truth_flag,
        pred_changed=pred_flag,
        sim=sim,
        error=None,
    )
    return FetchOutcome(changed=changed, error=None, html=result.html)


def _enqueue_links(store: Store, source_url: str, html: str, page_cap: int) -> int:
    """Track newly discovered allowed links; returns how many were added."""
    added = 0
    for link in extract_links(html, source_url):
        if store.page_count() >= page_cap:
            break
        if not is_allowed_host(link):
            continue
        try:
            normalized = normalize_url(link)
        except ValueError:
            continue
        before = store.page_count()
        store.add_page(normalized, host_of(normalized))
        store.add_backlink(source_url, normalized)
        added += store.page_count() - before
    return added
