"""Tests for the politeness frontier."""

from recrawl.frontier import Candidate, Frontier


def _candidate(url: str, priority: float) -> Candidate:
    host = url.split("/")[2]
    return Candidate(url=url, host=host, priority=priority)


def test_selects_highest_priority_ready_candidate() -> None:
    frontier = Frontier(host_delay=1.0)
    frontier.add(_candidate("https://a.example/x", 1.0))
    frontier.add(_candidate("https://b.example/y", 5.0))
    chosen = frontier.next_ready(now=100.0)
    assert chosen is not None
    assert chosen.url == "https://b.example/y"


def test_politeness_blocks_host_until_delay_passes() -> None:
    frontier = Frontier(host_delay=10.0)
    first = _candidate("https://a.example/1", 1.0)
    second = _candidate("https://a.example/2", 2.0)
    frontier.add(first)
    frontier.add(second)
    frontier.mark_fetched(first, now=100.0)
    assert frontier.next_ready(now=105.0) is None
    assert frontier.next_ready(now=110.0) is not None


def test_other_hosts_remain_ready() -> None:
    frontier = Frontier(host_delay=10.0)
    first = _candidate("https://a.example/1", 1.0)
    other = _candidate("https://b.example/2", 0.5)
    frontier.add(first)
    frontier.add(other)
    frontier.mark_fetched(first, now=100.0)
    chosen = frontier.next_ready(now=101.0)
    assert chosen is not None
    assert chosen.url == "https://b.example/2"


def test_earliest_ready_and_wait() -> None:
    frontier = Frontier(host_delay=4.0)
    first = _candidate("https://a.example/1", 1.0)
    second = _candidate("https://a.example/2", 0.5)
    frontier.add(first)
    frontier.add(second)
    frontier.mark_fetched(first, now=100.0)
    assert frontier.earliest_ready_ts() == 104.0
    assert frontier.wait_seconds(now=100.0) == 4.0
    assert frontier.wait_seconds(now=110.0) == 0.0


def test_mark_skipped_drops_candidate() -> None:
    frontier = Frontier(host_delay=1.0)
    candidate = _candidate("https://a.example/1", 1.0)
    frontier.add(candidate)
    frontier.mark_skipped(candidate)
    assert frontier.remaining() == 0
    assert frontier.next_ready(now=0.0) is None


def test_host_delay_override() -> None:
    frontier = Frontier(host_delay=1.0)
    first = _candidate("https://a.example/1", 1.0)
    second = _candidate("https://a.example/2", 0.5)
    frontier.add(first)
    frontier.add(second)
    frontier.host_delay_override["a.example"] = 60.0
    frontier.mark_fetched(first, now=100.0)
    assert frontier.next_ready(now=130.0) is None
    assert frontier.next_ready(now=160.0) is not None
