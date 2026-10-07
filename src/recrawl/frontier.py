"""Priority frontier with per-host politeness (Mercator-style ready queues)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A URL eligible for fetching in the current pass."""

    url: str
    host: str
    priority: float
    last_fetch_ts: float | None = None


@dataclass
class Frontier:
    """Selects the highest-priority URL whose host is ready.

    Callers add the due candidates for a pass, then repeatedly call
    :meth:`next_ready` / :meth:`earliest_ready_ts` while fetching; each fetch
    is acknowledged with :meth:`mark_fetched`, which enforces the per-host
    politeness delay.
    """

    host_delay: float
    candidates: dict[str, Candidate] = field(default_factory=dict)
    host_ready_ts: dict[str, float] = field(default_factory=dict)
    host_delay_override: dict[str, float] = field(default_factory=dict)

    def add(self, candidate: Candidate) -> None:
        """Enqueue *candidate* for this pass (idempotent per URL)."""
        self.candidates[candidate.url] = candidate

    def next_ready(self, now: float) -> Candidate | None:
        """Return the highest-priority candidate whose host is ready at *now*."""
        ready = [
            candidate
            for candidate in self.candidates.values()
            if self.host_ready_ts.get(candidate.host, 0.0) <= now
        ]
        if not ready:
            return None
        return max(ready, key=lambda candidate: candidate.priority)

    def earliest_ready_ts(self) -> float | None:
        """Earliest timestamp at which some remaining candidate's host is ready."""
        hosts = {candidate.host for candidate in self.candidates.values()}
        if not hosts:
            return None
        return min(self.host_ready_ts.get(host, 0.0) for host in hosts)

    def mark_fetched(self, candidate: Candidate, now: float) -> None:
        """Record that *candidate* was fetched at *now*; set the host's next-ready time."""
        delay = self.host_delay_override.get(candidate.host, self.host_delay)
        self.host_ready_ts[candidate.host] = now + delay
        self.candidates.pop(candidate.url, None)

    def mark_skipped(self, candidate: Candidate) -> None:
        """Drop *candidate* from this pass without fetching it."""
        self.candidates.pop(candidate.url, None)

    def remaining(self) -> int:
        """Number of candidates still awaiting selection."""
        return len(self.candidates)

    def wait_seconds(self, now: float) -> float:
        """Seconds to sleep before any candidate can be selected (0 if one is ready)."""
        if self.next_ready(now) is not None:
            return 0.0
        earliest = self.earliest_ready_ts()
        if earliest is None:
            return 0.0
        return max(0.0, earliest - now)


def sleep_until(earliest_ts: float) -> None:
    """Sleep until *earliest_ts*, never less than 0.1 seconds."""
    time.sleep(max(0.1, earliest_ts - time.time()))
