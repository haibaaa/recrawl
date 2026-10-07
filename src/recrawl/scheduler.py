"""Recrawl policies and the replay simulator.

Policies rank URLs for recrawl; the simulator replays the live crawl history
(complete observations from uniform hourly passes) and lets each policy spend a
fetch budget counterfactually, measuring index freshness at the horizon.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

POLICIES: tuple[str, ...] = ("rr", "random", "importance", "proposed", "oracle")

_POLICY_LABELS = {
    "rr": "round-robin (oldest fetch first)",
    "random": "random",
    "importance": "static importance only",
    "proposed": "proposed: change-rate x staleness x importance",
    "oracle": "oracle (knows the future)",
}


@dataclass
class UrlState:
    """The view a policy has of one URL."""

    url: str
    importance: float = 0.0
    last_fetch_ts: float | None = None
    available_ts: float = 0.0
    fetch_count: int = 0
    change_count: int = 0


def lambda_hat(state: UrlState) -> float:
    """Laplace-smoothed estimate of a URL's per-fetch change probability."""
    return (state.change_count + 1) / (state.fetch_count + 2)


def staleness_hours(state: UrlState, now: float) -> float:
    """Hours since the state's last fetch (or since it became available)."""
    reference = state.last_fetch_ts if state.last_fetch_ts is not None else state.available_ts
    return max(0.0, (now - reference) / 3600.0)


def score(state: UrlState, now: float, policy: str) -> float:
    """Priority of *state* under *policy*; higher means fetch sooner."""
    if policy == "rr":
        if state.last_fetch_ts is None:
            return float("inf")
        return staleness_hours(state, now)
    if policy == "importance":
        return state.importance
    if policy == "proposed":
        if state.last_fetch_ts is None:
            return float("inf")
        return (
            (0.05 + lambda_hat(state))
            * (1.0 + staleness_hours(state, now))
            * (0.25 + state.importance)
        )
    if policy == "random":
        return 0.0
    raise ValueError(f"Unknown policy: {policy!r}")


def rank(
    states: list[UrlState], now: float, policy: str, rng: random.Random | None = None
) -> list[UrlState]:
    """Order *states* best-first under *policy* (ties broken by URL)."""
    if policy == "random":
        shuffled = list(states)
        (rng or random.Random()).shuffle(shuffled)
        return shuffled
    return sorted(states, key=lambda state: (-score(state, now, policy), state.url))


@dataclass
class SimResult:
    """Endpoint metrics of one simulated policy run.

    ``stale_change_hours`` (the headline) is total content age: for every true
    change, the hours it went un-fetched. ``fresh_fraction`` and
    ``mean_fetch_age_hours`` are complementary point-in-time views at the horizon.
    """

    policy: str
    budget: int
    fetches: int
    fresh_fraction: float
    missed_changes: int
    stale_change_hours: float
    mean_fetch_age_hours: float


@dataclass
class SimulationInput:
    """Complete live history replayed counterfactually by the simulator."""

    pass_times: list[float]
    truth_events: dict[str, list[float]] = field(default_factory=dict)
    available_from: dict[str, float] = field(default_factory=dict)
    importance: dict[str, float] = field(default_factory=dict)


def simulate(
    data: SimulationInput,
    policy: str,
    budget: int,
    seed: int = 0,
) -> SimResult:
    """Run *policy* over *data*'s passes, spending up to *budget* fetches per pass.

    The per-pass budget mirrors the live loop's ``--budget`` flag, so annualised
    fetch volume is ``budget x passes``. Freshness at the horizon: a URL is fresh
    when the policy has fetched it after its last true content change.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy!r}")
    rounds = data.pass_times
    if not rounds:
        raise ValueError("Simulation needs at least one pass")
    horizon = rounds[-1]
    rng = random.Random(seed)

    base = {
        url: UrlState(url=url, importance=data.importance.get(url, 0.0), available_ts=available)
        for url, available in data.available_from.items()
    }
    fetches = 0
    stale_change_hours = 0.0
    prev_ts: float | None = None

    for now in rounds:
        states = [s for s in base.values() if s.available_ts <= now]
        if not states:
            continue
        per_round = budget
        if policy == "oracle":
            ordered = _oracle_rank(states, data, prev_ts, now, rng)
        else:
            ordered = rank(states, now, policy, rng)
        for state in ordered[:per_round]:
            previous_fetch = state.last_fetch_ts
            fetches += 1
            state.fetch_count += 1
            state.last_fetch_ts = now
            if previous_fetch is not None:
                for event in data.truth_events.get(state.url, []):
                    if previous_fetch < event <= now:
                        state.change_count += 1
                        stale_change_hours += (now - event) / 3600.0
        prev_ts = now

    fresh = 0
    missed = 0
    fetch_age_sum = 0.0
    for url, state in base.items():
        events = data.truth_events.get(url, [])
        events_after = [event for event in events if event > (state.last_fetch_ts or 0.0)]
        if state.last_fetch_ts is not None and not events_after:
            fresh += 1
        missed += len(events_after)
        for event in events_after:
            stale_change_hours += (horizon - event) / 3600.0
        reference = state.last_fetch_ts if state.last_fetch_ts is not None else state.available_ts
        fetch_age_sum += max(0.0, horizon - reference) / 3600.0

    total = len(base) or 1
    return SimResult(
        policy=policy,
        budget=budget,
        fetches=fetches,
        fresh_fraction=fresh / total,
        missed_changes=missed,
        stale_change_hours=stale_change_hours,
        mean_fetch_age_hours=fetch_age_sum / total,
    )


def _oracle_rank(
    states: list[UrlState],
    data: SimulationInput,
    prev_ts: float | None,
    now: float,
    rng: random.Random,
) -> list[UrlState]:
    """Perfect-change oracle: fetch pages that just changed since the last pass first.

    A page's change is observable at the first pass after it happened, so the
    oracle schedules those changed pages to the front. Everything else falls back
    to the proposed order, so the oracle upper-bounds the best feasible policy.
    """

    def changed_since(state: UrlState) -> int:
        if prev_ts is None:
            return 0
        return int(any(prev_ts < event <= now for event in data.truth_events.get(state.url, [])))

    ordered = sorted(states, key=lambda s: (-changed_since(s), -score(s, now, "proposed"), s.url))
    return ordered


def policy_label(policy: str) -> str:
    """Human-readable label of *policy* for reports and plots."""
    return _POLICY_LABELS[policy]
