"""Tests for recrawl policies and the replay simulator."""

import random

import pytest

from recrawl.scheduler import (
    SimulationInput,
    UrlState,
    lambda_hat,
    rank,
    score,
    simulate,
    staleness_hours,
)


def _state(
    url: str,
    *,
    importance: float = 0.5,
    last_fetch: float | None = None,
    available: float = 0.0,
    fetches: int = 0,
    changes: int = 0,
) -> UrlState:
    return UrlState(
        url=url,
        importance=importance,
        last_fetch_ts=last_fetch,
        available_ts=available,
        fetch_count=fetches,
        change_count=changes,
    )


def test_lambda_hat_is_laplace_smoothed() -> None:
    assert lambda_hat(_state("a")) == pytest.approx(0.5)
    assert lambda_hat(_state("a", fetches=10, changes=4)) == pytest.approx(5 / 12)


def test_staleness_hours_uses_available_when_never_fetched() -> None:
    state = _state("a", available=3600.0)
    assert staleness_hours(state, now=3600.0 * 3) == pytest.approx(2.0)


def test_rank_rr_prefers_oldest_fetch() -> None:
    states = [
        _state("a", last_fetch=100.0),
        _state("b", last_fetch=10.0),
        _state("c", last_fetch=50.0),
    ]
    ordered = [state.url for state in rank(states, now=200.0, policy="rr")]
    assert ordered == ["b", "c", "a"]


def test_rank_importance_descending() -> None:
    states = [_state("a", importance=0.2), _state("b", importance=0.9), _state("c", importance=0.5)]
    ordered = [state.url for state in rank(states, now=0.0, policy="importance")]
    assert ordered == ["b", "c", "a"]


def test_rank_proposed_prefers_high_change_rate() -> None:
    stable = _state("stable", fetches=10, changes=0, last_fetch=0.0)
    volatile = _state("volatile", fetches=10, changes=6, last_fetch=0.0)
    ordered = [state.url for state in rank([stable, volatile], now=3600.0, policy="proposed")]
    assert ordered[0] == "volatile"


def test_rank_random_is_seeded() -> None:
    states = [_state(f"u{i}") for i in range(10)]
    first = [s.url for s in rank(states, now=0.0, policy="random", rng=random.Random(7))]
    second = [s.url for s in rank(states, now=0.0, policy="random", rng=random.Random(7))]
    assert first == second
    assert sorted(first) == sorted(s.url for s in states)


def test_score_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        score(_state("a"), now=0.0, policy="nope")


def _history_input() -> SimulationInput:
    passes = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    urls = ["a", "b", "c", "d"]
    return SimulationInput(
        pass_times=passes,
        truth_events={"a": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]},
        available_from={url: 0.0 for url in urls},
        importance={url: 0.5 for url in urls},
    )


def test_full_budget_yields_fresh_index() -> None:
    result = simulate(_history_input(), policy="proposed", budget=10)
    assert result.fetches == 24
    assert result.fresh_fraction == pytest.approx(1.0)
    assert result.missed_changes == 0
    assert result.mean_fetch_age_hours == pytest.approx(0.0)
    assert result.stale_change_hours == pytest.approx(0.0)


def test_zero_budget_yields_stale_index() -> None:
    result = simulate(_history_input(), policy="proposed", budget=0)
    assert result.fetches == 0
    assert result.fresh_fraction == pytest.approx(0.0)
    assert result.missed_changes == 5


def test_budget_is_fetches_per_pass_not_total() -> None:
    data = _history_input()
    result = simulate(data, policy="proposed", budget=1)
    # 1 fetch per pass, every pass -> 6 total (not 1), exactly the bug that
    # previously swallowed per-pass budgets smaller than the pass count.
    assert result.fetches == 6


def test_proposed_misses_fewer_changes_than_round_robin() -> None:
    data = _history_input()
    proposed = simulate(data, policy="proposed", budget=2)
    round_robin = simulate(data, policy="rr", budget=2)
    assert proposed.missed_changes < round_robin.missed_changes


def test_oracle_is_at_least_as_good_as_proposed() -> None:
    data = _history_input()
    proposed = simulate(data, policy="proposed", budget=2)
    oracle = simulate(data, policy="oracle", budget=2)
    assert oracle.missed_changes <= proposed.missed_changes


def test_urls_become_available_only_from_their_first_pass() -> None:
    data = _history_input()
    data.available_from["d"] = 3000.0
    result = simulate(data, policy="proposed", budget=10)
    # d only exists from pass 3 onward: 3 URLs x 6 passes + d's 3 passes = 21 fetches.
    assert result.fetches == 21
    assert result.fresh_fraction == pytest.approx(1.0)


def test_simulate_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        simulate(_history_input(), policy="nope", budget=5)


def test_simulate_requires_passes() -> None:
    with pytest.raises(ValueError):
        simulate(SimulationInput(pass_times=[]), policy="rr", budget=1)
