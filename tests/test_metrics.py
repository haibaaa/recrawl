"""Tests for evaluation metrics."""

import pytest

from recrawl.metrics import (
    DetectorScores,
    RankingScores,
    average_precision,
    f1,
    precision_at_k,
    recall,
)


def test_precision_at_k() -> None:
    relevant = {"a", "b", "c"}
    retrieved = ["a", "x", "b", "y", "c"]
    assert precision_at_k(relevant, retrieved, 2) == pytest.approx(0.5)
    assert precision_at_k(relevant, retrieved, 5) == pytest.approx(0.6)


def test_precision_at_k_empty_retrieval() -> None:
    assert precision_at_k({"a"}, [], 5) == 0.0


def test_precision_at_k_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k({"a"}, ["a"], 0)


def test_recall() -> None:
    assert recall({"a", "b"}, ["a"]) == pytest.approx(0.5)
    assert recall(set(), ["a"]) == 0.0


def test_f1() -> None:
    assert f1(1.0, 1.0) == pytest.approx(1.0)
    assert f1(0.0, 0.5) == 0.0


def test_average_precision() -> None:
    relevant = {"a", "b"}
    assert average_precision(relevant, ["a", "x", "b"]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(set(), ["a"]) == 0.0


def test_detector_scores() -> None:
    scores = DetectorScores()
    scores.add(predicted=True, truth=True)
    scores.add(predicted=True, truth=False)
    scores.add(predicted=False, truth=True)
    scores.add(predicted=False, truth=False)
    assert scores.precision == pytest.approx(0.5)
    assert scores.recall == pytest.approx(0.5)
    assert scores.f1 == pytest.approx(0.5)
    assert scores.accuracy == pytest.approx(0.5)
    assert scores.as_dict()["tp"] == 1.0


def test_detector_scores_empty_is_zero() -> None:
    scores = DetectorScores()
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.accuracy == 0.0


def test_ranking_scores() -> None:
    scores = RankingScores(relevant={"a", "b"}, retrieved=["a", "x", "b"])
    assert scores.precision_at(1) == pytest.approx(1.0)
    assert scores.precision_at(3) == pytest.approx(2 / 3)
    assert scores.recall_full == pytest.approx(1.0)
    assert scores.average_precision == pytest.approx((1 + 2 / 3) / 2)
