"""Evaluation metrics: P@k, recall, F1, average precision, detector confusion."""

from __future__ import annotations

from dataclasses import dataclass


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Precision among the top *k* retrieved identifiers."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def recall(relevant: set[str], retrieved: list[str]) -> float:
    """Recall of *relevant* items over the full retrieved list."""
    if not relevant:
        return 0.0
    return sum(1 for item in retrieved if item in relevant) / len(relevant)


def f1(precision: float, recall_value: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall_value == 0:
        return 0.0
    return 2 * precision * recall_value / (precision + recall_value)


def average_precision(relevant: set[str], retrieved: list[str]) -> float:
    """Average precision over *retrieved* (AP)."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


@dataclass
class DetectorScores:
    """Accumulated binary decisions of a change detector."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted: bool, truth: bool) -> None:
        """Record one (prediction, truth) pair."""
        if predicted and truth:
            self.tp += 1
        elif predicted and not truth:
            self.fp += 1
        elif not predicted and not truth:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def n(self) -> int:
        """Total number of scored decisions."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        """Precision of the changed prediction."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        """Recall of the changed prediction."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        """F1 of the changed prediction."""
        return f1(self.precision, self.recall)

    @property
    def accuracy(self) -> float:
        """Fraction of correct decisions."""
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def as_dict(self) -> dict[str, float]:
        """All scores as a plain dict, for tables and CSV."""
        return {
            "tp": float(self.tp),
            "fp": float(self.fp),
            "tn": float(self.tn),
            "fn": float(self.fn),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


@dataclass
class RankingScores:
    """P@k and related measures for one query."""

    relevant: set[str]
    retrieved: list[str]

    def precision_at(self, k: int) -> float:
        """Precision at cutoff *k*."""
        return precision_at_k(self.relevant, self.retrieved, k)

    @property
    def recall_full(self) -> float:
        """Recall over the whole retrieved list."""
        return recall(self.relevant, self.retrieved)

    @property
    def average_precision(self) -> float:
        """Average precision over the retrieved list."""
        return average_precision(self.relevant, self.retrieved)
