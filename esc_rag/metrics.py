"""Ordinal classification metrics for prognosis evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from statistics import mean
from typing import Iterable


def prognosis_metrics(pairs: Iterable[tuple[int, int]]) -> dict[str, float | int]:
    pairs = list(pairs)
    if not pairs:
        return {"n": 0}
    diffs = [pred - actual for pred, actual in pairs]
    absdiff = [abs(d) for d in diffs]
    return {
        "n": len(pairs),
        "exact_accuracy": sum(d == 0 for d in diffs) / len(diffs),
        "mae": mean(absdiff),
        "rmse": sqrt(mean([d * d for d in diffs])),
        "within_1": sum(d <= 1 for d in absdiff) / len(absdiff),
        "within_2": sum(d <= 2 for d in absdiff) / len(absdiff),
        "bias": mean(diffs),
        "overcall_rate": sum(d > 0 for d in diffs) / len(diffs),
        "undercall_rate": sum(d < 0 for d in diffs) / len(diffs),
    }


def per_class_recall(pairs: Iterable[tuple[int, int]]) -> dict[int, float]:
    totals: dict[int, int] = defaultdict(int)
    hits: dict[int, int] = defaultdict(int)
    for pred, actual in pairs:
        totals[actual] += 1
        if pred == actual:
            hits[actual] += 1
    return {klass: hits[klass] / total for klass, total in sorted(totals.items())}


def confusion_matrix(pairs: Iterable[tuple[int, int]]) -> dict[int, dict[int, int]]:
    matrix: dict[int, Counter[int]] = defaultdict(Counter)
    for pred, actual in pairs:
        matrix[actual][pred] += 1
    return {actual: dict(counts) for actual, counts in sorted(matrix.items())}
