"""Small, auditable statistical primitives used by the evaluators."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any


def safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def accuracy(expected: Sequence[Any], predicted: Sequence[Any]) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    return safe_div(sum(a == b for a, b in zip(expected, predicted)), len(expected))


def classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, float]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    labels = sorted(set(expected) | set(predicted))
    f1_by_label: list[float] = []
    tp_total = fp_total = fn_total = 0
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1_by_label.append(safe_div(2 * precision * recall, precision + recall))
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_precision = safe_div(tp_total, tp_total + fp_total)
    micro_recall = safe_div(tp_total, tp_total + fn_total)
    return {
        "accuracy": accuracy(expected, predicted),
        "macro_f1": mean(f1_by_label),
        "micro_f1": safe_div(
            2 * micro_precision * micro_recall,
            micro_precision + micro_recall,
        ),
    }


def cohen_kappa(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """Cohen's kappa for aligned nominal ratings."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    if not expected:
        return 0.0
    observed = accuracy(expected, predicted)
    expected_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    n = len(expected)
    chance = sum(
        (expected_counts[label] / n) * (predicted_counts[label] / n)
        for label in set(expected_counts) | set(predicted_counts)
    )
    if chance == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - chance) / (1 - chance)


def krippendorff_alpha_nominal(units: Sequence[Sequence[str | None]]) -> float:
    """Krippendorff's alpha for nominal data with optional missing ratings.

    `units` is a sequence of decision units; each unit contains ratings from any
    number of annotators. Missing ratings are represented by ``None``.
    """

    cleaned = [[rating for rating in unit if rating is not None] for unit in units]
    cleaned = [unit for unit in cleaned if len(unit) >= 2]
    if not cleaned:
        return 0.0

    observed_disagreements = 0.0
    observed_pairs = 0.0
    marginal = Counter()
    for unit in cleaned:
        marginal.update(unit)
        for left_index, left in enumerate(unit):
            for right_index, right in enumerate(unit):
                if left_index == right_index:
                    continue
                observed_pairs += 1
                observed_disagreements += left != right
    observed = safe_div(observed_disagreements, observed_pairs)

    total = sum(marginal.values())
    expected_pairs = total * (total - 1)
    if expected_pairs == 0:
        return 1.0
    agreement_pairs = sum(count * (count - 1) for count in marginal.values())
    expected = 1 - safe_div(agreement_pairs, expected_pairs)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1 - observed / expected


def multilabel_metrics(
    expected_sets: Sequence[set[str]],
    predicted_sets: Sequence[set[str]],
) -> dict[str, float]:
    if len(expected_sets) != len(predicted_sets):
        raise ValueError("expected and predicted lengths differ")
    labels = sorted(set().union(*expected_sets, *predicted_sets)) if expected_sets else []
    tp_total = fp_total = fn_total = 0
    per_label_f1: list[float] = []
    jaccards: list[float] = []
    exact = 0
    for expected, predicted in zip(expected_sets, predicted_sets):
        intersection = expected & predicted
        union = expected | predicted
        exact += expected == predicted
        jaccards.append(safe_div(len(intersection), len(union), default=1.0))
    for label in labels:
        tp = sum(label in expected and label in predicted for expected, predicted in zip(expected_sets, predicted_sets))
        fp = sum(label not in expected and label in predicted for expected, predicted in zip(expected_sets, predicted_sets))
        fn = sum(label in expected and label not in predicted for expected, predicted in zip(expected_sets, predicted_sets))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_label_f1.append(safe_div(2 * precision * recall, precision + recall))
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = safe_div(tp_total, tp_total + fp_total, default=1.0)
    recall = safe_div(tp_total, tp_total + fn_total, default=1.0)
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": safe_div(2 * precision * recall, precision + recall, default=1.0),
        "macro_f1": mean(per_label_f1) if per_label_f1 else 1.0,
        "exact_match": safe_div(exact, len(expected_sets)),
        "mean_jaccard": mean(jaccards),
    }


def mean_absolute_error(expected: Sequence[float], predicted: Sequence[float]) -> float:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    return mean(abs(a - b) for a, b in zip(expected, predicted))


def bootstrap_interval(
    items: Sequence[Any],
    statistic: Callable[[list[Any]], float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not items:
        return {
            "low": 0.0,
            "high": 0.0,
            "confidence": confidence,
            "resamples": resamples,
        }
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [items[rng.randrange(len(items))] for _ in items]
        estimates.append(float(statistic(sample)))
    tail = (1 - confidence) / 2
    return {
        "low": percentile(estimates, tail),
        "high": percentile(estimates, 1 - tail),
        "confidence": confidence,
        "resamples": resamples,
    }

