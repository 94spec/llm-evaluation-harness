"""Human annotation reliability over comparable nominal decision units."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .metrics import accuracy, cohen_kappa, krippendorff_alpha_nominal


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _decision_map(family: str, output: dict[str, Any]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    if family == "qa_scorecard":
        for criterion, state in output.get("criteria", {}).items():
            decisions[f"criterion:{criterion}"] = str(state)
    elif family == "deal_summary":
        decisions["outcome"] = str(output.get("outcome"))
        decisions["primary_need"] = _normalize(str(output.get("primary_need", "")))
        decisions["next_step_present"] = str(output.get("next_step") is not None)
        for objection in output.get("objections", []):
            decisions[f"objection:{_normalize(objection)}"] = "present"
    elif family == "semantic_analytics":
        decisions["readiness"] = str(output.get("readiness"))
        for dimension in ("needs", "objections", "tariff_interest", "competitor_mentions"):
            for label in output.get(dimension, []):
                decisions[f"{dimension}:{_normalize(label)}"] = "present"
    elif family == "violation_flags":
        for flag, state in output.get("flags", {}).items():
            decisions[f"flag:{flag}"] = str(state)
    return decisions


def annotation_agreement(gold_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute agreement before adjudication when raw annotations are present.

    Set-valued annotations are expanded to binary present/absent units over the union
    of labels supplied by annotators for the same record.
    """

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold_records:
        if len(record.get("annotations", [])) >= 2:
            by_family[record["task_family"]].append(record)

    result: dict[str, Any] = {}
    for family, records in sorted(by_family.items()):
        alpha_units: list[list[str | None]] = []
        two_rater_expected: list[str] = []
        two_rater_predicted: list[str] = []
        record_count = 0
        for record in records:
            annotations = sorted(
                record["annotations"],
                key=lambda item: item["annotator_id"],
            )
            maps = [_decision_map(family, annotation["output"]) for annotation in annotations]
            keys = sorted(set().union(*(set(mapping) for mapping in maps)))
            if not keys:
                continue
            record_count += 1
            for key in keys:
                ratings = [mapping.get(key, "absent") for mapping in maps]
                alpha_units.append(ratings)
                two_rater_expected.append(ratings[0])
                two_rater_predicted.append(ratings[1])
        if alpha_units:
            result[family] = {
                "records_with_multiple_annotations": record_count,
                "decision_units": len(alpha_units),
                "two_rater_raw_agreement": accuracy(
                    two_rater_expected,
                    two_rater_predicted,
                ),
                "two_rater_cohen_kappa": cohen_kappa(
                    two_rater_expected,
                    two_rater_predicted,
                ),
                "krippendorff_alpha_nominal": krippendorff_alpha_nominal(alpha_units),
                "note": (
                    "Annotation reliability only; model performance uses the adjudicated "
                    "reference field."
                ),
            }
    return result

