"""Regression-policy evaluation over immutable JSON reports."""

from __future__ import annotations

from typing import Any


def get_metric(document: dict[str, Any], dotted_path: str) -> float:
    value: Any = document
    traversed: list[str] = []
    for segment in dotted_path.split("."):
        traversed.append(segment)
        if not isinstance(value, dict) or segment not in value:
            raise KeyError(".".join(traversed))
        value = value[segment]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{dotted_path} is not numeric")
    return float(value)


def evaluate_gates(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for rule in config.get("rules", []):
        rule_id = rule["id"]
        path = rule["metric"]
        operator = rule["operator"]
        threshold = float(rule["value"])
        severity = rule.get("severity", "block")
        try:
            candidate_value = get_metric(candidate, path)
            baseline_value = (
                get_metric(baseline, path)
                if baseline is not None and operator in {"max_drop", "max_increase"}
                else None
            )
            if operator == "gte":
                passed = candidate_value >= threshold
                boundary = threshold
            elif operator == "lte":
                passed = candidate_value <= threshold
                boundary = threshold
            elif operator == "max_drop":
                if baseline_value is None:
                    raise ValueError("max_drop requires a baseline")
                boundary = baseline_value - threshold
                passed = candidate_value >= boundary
            elif operator == "max_increase":
                if baseline_value is None:
                    raise ValueError("max_increase requires a baseline")
                boundary = baseline_value + threshold
                passed = candidate_value <= boundary
            else:
                raise ValueError(f"unsupported operator: {operator}")
            result = {
                "id": rule_id,
                "metric": path,
                "operator": operator,
                "threshold": threshold,
                "boundary": boundary,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "severity": severity,
                "passed": passed,
            }
        except (KeyError, TypeError, ValueError) as exc:
            result = {
                "id": rule_id,
                "metric": path,
                "operator": operator,
                "threshold": threshold,
                "severity": severity,
                "passed": False,
                "error": str(exc),
            }
        results.append(result)

    blocking_failures = [
        result
        for result in results
        if not result["passed"] and result["severity"] == "block"
    ]
    warnings = [
        result
        for result in results
        if not result["passed"] and result["severity"] != "block"
    ]
    return {
        "policy": {
            "name": config.get("name"),
            "version": config.get("version"),
        },
        "passed": not blocking_failures,
        "blocking_failures": len(blocking_failures),
        "warnings": len(warnings),
        "rules": results,
    }

