"""Versioned, strict schema validation without third-party runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

TASK_FAMILIES = (
    "qa_scorecard",
    "deal_summary",
    "semantic_analytics",
    "violation_flags",
)

CRITERION_STATES = {"pass", "fail", "not_applicable", "insufficient_data"}
OUTCOMES = {"won", "lost", "follow_up", "undecided", "not_applicable"}
READINESS_STATES = {"cold", "warm", "hot", "insufficient_data"}
FLAG_STATES = {"yes", "no", "insufficient_data"}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _strict_keys(
    value: Any,
    required: set[str],
    allowed: set[str],
    path: str,
) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return [ValidationIssue(path, "must be an object")]
    issues: list[ValidationIssue] = []
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    for key in missing:
        issues.append(ValidationIssue(f"{path}.{key}", "required field is missing"))
    for key in unknown:
        issues.append(ValidationIssue(f"{path}.{key}", "unknown field is not allowed"))
    return issues


def _string(value: Any, path: str, *, nullable: bool = False) -> list[ValidationIssue]:
    if nullable and value is None:
        return []
    if not isinstance(value, str):
        return [ValidationIssue(path, "must be a string")]
    if not value.strip():
        return [ValidationIssue(path, "must not be blank")]
    return []


def _enum(value: Any, options: set[str], path: str) -> list[ValidationIssue]:
    if not isinstance(value, str) or value not in options:
        return [ValidationIssue(path, f"must be one of {sorted(options)}")]
    return []


def _string_list(value: Any, path: str) -> list[ValidationIssue]:
    if not isinstance(value, list):
        return [ValidationIssue(path, "must be an array")]
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        issues.extend(_string(item, item_path))
        if isinstance(item, str):
            normalized = item.strip().casefold()
            if normalized in seen:
                issues.append(ValidationIssue(item_path, "duplicate array item"))
            seen.add(normalized)
    return issues


def _state_map(value: Any, states: set[str], path: str) -> list[ValidationIssue]:
    if not isinstance(value, dict) or not value:
        return [ValidationIssue(path, "must be a non-empty object")]
    issues: list[ValidationIssue] = []
    for key, state in value.items():
        if not isinstance(key, str) or not key.strip():
            issues.append(ValidationIssue(path, "keys must be non-blank strings"))
            continue
        issues.extend(_enum(state, states, f"{path}.{key}"))
    return issues


def _qa_reference(value: Any) -> list[ValidationIssue]:
    path = "$.reference"
    issues = _strict_keys(
        value,
        {"criteria", "weights", "total_score"},
        {"criteria", "weights", "total_score"},
        path,
    )
    if not isinstance(value, dict):
        return issues
    if "criteria" in value:
        issues.extend(_state_map(value["criteria"], CRITERION_STATES, f"{path}.criteria"))
    if "weights" in value:
        weights = value["weights"]
        if not isinstance(weights, dict) or not weights:
            issues.append(ValidationIssue(f"{path}.weights", "must be a non-empty object"))
        else:
            for key, weight in weights.items():
                if not isinstance(key, str) or not key.strip():
                    issues.append(
                        ValidationIssue(f"{path}.weights", "keys must be non-blank strings")
                    )
                if not _is_number(weight) or weight < 0:
                    issues.append(
                        ValidationIssue(f"{path}.weights.{key}", "must be a non-negative number")
                    )
        criteria = value.get("criteria")
        if isinstance(criteria, dict) and isinstance(weights, dict):
            if set(criteria) != set(weights):
                issues.append(
                    ValidationIssue(
                        f"{path}.weights",
                        "weight keys must exactly match criterion keys",
                    )
                )
    if "total_score" in value:
        score = value["total_score"]
        if not _is_number(score) or not 0 <= score <= 100:
            issues.append(ValidationIssue(f"{path}.total_score", "must be between 0 and 100"))
    return issues


def _qa_prediction(value: Any) -> list[ValidationIssue]:
    path = "$.output"
    issues = _strict_keys(
        value,
        {"criteria", "total_score"},
        {"criteria", "total_score"},
        path,
    )
    if not isinstance(value, dict):
        return issues
    if "criteria" in value:
        issues.extend(_state_map(value["criteria"], CRITERION_STATES, f"{path}.criteria"))
    if "total_score" in value:
        score = value["total_score"]
        if not _is_number(score) or not 0 <= score <= 100:
            issues.append(ValidationIssue(f"{path}.total_score", "must be between 0 and 100"))
    return issues


def _deal_summary(value: Any, role: str) -> list[ValidationIssue]:
    path = "$.reference" if role == "reference" else "$.output"
    keys = {"outcome", "primary_need", "objections", "next_step", "facts"}
    issues = _strict_keys(value, keys, keys, path)
    if not isinstance(value, dict):
        return issues
    if "outcome" in value:
        issues.extend(_enum(value["outcome"], OUTCOMES, f"{path}.outcome"))
    if "primary_need" in value:
        issues.extend(_string(value["primary_need"], f"{path}.primary_need"))
    if "objections" in value:
        issues.extend(_string_list(value["objections"], f"{path}.objections"))
    if "next_step" in value:
        issues.extend(_string(value["next_step"], f"{path}.next_step", nullable=True))
    if "facts" in value:
        issues.extend(_string_list(value["facts"], f"{path}.facts"))
    return issues


def _semantic_analytics(value: Any, role: str) -> list[ValidationIssue]:
    path = "$.reference" if role == "reference" else "$.output"
    keys = {"needs", "objections", "readiness", "tariff_interest", "competitor_mentions"}
    issues = _strict_keys(value, keys, keys, path)
    if not isinstance(value, dict):
        return issues
    for key in ("needs", "objections", "tariff_interest", "competitor_mentions"):
        if key in value:
            issues.extend(_string_list(value[key], f"{path}.{key}"))
    if "readiness" in value:
        issues.extend(_enum(value["readiness"], READINESS_STATES, f"{path}.readiness"))
    return issues


def _violation_reference(value: Any) -> list[ValidationIssue]:
    path = "$.reference"
    issues = _strict_keys(
        value,
        {"flags", "critical_flags"},
        {"flags", "critical_flags"},
        path,
    )
    if not isinstance(value, dict):
        return issues
    if "flags" in value:
        issues.extend(_state_map(value["flags"], FLAG_STATES, f"{path}.flags"))
    if "critical_flags" in value:
        issues.extend(_string_list(value["critical_flags"], f"{path}.critical_flags"))
        flags = value.get("flags")
        if isinstance(flags, dict) and isinstance(value["critical_flags"], list):
            for index, flag in enumerate(value["critical_flags"]):
                if isinstance(flag, str) and flag not in flags:
                    issues.append(
                        ValidationIssue(
                            f"{path}.critical_flags[{index}]",
                            "critical flag must exist in flags",
                        )
                    )
    return issues


def _violation_prediction(value: Any) -> list[ValidationIssue]:
    path = "$.output"
    issues = _strict_keys(value, {"flags"}, {"flags"}, path)
    if isinstance(value, dict) and "flags" in value:
        issues.extend(_state_map(value["flags"], FLAG_STATES, f"{path}.flags"))
    return issues


_REFERENCE_VALIDATORS: dict[str, Callable[[Any], list[ValidationIssue]]] = {
    "qa_scorecard": _qa_reference,
    "deal_summary": lambda value: _deal_summary(value, "reference"),
    "semantic_analytics": lambda value: _semantic_analytics(value, "reference"),
    "violation_flags": _violation_reference,
}

_PREDICTION_VALIDATORS: dict[str, Callable[[Any], list[ValidationIssue]]] = {
    "qa_scorecard": _qa_prediction,
    "deal_summary": lambda value: _deal_summary(value, "prediction"),
    "semantic_analytics": lambda value: _semantic_analytics(value, "prediction"),
    "violation_flags": _violation_prediction,
}


def validate_payload(family: str, value: Any, *, role: str) -> list[ValidationIssue]:
    """Validate a version-1 reference or model output for one task family."""
    if family not in TASK_FAMILIES:
        return [ValidationIssue("$.task_family", f"unsupported task family: {family!r}")]
    if role == "reference":
        return _REFERENCE_VALIDATORS[family](value)
    if role == "prediction":
        return _PREDICTION_VALIDATORS[family](value)
    raise ValueError(f"unsupported role: {role!r}")


def validate_gold_record(record: Any) -> list[ValidationIssue]:
    path = "$"
    required = {"record_id", "deal_id", "task_family", "schema_version", "reference", "metadata"}
    allowed = required | {"annotations"}
    issues = _strict_keys(record, required, allowed, path)
    if not isinstance(record, dict):
        return issues
    for key in ("record_id", "deal_id", "schema_version"):
        if key in record:
            issues.extend(_string(record[key], f"$.{key}"))
    family = record.get("task_family")
    if family not in TASK_FAMILIES:
        issues.append(ValidationIssue("$.task_family", f"must be one of {TASK_FAMILIES}"))
    if "metadata" in record and not isinstance(record["metadata"], dict):
        issues.append(ValidationIssue("$.metadata", "must be an object"))
    if family in TASK_FAMILIES and "reference" in record:
        issues.extend(validate_payload(family, record["reference"], role="reference"))
    annotations = record.get("annotations")
    if annotations is not None:
        if not isinstance(annotations, list):
            issues.append(ValidationIssue("$.annotations", "must be an array"))
        else:
            for index, annotation in enumerate(annotations):
                annotation_path = f"$.annotations[{index}]"
                annotation_issues = _strict_keys(
                    annotation,
                    {"annotator_id", "output"},
                    {"annotator_id", "output"},
                    annotation_path,
                )
                issues.extend(annotation_issues)
                if isinstance(annotation, dict):
                    if "annotator_id" in annotation:
                        issues.extend(
                            _string(annotation["annotator_id"], f"{annotation_path}.annotator_id")
                        )
                    if family in TASK_FAMILIES and "output" in annotation:
                        # Human annotations use the reference shape because they include policy
                        # fields such as score weights or critical-flag metadata.
                        issues.extend(
                            validate_payload(family, annotation["output"], role="reference")
                        )
    return issues


def validate_prediction_record(record: Any) -> list[ValidationIssue]:
    path = "$"
    required = {"record_id", "task_family", "output"}
    allowed = required | {"telemetry"}
    issues = _strict_keys(record, required, allowed, path)
    if not isinstance(record, dict):
        return issues
    if "record_id" in record:
        issues.extend(_string(record["record_id"], "$.record_id"))
    family = record.get("task_family")
    if family not in TASK_FAMILIES:
        issues.append(ValidationIssue("$.task_family", f"must be one of {TASK_FAMILIES}"))
    if family in TASK_FAMILIES and "output" in record:
        issues.extend(validate_payload(family, record["output"], role="prediction"))
    telemetry = record.get("telemetry")
    if telemetry is not None:
        telemetry_keys = {"latency_ms", "input_tokens", "output_tokens"}
        issues.extend(_strict_keys(telemetry, telemetry_keys, telemetry_keys, "$.telemetry"))
        if isinstance(telemetry, dict):
            latency = telemetry.get("latency_ms")
            if not _is_number(latency) or latency < 0:
                issues.append(ValidationIssue("$.telemetry.latency_ms", "must be non-negative"))
            for key in ("input_tokens", "output_tokens"):
                value = telemetry.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    issues.append(ValidationIssue(f"$.telemetry.{key}", "must be a non-negative int"))
    return issues

