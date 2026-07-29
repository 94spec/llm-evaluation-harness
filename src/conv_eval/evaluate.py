"""Task-aware evaluation, uncertainty, operations, and error-ledger assembly."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import random
from typing import Any

from .agreement import annotation_agreement
from .io import DataError
from .metrics import (
    accuracy,
    classification_metrics,
    cohen_kappa,
    mean,
    mean_absolute_error,
    multilabel_metrics,
    percentile,
    safe_div,
)
from .schema import (
    TASK_FAMILIES,
    ValidationIssue,
    validate_gold_record,
    validate_prediction_record,
)


@dataclass
class EvalPair:
    gold: dict[str, Any]
    prediction: dict[str, Any] | None
    schema_issues: list[ValidationIssue]

    @property
    def family(self) -> str:
        return self.gold["task_family"]

    @property
    def valid(self) -> bool:
        return self.prediction is not None and not self.schema_issues

    @property
    def reference(self) -> dict[str, Any]:
        return self.gold["reference"]

    @property
    def output(self) -> dict[str, Any]:
        if self.prediction is None:
            return {}
        value = self.prediction.get("output")
        return value if isinstance(value, dict) else {}


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _error(
    pair: EvalPair,
    code: str,
    *,
    severity: str,
    field: str | None = None,
    expected: Any = None,
    predicted: Any = None,
    detail: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "record_id": pair.gold["record_id"],
        "deal_id": pair.gold["deal_id"],
        "task_family": pair.family,
    }
    if field is not None:
        value["field"] = field
    if expected is not None:
        value["expected"] = expected
    if predicted is not None:
        value["predicted"] = predicted
    if detail is not None:
        value["detail"] = detail
    return value


def _schema_errors(pair: EvalPair) -> list[dict[str, Any]]:
    if pair.prediction is None:
        return [
            _error(
                pair,
                "MISSING_PREDICTION",
                severity="critical",
                detail="No prediction record matched this golden record.",
            )
        ]
    if pair.schema_issues:
        return [
            _error(
                pair,
                "SCHEMA_INVALID",
                severity="high",
                field=issue.path,
                detail=issue.message,
            )
            for issue in pair.schema_issues
        ]
    return []


def recompute_weighted_score(states: dict[str, str], weights: dict[str, float]) -> float:
    applicable = [
        float(weight)
        for criterion, weight in weights.items()
        if states.get(criterion) != "not_applicable"
    ]
    denominator = sum(applicable)
    if denominator == 0:
        return 0.0
    numerator = sum(
        float(weight)
        for criterion, weight in weights.items()
        if states.get(criterion) == "pass"
    )
    return 100 * numerator / denominator


def _qa_scorecard(
    pairs: list[EvalPair],
    *,
    collect_errors: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    expected_states: list[str] = []
    predicted_states: list[str] = []
    expected_totals: list[float] = []
    model_totals: list[float] = []
    recomputed_totals: list[float] = []
    model_consistency_targets: list[float] = []
    errors: list[dict[str, Any]] = []

    for pair in pairs:
        if collect_errors:
            errors.extend(_schema_errors(pair))
        reference = pair.reference
        output = pair.output
        expected_criteria = reference["criteria"]
        predicted_criteria = output.get("criteria", {})
        if not isinstance(predicted_criteria, dict):
            predicted_criteria = {}
        for criterion, expected in expected_criteria.items():
            predicted = predicted_criteria.get(criterion, "__missing__")
            expected_states.append(expected)
            predicted_states.append(predicted)
            if collect_errors and expected != predicted:
                if expected == "pass":
                    code, severity = "CRITERION_FALSE_NEGATIVE", "medium"
                elif predicted == "pass":
                    code, severity = "CRITERION_FALSE_POSITIVE", "high"
                else:
                    code, severity = "CRITERION_STATE_MISMATCH", "medium"
                errors.append(
                    _error(
                        pair,
                        code,
                        severity=severity,
                        field=f"criteria.{criterion}",
                        expected=expected,
                        predicted=predicted,
                    )
                )
        expected_total = float(reference["total_score"])
        model_total_value = output.get("total_score", 0.0)
        model_total = (
            float(model_total_value)
            if isinstance(model_total_value, (int, float))
            and not isinstance(model_total_value, bool)
            else 0.0
        )
        recomputed = recompute_weighted_score(predicted_criteria, reference["weights"])
        expected_totals.append(expected_total)
        model_totals.append(model_total)
        recomputed_totals.append(recomputed)
        model_consistency_targets.append(recomputed)
        if collect_errors and abs(model_total - recomputed) > 1.0:
            errors.append(
                _error(
                    pair,
                    "MODEL_SCORE_DRIFT",
                    severity="high",
                    field="total_score",
                    expected=round(recomputed, 4),
                    predicted=round(model_total, 4),
                    detail="Model total differs from score recomputed from criterion states.",
                )
            )

    state_metrics = classification_metrics(expected_states, predicted_states)
    metrics = {
        "schema_validity": safe_div(sum(pair.valid for pair in pairs), len(pairs)),
        "criterion_accuracy": state_metrics["accuracy"],
        "criterion_macro_f1": state_metrics["macro_f1"],
        "criterion_micro_f1": state_metrics["micro_f1"],
        "criterion_cohen_kappa": cohen_kappa(expected_states, predicted_states),
        "model_reported_score_mae": mean_absolute_error(expected_totals, model_totals),
        "recomputed_weighted_score_mae": mean_absolute_error(
            expected_totals,
            recomputed_totals,
        ),
        "model_score_internal_consistency_mae": mean_absolute_error(
            model_totals,
            model_consistency_targets,
        ),
    }
    return metrics, errors


def _deal_summary(
    pairs: list[EvalPair],
    *,
    collect_errors: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    outcomes_expected: list[str] = []
    outcomes_predicted: list[str] = []
    primary_need_matches: list[float] = []
    next_presence_matches: list[float] = []
    next_exact_matches: list[float] = []
    objection_expected: list[set[str]] = []
    objection_predicted: list[set[str]] = []
    fact_expected: list[set[str]] = []
    fact_predicted: list[set[str]] = []
    full_exact: list[float] = []
    errors: list[dict[str, Any]] = []

    for pair in pairs:
        if collect_errors:
            errors.extend(_schema_errors(pair))
        reference = pair.reference
        output = pair.output
        expected_outcome = reference["outcome"]
        predicted_outcome = str(output.get("outcome", "__missing__"))
        outcomes_expected.append(expected_outcome)
        outcomes_predicted.append(predicted_outcome)
        if collect_errors and expected_outcome != predicted_outcome:
            errors.append(
                _error(
                    pair,
                    "OUTCOME_MISMATCH",
                    severity="high",
                    field="outcome",
                    expected=expected_outcome,
                    predicted=predicted_outcome,
                )
            )

        expected_need = _normalize(reference["primary_need"])
        predicted_need = _normalize(str(output.get("primary_need", "")))
        primary_need_matches.append(float(expected_need == predicted_need))
        if collect_errors and expected_need != predicted_need:
            errors.append(
                _error(
                    pair,
                    "PRIMARY_NEED_MISMATCH",
                    severity="medium",
                    field="primary_need",
                    expected=reference["primary_need"],
                    predicted=output.get("primary_need"),
                )
            )

        expected_objections = {_normalize(item) for item in reference["objections"]}
        predicted_objections = {
            _normalize(str(item))
            for item in output.get("objections", [])
            if isinstance(item, str)
        }
        objection_expected.append(expected_objections)
        objection_predicted.append(predicted_objections)

        expected_facts = {_normalize(item) for item in reference["facts"]}
        predicted_facts = {
            _normalize(str(item))
            for item in output.get("facts", [])
            if isinstance(item, str)
        }
        fact_expected.append(expected_facts)
        fact_predicted.append(predicted_facts)
        if collect_errors:
            for fact in sorted(expected_facts - predicted_facts):
                errors.append(
                    _error(
                        pair,
                        "SUMMARY_FACT_OMISSION",
                        severity="medium",
                        field="facts",
                        expected=fact,
                    )
                )
            for fact in sorted(predicted_facts - expected_facts):
                errors.append(
                    _error(
                        pair,
                        "SUMMARY_UNSUPPORTED_FACT",
                        severity="high",
                        field="facts",
                        predicted=fact,
                    )
                )

        expected_next = reference["next_step"]
        predicted_next = output.get("next_step")
        expected_present = expected_next is not None
        predicted_present = predicted_next is not None
        next_presence_matches.append(float(expected_present == predicted_present))
        normalized_expected_next = _normalize(expected_next) if expected_next else ""
        normalized_predicted_next = (
            _normalize(predicted_next) if isinstance(predicted_next, str) else ""
        )
        next_exact_matches.append(
            float(normalized_expected_next == normalized_predicted_next)
        )
        if collect_errors and normalized_expected_next != normalized_predicted_next:
            errors.append(
                _error(
                    pair,
                    "NEXT_STEP_MISMATCH",
                    severity="medium",
                    field="next_step",
                    expected=expected_next,
                    predicted=predicted_next,
                )
            )
        full_exact.append(
            float(
                expected_outcome == predicted_outcome
                and expected_need == predicted_need
                and expected_objections == predicted_objections
                and expected_facts == predicted_facts
                and normalized_expected_next == normalized_predicted_next
            )
        )

    objections = multilabel_metrics(objection_expected, objection_predicted)
    facts = multilabel_metrics(fact_expected, fact_predicted)
    metrics = {
        "schema_validity": safe_div(sum(pair.valid for pair in pairs), len(pairs)),
        "full_normalized_exact_match": mean(full_exact),
        "outcome_accuracy": accuracy(outcomes_expected, outcomes_predicted),
        "outcome_cohen_kappa": cohen_kappa(outcomes_expected, outcomes_predicted),
        "primary_need_exact_match": mean(primary_need_matches),
        "objection_micro_f1": objections["micro_f1"],
        "content_unit_precision": facts["precision"],
        "content_unit_recall": facts["recall"],
        "content_unit_f1": facts["micro_f1"],
        "next_step_presence_accuracy": mean(next_presence_matches),
        "next_step_exact_match": mean(next_exact_matches),
    }
    return metrics, errors


def _semantic_labels(output: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for dimension in ("needs", "objections", "tariff_interest", "competitor_mentions"):
        for item in output.get(dimension, []):
            if isinstance(item, str):
                labels.add(f"{dimension}:{_normalize(item)}")
    return labels


def _semantic_analytics(
    pairs: list[EvalPair],
    *,
    collect_errors: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    expected_sets: list[set[str]] = []
    predicted_sets: list[set[str]] = []
    readiness_expected: list[str] = []
    readiness_predicted: list[str] = []
    dimensions: dict[str, tuple[list[set[str]], list[set[str]]]] = {
        key: ([], [])
        for key in ("needs", "objections", "tariff_interest", "competitor_mentions")
    }
    errors: list[dict[str, Any]] = []

    for pair in pairs:
        if collect_errors:
            errors.extend(_schema_errors(pair))
        reference = pair.reference
        output = pair.output
        expected = _semantic_labels(reference)
        predicted = _semantic_labels(output)
        expected_sets.append(expected)
        predicted_sets.append(predicted)
        if collect_errors:
            for label in sorted(expected - predicted):
                errors.append(
                    _error(
                        pair,
                        "SEMANTIC_LABEL_MISSED",
                        severity="medium",
                        field=label.split(":", 1)[0],
                        expected=label,
                    )
                )
            for label in sorted(predicted - expected):
                errors.append(
                    _error(
                        pair,
                        "SEMANTIC_LABEL_SPURIOUS",
                        severity="medium",
                        field=label.split(":", 1)[0],
                        predicted=label,
                    )
                )
        for dimension, (expected_dimension, predicted_dimension) in dimensions.items():
            expected_dimension.append(
                {_normalize(item) for item in reference.get(dimension, [])}
            )
            predicted_dimension.append(
                {
                    _normalize(str(item))
                    for item in output.get(dimension, [])
                    if isinstance(item, str)
                }
            )
        expected_readiness = reference["readiness"]
        predicted_readiness = str(output.get("readiness", "__missing__"))
        readiness_expected.append(expected_readiness)
        readiness_predicted.append(predicted_readiness)
        if collect_errors and expected_readiness != predicted_readiness:
            errors.append(
                _error(
                    pair,
                    "READINESS_MISMATCH",
                    severity="medium",
                    field="readiness",
                    expected=expected_readiness,
                    predicted=predicted_readiness,
                )
            )

    labels = multilabel_metrics(expected_sets, predicted_sets)
    metrics = {
        "schema_validity": safe_div(sum(pair.valid for pair in pairs), len(pairs)),
        "label_precision": labels["precision"],
        "label_recall": labels["recall"],
        "label_micro_f1": labels["micro_f1"],
        "label_macro_f1": labels["macro_f1"],
        "exact_set_match": labels["exact_match"],
        "mean_jaccard": labels["mean_jaccard"],
        "readiness_accuracy": accuracy(readiness_expected, readiness_predicted),
        "readiness_cohen_kappa": cohen_kappa(
            readiness_expected,
            readiness_predicted,
        ),
    }
    for dimension, (expected_dimension, predicted_dimension) in dimensions.items():
        dimension_metrics = multilabel_metrics(expected_dimension, predicted_dimension)
        metrics[f"{dimension}_micro_f1"] = dimension_metrics["micro_f1"]
    return metrics, errors


def _violation_flags(
    pairs: list[EvalPair],
    *,
    collect_errors: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    expected_positive: list[set[str]] = []
    predicted_positive: list[set[str]] = []
    expected_states: list[str] = []
    predicted_states: list[str] = []
    critical_expected = 0
    critical_detected = 0
    positive_expected = 0
    false_negatives = 0
    errors: list[dict[str, Any]] = []

    for pair in pairs:
        if collect_errors:
            errors.extend(_schema_errors(pair))
        reference_flags = pair.reference["flags"]
        output_flags = pair.output.get("flags", {})
        if not isinstance(output_flags, dict):
            output_flags = {}
        expected_set = {flag for flag, state in reference_flags.items() if state == "yes"}
        predicted_set = {flag for flag, state in output_flags.items() if state == "yes"}
        expected_positive.append(expected_set)
        predicted_positive.append(predicted_set)
        critical = set(pair.reference["critical_flags"])
        for flag, expected in reference_flags.items():
            predicted = str(output_flags.get(flag, "__missing__"))
            expected_states.append(expected)
            predicted_states.append(predicted)
            if expected == "yes":
                positive_expected += 1
                if flag in critical:
                    critical_expected += 1
                if predicted == "yes" and flag in critical:
                    critical_detected += 1
                if predicted != "yes":
                    false_negatives += 1
                    code = (
                        "CRITICAL_FLAG_MISSED"
                        if flag in critical
                        else "VIOLATION_FALSE_NEGATIVE"
                    )
                    severity = "critical" if flag in critical else "high"
                    if collect_errors:
                        errors.append(
                            _error(
                                pair,
                                code,
                                severity=severity,
                                field=f"flags.{flag}",
                                expected=expected,
                                predicted=predicted,
                            )
                        )
            elif predicted == "yes":
                if collect_errors:
                    errors.append(
                        _error(
                            pair,
                            "VIOLATION_FALSE_POSITIVE",
                            severity="high",
                            field=f"flags.{flag}",
                            expected=expected,
                            predicted=predicted,
                        )
                    )
            elif collect_errors and expected != predicted:
                errors.append(
                    _error(
                        pair,
                        "VIOLATION_STATE_MISMATCH",
                        severity="medium",
                        field=f"flags.{flag}",
                        expected=expected,
                        predicted=predicted,
                    )
                )

    positives = multilabel_metrics(expected_positive, predicted_positive)
    state_metrics = classification_metrics(expected_states, predicted_states)
    metrics = {
        "schema_validity": safe_div(sum(pair.valid for pair in pairs), len(pairs)),
        "positive_precision": positives["precision"],
        "positive_recall": positives["recall"],
        "positive_micro_f1": positives["micro_f1"],
        "positive_macro_f1": positives["macro_f1"],
        "exact_flag_set_match": positives["exact_match"],
        "critical_recall": safe_div(
            critical_detected,
            critical_expected,
            default=1.0,
        ),
        "false_negative_rate": safe_div(
            false_negatives,
            positive_expected,
            default=0.0,
        ),
        "flag_state_accuracy": state_metrics["accuracy"],
        "flag_state_cohen_kappa": cohen_kappa(expected_states, predicted_states),
    }
    return metrics, errors


_FAMILY_EVALUATORS = {
    "qa_scorecard": _qa_scorecard,
    "deal_summary": _deal_summary,
    "semantic_analytics": _semantic_analytics,
    "violation_flags": _violation_flags,
}


def _index_records(
    records: list[dict[str, Any]],
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise DataError(f"{kind} record {position} has no usable record_id")
        if record_id in indexed:
            raise DataError(f"duplicate {kind} record_id: {record_id}")
        indexed[record_id] = record
    return indexed


def _build_pairs(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> tuple[list[EvalPair], list[dict[str, Any]]]:
    gold_index = _index_records(gold_records, kind="gold")
    prediction_index = _index_records(prediction_records, kind="prediction")
    pairs: list[EvalPair] = []
    for record_id, gold in gold_index.items():
        gold_issues = validate_gold_record(gold)
        if gold_issues:
            rendered = "; ".join(f"{item.path}: {item.message}" for item in gold_issues[:10])
            raise DataError(f"invalid gold record {record_id}: {rendered}")
        prediction = prediction_index.get(record_id)
        issues: list[ValidationIssue] = []
        if prediction is not None:
            issues.extend(validate_prediction_record(prediction))
            predicted_family = prediction.get("task_family")
            if predicted_family != gold["task_family"]:
                issues.append(
                    ValidationIssue(
                        "$.task_family",
                        f"expected {gold['task_family']!r}, received {predicted_family!r}",
                    )
                )
            output = prediction.get("output")
            if isinstance(output, dict):
                if gold["task_family"] == "qa_scorecard":
                    predicted_keys = set(output.get("criteria", {}))
                    expected_keys = set(gold["reference"]["criteria"])
                    if predicted_keys != expected_keys:
                        issues.append(
                            ValidationIssue(
                                "$.output.criteria",
                                "criterion keys must exactly match the versioned rubric",
                            )
                        )
                elif gold["task_family"] == "violation_flags":
                    predicted_keys = set(output.get("flags", {}))
                    expected_keys = set(gold["reference"]["flags"])
                    if predicted_keys != expected_keys:
                        issues.append(
                            ValidationIssue(
                                "$.output.flags",
                                "flag keys must exactly match the versioned policy",
                            )
                        )
        pairs.append(EvalPair(gold=gold, prediction=prediction, schema_issues=issues))

    unexpected: list[dict[str, Any]] = []
    for record_id in sorted(set(prediction_index) - set(gold_index)):
        prediction = prediction_index[record_id]
        unexpected.append(
            {
                "code": "UNEXPECTED_PREDICTION",
                "severity": "medium",
                "record_id": record_id,
                "task_family": prediction.get("task_family"),
                "detail": "Prediction has no matching golden record.",
            }
        )
    return pairs, unexpected


def _confidence_intervals(
    family: str,
    pairs: list[EvalPair],
    metric_names: list[str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    evaluator = _FAMILY_EVALUATORS[family]
    rng = random.Random(seed)
    estimates: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        sample_metrics, _ = evaluator(sample, collect_errors=False)
        for metric_name in metric_names:
            estimates[metric_name].append(float(sample_metrics[metric_name]))
    return {
        metric_name: {
            "low": percentile(values, 0.025),
            "high": percentile(values, 0.975),
            "confidence": 0.95,
            "resamples": resamples,
        }
        for metric_name, values in estimates.items()
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "run_id",
        "dataset_id",
        "provider",
        "model",
        "prompt_version",
        "output_schema_version",
        "code_revision",
        "created_at",
        "generation",
        "pricing",
        "is_demo",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise DataError(f"manifest missing required fields: {missing}")
    if not isinstance(manifest["pricing"], dict):
        raise DataError("manifest pricing must be an object")
    pricing_required = {
        "currency",
        "input_per_1m_tokens",
        "output_per_1m_tokens",
        "effective_at",
        "source",
    }
    pricing_missing = sorted(pricing_required - set(manifest["pricing"]))
    if pricing_missing:
        raise DataError(f"manifest pricing missing fields: {pricing_missing}")


def _operations(
    pairs: list[EvalPair],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    telemetry = [
        pair.prediction["telemetry"]
        for pair in pairs
        if pair.prediction is not None
        and isinstance(pair.prediction.get("telemetry"), dict)
        and not any(issue.path.startswith("$.telemetry") for issue in pair.schema_issues)
    ]
    latencies = [float(item["latency_ms"]) for item in telemetry]
    input_tokens = sum(int(item["input_tokens"]) for item in telemetry)
    output_tokens = sum(int(item["output_tokens"]) for item in telemetry)
    pricing = manifest["pricing"]
    input_rate = float(pricing["input_per_1m_tokens"])
    output_rate = float(pricing["output_per_1m_tokens"])
    estimated_cost = input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
    return {
        "telemetry_records": len(telemetry),
        "latency_ms": {
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "mean": mean(latencies) if latencies else None,
        },
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "total": input_tokens + output_tokens,
        },
        "estimated_cost": {
            "currency": pricing["currency"],
            "total": estimated_cost,
            "cost_per_record": safe_div(estimated_cost, len(telemetry)),
            "cost_per_1000_records": safe_div(estimated_cost * 1000, len(telemetry)),
            "pricing_effective_at": pricing["effective_at"],
            "pricing_source": pricing["source"],
        },
    }


def evaluate_run(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 20260728,
    max_error_examples: int = 100,
) -> dict[str, Any]:
    """Evaluate one immutable prediction run against adjudicated golden records."""

    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    _validate_manifest(manifest)
    pairs, unexpected_errors = _build_pairs(gold_records, prediction_records)
    by_family: dict[str, list[EvalPair]] = defaultdict(list)
    for pair in pairs:
        by_family[pair.family].append(pair)

    families: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = list(unexpected_errors)
    for family_index, family in enumerate(TASK_FAMILIES):
        family_pairs = by_family.get(family, [])
        if not family_pairs:
            continue
        metrics, errors = _FAMILY_EVALUATORS[family](
            family_pairs,
            collect_errors=True,
        )
        intervals = _confidence_intervals(
            family,
            family_pairs,
            list(metrics),
            resamples=bootstrap_resamples,
            seed=seed + family_index * 104729,
        )
        families[family] = {
            "records": len(family_pairs),
            "unique_deals": len({pair.gold["deal_id"] for pair in family_pairs}),
            "metrics": metrics,
            "confidence_intervals_95": intervals,
        }
        all_errors.extend(errors)

    synthetic = bool(gold_records) and all(
        record.get("metadata", {}).get("synthetic") is True for record in gold_records
    )
    schema_validity = safe_div(sum(pair.valid for pair in pairs), len(pairs))
    error_counts = Counter(error["code"] for error in all_errors)
    severity_counts = Counter(error["severity"] for error in all_errors)
    report = {
        "report_schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "disclaimer": (
            "Synthetic fixture run generated from bundled test data."
            if synthetic or manifest.get("is_demo") is True
            else "Evaluation report generated from the referenced immutable dataset and run manifest."
        ),
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "synthetic": synthetic,
            "records": len(gold_records),
            "unique_deals": len({record["deal_id"] for record in gold_records}),
            "records_by_family": dict(
                sorted(Counter(record["task_family"] for record in gold_records).items())
            ),
        },
        "run": manifest,
        "overall": {
            "records": len(pairs),
            "matched_predictions": sum(pair.prediction is not None for pair in pairs),
            "schema_validity": schema_validity,
        },
        "families": families,
        "annotation_agreement": annotation_agreement(gold_records),
        "operations": _operations(pairs, manifest),
        "errors": {
            "total": len(all_errors),
            "counts_by_code": dict(sorted(error_counts.items())),
            "counts_by_severity": dict(sorted(severity_counts.items())),
            "examples": all_errors[:max_error_examples],
            "examples_truncated": len(all_errors) > max_error_examples,
        },
        "method_notes": [
            "Schema-invalid and missing predictions remain in all task denominators.",
            "Quality confidence intervals use deal-record bootstrap resampling.",
            "Cost is estimated from the immutable pricing snapshot in the run manifest.",
            "Human annotation agreement is separate from model-vs-adjudicated-gold quality.",
        ],
    }
    return report
