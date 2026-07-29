"""Compact quality/cost/latency comparison across evaluation reports."""

from __future__ import annotations

from typing import Any

from .gates import get_metric

DEFAULT_METRICS = (
    "overall.schema_validity",
    "families.qa_scorecard.metrics.criterion_macro_f1",
    "families.qa_scorecard.metrics.recomputed_weighted_score_mae",
    "families.deal_summary.metrics.content_unit_recall",
    "families.semantic_analytics.metrics.label_micro_f1",
    "families.violation_flags.metrics.critical_recall",
    "operations.latency_ms.p95",
    "operations.estimated_cost.cost_per_1000_records",
)


def compare_reports(
    reports: list[dict[str, Any]],
    *,
    metric_paths: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        run = report["run"]
        metrics: dict[str, float | None] = {}
        for path in metric_paths:
            try:
                metrics[path] = get_metric(report, path)
            except (KeyError, TypeError):
                metrics[path] = None
        rows.append(
            {
                "run_id": run["run_id"],
                "provider": run["provider"],
                "model": run["model"],
                "prompt_version": run["prompt_version"],
                "output_schema_version": run["output_schema_version"],
                "dataset_id": run["dataset_id"],
                "metrics": metrics,
            }
        )
    dataset_ids = sorted({row["dataset_id"] for row in rows})
    return {
        "comparable": len(dataset_ids) == 1,
        "dataset_ids": dataset_ids,
        "warning": (
            None
            if len(dataset_ids) == 1
            else "Runs use different dataset IDs; metric ranking is not valid."
        ),
        "metric_paths": list(metric_paths),
        "runs": rows,
    }

