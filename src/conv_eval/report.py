"""Human-readable report rendering."""

from __future__ import annotations

from typing import Any


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) <= 1:
            return f"{value:.4f}"
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"> **{report['disclaimer']}**",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    run = report["run"]
    for key in (
        "run_id",
        "dataset_id",
        "provider",
        "model",
        "prompt_version",
        "output_schema_version",
        "code_revision",
        "created_at",
    ):
        lines.append(f"| `{key}` | {_format(run.get(key))} |")

    overall = report["overall"]
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Golden records: **{report['dataset']['records']}**",
            f"- Unique deals: **{report['dataset']['unique_deals']}**",
            f"- Matched predictions: **{overall['matched_predictions']}**",
            f"- Schema validity: **{overall['schema_validity']:.2%}**",
        ]
    )

    for family, family_report in report["families"].items():
        lines.extend(
            [
                "",
                f"## {family}",
                "",
                f"Records: **{family_report['records']}**",
                "",
                "| Metric | Value | Bootstrap 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for metric, value in family_report["metrics"].items():
            interval = family_report["confidence_intervals_95"].get(metric, {})
            ci = (
                f"{_format(interval.get('low'))}–{_format(interval.get('high'))}"
                if interval
                else "n/a"
            )
            lines.append(f"| `{metric}` | {_format(value)} | {ci} |")

    operations = report["operations"]
    cost = operations["estimated_cost"]
    lines.extend(
        [
            "",
            "## Operations",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Telemetry records | {operations['telemetry_records']} |",
            f"| Latency p50, ms | {_format(operations['latency_ms']['p50'])} |",
            f"| Latency p95, ms | {_format(operations['latency_ms']['p95'])} |",
            f"| Input tokens | {operations['tokens']['input_total']} |",
            f"| Output tokens | {operations['tokens']['output_total']} |",
            f"| Estimated total cost ({cost['currency']}) | {_format(cost['total'])} |",
            f"| Estimated cost / 1,000 records | {_format(cost['cost_per_1000_records'])} |",
            "",
            (
                f"Pricing snapshot: `{cost['pricing_source']}`, effective "
                f"`{cost['pricing_effective_at']}`."
            ),
        ]
    )

    errors = report["errors"]
    lines.extend(
        [
            "",
            "## Error ledger",
            "",
            f"Total typed errors: **{errors['total']}**",
            "",
            "| Code | Count |",
            "|---|---:|",
        ]
    )
    if errors["counts_by_code"]:
        for code, count in errors["counts_by_code"].items():
            lines.append(f"| `{code}` | {count} |")
    else:
        lines.append("| — | 0 |")

    if report["annotation_agreement"]:
        lines.extend(
            [
                "",
                "## Human annotation reliability",
                "",
                "These values describe pre-adjudication human agreement, not model quality.",
                "",
                "| Family | Units | Raw agreement | Cohen κ | Krippendorff α |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for family, values in report["annotation_agreement"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        family,
                        str(values["decision_units"]),
                        _format(values["two_rater_raw_agreement"]),
                        _format(values["two_rater_cohen_kappa"]),
                        _format(values["krippendorff_alpha_nominal"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Do not generalize synthetic results to production.",
            "- Do not compare runs from different dataset or rubric versions.",
            "- Inspect critical and high-severity examples before a release decision.",
            "- Treat cost as an estimate tied to the manifest pricing snapshot.",
            "",
        ]
    )
    return "\n".join(lines)

