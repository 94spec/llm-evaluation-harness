"""Release gates: the code that decides whether a change may ship.

Every branch here has a cost attached. A gate that passes when it should block
lets a regression into production; a gate that blocks on a missing metric stops
a release for a reporting bug. Both are tested.
"""
import unittest

from conv_eval.gates import evaluate_gates, get_metric

CANDIDATE = {
    "overall": {"schema_validity": 0.98},
    "families": {
        "violation_flags": {"metrics": {"critical_recall": 0.94}},
        "qa_scorecard": {"metrics": {"criterion_macro_f1": 0.88}},
    },
    "operations": {"latency_ms": {"p95": 2400.0}, "flag": True},
}

BASELINE = {
    "overall": {"schema_validity": 1.0},
    "families": {
        "violation_flags": {"metrics": {"critical_recall": 0.97}},
        "qa_scorecard": {"metrics": {"criterion_macro_f1": 0.90}},
    },
    "operations": {"latency_ms": {"p95": 2000.0}, "flag": True},
}


def rule(rule_id: str, metric: str, operator: str, value: float, **extra) -> dict:
    return {"id": rule_id, "metric": metric, "operator": operator, "value": value, **extra}


def run(*rules, baseline=None) -> dict:
    return evaluate_gates(CANDIDATE, baseline, {"rules": list(rules)})


class MetricLookup(unittest.TestCase):
    def test_reads_a_nested_path(self) -> None:
        self.assertEqual(get_metric(CANDIDATE, "overall.schema_validity"), 0.98)

    def test_missing_path_names_where_it_stopped(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            get_metric(CANDIDATE, "overall.nothing.here")
        self.assertIn("overall.nothing", str(ctx.exception))

    def test_non_numeric_value_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            get_metric(CANDIDATE, "families.violation_flags.metrics")

    def test_boolean_is_not_a_metric(self) -> None:
        # True would silently become 1.0 and pass a "greater than zero" gate.
        with self.assertRaises(TypeError):
            get_metric(CANDIDATE, "operations.flag")


class Thresholds(unittest.TestCase):
    def test_gte_passes_at_the_boundary(self) -> None:
        result = run(rule("r1", "overall.schema_validity", "gte", 0.98))
        self.assertTrue(result["passed"])

    def test_gte_fails_below(self) -> None:
        result = run(rule("r1", "overall.schema_validity", "gte", 0.99))
        self.assertFalse(result["passed"])

    def test_lte_passes_at_the_boundary(self) -> None:
        self.assertTrue(run(rule("r1", "operations.latency_ms.p95", "lte", 2400.0))["passed"])

    def test_lte_fails_above(self) -> None:
        self.assertFalse(run(rule("r1", "operations.latency_ms.p95", "lte", 2000.0))["passed"])


class RelativeToBaseline(unittest.TestCase):
    def test_max_drop_allows_a_small_regression(self) -> None:
        result = run(rule("r1", "families.violation_flags.metrics.critical_recall",
                          "max_drop", 0.05), baseline=BASELINE)
        self.assertTrue(result["passed"])

    def test_max_drop_blocks_a_large_one(self) -> None:
        result = run(rule("r1", "families.violation_flags.metrics.critical_recall",
                          "max_drop", 0.01), baseline=BASELINE)
        self.assertFalse(result["passed"])

    def test_max_increase_blocks_a_latency_jump(self) -> None:
        result = run(rule("r1", "operations.latency_ms.p95", "max_increase", 100.0),
                     baseline=BASELINE)
        self.assertFalse(result["passed"])

    def test_max_increase_allows_one_within_budget(self) -> None:
        result = run(rule("r1", "operations.latency_ms.p95", "max_increase", 600.0),
                     baseline=BASELINE)
        self.assertTrue(result["passed"])

    def test_relative_rule_without_a_baseline_does_not_silently_pass(self) -> None:
        # No baseline means the comparison cannot be made. Reporting success
        # here would wave through exactly the change the gate exists to catch.
        result = run(rule("r1", "operations.latency_ms.p95", "max_increase", 10.0))
        self.assertFalse(result["passed"])


class SeverityAndReporting(unittest.TestCase):
    def test_warning_severity_does_not_block(self) -> None:
        result = run(rule("r1", "overall.schema_validity", "gte", 0.999, severity="warn"))
        self.assertTrue(result["passed"])
        self.assertTrue(any(item.get("severity") == "warn" for item in result["rules"]))

    def test_blocking_rule_is_reported_by_id(self) -> None:
        result = run(rule("critical-recall", "families.violation_flags.metrics.critical_recall",
                          "gte", 0.99))
        failed = [item for item in result["rules"] if not item["passed"]]
        self.assertEqual([item["id"] for item in failed], ["critical-recall"])

    def test_missing_metric_is_reported_rather_than_ignored(self) -> None:
        result = run(rule("r1", "families.nothing.metrics.x", "gte", 0.5))
        self.assertFalse(result["passed"])
        self.assertTrue(any("error" in item or not item["passed"] for item in result["rules"]))

    def test_every_rule_is_evaluated_even_after_a_failure(self) -> None:
        result = run(rule("r1", "overall.schema_validity", "gte", 0.999),
                     rule("r2", "overall.schema_validity", "gte", 0.5))
        self.assertEqual(len(result["rules"]), 2)

    def test_empty_rule_set_passes_and_says_so(self) -> None:
        result = evaluate_gates(CANDIDATE, None, {"rules": []})
        self.assertTrue(result["passed"])
        self.assertEqual(result["rules"], [])


if __name__ == "__main__":
    unittest.main()
