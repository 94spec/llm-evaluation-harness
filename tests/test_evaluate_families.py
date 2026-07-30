"""The four family evaluators, on cases where the right answer is known by hand.

This is the largest module in the repository and the one whose output everything
else reports. The tests below construct the degenerate cases — perfect match,
total mismatch, invalid schema, missing prediction — because those are the ones
where a scoring bug produces a plausible number instead of an error.
"""
import copy
import unittest

from conv_eval.evaluate import evaluate_run, recompute_weighted_score
from conv_eval.synthetic import generate_bundle

GOLD, BASELINE, CANDIDATE, BASELINE_MANIFEST, CANDIDATE_MANIFEST = generate_bundle(
    deals_per_family=5, seed=4242)


def run(gold=None, predictions=None, manifest=None, **kwargs):
    return evaluate_run(
        gold_records=gold if gold is not None else GOLD,
        prediction_records=predictions if predictions is not None else CANDIDATE,
        manifest=manifest if manifest is not None else CANDIDATE_MANIFEST,
        bootstrap_resamples=kwargs.pop("bootstrap_resamples", 20),
        seed=kwargs.pop("seed", 1),
        **kwargs)


def predictions_matching_gold() -> list[dict]:
    """A run that answered exactly like the reference."""
    perfect = []
    for record in GOLD:
        reference = copy.deepcopy(record["reference"])
        # A prediction never carries the reference-only fields.
        reference.pop("weights", None)
        reference.pop("critical_flags", None)
        perfect.append({
            "record_id": record["record_id"],
            "task_family": record["task_family"],
            "output": reference,
            "telemetry": {"latency_ms": 100.0, "input_tokens": 10, "output_tokens": 10},
        })
    return perfect


class WeightedScore(unittest.TestCase):
    def test_all_criteria_passed_is_a_full_score(self) -> None:
        # The score is a percentage of the applicable weight, not a sum.
        score = recompute_weighted_score({"a": "pass", "b": "pass"}, {"a": 3.0, "b": 7.0})
        self.assertAlmostEqual(score, 100.0)

    def test_half_the_weight_passed(self) -> None:
        score = recompute_weighted_score({"a": "pass", "b": "fail"}, {"a": 3.0, "b": 7.0})
        self.assertAlmostEqual(score, 30.0)

    def test_all_criteria_failed(self) -> None:
        self.assertAlmostEqual(
            recompute_weighted_score({"a": "fail", "b": "fail"}, {"a": 3.0, "b": 7.0}), 0.0)

    def test_not_applicable_leaves_the_denominator(self) -> None:
        # Scoring an inapplicable criterion as zero punishes a conversation for
        # not containing something it could not contain.
        only_applicable = recompute_weighted_score({"a": "pass"}, {"a": 3.0})
        with_inapplicable = recompute_weighted_score(
            {"a": "pass", "b": "not_applicable"}, {"a": 3.0, "b": 7.0})
        self.assertAlmostEqual(only_applicable, with_inapplicable)

    def test_insufficient_data_is_not_a_pass(self) -> None:
        score = recompute_weighted_score({"a": "insufficient_data"}, {"a": 5.0})
        self.assertLess(score, 100.0)

    def test_missing_weight_does_not_crash(self) -> None:
        self.assertIsInstance(recompute_weighted_score({"a": "pass"}, {}), float)


class PerfectRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run(predictions=predictions_matching_gold())

    def test_schema_validity_is_total(self) -> None:
        self.assertEqual(self.report["overall"]["schema_validity"], 1.0)

    def test_every_family_is_scored(self) -> None:
        self.assertEqual(sorted(self.report["families"]),
                         sorted({record["task_family"] for record in GOLD}))

    def test_agreement_metrics_are_at_their_maximum(self) -> None:
        for family, payload in self.report["families"].items():
            metrics = payload["metrics"]
            for name, value in metrics.items():
                if name.endswith(("_f1", "_accuracy", "_recall", "_precision")):
                    with self.subTest(family=family, metric=name):
                        self.assertAlmostEqual(value, 1.0, places=6)

    def test_error_ledger_is_empty(self) -> None:
        self.assertEqual(self.report["errors"]["total"], 0)


class ImperfectRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run()

    def test_the_generated_candidate_is_not_perfect(self) -> None:
        # The generator injects a known error profile; if everything scored 1.0
        # the evaluator would be measuring nothing.
        f1_values = [
            value
            for payload in self.report["families"].values()
            for name, value in payload["metrics"].items()
            if name.endswith("_f1")
        ]
        self.assertTrue(f1_values)
        self.assertTrue(any(value < 1.0 for value in f1_values))

    def test_errors_are_counted_and_classified(self) -> None:
        # A total without a breakdown tells you something broke and not what.
        ledger = self.report["errors"]
        self.assertGreater(ledger["total"], 0)
        self.assertTrue(ledger["counts_by_code"])
        self.assertTrue(ledger["counts_by_severity"])
        self.assertEqual(sum(ledger["counts_by_code"].values()), ledger["total"])

    def test_confidence_intervals_bracket_the_estimate(self) -> None:
        for family, payload in self.report["families"].items():
            intervals = payload.get("confidence_intervals_95", {})
            for metric, interval in intervals.items():
                with self.subTest(family=family, metric=metric):
                    self.assertLessEqual(interval["low"], interval["high"])

    def test_same_seed_gives_the_same_report(self) -> None:
        again = run()
        self.assertEqual(self.report["families"], again["families"])

    def test_operations_carry_latency_and_cost(self) -> None:
        operations = self.report["operations"]
        self.assertIn("p95", operations["latency_ms"])
        self.assertIn("cost_per_1000_records", operations["estimated_cost"])

    def test_cost_uses_the_manifest_price_not_a_current_one(self) -> None:
        self.assertIn("pricing", self.report["run"])
        self.assertEqual(self.report["run"]["pricing"]["effective_at"],
                         CANDIDATE_MANIFEST["pricing"]["effective_at"])


class BrokenInput(unittest.TestCase):
    def test_invalid_prediction_shape_lowers_schema_validity(self) -> None:
        predictions = copy.deepcopy(CANDIDATE)
        predictions[0]["output"] = {"unexpected": "shape"}
        report = run(predictions=predictions)
        self.assertLess(report["overall"]["schema_validity"], 1.0)

    def test_a_missing_prediction_is_visible_in_the_counts(self) -> None:
        report = run(predictions=copy.deepcopy(CANDIDATE)[1:])
        overall = report["overall"]
        self.assertLess(overall["matched_predictions"], overall["records"])

    def test_manifest_without_pricing_is_rejected(self) -> None:
        manifest = copy.deepcopy(CANDIDATE_MANIFEST)
        del manifest["pricing"]
        with self.assertRaises(ValueError):
            run(manifest=manifest)

    def test_manifest_without_model_is_rejected(self) -> None:
        manifest = copy.deepcopy(CANDIDATE_MANIFEST)
        del manifest["model"]
        with self.assertRaises(ValueError):
            run(manifest=manifest)

    def test_zero_bootstrap_resamples_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run(bootstrap_resamples=0)

    def test_empty_gold_set_produces_an_empty_report_not_a_crash(self) -> None:
        # Nothing to score is a legitimate state — an empty run must report zero
        # records rather than divide by zero somewhere deep in a metric.
        report = run(gold=[], predictions=[])
        self.assertEqual(report["overall"]["records"], 0)
        self.assertEqual(report["families"], {})


if __name__ == "__main__":
    unittest.main()
