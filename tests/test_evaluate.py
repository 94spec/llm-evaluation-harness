import unittest

from conv_eval.evaluate import evaluate_run
from conv_eval.synthetic import generate_bundle


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.gold,
            self.baseline,
            self.candidate,
            self.baseline_manifest,
            self.candidate_manifest,
        ) = generate_bundle(deals_per_family=8, seed=42)

    def test_all_four_families_are_scored(self) -> None:
        report = evaluate_run(
            self.gold,
            self.candidate,
            self.candidate_manifest,
            bootstrap_resamples=20,
            seed=42,
        )
        self.assertEqual(
            set(report["families"]),
            {
                "qa_scorecard",
                "deal_summary",
                "semantic_analytics",
                "violation_flags",
            },
        )
        self.assertTrue(report["dataset"]["synthetic"])
        self.assertIn("Synthetic fixture run", report["disclaimer"])
        self.assertEqual(report["overall"]["schema_validity"], 1.0)
        self.assertEqual(report["dataset"]["records"], 32)
        self.assertGreater(report["operations"]["tokens"]["total"], 0)

    def test_missing_prediction_is_in_denominator(self) -> None:
        predictions = self.candidate[:-1]
        report = evaluate_run(
            self.gold,
            predictions,
            self.candidate_manifest,
            bootstrap_resamples=10,
            seed=42,
        )
        self.assertLess(report["overall"]["schema_validity"], 1.0)
        self.assertEqual(report["errors"]["counts_by_code"]["MISSING_PREDICTION"], 1)

    def test_dynamic_criterion_keys_are_schema_checked(self) -> None:
        predictions = [dict(record) for record in self.candidate]
        first = dict(predictions[0])
        first["output"] = dict(first["output"])
        first["output"]["criteria"] = dict(first["output"]["criteria"])
        first["output"]["criteria"].pop(next(iter(first["output"]["criteria"])))
        predictions[0] = first
        report = evaluate_run(
            self.gold,
            predictions,
            self.candidate_manifest,
            bootstrap_resamples=10,
            seed=42,
        )
        self.assertLess(report["overall"]["schema_validity"], 1.0)
        self.assertGreater(report["errors"]["counts_by_code"]["SCHEMA_INVALID"], 0)


if __name__ == "__main__":
    unittest.main()

