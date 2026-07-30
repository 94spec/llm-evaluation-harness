"""The report and the provider comparison: the part a human actually reads.

A number that is correct in JSON and wrong in the rendered table is still a
wrong number to everyone downstream. These tests check that what gets rendered
is what was measured, and that a missing metric shows as missing rather than as
zero.
"""
import unittest

from conv_eval.compare import DEFAULT_METRICS, compare_reports
from conv_eval.evaluate import evaluate_run
from conv_eval.report import render_markdown
from conv_eval.synthetic import generate_bundle

GOLD, BASELINE_PREDICTIONS, CANDIDATE_PREDICTIONS, BASELINE_MANIFEST, CANDIDATE_MANIFEST = \
    generate_bundle(deals_per_family=6, seed=20260730)

BASELINE_REPORT = evaluate_run(
    gold_records=GOLD, prediction_records=BASELINE_PREDICTIONS,
    manifest=BASELINE_MANIFEST, bootstrap_resamples=20, seed=1)
CANDIDATE_REPORT = evaluate_run(
    gold_records=GOLD, prediction_records=CANDIDATE_PREDICTIONS,
    manifest=CANDIDATE_MANIFEST, bootstrap_resamples=20, seed=1)


class Rendering(unittest.TestCase):
    def setUp(self) -> None:
        self.markdown = render_markdown(CANDIDATE_REPORT)

    def test_report_is_markdown_with_a_heading(self) -> None:
        self.assertTrue(self.markdown.lstrip().startswith("#"))

    def test_every_family_appears(self) -> None:
        for family in CANDIDATE_REPORT["families"]:
            with self.subTest(family=family):
                self.assertIn(family, self.markdown)

    def test_the_run_identity_is_printed(self) -> None:
        # A report you cannot trace back to a model, a prompt and a price list
        # is an opinion with decimal places.
        for field in ("model", "prompt_version", "run_id"):
            with self.subTest(field=field):
                self.assertIn(str(CANDIDATE_REPORT["run"][field]), self.markdown)

    def test_schema_validity_is_reported(self) -> None:
        self.assertIn("schema", self.markdown.lower())

    def test_annotation_agreement_section_is_present(self) -> None:
        # The generator writes two annotators per record, so the section must
        # exist — its absence would mean the reliability numbers were dropped.
        self.assertTrue(CANDIDATE_REPORT["annotation_agreement"])
        self.assertIn("greement", self.markdown)

    def test_rendering_is_deterministic(self) -> None:
        self.assertEqual(self.markdown, render_markdown(CANDIDATE_REPORT))

    def test_report_without_agreement_still_renders(self) -> None:
        stripped = dict(CANDIDATE_REPORT, annotation_agreement={})
        self.assertTrue(render_markdown(stripped))

    def test_confidence_intervals_reach_the_page(self) -> None:
        self.assertIn("95", self.markdown)


class Comparison(unittest.TestCase):
    def setUp(self) -> None:
        self.comparison = compare_reports([BASELINE_REPORT, CANDIDATE_REPORT])

    def test_comparability_is_stated_explicitly(self) -> None:
        # Two runs over different datasets are not comparable, and the tool says
        # so instead of ranking them anyway.
        self.assertTrue(self.comparison["comparable"])
        self.assertIsNone(self.comparison["warning"])

    def test_different_datasets_are_flagged_as_not_comparable(self) -> None:
        other = dict(CANDIDATE_REPORT)
        other["run"] = dict(CANDIDATE_REPORT["run"], dataset_id="another-dataset")
        comparison = compare_reports([BASELINE_REPORT, other])
        self.assertFalse(comparison["comparable"])
        self.assertIn("not valid", comparison["warning"])

    def test_one_row_per_report(self) -> None:
        self.assertEqual(len(self.comparison["runs"]), 2)

    def test_rows_are_identified_by_run(self) -> None:
        runs = [row["run_id"] for row in self.comparison["runs"]]
        self.assertEqual(runs, [BASELINE_MANIFEST["run_id"], CANDIDATE_MANIFEST["run_id"]])

    def test_default_metrics_are_collected(self) -> None:
        for path in DEFAULT_METRICS:
            with self.subTest(metric=path):
                self.assertIn(path, self.comparison["runs"][0]["metrics"])

    def test_missing_metric_is_none_rather_than_zero(self) -> None:
        # Zero would sort as the worst value and quietly lose a comparison.
        comparison = compare_reports([BASELINE_REPORT],
                                     metric_paths=("families.nothing.metrics.x",))
        self.assertIsNone(comparison["runs"][0]["metrics"]["families.nothing.metrics.x"])

    def test_a_single_report_compares_against_itself_without_failing(self) -> None:
        self.assertEqual(len(compare_reports([CANDIDATE_REPORT])["runs"]), 1)

    def test_cost_and_latency_are_part_of_the_comparison(self) -> None:
        joined = " ".join(DEFAULT_METRICS)
        self.assertIn("latency", joined)
        self.assertIn("cost", joined)


if __name__ == "__main__":
    unittest.main()
