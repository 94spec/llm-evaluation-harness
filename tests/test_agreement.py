"""Annotation reliability: the number that caps every model score on this set.

This module had no tests while its output was already being printed in reports —
the worst combination, because a wrong reliability figure does not crash anything,
it just quietly raises the ceiling everyone measures against.
"""
import unittest

from conv_eval.agreement import annotation_agreement


def record(family: str, *outputs: dict, deal_id: str = "SYN-0001") -> dict:
    return {
        "deal_id": deal_id,
        "task_family": family,
        "annotations": [
            {"annotator_id": f"rater-{index}", "output": output}
            for index, output in enumerate(outputs)
        ],
    }


class WhatCounts(unittest.TestCase):
    def test_single_annotation_is_ignored(self) -> None:
        # One rater cannot disagree with anyone: including such records would
        # inflate agreement with units that were never contested.
        result = annotation_agreement([record("qa_scorecard", {"criteria": {"greeting": "pass"}})])
        self.assertEqual(result, {})

    def test_two_annotators_produce_a_family_report(self) -> None:
        result = annotation_agreement([
            record("qa_scorecard",
                   {"criteria": {"greeting": "pass", "next_step": "fail"}},
                   {"criteria": {"greeting": "pass", "next_step": "fail"}}),
        ])
        self.assertIn("qa_scorecard", result)
        family = result["qa_scorecard"]
        self.assertEqual(family["records_with_multiple_annotations"], 1)
        self.assertEqual(family["decision_units"], 2)

    def test_empty_input(self) -> None:
        self.assertEqual(annotation_agreement([]), {})

    def test_record_without_decisions_is_skipped(self) -> None:
        result = annotation_agreement([record("qa_scorecard", {"criteria": {}}, {"criteria": {}})])
        self.assertEqual(result, {})


class Values(unittest.TestCase):
    def test_perfect_agreement(self) -> None:
        output = {"criteria": {"a": "pass", "b": "fail", "c": "pass"}}
        result = annotation_agreement([record("qa_scorecard", output, dict(output))])
        family = result["qa_scorecard"]
        self.assertEqual(family["two_rater_raw_agreement"], 1.0)
        self.assertAlmostEqual(family["two_rater_cohen_kappa"], 1.0)
        self.assertAlmostEqual(family["krippendorff_alpha_nominal"], 1.0)

    def test_total_disagreement_is_not_positive(self) -> None:
        result = annotation_agreement([
            record("qa_scorecard",
                   {"criteria": {"a": "pass", "b": "fail"}},
                   {"criteria": {"a": "fail", "b": "pass"}}),
        ])
        family = result["qa_scorecard"]
        self.assertEqual(family["two_rater_raw_agreement"], 0.0)
        self.assertLessEqual(family["two_rater_cohen_kappa"], 0.0)

    def test_kappa_is_stricter_than_raw_agreement_when_labels_are_skewed(self) -> None:
        # Nine criteria agreed as "pass", one disagreed: raw agreement flatters.
        first = {"criteria": {f"c{i}": "pass" for i in range(9)} | {"c9": "pass"}}
        second = {"criteria": {f"c{i}": "pass" for i in range(9)} | {"c9": "fail"}}
        family = annotation_agreement([record("qa_scorecard", first, second)])["qa_scorecard"]
        self.assertGreater(family["two_rater_raw_agreement"], 0.85)
        self.assertLess(family["two_rater_cohen_kappa"], family["two_rater_raw_agreement"])


class SetValuedAnnotations(unittest.TestCase):
    """Lists are expanded to present/absent units over the union of labels."""

    def test_matching_lists_agree(self) -> None:
        output = {"readiness": "high", "needs": ["price", "terms"], "objections": [],
                  "tariff_interest": [], "competitor_mentions": []}
        family = annotation_agreement([
            record("semantic_analytics", output, dict(output))])["semantic_analytics"]
        self.assertEqual(family["two_rater_raw_agreement"], 1.0)
        # readiness + two needs
        self.assertEqual(family["decision_units"], 3)

    def test_label_present_for_one_rater_only_counts_as_a_disagreement(self) -> None:
        first = {"readiness": "high", "needs": ["price"], "objections": [],
                 "tariff_interest": [], "competitor_mentions": []}
        second = {"readiness": "high", "needs": ["price", "timeline"], "objections": [],
                  "tariff_interest": [], "competitor_mentions": []}
        family = annotation_agreement([
            record("semantic_analytics", first, second)])["semantic_analytics"]
        self.assertEqual(family["decision_units"], 3)
        self.assertLess(family["two_rater_raw_agreement"], 1.0)

    def test_label_text_is_normalised_before_comparison(self) -> None:
        # "  Цена   высокая " and "цена высокая" are the same label; without
        # normalisation every whitespace difference becomes a disagreement.
        first = {"readiness": "high", "needs": ["  Цена   высокая "], "objections": [],
                 "tariff_interest": [], "competitor_mentions": []}
        second = {"readiness": "high", "needs": ["цена высокая"], "objections": [],
                  "tariff_interest": [], "competitor_mentions": []}
        family = annotation_agreement([
            record("semantic_analytics", first, second)])["semantic_analytics"]
        self.assertEqual(family["two_rater_raw_agreement"], 1.0)

    def test_deal_summary_tracks_presence_of_a_next_step(self) -> None:
        first = {"outcome": "won", "primary_need": "обучение", "next_step": {"action": "call"},
                 "objections": []}
        second = {"outcome": "won", "primary_need": "обучение", "next_step": None,
                  "objections": []}
        family = annotation_agreement([record("deal_summary", first, second)])["deal_summary"]
        self.assertLess(family["two_rater_raw_agreement"], 1.0)


class FamilySeparation(unittest.TestCase):
    def test_families_are_reported_separately(self) -> None:
        result = annotation_agreement([
            record("qa_scorecard", {"criteria": {"a": "pass"}}, {"criteria": {"a": "pass"}},
                   deal_id="SYN-0001"),
            record("violation_flags", {"flags": {"promise": True}}, {"flags": {"promise": False}},
                   deal_id="SYN-0002"),
        ])
        self.assertEqual(sorted(result), ["qa_scorecard", "violation_flags"])
        self.assertEqual(result["qa_scorecard"]["two_rater_raw_agreement"], 1.0)
        self.assertEqual(result["violation_flags"]["two_rater_raw_agreement"], 0.0)

    def test_report_states_what_the_number_is_not(self) -> None:
        result = annotation_agreement([
            record("qa_scorecard", {"criteria": {"a": "pass"}}, {"criteria": {"a": "pass"}})])
        self.assertIn("model performance", result["qa_scorecard"]["note"])


if __name__ == "__main__":
    unittest.main()
