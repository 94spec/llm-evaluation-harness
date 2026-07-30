"""The output contract: what a record must look like before anything scores it.

A loose schema is how a renamed field becomes a null in a dashboard nobody
checks. Fixtures come from the generator rather than being written by hand, so
these tests describe the contract the tool actually ships, not the one someone
remembered while writing tests.
"""
import copy
import unittest

from conv_eval.schema import (TASK_FAMILIES, validate_gold_record,
                              validate_payload, validate_prediction_record)
from conv_eval.synthetic import generate_bundle

GOLD, BASELINE, CANDIDATE, _, _ = generate_bundle(deals_per_family=1, seed=1)
BY_FAMILY = {record["task_family"]: record for record in GOLD}
PREDICTION_BY_FAMILY = {record["task_family"]: record for record in CANDIDATE}


def gold_for(family: str) -> dict:
    return copy.deepcopy(BY_FAMILY[family])


def prediction_for(family: str) -> dict:
    return copy.deepcopy(PREDICTION_BY_FAMILY[family])


def paths(issues) -> list[str]:
    return [issue.path for issue in issues]


class GeneratedFixtures(unittest.TestCase):
    """Whatever the generator writes must satisfy the validator, always."""

    def test_every_generated_gold_record_is_valid(self) -> None:
        for record in GOLD:
            with self.subTest(family=record["task_family"]):
                self.assertEqual(validate_gold_record(record), [])

    def test_every_generated_prediction_is_valid(self) -> None:
        for record in BASELINE + CANDIDATE:
            with self.subTest(record=record["record_id"]):
                self.assertEqual(validate_prediction_record(record), [])

    def test_all_four_families_are_generated(self) -> None:
        self.assertEqual(sorted(BY_FAMILY), sorted(TASK_FAMILIES))


class RecordShape(unittest.TestCase):
    def test_missing_required_key_is_reported(self) -> None:
        record = gold_for("qa_scorecard")
        del record["deal_id"]
        self.assertTrue(any("deal_id" in path for path in paths(validate_gold_record(record))))

    def test_unexpected_key_is_reported(self) -> None:
        # An extra field usually means a producer changed and a consumer did not.
        record = gold_for("qa_scorecard")
        record["extra_field"] = "surprise"
        self.assertTrue(validate_gold_record(record))

    def test_non_object_record_does_not_crash_the_validator(self) -> None:
        self.assertTrue(validate_gold_record(["not", "an", "object"]))

    def test_metadata_must_be_an_object(self) -> None:
        record = gold_for("qa_scorecard")
        record["metadata"] = "none"
        self.assertTrue(any("metadata" in path for path in paths(validate_gold_record(record))))

    def test_identifiers_must_be_strings(self) -> None:
        record = gold_for("qa_scorecard")
        record["deal_id"] = 12345
        self.assertTrue(any("deal_id" in path for path in paths(validate_gold_record(record))))

    def test_unknown_task_family_is_rejected(self) -> None:
        record = gold_for("qa_scorecard")
        record["task_family"] = "astrology"
        self.assertTrue(any("task_family" in path for path in paths(validate_gold_record(record))))


class Annotations(unittest.TestCase):
    def test_annotations_are_optional(self) -> None:
        record = gold_for("qa_scorecard")
        del record["annotations"]
        self.assertEqual(validate_gold_record(record), [])

    def test_annotations_must_be_a_list(self) -> None:
        record = gold_for("qa_scorecard")
        record["annotations"] = {"annotator_id": "a"}
        self.assertTrue(any("annotations" in path for path in paths(validate_gold_record(record))))

    def test_annotation_output_is_validated_like_a_reference(self) -> None:
        record = gold_for("qa_scorecard")
        record["annotations"][0]["output"]["criteria"]["criterion_01"] = "almost"
        self.assertTrue(validate_gold_record(record))


class Vocabularies(unittest.TestCase):
    """Every free-text-looking field is actually a closed vocabulary."""

    def test_criterion_state_outside_the_vocabulary(self) -> None:
        reference = gold_for("qa_scorecard")["reference"]
        reference["criteria"]["criterion_01"] = "almost"
        self.assertTrue(validate_payload("qa_scorecard", reference, role="reference"))

    def test_outcome_outside_the_vocabulary(self) -> None:
        reference = gold_for("deal_summary")["reference"]
        reference["outcome"] = "maybe"
        self.assertTrue(validate_payload("deal_summary", reference, role="reference"))

    def test_readiness_outside_the_vocabulary(self) -> None:
        reference = gold_for("semantic_analytics")["reference"]
        reference["readiness"] = "lukewarm"
        self.assertTrue(validate_payload("semantic_analytics", reference, role="reference"))

    def test_flag_state_outside_the_vocabulary(self) -> None:
        reference = gold_for("violation_flags")["reference"]
        reference["flags"]["pressure"] = "probably"
        self.assertTrue(validate_payload("violation_flags", reference, role="reference"))


class Types(unittest.TestCase):
    def test_score_must_be_a_number(self) -> None:
        reference = gold_for("qa_scorecard")["reference"]
        reference["total_score"] = "60.0"
        self.assertTrue(validate_payload("qa_scorecard", reference, role="reference"))

    def test_label_list_must_hold_strings(self) -> None:
        reference = gold_for("semantic_analytics")["reference"]
        reference["needs"] = [42]
        self.assertTrue(validate_payload("semantic_analytics", reference, role="reference"))

    def test_objections_must_be_a_list_not_an_object(self) -> None:
        reference = gold_for("deal_summary")["reference"]
        reference["objections"] = {"type": "price"}
        self.assertTrue(validate_payload("deal_summary", reference, role="reference"))

    def test_flags_must_be_an_object_not_a_list(self) -> None:
        reference = gold_for("violation_flags")["reference"]
        reference["flags"] = ["pressure"]
        self.assertTrue(validate_payload("violation_flags", reference, role="reference"))


class Roles(unittest.TestCase):
    """A reference carries what a prediction must not be asked to produce."""

    def test_reference_and_prediction_differ_for_the_scorecard(self) -> None:
        reference = gold_for("qa_scorecard")["reference"]
        self.assertIn("weights", reference)
        self.assertNotIn("weights", prediction_for("qa_scorecard")["output"])

    def test_prediction_with_reference_only_fields_is_rejected(self) -> None:
        output = prediction_for("qa_scorecard")["output"]
        output["weights"] = {"criterion_01": 8}
        self.assertTrue(validate_payload("qa_scorecard", output, role="prediction"))

    def test_unknown_role_is_a_programming_error(self) -> None:
        reference = gold_for("qa_scorecard")["reference"]
        with self.assertRaises(ValueError):
            validate_payload("qa_scorecard", reference, role="opinion")

    def test_unknown_family_is_reported_not_raised(self) -> None:
        self.assertTrue(validate_payload("astrology", {}, role="reference"))


class PredictionRecords(unittest.TestCase):
    def test_missing_output_is_reported(self) -> None:
        record = prediction_for("qa_scorecard")
        del record["output"]
        self.assertTrue(validate_prediction_record(record))

    def test_unknown_family_is_reported(self) -> None:
        record = prediction_for("qa_scorecard")
        record["task_family"] = "astrology"
        self.assertTrue(validate_prediction_record(record))

    def test_non_object_prediction_does_not_crash(self) -> None:
        self.assertTrue(validate_prediction_record("nope"))

    def test_telemetry_is_part_of_the_record(self) -> None:
        record = prediction_for("qa_scorecard")
        self.assertIn("telemetry", record)
        self.assertIn("latency_ms", record["telemetry"])


if __name__ == "__main__":
    unittest.main()
