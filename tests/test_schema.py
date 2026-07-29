import unittest

from conv_eval.schema import validate_payload


class SchemaTests(unittest.TestCase):
    def test_qa_prediction_is_strict(self) -> None:
        issues = validate_payload(
            "qa_scorecard",
            {
                "criteria": {"c1": "pass"},
                "total_score": 100,
                "explanation": "not allowed",
            },
            role="prediction",
        )
        self.assertTrue(any("unknown field" in issue.message for issue in issues))

    def test_violation_state_is_controlled(self) -> None:
        issues = validate_payload(
            "violation_flags",
            {"flags": {"pressure": "maybe"}},
            role="prediction",
        )
        self.assertTrue(any("must be one of" in issue.message for issue in issues))

    def test_summary_valid_shape(self) -> None:
        issues = validate_payload(
            "deal_summary",
            {
                "outcome": "follow_up",
                "primary_need": "career_growth",
                "objections": ["price"],
                "next_step": "call tomorrow",
                "facts": ["client compared options"],
            },
            role="prediction",
        )
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()

