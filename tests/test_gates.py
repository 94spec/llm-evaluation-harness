import unittest

from conv_eval.gates import evaluate_gates


class GateTests(unittest.TestCase):
    def test_absolute_and_relative_rules(self) -> None:
        baseline = {"quality": {"f1": 0.90, "mae": 2.0}}
        candidate = {"quality": {"f1": 0.895, "mae": 2.4}}
        config = {
            "name": "test",
            "version": "1",
            "rules": [
                {
                    "id": "f1",
                    "metric": "quality.f1",
                    "operator": "max_drop",
                    "value": 0.01,
                    "severity": "block",
                },
                {
                    "id": "mae",
                    "metric": "quality.mae",
                    "operator": "max_increase",
                    "value": 0.5,
                    "severity": "block",
                },
                {
                    "id": "absolute",
                    "metric": "quality.f1",
                    "operator": "gte",
                    "value": 0.89,
                    "severity": "block",
                },
            ],
        }
        result = evaluate_gates(candidate, baseline, config)
        self.assertTrue(result["passed"])

    def test_missing_metric_fails_closed(self) -> None:
        result = evaluate_gates(
            {},
            None,
            {
                "rules": [
                    {
                        "id": "required",
                        "metric": "missing.value",
                        "operator": "gte",
                        "value": 1,
                        "severity": "block",
                    }
                ]
            },
        )
        self.assertFalse(result["passed"])
        self.assertIn("error", result["rules"][0])


if __name__ == "__main__":
    unittest.main()

