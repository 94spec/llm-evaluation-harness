import unittest

from conv_eval.metrics import (
    cohen_kappa,
    krippendorff_alpha_nominal,
    multilabel_metrics,
    percentile,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_kappa(self) -> None:
        labels = ["pass", "fail", "pass", "not_applicable"]
        self.assertEqual(cohen_kappa(labels, labels), 1.0)

    def test_kappa_accounts_for_chance(self) -> None:
        expected = ["yes", "yes", "no", "no"]
        predicted = ["yes", "no", "yes", "no"]
        self.assertAlmostEqual(cohen_kappa(expected, predicted), 0.0)

    def test_perfect_krippendorff_alpha(self) -> None:
        units = [["yes", "yes", "yes"], ["no", "no", None], ["yes", "yes", "yes"]]
        self.assertEqual(krippendorff_alpha_nominal(units), 1.0)

    def test_multilabel_metrics(self) -> None:
        result = multilabel_metrics(
            [{"a", "b"}, {"c"}],
            [{"a"}, {"c", "d"}],
        )
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["micro_f1"], 2 / 3)
        self.assertEqual(result["exact_match"], 0.0)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0, 10], 0.5), 5)


if __name__ == "__main__":
    unittest.main()

