"""Unit tests locking the macro F0.5 scorer against the PDF worked example (§6)."""

import unittest

from src.matching.scorer import f05_entity, macro_f05


class TestScorer(unittest.TestCase):
    def test_pdf_worked_example(self):
        # PDF: P=2/3, R=1 -> 0.714...
        self.assertAlmostEqual(f05_entity(["a", "b", "c"], ["a", "b"]), 5 / 7, places=6)

    def test_perfect(self):
        self.assertEqual(f05_entity(["a"], ["a"]), 1.0)

    def test_empty_empty_is_one(self):
        self.assertEqual(f05_entity([], []), 1.0)

    def test_empty_prediction_nonempty_truth_is_zero(self):
        self.assertEqual(f05_entity([], ["a"]), 0.0)

    def test_nonempty_prediction_empty_truth_is_zero(self):
        self.assertEqual(f05_entity(["a"], []), 0.0)

    def test_precision_weighting(self):
        # precision 0.5, recall 1.0 -> 1.25*0.5/(0.125+1) = 0.5556
        self.assertAlmostEqual(f05_entity(["a", "b"], ["a"]), 0.5556, places=3)

    def test_macro_includes_zero_candidate_entities(self):
        pred = {"S1-1": ["x"], "S1-2": [], "S1-3": []}
        true = {"S1-1": ["x"], "S1-2": ["y"], "S1-3": []}
        self.assertAlmostEqual(macro_f05(pred, true), (1.0 + 0.0 + 1.0) / 3)


if __name__ == "__main__":
    unittest.main()
