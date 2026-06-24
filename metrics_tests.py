"""Tests for the held-out metric functions."""

import unittest

import numpy as np

import metrics


class QMetricsTest(unittest.TestCase):
    def test_perfect_lcf(self):
        eh = np.array([12.0, 8.0])
        true = np.array([10.0, 10.0])
        true_pred = true.copy()           # LCF recovered truth exactly
        m = metrics.q_metrics(eh, true, true_pred, np.log(eh / true), np.log(eh / true),
                              tol=np.array([0, 0]))
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["mae_true"], 0.0)
        self.assertAlmostEqual(m["mae_eh"], 2.0)       # |10-12| and |10-8|
        self.assertAlmostEqual(m["dist_reduction"], 1.0)
        self.assertAlmostEqual(m["exact_match_rate"], 1.0)
        self.assertAlmostEqual(m["eh_exact_match_rate"], 0.0)

    def test_empty(self):
        m = metrics.q_metrics([], [], [], [], [], [])
        self.assertEqual(m["n"], 0)


class DirectionMetricsTest(unittest.TestCase):
    def test_confident_correct(self):
        dir_code = np.array([0, 1, 2])
        proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        m = metrics.direction_metrics(dir_code, proba)
        self.assertEqual(m["p_ok_accuracy"], 1.0)
        self.assertEqual(m["confusion"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class GatedMaeTest(unittest.TestCase):
    def test_gate(self):
        eh = np.array([12.0, 20.0])
        true = np.array([10.0, 10.0])
        true_pred = np.array([10.0, 10.0])
        p_ok = np.array([0.9, 0.1])  # only the 2nd allele is corrected (p_ok < 0.5)
        m = metrics.gated_mae(eh, true, true_pred, p_ok)
        self.assertAlmostEqual(m["mae_raw"], (2.0 + 10.0) / 2.0)
        # corrected = [eh=12 (kept), true_pred=10] -> errors |10-12|=2, |10-10|=0
        self.assertAlmostEqual(m["mae_gated"], (2.0 + 0.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
