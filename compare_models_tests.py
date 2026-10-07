"""Tests for compare_models' per-sample paired comparison helpers."""

import unittest

import numpy as np

import compare_models


PREV = {"S1": {"n": 10, "sum_gated_abs_error": 20.0}, "S2": {"n": 5, "sum_gated_abs_error": 5.0},
        "S3": {"n": 0, "sum_gated_abs_error": 0.0}}
NEW = {"S1": {"n": 10, "sum_gated_abs_error": 10.0}, "S2": {"n": 5, "sum_gated_abs_error": 10.0},
       "S3": {"n": 0, "sum_gated_abs_error": 0.0}}


class PairedComparisonTest(unittest.TestCase):
    def test_deltas_are_new_minus_prev_per_sample_and_skip_empty_samples(self):
        deltas = compare_models._per_sample_gated_mae_deltas(
            PREV, NEW, "n", "sum_gated_abs_error", ["S1", "S2", "S3", "S4"])
        np.testing.assert_allclose(deltas, [1.0 - 2.0, 2.0 - 1.0])

    def test_pooled_mae_weights_alleles_and_can_leave_samples_out(self):
        self.assertAlmostEqual(compare_models._pooled_gated_mae(PREV, "n", "sum_gated_abs_error", ["S1", "S2"]),
                               25.0 / 15)
        self.assertAlmostEqual(compare_models._pooled_gated_mae(PREV, "n", "sum_gated_abs_error", ["S2"]), 1.0)

    def test_bootstrap_interval_is_deterministic_and_brackets_the_mean(self):
        values = np.array([-0.3, -0.1, -0.2, 0.05, -0.15])
        first = compare_models._bootstrap_mean_interval(values)
        np.testing.assert_array_equal(first, compare_models._bootstrap_mean_interval(values))
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())


if __name__ == "__main__":
    unittest.main()
