"""Unit tests for ``size_bins`` (allele-size bins + equal-per-bin weights).

Run with:  python3 -m unittest size_bins_tests -v
"""

import unittest

import numpy as np

import size_bins as SB


class AssignBinsTest(unittest.TestCase):
    def test_zero_delta_lands_in_the_no_change_bin(self):
        # delta 0 -> the "0" bin, which is index 13 (BIN_LABELS[13] == "0").
        idx = SB.assign_bins([10.0], [10.0])
        self.assertEqual(int(idx[0]), 13)
        self.assertEqual(SB.BIN_LABELS[13], "0")

    def test_boundary_deltas_map_to_expected_labels(self):
        # (true, ref) chosen to hit each named edge of a few bins.
        cases = {
            (-2, "-2:-1"), (-1, "-2:-1"), (1, "1:2"), (2, "1:2"),
            (3, "3:4"), (-3, "-4:-3"), (-31, "<=-31"), (-32, "<=-31"),
            (31, ">=31"), (40, ">=31"), (25, "21:25"), (26, "26:30"),
            (30, "26:30"), (20, "19:20"),
        }
        for delta, label in cases:
            idx = int(SB.assign_bins([100.0 + delta], [100.0])[0])
            self.assertEqual(SB.BIN_LABELS[idx], label, "delta=%d" % delta)

    def test_rounds_fractional_truth(self):
        # 10.4 rounds to 10 (delta 0), 10.6 rounds to 11 (delta +1 -> "1:2").
        self.assertEqual(SB.BIN_LABELS[int(SB.assign_bins([10.4], [10.0])[0])], "0")
        self.assertEqual(SB.BIN_LABELS[int(SB.assign_bins([10.6], [10.0])[0])], "1:2")

    def test_indices_in_range_and_monotone_in_delta(self):
        deltas = np.arange(-60, 61)
        idx = SB.assign_bins(100 + deltas, np.full_like(deltas, 100))
        self.assertTrue(np.all((idx >= 0) & (idx < SB.N_BINS)))
        self.assertTrue(np.all(np.diff(idx) >= 0))  # non-decreasing with delta


class BinWeightsTest(unittest.TestCase):
    def test_mean_weight_is_one(self):
        bins = np.array([0] * 1000 + [5] * 10 + [13] * 50000)
        w = SB.bin_weights(bins, cap=50.0)
        self.assertAlmostEqual(float(np.mean(w)), 1.0, places=6)

    def test_uncapped_equalizes_total_weight_per_bin(self):
        bins = np.array([0] * 1000 + [5] * 10 + [13] * 50000)
        w = SB.bin_weights(bins, cap=0.0)  # pure inverse-frequency
        totals = {b: float(w[bins == b].sum()) for b in np.unique(bins)}
        self.assertAlmostEqual(totals[0], totals[5], places=4)
        self.assertAlmostEqual(totals[5], totals[13], places=4)

    def test_cap_is_a_hard_ceiling(self):
        bins = np.array([0] * 1000 + [5] * 2 + [13] * 50000)  # bin 5 is tiny -> huge raw weight
        w_capped = SB.bin_weights(bins, cap=50.0)
        w_pure = SB.bin_weights(bins, cap=0.0)
        # No weight exceeds the cap, the tiny bin is the one clamped, and capping shrank it.
        self.assertLessEqual(float(w_capped.max()), 50.0 + 1e-9)
        self.assertAlmostEqual(float(w_capped[bins == 5].max()), 50.0, places=6)
        self.assertLess(w_capped[bins == 5].max(), w_pure[bins == 5].max())
        self.assertAlmostEqual(float(np.mean(w_capped)), 1.0, places=6)

    def test_uniform_bins_give_unit_weights(self):
        w = SB.bin_weights(np.full(100, 13), cap=50.0)
        self.assertTrue(np.allclose(w, 1.0))

    def test_empty_input(self):
        self.assertEqual(SB.bin_weights(np.array([], dtype=int)).size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
