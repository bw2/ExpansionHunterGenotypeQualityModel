"""Unit tests for ``size_tolerance`` (size-dependent tolerance + regime routing).

Run with:  python3 -m unittest size_tolerance_tests -v
"""

import unittest

import numpy as np

import size_tolerance as ST


class TolRepeatsTest(unittest.TestCase):
    def test_tier_boundaries(self):
        # (bp, expected tol) at and around every edge per the boundary rule.
        cases = [(0, 0), (49, 0), (50, 1), (120, 1), (121, 2), (269, 2), (270, 4),
                 (599, 4), (600, 8), (5000, 8)]
        bp = np.array([b for b, _ in cases], dtype=float)
        got = ST.tol_repeats(bp)
        for (b, exp), g in zip(cases, got):
            self.assertEqual(int(g), exp, "bp=%d" % b)


class DirectionCodesTest(unittest.TestCase):
    def test_exact_band_when_tol_zero(self):
        # tol 0: only an exact call is OK; off-by-one is a miscall.
        self.assertEqual(int(ST.direction_codes([10], [10], [0])[0]), ST.OK)
        self.assertEqual(int(ST.direction_codes([11], [10], [0])[0]), ST.TOO_LONG)
        self.assertEqual(int(ST.direction_codes([9], [10], [0])[0]), ST.TOO_SHORT)

    def test_wide_band_tolerates_within(self):
        # tol 4: +/-4 is OK, beyond is TOO_LONG/TOO_SHORT.
        eh = np.array([104, 96, 105, 95, 100])
        out = ST.direction_codes(eh, np.full(5, 100), np.full(5, 4))
        self.assertEqual(out.tolist(), [ST.OK, ST.OK, ST.TOO_LONG, ST.TOO_SHORT, ST.OK])


class RegimeTest(unittest.TestCase):
    def test_fast_branch_routes_fast(self):
        out = ST.regime_of(["fast", "fast"], [0, 5])
        self.assertEqual(out.tolist(), [ST.REGIME_FAST, ST.REGIME_FAST])

    def test_full_split_on_spanning_support(self):
        out = ST.regime_of(["full", "full", "full"], [1, 0, np.nan])
        self.assertEqual(out.tolist(), [ST.REGIME_FULL_SPANNING,
                                        ST.REGIME_FULL_NONSPANNING,
                                        ST.REGIME_FULL_NONSPANNING])  # NaN spanning -> non-spanning


if __name__ == "__main__":
    unittest.main(verbosity=2)
