"""Tests for features: tolerance tiers, direction codes, genotyping_regime routing, engineered cols, labels."""

import unittest

import numpy as np
import pandas as pd

import features


class TolRepeatsTest(unittest.TestCase):
    def test_tiers(self):
        # boundaries: <50->0, [50,120]->1, (120,270)->2, [270,600)->4, >=600->8
        bp = np.array([0, 49, 50, 120, 121, 269, 270, 599, 600, 1000])
        self.assertEqual(list(features.tol_repeats(bp)), [0, 0, 1, 1, 2, 2, 4, 4, 8, 8])


class DirectionCodesTest(unittest.TestCase):
    def test_ok_long_short(self):
        eh = np.array([10, 14, 6, 12])
        true = np.array([10, 10, 10, 10])
        tol = np.array([0, 1, 1, 1])
        # dr = 0,+4,-4,+2 ; tol = 0,1,1,1 -> OK, TOO_LONG, TOO_SHORT, TOO_LONG
        self.assertEqual(list(features.direction_codes(eh, true, tol)),
                         [features.OK, features.TOO_LONG, features.TOO_SHORT, features.TOO_LONG])


class GenotypingRegimeOfTest(unittest.TestCase):
    def test_routing(self):
        branch = np.array(["quick", "full", "full"], dtype=object)
        span = np.array([5.0, 2.0, 0.0])
        self.assertEqual(list(features.genotyping_regime_of(branch, span)),
                         [features.GENOTYPING_REGIME_QUICK, features.GENOTYPING_REGIME_FULL_SPANNING,
                          features.GENOTYPING_REGIME_FULL_NONSPANNING])

    def test_nan_spanning_is_nonspanning(self):
        self.assertEqual(features.genotyping_regime_of(np.array(["full"], dtype=object),
                                            np.array([np.nan]))[0],
                         features.GENOTYPING_REGIME_FULL_NONSPANNING)


class EngineeredTest(unittest.TestCase):
    def test_values_and_missing_fill(self):
        df = pd.DataFrame({"eh": [10, 10], "ci_start": [8, np.nan], "ci_end": [14, 20],
                           "ci_width": [6, np.nan]})
        out = features.add_engineered(df)
        # row0: ((14-10)-(10-8))/(6+1) = 2/7 ; ci_over_eh = 6/11
        self.assertAlmostEqual(out["ci_asymmetry"].iloc[0], 2.0 / 7.0)
        self.assertAlmostEqual(out["ci_over_eh"].iloc[0], 6.0 / 11.0)
        # row1: ci inputs missing -> 0
        self.assertEqual(out["ci_asymmetry"].iloc[1], 0.0)
        self.assertEqual(out["ci_over_eh"].iloc[1], 0.0)


class BuildMatrixTest(unittest.TestCase):
    def _raw_row(self):
        row = {c: 1.0 for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")}
        row.update({"ci_start": 1.0, "ci_end": 3.0})  # engineered inputs
        return pd.DataFrame([row])

    def test_full_order_and_count(self):
        X, names = features.build_matrix(self._raw_row(), "full")
        self.assertEqual(names, features.FULL_FEATURES)
        self.assertEqual(list(X.columns), features.FULL_FEATURES)
        self.assertEqual(X.shape[1], 24)

    def test_quick_count(self):
        _, names = features.build_matrix(self._raw_row(), "quick")
        self.assertEqual(names, features.QUICK_FEATURES)
        self.assertEqual(len(names), 22)

    def test_bad_branch(self):
        with self.assertRaises(ValueError):
            features.build_matrix(self._raw_row(), "fast")


class AddLabelsTest(unittest.TestCase):
    def test_labels(self):
        df = pd.DataFrame({"eh": [10.0, 30.0], "true": [10.0, 10.0], "motif_size": [3, 3],
                           "genotyping_branch": ["quick", "full"], "spanning_at_called": [4, 0]})
        features.add_labels(df)
        self.assertAlmostEqual(df["t"].iloc[0], 0.0)
        self.assertAlmostEqual(df["t"].iloc[1], np.log(3.0))  # eh/true = 30/10 = 3
        # allele_bp = 3 * round(10) = 30 -> tol 0 -> eh==true OK ; row1 dr=+20 -> TOO_LONG
        self.assertEqual(df["dir_code"].iloc[0], features.OK)
        self.assertEqual(df["direction"].iloc[1], "TOO_LONG")
        self.assertEqual(df["genotyping_regime"].iloc[0], features.GENOTYPING_REGIME_QUICK)
        self.assertEqual(df["genotyping_regime"].iloc[1], features.GENOTYPING_REGIME_FULL_NONSPANNING)


if __name__ == "__main__":
    unittest.main()
