"""Unit tests for ``features`` on synthetic DataFrames.

No fixtures: a small frame is built in-memory carrying every raw column the
feature matrix needs (all ``FAST``/``FULL`` inputs plus ``eh`` and the CI bounds
used by the engineered columns), then ``build_matrix`` / ``add_engineered`` are
checked for column membership, branch sizes, bool casting, NaN handling, and the
hand-computed engineered values.

Run with:  python3 -m unittest features_tests -v
"""

import unittest

import numpy as np
import pandas as pd

import features as F


# Raw feature columns that are NOT engineered (engineered = ci_asymmetry/ci_over_eh).
_RAW_FAST = [c for c in F.FAST_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")]
_FULL_ONLY = ["left_flank_norm_depth", "right_flank_norm_depth"]


def _raw_df(n=4):
    """Builds a frame with every raw column build_matrix needs, plus CI bounds."""
    df = pd.DataFrame({c: np.arange(1.0, n + 1.0) for c in _RAW_FAST + _FULL_ONLY})
    # CI bounds consumed by add_engineered (not themselves features).
    df["eh"] = [10.0, 50.0, 30.0, 100.0][:n]
    df["ci_start"] = [8.0, 45.0, 30.0, 90.0][:n]
    df["ci_end"] = [20.0, 60.0, 30.0, 130.0][:n]
    df["ci_width"] = [12.0, 15.0, 0.0, 40.0][:n]
    return df


class FeatureListTest(unittest.TestCase):
    def test_full_has_two_more_than_fast(self):
        self.assertEqual(len(F.FULL_FEATURES), len(F.FAST_FEATURES) + 2)
        self.assertEqual(F.FULL_FEATURES[:len(F.FAST_FEATURES)], F.FAST_FEATURES)
        self.assertEqual(F.FULL_FEATURES[len(F.FAST_FEATURES):], _FULL_ONLY)

    def test_no_duplicate_feature_names(self):
        self.assertEqual(len(set(F.FAST_FEATURES)), len(F.FAST_FEATURES))
        self.assertEqual(len(set(F.FULL_FEATURES)), len(F.FULL_FEATURES))

    def test_engineered_in_fast(self):
        self.assertIn("ci_asymmetry", F.FAST_FEATURES)
        self.assertIn("ci_over_eh", F.FAST_FEATURES)


class AddEngineeredTest(unittest.TestCase):
    def test_hand_computed_values(self):
        out = F.add_engineered(_raw_df())
        # Row 0: ((20-10)-(10-8))/(12+1) = 8/13 ; 12/(10+1) = 12/11.
        self.assertAlmostEqual(out["ci_asymmetry"].iloc[0], 8.0 / 13.0)
        self.assertAlmostEqual(out["ci_over_eh"].iloc[0], 12.0 / 11.0)
        # Row 2: symmetric CI of width 0 -> asymmetry 0, ci_over_eh 0/(31)=0.
        self.assertAlmostEqual(out["ci_asymmetry"].iloc[2], 0.0)
        self.assertAlmostEqual(out["ci_over_eh"].iloc[2], 0.0 / 31.0)

    def test_missing_ci_filled_with_zero(self):
        df = _raw_df()
        df.loc[1, "ci_start"] = np.nan   # missing CI bound
        df.loc[3, "ci_width"] = np.nan   # missing width
        out = F.add_engineered(df)
        self.assertEqual(out["ci_asymmetry"].iloc[1], 0.0)
        self.assertEqual(out["ci_over_eh"].iloc[3], 0.0)
        # ci_asymmetry at row 3 stays computed (ci_start/ci_end present, width NaN
        # -> width missing too, so it is filled with 0).
        self.assertEqual(out["ci_asymmetry"].iloc[3], 0.0)

    def test_does_not_mutate_input(self):
        df = _raw_df()
        F.add_engineered(df)
        self.assertNotIn("ci_asymmetry", df.columns)
        self.assertNotIn("ci_over_eh", df.columns)


class BuildMatrixTest(unittest.TestCase):
    def test_fast_columns(self):
        X, names = F.build_matrix(_raw_df(), "fast")
        self.assertEqual(names, F.FAST_FEATURES)
        self.assertEqual(list(X.columns), F.FAST_FEATURES)
        self.assertEqual(len(X), 4)

    def test_full_columns(self):
        X, names = F.build_matrix(_raw_df(), "full")
        self.assertEqual(names, F.FULL_FEATURES)
        self.assertEqual(list(X.columns), F.FULL_FEATURES)
        self.assertEqual(X.shape[1], len(F.FAST_FEATURES) + 2)

    def test_all_float(self):
        X, _ = F.build_matrix(_raw_df(), "full")
        self.assertTrue(all(dt == np.float64 for dt in X.dtypes))

    def test_nan_preserved(self):
        df = _raw_df()
        df.loc[2, "depth"] = np.nan
        X, _ = F.build_matrix(df, "full")
        self.assertTrue(np.isnan(X["depth"].iloc[2]))

    def test_engineered_columns_present(self):
        X, _ = F.build_matrix(_raw_df(), "fast")
        self.assertAlmostEqual(X["ci_asymmetry"].iloc[0], 8.0 / 13.0)
        self.assertAlmostEqual(X["ci_over_eh"].iloc[0], 12.0 / 11.0)

    def test_missing_column_asserts(self):
        with self.assertRaises(AssertionError):
            F.build_matrix(_raw_df().drop(columns=["left_flank_norm_depth"]), "full")

    def test_fast_branch_ignores_missing_full_only(self):
        # Fast df lacks the full-only flank-depth columns but build_matrix("fast") must still work.
        X, _ = F.build_matrix(_raw_df().drop(columns=_FULL_ONLY), "fast")
        self.assertEqual(list(X.columns), F.FAST_FEATURES)

    def test_bad_branch_raises(self):
        with self.assertRaises(ValueError):
            F.build_matrix(_raw_df(), "medium")


class FeatureFamiliesTest(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(set(F.FEATURE_FAMILIES),
                         {"tier2_aqm", "ci", "read_fractions"})

    def test_family_members_are_features(self):
        # Every family member is a real full-branch feature.
        for members in F.FEATURE_FAMILIES.values():
            self.assertTrue(set(members) <= set(F.FULL_FEATURES))

    def test_full_branch_keeps_full_only_members(self):
        fams = F.feature_families("full")
        # The flank-normalized depths are full-only members of tier2_aqm.
        self.assertIn("left_flank_norm_depth", fams["tier2_aqm"])
        self.assertIn("right_flank_norm_depth", fams["tier2_aqm"])

    def test_fast_branch_drops_full_only(self):
        fams = F.feature_families("fast")
        # The full-only flank-depth members are dropped from tier2_aqm on fast.
        for full_only in _FULL_ONLY:
            self.assertNotIn(full_only, fams.get("tier2_aqm", []))
        # CI / read-fraction families are unchanged on fast.
        self.assertEqual(fams["ci"], F.FEATURE_FAMILIES["ci"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
