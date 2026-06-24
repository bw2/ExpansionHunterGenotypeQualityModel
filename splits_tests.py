"""Unit tests for ``splits`` on synthetic DataFrames.

No fixtures: small frames are built in-memory with the columns the splitters
read (``chrom``, ``sample``, ``coverage``, ``source``). The fold tests assert the
Monte-Carlo partition properties and reproducibility; the cross-split tests
assert the right rows land on each side.

Run with:  python3 -m unittest splits_tests -v
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import splits as S


def _row_df():
    """Builds a small frame spanning chroms, samples, coverages, and sources."""
    return pd.DataFrame({
        "chrom": ["1", "1", "5", "X", "Y", "22", "9", "13"],
        "sample": ["HG002", "HG002", "HG002", "HG002",
                   "CHM1_CHM13", "CHM1_CHM13", "HG002", "HG002"],
        "coverage": [10.0, 20.0, 31.0, 10.0, 46.0, 46.0, 20.0, 31.0],
        "source": ["real", "real", "real", "sim", "real", "real", "sim", "real"],
    })


class MakeCvFoldsTest(unittest.TestCase):
    def setUp(self):
        self.folds = S.make_cv_folds()

    def test_ten_folds(self):
        self.assertEqual(len(self.folds), 10)

    def test_partition_sizes_and_disjoint(self):
        for fold in self.folds:
            self.assertEqual(len(fold["test"]), 5)
            self.assertEqual(len(fold["calib"]), 2)
            self.assertEqual(len(fold["train"]), 17)
            # The three sets are disjoint and together cover all 24 chroms.
            self.assertEqual(
                set(fold["test"]) | set(fold["calib"]) | set(fold["train"]),
                set(S.ALL_CHROMS))
            self.assertEqual(
                len(fold["test"]) + len(fold["calib"]) + len(fold["train"]),
                len(S.ALL_CHROMS))
            self.assertFalse(set(fold["test"]) & set(fold["calib"]))
            self.assertFalse(set(fold["test"]) & set(fold["train"]))
            self.assertFalse(set(fold["calib"]) & set(fold["train"]))

    def test_only_known_chroms(self):
        for fold in self.folds:
            for group in ("test", "calib", "train"):
                self.assertTrue(set(fold[group]) <= set(S.ALL_CHROMS))
                # All chrom names are plain JSON-serializable strings.
                self.assertTrue(all(isinstance(c, str) for c in fold[group]))

    def test_reproducible_same_seed(self):
        self.assertEqual(self.folds, S.make_cv_folds())

    def test_different_seed_differs(self):
        self.assertNotEqual(self.folds, S.make_cv_folds(seed=S.SEED + 1))

    def test_monte_carlo_overlap_across_folds(self):
        # Independent draws: test chroms are not a single partition of the 24.
        self.assertTrue(
            sum(len(f["test"]) for f in self.folds) > len(set(
                c for f in self.folds for c in f["test"])))


class SaveLoadFoldsTest(unittest.TestCase):
    def test_roundtrip(self):
        folds = S.make_cv_folds()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "folds.json")
        S.save_folds(folds, path)
        self.assertEqual(S.load_folds(path), folds)


class FoldMasksTest(unittest.TestCase):
    def test_masks_partition_rows(self):
        df = _row_df()
        fold = {"train": ["1", "9", "13"], "calib": ["5"], "test": ["X", "Y", "22"]}
        train_mask, calib_mask, test_mask = S.fold_masks(df, fold)
        for mask in (train_mask, calib_mask, test_mask):
            self.assertEqual(mask.dtype, bool)
            self.assertEqual(mask.shape, (len(df),))
        # chrom "1" rows -> train, "5" -> calib, "X"/"Y"/"22" -> test.
        np.testing.assert_array_equal(
            train_mask, df["chrom"].isin({"1", "9", "13"}).to_numpy())
        np.testing.assert_array_equal(test_mask, df["chrom"].isin({"X", "Y", "22"}).to_numpy())
        # Disjoint and (here) covering every row exactly once.
        self.assertFalse(np.any(train_mask & calib_mask))
        self.assertFalse(np.any(train_mask & test_mask))
        self.assertFalse(np.any(calib_mask & test_mask))
        np.testing.assert_array_equal(train_mask | calib_mask | test_mask,
                                      np.ones(len(df), dtype=bool))


class CrossSampleSplitTest(unittest.TestCase):
    def test_hg002_train_chm_test(self):
        df = _row_df()
        train_mask, test_mask = S.cross_sample_split(df)
        np.testing.assert_array_equal(
            train_mask, df["sample"].str.startswith("HG002").to_numpy())
        np.testing.assert_array_equal(
            test_mask, df["sample"].str.startswith("CHM").to_numpy())
        # No row is on both sides.
        self.assertFalse(np.any(train_mask & test_mask))


class CrossCoverageSplitsTest(unittest.TestCase):
    def test_leave_one_coverage_out(self):
        df = _row_df()
        splits = S.cross_coverage_splits(df)
        self.assertEqual([cov for cov, _, _ in splits], [10, 20, 31])
        is_hg002 = df["sample"].str.startswith("HG002").to_numpy()
        for held, train_mask, test_mask in splits:
            # Test = HG002 rows at the held coverage.
            np.testing.assert_array_equal(
                test_mask, is_hg002 & (df["coverage"].to_numpy() == held))
            # Train = HG002 rows at the other coverages.
            np.testing.assert_array_equal(
                train_mask, is_hg002 & (df["coverage"].to_numpy() != held))
            self.assertFalse(np.any(train_mask & test_mask))

    def test_chm_never_included(self):
        df = _row_df()
        is_chm = df["sample"].str.startswith("CHM").to_numpy()
        for _, train_mask, test_mask in S.cross_coverage_splits(df):
            self.assertFalse(np.any(is_chm & train_mask))
            self.assertFalse(np.any(is_chm & test_mask))


class CrossDomainSplitsTest(unittest.TestCase):
    def test_two_directions(self):
        df = _row_df()
        splits = S.cross_domain_splits(df)
        self.assertEqual([name for name, _, _ in splits],
                         ["train_real_test_sim", "train_sim_test_real"])
        is_real = (df["source"] == "real").to_numpy()
        is_sim = (df["source"] == "sim").to_numpy()
        (_, train0, test0), (_, train1, test1) = splits
        np.testing.assert_array_equal(train0, is_real)
        np.testing.assert_array_equal(test0, is_sim)
        np.testing.assert_array_equal(train1, is_sim)
        np.testing.assert_array_equal(test1, is_real)


if __name__ == "__main__":
    unittest.main(verbosity=2)
