"""Tests for train's held-out calibration rows and per-regime model sizes."""

import unittest

import numpy as np

import features
import train


class CalibMaskTest(unittest.TestCase):
    def setUp(self):
        people = ["HG002", "CHM1_CHM13"] + ["S%02d" % i for i in range(20)]
        chroms = [str(c) for c in range(1, 23)] + ["X"]
        self.individual = np.array([p for p in people for _ in chroms], dtype=object)
        self.chrom = np.array([c for _ in people for c in chroms], dtype=object)
        self.always = np.isin(self.individual, train.ALWAYS_TRAIN_INDIVIDUALS)

    def test_other_people_are_held_out_whole(self):
        calib = train._calib_mask(self.individual, self.chrom, 7)
        people = set(self.individual[calib & ~self.always])
        self.assertEqual(len(people), train.N_CALIB_INDIVIDUALS)
        for p in people:  # a held-out person is held out entirely
            self.assertTrue(calib[self.individual == p].all())

    def test_always_train_people_lose_only_one_autosome(self):
        calib = train._calib_mask(self.individual, self.chrom, 7)
        held = calib & self.always
        self.assertEqual(len(set(self.chrom[held])), 1)          # one chromosome ...
        self.assertTrue(str(self.chrom[held][0]).isdigit())      # ... an autosome ...
        self.assertEqual(set(self.individual[held]), set(train.ALWAYS_TRAIN_INDIVIDUALS))  # of both

    def test_deterministic(self):
        np.testing.assert_array_equal(train._calib_mask(self.individual, self.chrom, 7),
                                      train._calib_mask(self.individual, self.chrom, 7))


class IterationsByRegimeTest(unittest.TestCase):
    def test_every_regime_has_positive_direction_and_q_median_sizes(self):
        self.assertEqual(set(train.ITERATIONS_BY_REGIME), set(features.GENOTYPING_REGIMES))
        for direction, q_median in train.ITERATIONS_BY_REGIME.values():
            self.assertGreater(direction, 0)
            self.assertGreater(q_median, 0)


if __name__ == "__main__":
    unittest.main()
