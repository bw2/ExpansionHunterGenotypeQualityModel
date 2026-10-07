"""Tests for model_size_curves' size rule and validation split."""

import unittest

import numpy as np
import pandas as pd

import model_size_curves as C
import train


def _result(nonhomo, homo):
    ks = [100, 500, 1000, 4000]
    return {"gated_mae": {"val_known": {"nonhomo": [[k, v] for k, v in zip(ks, nonhomo)],
                                        "homo": [[k, v] for k, v in zip(ks, homo)]}}}


class SmallestWithinTest(unittest.TestCase):
    def test_both_strata_must_be_within_the_tolerance(self):
        r = _result([10.5, 10.1, 10.0, 10.0], [2.0, 1.04, 1.02, 1.0])
        self.assertEqual(C.smallest_within(r, 0.02), 1000)  # 500 is close on nonhomo but not homo
        self.assertEqual(C.smallest_within(r, 0.05), 500)
        self.assertEqual(C.smallest_within(r, 0.0), 4000)  # homopolymer error still falls after 1,000


class SplitTest(unittest.TestCase):
    def test_validation_people_are_never_calibration_and_unseen_is_by_chromosome(self):
        people = ["HG002", "CHM1_CHM13"] + ["S%02d" % i for i in range(25)]
        chroms = [str(c) for c in range(1, 23)]
        df = pd.DataFrame({"individual": [p for p in people for _ in chroms],
                           "chrom": [c for _ in people for c in chroms]})
        validation, unseen, calib = C.split(df)
        self.assertFalse((validation & calib).any())
        self.assertEqual(len(set(df["individual"][validation])), C.N_VALIDATION_PEOPLE)
        self.assertFalse(np.isin(df["individual"][validation], train.ALWAYS_TRAIN_INDIVIDUALS).any())
        self.assertEqual(set(df["chrom"][unseen]), set(C.UNSEEN_CHROMS))


if __name__ == "__main__":
    unittest.main()
