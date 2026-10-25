"""Tests for train's held-out calibration rows and per-regime model sizes."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

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


class DirectionLabelOptionTest(unittest.TestCase):
    """fit_genotyping_regime reads the chosen population's parquet and trains on the chosen label."""

    def _fit(self, direction_label, parquet_subdir="parquet_all_loci"):
        people = ["HG002", "CHM1_CHM13"] + ["S%02d" % i for i in range(8)]
        rows = []
        for person in people:
            for chrom in ("1", "2"):
                row = {c: 1.0 for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")}
                row.update({"ci_start": 1.0, "ci_end": 2.0, "t": 0.0, "dir_code": features.OK,
                            "dir_code_genotype": features.TOO_SHORT, "chrom": chrom, "genotyping_regime": "quick",
                            "individual": person, "representative": True})
                rows.append(row)
        captured = {}

        def fake_fit_heads(df, *args, **kwargs):
            captured["dir_code"] = df["dir_code"].unique().tolist()
            raise StopIteration  # stop before serialization; only the label matters here

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, parquet_subdir))
            pd.DataFrame(rows).to_parquet(os.path.join(d, parquet_subdir, "quick.parquet"))
            with mock.patch.object(train, "fit_heads", side_effect=fake_fit_heads), self.assertRaises(StopIteration):
                train.fit_genotyping_regime("quick", d, None, parquet_subdir=parquet_subdir,
                                            direction_label=direction_label)
        return captured["dir_code"]

    def test_the_genotype_level_label_replaces_dir_code(self):
        self.assertEqual(self._fit("dir_code_genotype"), [features.TOO_SHORT])
        self.assertEqual(self._fit("dir_code"), [features.OK])

    def test_non_default_choices_get_their_own_default_output_name(self):
        today = train._default_out()
        names = {train._default_out(population, label) for population in ("variant_loci", "all_loci")
                 for label in train.DIRECTION_LABEL_COLUMNS}
        self.assertEqual(len(names), 4)  # no two choices overwrite each other's model
        self.assertIn(today, names)
        date = os.path.basename(today).split(".")[1]
        for name in names:  # the date stays the last token before .json.gz, where compare_models reads it
            self.assertTrue(name.endswith(".%s.json.gz" % date), name)


class RelativeToleranceOptionTest(unittest.TestCase):
    """fit_genotyping_regime with --relative-tolerance: change B's label, 4 classes and the fragment-length input."""

    def _fit(self, direction_label, with_fragment_length=True):
        people = ["HG002", "CHM1_CHM13"] + ["S%02d" % i for i in range(8)]
        rows = []
        for person in people:
            for chrom in ("1", "2"):
                row = {c: 1.0 for c in features.FULL_FEATURES if c not in features.ENGINEERED_FEATURES}
                # A homozygous 100 bp call: OK against its own 100 bp truth allele, TOO_SHORT against the other
                # 1000 bp one (past the 500 bp fragment length).
                row.update({"eh": 20.0, "motif_size": 5.0, "true": 20.0, "true_other_allele": 200.0,
                            "ci_start": 19.0, "ci_end": 21.0, "t": 0.0, "dir_code": features.TOO_LONG,
                            "dir_code_genotype": features.TOO_LONG, "chrom": chrom, "genotyping_regime": "quick",
                            "individual": person, "representative": True})
                if with_fragment_length:
                    row["fragment_length"] = 500.0
                rows.append(row)
        captured = {}

        def fake_fit_heads(df, *args, **kwargs):
            captured.update(dir_code=df["dir_code"].unique().tolist(), **kwargs)
            raise StopIteration  # stop before serialization; only what fit_heads gets matters here

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "parquet_all_loci"))
            pd.DataFrame(rows).to_parquet(os.path.join(d, "parquet_all_loci", "quick.parquet"))
            with mock.patch.object(train, "fit_heads", side_effect=fake_fit_heads), self.assertRaises(StopIteration):
                train.fit_genotyping_regime("quick", d, None, parquet_subdir="parquet_all_loci",
                                            direction_label=direction_label, relative_tolerance=0.10)
        return captured

    def test_labels_are_computed_from_the_stored_columns_not_read(self):
        self.assertEqual(self._fit("dir_code")["dir_code"], [features.OK])
        self.assertEqual(self._fit("dir_code_genotype")["dir_code"], [features.TOO_SHORT])

    def test_the_head_gets_four_classes_and_the_fragment_length_ratio(self):
        captured = self._fit("dir_code")
        self.assertEqual(captured["n_classes"], 4)
        self.assertEqual(captured["names"], features.feature_names(features.BRANCH_QUICK, with_fragment_length_ratio=True))

    def test_a_table_without_fragment_length_is_refused(self):
        with self.assertRaises(SystemExit) as ctx:
            self._fit("dir_code", with_fragment_length=False)
        self.assertIn("fragment_length", str(ctx.exception))

    def test_each_tolerance_gets_its_own_default_output_name(self):
        names = {train._default_out("all_loci", "dir_code_genotype", x) for x in (None, 0.10, 0.15)}
        self.assertEqual(len(names), 3)
        self.assertTrue(any(name.endswith("_all_loci_dir_code_genotype_relative_tolerance_10pct.%s.json.gz"
                                          % os.path.basename(train._default_out()).split(".")[1]) for name in names))


class IterationsByRegimeTest(unittest.TestCase):
    def test_every_regime_has_positive_direction_and_q_median_sizes(self):
        self.assertEqual(set(train.ITERATIONS_BY_REGIME), set(features.GENOTYPING_REGIMES))
        for direction, q_median in train.ITERATIONS_BY_REGIME.values():
            self.assertGreater(direction, 0)
            self.assertGreater(q_median, 0)


if __name__ == "__main__":
    unittest.main()
