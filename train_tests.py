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

    def test_spiked_samples_are_never_held_out_and_do_not_change_who_is(self):
        spiked = ["S%02d_spiked_%d" % (i, k) for i in range(5) for k in range(3)]
        individual = np.concatenate([self.individual, np.repeat(spiked, 23)]).astype(object)
        chrom = np.concatenate([self.chrom, np.tile(self.chrom[:23], len(spiked))]).astype(object)
        self.assertEqual(train._calibration_people(individual, chrom, 7),
                         train._calibration_people(self.individual, self.chrom, 7))
        self.assertFalse(train._calib_mask(individual, chrom, 7)[len(self.individual):].any())

    def test_a_held_out_person_with_spiked_samples_is_refused(self):
        held, _ = train._calibration_people(self.individual, self.chrom, 7)
        individual = np.concatenate([self.individual, ["%s_spiked_0" % held[0]] * 23]).astype(object)
        chrom = np.concatenate([self.chrom, self.chrom[:23]]).astype(object)
        with self.assertRaisesRegex(RuntimeError, held[0]):
            train._calibration_people(individual, chrom, 7)

    def test_spiked_samples_of_the_always_trained_people_are_held_out_on_their_calibration_autosome(self):
        _, autosome = train._calibration_people(self.individual, self.chrom, 7)
        individual = np.array(["HG002_spiked_0", "HG002_spiked_0", "CHM1_CHM13_spiked_2"], dtype=object)
        chrom = np.array([autosome, "22" if autosome != "22" else "21", autosome], dtype=object)
        people = train._calibration_people(self.individual, self.chrom, 7)
        self.assertEqual(train._calib_mask(individual, chrom, 7, people).tolist(), [True, False, True])

    def test_a_spiked_sample_joins_its_persons_out_of_fold_group(self):
        groups = train._fold_groups(np.array(["S01", "S01_spiked_3", "HG002_spiked_0", "S02"], dtype=object),
                                    np.array(["1", "2", "7", "1"], dtype=object))
        self.assertEqual(groups.tolist(), ["S01", "S01", "HG002|7", "S02"])

    def test_another_table_of_the_same_people_holds_out_the_same_people(self):
        people = train._calibration_people(self.individual, self.chrom, 7)
        np.testing.assert_array_equal(train._calib_mask(self.individual, self.chrom, 7, people),
                                      train._calib_mask(self.individual, self.chrom, 7))
        # Another table with an extra person, where a fresh draw would hold out other people: with ``people`` it
        # holds out the same people and autosome.
        individual = np.concatenate([self.individual, ["S99"] * 23]).astype(object)
        chrom = np.concatenate([self.chrom, self.chrom[:23]]).astype(object)
        self.assertNotEqual(train._calibration_people(individual, chrom, 7), people)
        held, autosome = people
        always = np.isin(individual, train.ALWAYS_TRAIN_INDIVIDUALS)
        np.testing.assert_array_equal(train._calib_mask(individual, chrom, 7, people),
                                      np.isin(individual, held) | (always & (chrom == autosome)))


class OutOfFoldGateOnAnotherTableTest(unittest.TestCase):
    """_out_of_fold_gate_fires with ``scored``: heads fit on the direction table's halves judge the other table."""

    def test_each_scored_row_is_judged_by_the_head_fit_on_the_other_half(self):
        people = ["P%02d" % i for i in range(8)]
        fit_df = pd.DataFrame({"person": np.repeat(people, 5), "motif_size": 3.0, "eh": 10.0, "dir_code": 0})
        score_df = pd.DataFrame({"person": np.repeat(people, 3), "motif_size": 3.0, "eh": 10.0})
        fit_pool, score_pool = np.arange(len(fit_df)), np.arange(len(score_df))
        fit_groups, score_groups = fit_df["person"].to_numpy(object), score_df["person"].to_numpy(object)

        def fake_train_direction(X, *args, **kwargs):
            return set(X["person"])          # the head remembers whose rows it was fit on

        def fake_p_ok(dmodel, df, idx, branch):
            # pOk < 0.5 (the gate fires) only when the head never saw the scored row's person.
            return np.where(np.isin(df["person"].to_numpy(object)[idx], list(dmodel)), 1.0, 0.0)

        with mock.patch.object(train.features, "build_matrix", side_effect=lambda df, branch, names=None: (df, names)), \
             mock.patch.object(train.M, "train_direction", side_effect=fake_train_direction), \
             mock.patch.object(train, "_predict_p_ok_in_chunks", side_effect=fake_p_ok):
            fires = train._out_of_fold_gate_fires(fit_df, fit_pool, fit_groups, "quick", None, None, None,
                                                  scored=(score_df, score_pool, score_groups))
        self.assertEqual(fires.size, len(score_df))
        self.assertTrue(fires.all())


class TrainingPopulationOptionTest(unittest.TestCase):
    """--training-population: a non-default population goes into the default output name."""

    def test_non_default_choices_get_their_own_default_output_name(self):
        today = train._default_out()
        names = {train._default_out(population) for population in ("variant_loci", "all_loci")}
        self.assertEqual(len(names), 2)  # no two choices overwrite each other's model
        self.assertIn(today, names)
        date = os.path.basename(today).split(".")[1]
        for name in names:  # the date stays the last token before .json.gz, where compare_models reads it
            self.assertTrue(name.endswith(".%s.json.gz" % date), name)


class LengthHeadTableOptionTest(unittest.TestCase):
    """fit_genotyping_regime with length_head_parquet_subdir: the q-median head's rows come from the other table."""

    @staticmethod
    def _rows(table_marker, rows_per_person, extra_people=()):
        people = ["HG002", "CHM1_CHM13"] + ["S%02d" % i for i in range(8)] + list(extra_people)
        rows = []
        for person in people:
            for chrom in ("1", "2"):
                for _ in range(rows_per_person):
                    row = {c: 1.0 for c in features.FULL_FEATURES if c not in features.ENGINEERED_FEATURES}
                    row.update({"ci_start": 1.0, "ci_end": 2.0, "t": table_marker, "dir_code": features.OK,
                                "chrom": chrom, "genotyping_regime": "quick", "individual": person,
                                "representative": True})
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_q_median_rows_come_from_the_other_table_without_the_calibration_people(self):
        captured = {}

        def fake_fit_heads(df, train_pool, calib_rows, *args, **kwargs):
            captured.update(df=df, train_pool=train_pool, **kwargs)
            raise StopIteration

        with tempfile.TemporaryDirectory() as d:
            # The length table has extra people, so a fresh draw on it would hold out other people.
            for subdir, marker, n, extra in (("parquet_all_loci", 0.0, 3, ()), ("parquet", 9.0, 2, ("S98", "S99"))):
                os.makedirs(os.path.join(d, subdir))
                self._rows(marker, n, extra).to_parquet(os.path.join(d, subdir, "quick.parquet"))
            with mock.patch.object(train, "fit_heads", side_effect=fake_fit_heads), self.assertRaises(StopIteration):
                train.fit_genotyping_regime("quick", d, None, parquet_subdir="parquet_all_loci",
                                            length_head_parquet_subdir="parquet")
        length_df, length_pool, length_groups = captured["length_head_table"]
        self.assertEqual(set(length_df["t"]), {9.0})                  # the variant-locus table's rows
        direction_df, direction_pool = captured["df"], captured["train_pool"]
        others = set(direction_df["individual"]) - set(train.ALWAYS_TRAIN_INDIVIDUALS)
        held = others - set(direction_df["individual"].to_numpy()[direction_pool])
        self.assertEqual(len(held), train.N_CALIB_INDIVIDUALS)
        self.assertNotEqual(set(train._calibration_people(length_df["individual"].to_numpy(object),
                                                          length_df["chrom"].to_numpy(object), train.SEED)[0]), held)
        length_trained = set(length_df["individual"].to_numpy()[length_pool])
        self.assertFalse(held & length_trained)                       # the same people are held out ...
        self.assertTrue({"S98", "S99"} <= length_trained)             # ... and the extra people are trained on
        self.assertEqual(len(length_groups), len(length_df))

    def test_the_same_table_for_both_heads_passes_no_length_table(self):
        captured = {}

        def fake_fit_heads(df, *args, **kwargs):
            captured.update(kwargs)
            raise StopIteration

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "parquet"))
            self._rows(0.0, 1).to_parquet(os.path.join(d, "parquet", "quick.parquet"))
            with mock.patch.object(train, "fit_heads", side_effect=fake_fit_heads), self.assertRaises(StopIteration):
                train.fit_genotyping_regime("quick", d, None, parquet_subdir="parquet", length_head_parquet_subdir="parquet")
        self.assertIsNone(captured["length_head_table"])

    def test_a_different_length_head_population_gets_its_own_default_output_name(self):
        name = train._default_out("all_loci", "variant_loci")
        self.assertIn("_all_loci_length_head_variant_loci.", name)
        self.assertEqual(train._default_out("all_loci", "all_loci"), train._default_out("all_loci"))


class HomRefLocusDirectionTrainingRowsTest(unittest.TestCase):
    """--hom-ref-locus-direction-training-rows: the variant-locus recipe's direction rows plus hom-ref-locus rows."""

    @staticmethod
    def _rows(table_marker, variant_locus_pattern):
        """One table: per person and chromosome, one row per entry of ``variant_locus_pattern``."""
        people = ["HG002", "CHM1_CHM13"] + ["S%02d" % i for i in range(8)]
        rows = []
        for person in people:
            for chrom in ("1", "2"):
                for variant_locus in variant_locus_pattern:
                    row = {c: 1.0 for c in features.FULL_FEATURES if c not in features.ENGINEERED_FEATURES}
                    row.update({"ci_start": 1.0, "ci_end": 2.0, "t": table_marker, "dir_code": features.OK,
                                "chrom": chrom, "genotyping_regime": "quick", "individual": person,
                                "representative": True, "variant_locus": variant_locus})
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_the_frame_holds_the_variant_table_then_hom_ref_rows_and_calibration_stays_on_all_loci(self):
        captured = {}

        def fake_fit_heads(df, train_pool, calib_rows, fold_groups, *args, **kwargs):
            captured.update(df=df, train_pool=train_pool, calib_rows=calib_rows, fold_groups=fold_groups, **kwargs)
            raise StopIteration

        all_loci, variant_loci = self._rows(0.0, (True, False, False)), self._rows(9.0, (True,))
        with tempfile.TemporaryDirectory() as d:
            for subdir, table in (("parquet_all_loci", all_loci), ("parquet", variant_loci)):
                os.makedirs(os.path.join(d, subdir))
                table.to_parquet(os.path.join(d, subdir, "quick.parquet"))
            with mock.patch.object(train, "fit_heads", side_effect=fake_fit_heads), self.assertRaises(StopIteration):
                train.fit_genotyping_regime("quick", d, None, parquet_subdir="parquet_all_loci",
                                            length_head_parquet_subdir="parquet", hom_ref_locus_direction_training_rows=7)
        df, train_pool, calib_rows = captured["df"], captured["train_pool"], captured["calib_rows"]
        self.assertEqual(captured["hom_ref_locus_direction_training_rows"], 7)
        people = train._calibration_people(all_loci["individual"].to_numpy(object), all_loci["chrom"].to_numpy(object),
                                           train.SEED)
        held = people[0]
        # The variant table's pool keeps its own positions, so its quota sample is the variant-locus recipe's.
        variant_pool = np.where(~train._calib_mask(variant_loci["individual"].to_numpy(object),
                                                   variant_loci["chrom"].to_numpy(object), train.SEED, people))[0]
        np.testing.assert_array_equal(train_pool[:variant_pool.size], variant_pool)
        self.assertEqual(set(df["t"].iloc[:len(variant_loci)]), {9.0})
        hom_ref = df.iloc[train_pool[variant_pool.size:]]
        self.assertEqual(set(hom_ref["t"]), {0.0})
        self.assertFalse(hom_ref["variant_locus"].any())               # never the all-loci table's variant rows
        self.assertEqual(len(hom_ref), (8 - len(held)) * 2 * 2 + 2 * 2)  # both hom-ref rows of every trained chromosome
        self.assertFalse(set(df.iloc[train_pool]["individual"]) & set(held))
        calibration = df.iloc[calib_rows]
        self.assertEqual(set(calibration["t"]), {0.0})                  # calibration rows still come from all loci
        self.assertTrue(set(calibration["individual"]) <= set(held) | set(train.ALWAYS_TRAIN_INDIVIDUALS))
        self.assertEqual(len(calibration), 3 * (2 * len(held) + 2))      # their 3 rows per chromosome, all of them
        self.assertFalse(set(calib_rows) & set(train_pool))
        length_df, length_pool, length_groups = captured["length_head_table"]
        self.assertIs(length_df, df)                                    # q-median rows: the variant table's pool only
        np.testing.assert_array_equal(length_pool, variant_pool)
        self.assertIs(length_groups, captured["fold_groups"])

    def test_variant_locus_rows_keep_their_own_cap_and_hom_ref_rows_get_theirs_in_every_direction_head(self):
        people = ["P%02d" % i for i in range(8)]
        df = pd.DataFrame({"person": np.repeat(people, 4), "motif_size": 3.0, "eh": 10.0, "dir_code": features.OK,
                           "t": 0.0, "variant_locus": np.tile([True, False, False, False], len(people))})
        pool = np.arange(len(df))
        variant_locus_rows = pool[df["variant_locus"].to_numpy(bool)]
        np.testing.assert_array_equal(
            np.intersect1d(train.direction_training_rows(df, pool, 5, 3), variant_locus_rows),
            train.dataset.quota_sample(df["motif_size"].to_numpy(float), df["eh"].to_numpy(float), variant_locus_rows,
                                       5, train.SEED))
        direction_rows, q_median_rows = [], []

        def fake_train_direction(X, *args, **kwargs):
            direction_rows.append((len(X), int(X["variant_locus"].sum())))   # (rows, variant-locus rows)
            return None

        def fake_train_q_median(X, *args, **kwargs):
            q_median_rows.append(len(X))
            return None

        for hom_ref_locus_rows in (3, None):
            direction_rows.clear()
            q_median_rows.clear()
            with mock.patch.object(train.features, "build_matrix",
                                   side_effect=lambda df, branch, names=None: (df, names)), \
                 mock.patch.object(train.M, "train_direction", side_effect=fake_train_direction), \
                 mock.patch.object(train, "_predict_p_ok_in_chunks",
                                   side_effect=lambda dmodel, df, idx, branch: np.zeros(idx.size)), \
                 mock.patch.object(train.M, "train_q_median", side_effect=fake_train_q_median):
                train.fit_heads(df, pool, pool[:4], df["person"].to_numpy(object), "quick", 5, (5, 5), "test",
                                hom_ref_locus_direction_training_rows=hom_ref_locus_rows)
            if hom_ref_locus_rows is None:
                self.assertEqual([n for n, _ in direction_rows], [5, 5, 5])   # one quota sample of 5 rows each
            else:
                # Exported head: 5 of the 8 variant-locus rows + 3 hom-ref rows; each gating head (4 people): its 4
                # variant-locus rows + 3 hom-ref rows.
                self.assertEqual(direction_rows, [(8, 5), (7, 4), (7, 4)])
            self.assertEqual(q_median_rows, [5])                                # the cap still limits the q-median head

    def test_the_option_gets_its_own_default_output_name(self):
        name = train._default_out("all_loci", "variant_loci", 500000)
        self.assertIn("_length_head_variant_loci_variant_locus_direction_rows_plus_500000_hom_ref_locus_rows.", name)

    def test_the_option_is_refused_without_both_tables_or_with_no_rows(self):
        for argv in (["train.py", "--hom-ref-locus-direction-training-rows", "500000"],
                     ["train.py", "--training-population", "all_loci", "--hom-ref-locus-direction-training-rows", "5"],
                     ["train.py", "--training-population", "all_loci", "--length-head-training-population",
                      "variant_loci", "--hom-ref-locus-direction-training-rows", "0"]):
            with mock.patch("sys.argv", argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                train.main()


class QuotaByMotifSizeCategoryTest(unittest.TestCase):
    """--quota-by-motif-size-category: every head's quota sample uses the motif-size categories."""

    def test_every_head_samples_by_category_only_when_asked(self):
        people = ["P%02d" % i for i in range(8)]
        df = pd.DataFrame({"person": np.repeat(people, 4), "motif_size": 3.0, "eh": 10.0, "dir_code": features.OK,
                           "t": 0.0})
        pool = np.arange(len(df))
        real_quota_sample = train.dataset.quota_sample
        for by_category in (True, False):
            flags = []

            def recording_quota_sample(motif, eh, idx, cap, seed, by_motif_size_category=False):
                flags.append(by_motif_size_category)
                return real_quota_sample(motif, eh, idx, cap, seed, by_motif_size_category)

            with mock.patch.object(train.dataset, "quota_sample", side_effect=recording_quota_sample), \
                 mock.patch.object(train.features, "build_matrix",
                                   side_effect=lambda df, branch, names=None: (df, names)), \
                 mock.patch.object(train.M, "train_direction"), \
                 mock.patch.object(train, "_predict_p_ok_in_chunks",
                                   side_effect=lambda dmodel, df, idx, branch: np.zeros(idx.size)), \
                 mock.patch.object(train.M, "train_q_median"):
                train.fit_heads(df, pool, pool[:4], df["person"].to_numpy(object), "quick", 5, (5, 5), "test",
                                quota_by_motif_size_category=by_category)
            # The exported direction head, the two gating heads and the q-median head.
            self.assertEqual(flags, [by_category] * 4)

    def test_the_option_gets_its_own_default_output_name(self):
        self.assertIn("_quota_by_motif_size_category.", train._default_out(quota_by_motif_size_category=True))
        self.assertEqual(train._default_out(quota_by_motif_size_category=False), train._default_out())


class FullNonspanningTooLongTooShortBalancingTest(unittest.TestCase):
    """--full-nonspanning-balance-too-long-and-too-short-by-called-size: the full_nonspanning regime's direction heads
    and their TOO_LONG / TOO_SHORT calibrators give too-long and too-short rows equal weight per called-size bin."""

    def test_each_bin_gets_equal_too_long_and_too_short_weight(self):
        OK, LONG, SHORT = features.OK, features.TOO_LONG, features.TOO_SHORT
        called_bp = np.array([100, 100, 160, 160, 160, 170, 199, 260, 260, 900, 900, 900, 900, 1200, np.nan, np.nan])
        dir_code = np.array([LONG, SHORT, LONG, LONG, LONG, SHORT, OK, LONG, OK, LONG, LONG, LONG, SHORT, SHORT,
                             LONG, SHORT])
        weights = train.too_long_too_short_balancing_weights(called_bp, dir_code)
        # 150-200 bp: 3 too long, 1 too short -> each class carries 2; 250-300 bp has no too-short row; 450+ bp:
        # 3 too long, 2 too short -> each carries 2.5; below 150 bp, OK and no called size: 1.
        np.testing.assert_allclose(weights, [1, 1, 2 / 3, 2 / 3, 2 / 3, 2, 1, 1, 1, 2.5 / 3, 2.5 / 3, 2.5 / 3, 1.25,
                                             1.25, 1, 1])
        balanced = called_bp >= 450
        self.assertAlmostEqual(weights[balanced & (dir_code == LONG)].sum(), weights[balanced & (dir_code == SHORT)].sum())
        self.assertAlmostEqual(weights[balanced].sum(), balanced.sum())

    def _fit_heads_weights(self, balance):
        """Runs fit_heads on 8 people whose calls are 300 bp (3 too long, 1 too short each) and returns the training
        and calibration weights each direction head got: the exported head first, then the two gating heads."""
        people = ["P%02d" % i for i in range(8)]
        df = pd.DataFrame({"person": np.repeat(people, 4), "motif_size": 3.0, "eh": 100.0, "t": 0.0,
                           "dir_code": np.tile([features.TOO_LONG] * 3 + [features.TOO_SHORT], len(people))})
        pool = np.arange(len(df))
        weights_by_head = []

        def fake_train_direction(X, y, *args, **kwargs):
            weights_by_head.append((kwargs["sample_weight"], kwargs["too_long_too_short_calibration_weight"]))
            return None

        with mock.patch.object(train.features, "build_matrix", side_effect=lambda df, branch, names=None: (df, names)), \
             mock.patch.object(train.M, "train_direction", side_effect=fake_train_direction), \
             mock.patch.object(train, "_predict_p_ok_in_chunks",
                               side_effect=lambda dmodel, df, idx, branch: np.zeros(idx.size)), \
             mock.patch.object(train.M, "train_q_median", return_value=None), mock.patch("builtins.print"):
            train.fit_heads(df, pool, pool[:8], df["person"].to_numpy(object), "full", None, (5, 5), "test",
                            balance_too_long_and_too_short_by_called_size=balance)
        return weights_by_head

    def test_the_exported_and_the_gating_heads_and_their_calibrators_are_weighted(self):
        weights_by_head = self._fit_heads_weights(True)
        self.assertEqual(len(weights_by_head), 3)
        expected_calibration = np.tile([2 / 3] * 3 + [2.0], 2)
        for training_weights, calibration_weights in weights_by_head:
            np.testing.assert_allclose(training_weights, np.tile([2 / 3] * 3 + [2.0], training_weights.size // 4))
            np.testing.assert_allclose(calibration_weights, expected_calibration)
        self.assertEqual(self._fit_heads_weights(False), [(None, None)] * 3)

    def test_only_the_full_nonspanning_regime_is_balanced(self):
        balance_by_regime = {}

        def fake_fit_genotyping_regime(genotyping_regime, *args, **kwargs):
            balance_by_regime[genotyping_regime] = kwargs["balance_too_long_and_too_short_by_called_size"]
            return None, None, features.GENOTYPING_REGIME_BRANCH[genotyping_regime], ["motif_size"], None

        with tempfile.TemporaryDirectory() as d, \
             mock.patch("sys.argv", ["train.py", "--out", os.path.join(d, "model.json.gz"),
                                     "--full-nonspanning-balance-too-long-and-too-short-by-called-size"]), \
             mock.patch.object(train.dataset, "assert_parquets_up_to_date"), \
             mock.patch.object(train, "fit_genotyping_regime", side_effect=fake_fit_genotyping_regime), \
             mock.patch.object(train.M, "serialize_genotyping_regime",
                               return_value={"q_median": {"trees": []}, "direction": {"trees": []}}), \
             mock.patch.object(train.M, "verify_genotyping_regime"), mock.patch("builtins.print"):
            train.main()
        self.assertEqual(balance_by_regime, {features.GENOTYPING_REGIME_QUICK: False,
                                             features.GENOTYPING_REGIME_FULL_SPANNING: False,
                                             features.GENOTYPING_REGIME_FULL_NONSPANNING: True})

    def test_the_option_gets_its_own_default_output_name(self):
        self.assertIn("_full_nonspanning_too_long_too_short_balanced_by_called_size.",
                      train._default_out(full_nonspanning_balance_too_long_and_too_short_by_called_size=True))
        self.assertEqual(train._default_out(full_nonspanning_balance_too_long_and_too_short_by_called_size=False),
                         train._default_out())


class IterationsByRegimeTest(unittest.TestCase):
    def test_every_regime_has_positive_direction_and_q_median_sizes(self):
        self.assertEqual(set(train.ITERATIONS_BY_REGIME), set(features.GENOTYPING_REGIMES))
        for direction, q_median in train.ITERATIONS_BY_REGIME.values():
            self.assertGreater(direction, 0)
            self.assertGreater(q_median, 0)


if __name__ == "__main__":
    unittest.main()
