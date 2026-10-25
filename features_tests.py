"""Tests for features: tolerance tiers, direction codes, genotyping_regime routing, engineered cols, labels."""

import os
import re
import unittest

import numpy as np
import pandas as pd

import dataset
import eh_json
import features

# The C++ side's copy of the feature-name lists (gq::featureNamesForGenotypingRegime).
CPP_FEATURES_SOURCE = os.path.join(
    dataset.EXPANSIONHUNTER_BW2_REPO, "ehunter", "genotype_quality", "GenotypeQualityFeatures.cpp")


def _cpp_feature_names(source):
    """Parses ``featureNamesForGenotypingRegime``'s two name lists out of the C++ source text.

    Returns:
        ``(quick_names, full_names)`` in declaration order. ``full`` is built in the C++ as ``quick``
        plus ``push_back``ed names, so it is reassembled the same way here.
    """
    quick_block = re.search(r"std::vector<std::string>\s+quick\s*=\s*\{(.*?)\};", source, re.S)
    full_block = re.search(r"std::vector<std::string>\s+full\s*=\s*\[\]\s*\{(.*?)\}\(\);", source, re.S)
    assert quick_block and full_block, "could not locate the C++ feature-name lists"
    quick = re.findall(r'"([^"]*)"', quick_block.group(1))
    return quick, quick + re.findall(r'names\.push_back\("([^"]*)"\)', full_block.group(1))


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

    def test_quick_order(self):
        _, names = features.build_matrix(self._raw_row(), "quick")
        self.assertEqual(names, features.QUICK_FEATURES)

    def test_bad_branch(self):
        with self.assertRaises(ValueError):
            features.build_matrix(self._raw_row(), "fast")


class FeatureContractTest(unittest.TestCase):
    def test_full_is_quick_plus_flank_depths_and_inrepeat_total(self):
        self.assertEqual(features.FULL_FEATURES,
                         features.QUICK_FEATURES
                         + ["left_flank_norm_depth", "right_flank_norm_depth", "inrepeat_total"])

    def test_names_match_the_cpp_assembler(self):
        # The C++ consumer keeps its own copy of this list in GenotypeQualityFeatures.cpp
        # (featureNamesForGenotypingRegime) and nothing reads features.py from there, so a rename or
        # reorder on one side would otherwise go unnoticed until a retrained model was deployed and
        # GenotypeQualityAnnotator.cpp threw "requires feature 'x' that ExpansionHunter does not
        # produce". Spelling the names out here forces any edit to features.py to be a deliberate,
        # mirrored one. Counts are covered too, since an added/removed name changes the list.
        self.assertEqual(features.QUICK_FEATURES, [
            "motif_size",
            "num_repeats_in_reference", "ref_size_bp",
            "eh", "eh_minus_ref", "allele_rank",
            "n_alleles", "n_distinct_alleles",
            "ci_width", "ci_asymmetry", "ci_over_eh",
            "spanning_total", "hq_unamb_total", "flanking_total",
            "spanning_at_called", "spanning_above_called", "flanking_above_called",
            "support_frac", "flanking_frac",
            "coverage",
            "depth", "hq_unambiguous_reads", "strand_bias_phred",
            "mean_inserted_bases", "mean_deleted_bases",
            "reference_repeat_purity", "read_repeat_purity",
        ], "feature list changed -- mirror it in GenotypeQualityFeatures.cpp's "
           "featureNamesForGenotypingRegime and GenotypeQualityFeaturesTest.cpp, then update this list")

    @unittest.skipUnless(os.path.exists(CPP_FEATURES_SOURCE),
                         "no local ExpansionHunter-bw2 checkout at %s" % CPP_FEATURES_SOURCE)
    def test_names_match_the_cpp_source_on_disk(self):
        # The literal above only proves features.py and this file agree; both can be edited together
        # while GenotypeQualityFeatures.cpp is left behind. This reads the C++ list itself, so the
        # two repos are checked against each other whenever the checkout is present. (In CI, where
        # it is not, the test skips and the literal above is the remaining guard.)
        with open(CPP_FEATURES_SOURCE) as f:
            cpp_quick, cpp_full = _cpp_feature_names(f.read())
        message = ("features.py and %s disagree -- the model's feature_names are resolved by name at "
                   "load time, so a model trained under this contract would be rejected by that "
                   "binary with 'requires feature ... that ExpansionHunter does not produce'"
                   % CPP_FEATURES_SOURCE)
        self.assertEqual(features.QUICK_FEATURES, cpp_quick, message)
        self.assertEqual(features.FULL_FEATURES, cpp_full, message)

    def test_every_feature_is_emitted_by_the_extractor(self):
        # build_matrix asserts on the columns at fit/apply time; catching it here instead means a
        # feature added to the list without a matching eh_json column fails fast, not after a rebuild.
        row = next(iter(eh_json.extract_rows(
            {"LocusResults": {"x": {
                "LocusId": "1-1000-1030-CAG", "Coverage": 30.0,
                "Variants": {"L": {
                    "RepeatUnit": "CAG", "ReferenceRegion": "chr1:1000-1030", "Genotype": "20/97",
                    "GenotypeConfidenceInterval": "20-20/90-105",
                    "CountsOfSpanningReads": "(20, 8)", "CountsOfFlankingReads": "(98, 2)",
                    "CountsOfHighQualityUnambiguousReads": "(20, 6)"}}}}}, sample_id="S")))
        engineered = ("ci_asymmetry", "ci_over_eh")  # add_engineered rebuilds these from ci_start/ci_end
        self.assertEqual([f for f in features.FULL_FEATURES
                          if f not in engineered and f not in row], [])

    def test_names_are_unique(self):
        self.assertEqual(len(set(features.FULL_FEATURES)), len(features.FULL_FEATURES))

    def test_every_feature_is_documented(self):
        # The report's glossary renders FEATURE_DEFINITIONS; a feature added without one shows blank.
        self.assertEqual([f for f in features.FULL_FEATURES if f not in features.FEATURE_DEFINITIONS], [])


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


class RelativeToleranceDirectionCodesTest(unittest.TestCase):
    """Change B's label: relative tolerance below the fragment length, the one-sided rule at or above it."""

    @staticmethod
    def _codes(eh, true, motif_size, fragment_length, x=0.10):
        n = len(eh)
        return features.relative_tolerance_direction_codes(
            eh, np.broadcast_to(true, n), np.broadcast_to(motif_size, n), np.broadcast_to(fragment_length, n),
            x).tolist()

    def test_within_the_limit_the_tolerance_is_a_fraction_of_the_true_size(self):
        # true 20 repeats (60 bp < 500 bp): tolerance max(1, 2) = 2 repeats.
        self.assertEqual(self._codes([22, 23, 18, 17], 20, 3, 500),
                         [features.OK, features.TOO_LONG, features.OK, features.TOO_SHORT])
        # x = 15%: tolerance 3 repeats.
        self.assertEqual(self._codes([23, 24], 20, 3, 500, x=0.15), [features.OK, features.TOO_LONG])

    def test_short_alleles_keep_a_one_repeat_tolerance(self):
        # true 5 repeats: 10% is 0.5, so the 1-repeat floor applies.
        self.assertEqual(self._codes([6, 7, 4, 3], 5, 3, 500),
                         [features.OK, features.TOO_LONG, features.OK, features.TOO_SHORT])

    def test_past_the_limit_only_a_clear_over_call_or_a_call_short_of_the_limit_is_wrong(self):
        # true 200 repeats x 5 bp = 1000 bp >= 500 bp: TOO_LONG above 220 repeats, TOO_SHORT below 450 bp (90 repeats).
        self.assertEqual(self._codes([221, 220, 90, 89], 200, 5, 500),
                         [features.TOO_LONG, features.PAST_LIMIT, features.PAST_LIMIT, features.TOO_SHORT])

    def test_a_true_allele_exactly_at_the_fragment_length_is_past_the_limit(self):
        # 100 repeats x 5 bp = 500 bp: a 95-repeat call is PAST_LIMIT, not OK as the within-limit rule would say.
        self.assertEqual(self._codes([95], 100, 5, 500), [features.PAST_LIMIT])
        self.assertEqual(self._codes([95], 100, 5, 501), [features.OK])

    def test_without_a_fragment_length_there_is_no_limit(self):
        for fragment_length in (np.nan, 0.0):
            self.assertEqual(self._codes([100], 200, 5, fragment_length), [features.TOO_SHORT])
        self.assertEqual(self._codes([100], 200, 5, 500), [features.PAST_LIMIT])


class GenotypeLevelDirectionCodesTest(unittest.TestCase):
    """``genotype_level_direction_codes`` with change B's label (FL 500 bp, 5 bp motif, x = 10%)."""

    @staticmethod
    def _codes(eh, true, true_other_allele):
        eh = np.asarray(eh, dtype=float)

        def codes_against(truth):
            return features.relative_tolerance_direction_codes(eh, truth, np.full(eh.size, 5.0),
                                                               np.full(eh.size, 500.0), 0.10)
        return features.genotype_level_direction_codes(eh, true, true_other_allele, codes_against).tolist()

    def test_a_missed_long_allele_makes_a_matching_call_too_short(self):
        # truths 100 and 1000 bp, call 100 bp: OK against the short allele, TOO_SHORT against the long one.
        self.assertEqual(self._codes([20], [20], [200]), [features.TOO_SHORT])

    def test_when_both_are_missed_the_bigger_miss_wins(self):
        # call 300 bp: TOO_LONG by 40 repeats against 100 bp, TOO_SHORT by 140 repeats against 1000 bp.
        self.assertEqual(self._codes([60], [20], [200]), [features.TOO_SHORT])
        # call 40 repeats: TOO_LONG by 20 against 20 repeats, TOO_SHORT by 10 against 50 repeats.
        self.assertEqual(self._codes([40], [20], [50]), [features.TOO_LONG])

    def test_ok_against_one_allele_and_past_the_limit_against_the_other_is_past_limit(self):
        # truths 480 and 900 bp, call 470 bp.
        self.assertEqual(self._codes([94], [96], [180]), [features.PAST_LIMIT])

    def test_a_row_without_a_second_truth_allele_keeps_its_own_label(self):
        self.assertEqual(self._codes([60, 20], [20, 20], [np.nan, np.nan]), [features.TOO_LONG, features.OK])


class FragmentLengthRatioFeatureTest(unittest.TestCase):
    def test_called_size_in_bp_over_the_fragment_length_and_empty_without_one(self):
        df = pd.DataFrame({"eh": [10.4, 10.0, 10.0], "motif_size": [5, 5, 5], "fragment_length": [500.0, 0.0, np.nan],
                           "ci_start": [9, 9, 9], "ci_end": [11, 11, 11], "ci_width": [2, 2, 2]})
        ratio = features.add_engineered(df)[features.FRAGMENT_LENGTH_RATIO_FEATURE].tolist()
        self.assertAlmostEqual(ratio[0], 0.1)   # round(10.4) * 5 / 500
        self.assertTrue(np.isnan(ratio[1]) and np.isnan(ratio[2]))

    def test_only_the_relative_tolerance_feature_lists_have_it(self):
        self.assertNotIn(features.FRAGMENT_LENGTH_RATIO_FEATURE, features.add_engineered(
            pd.DataFrame({"eh": [10.0], "ci_start": [9], "ci_end": [11], "ci_width": [2]})).columns)
        for branch in (features.BRANCH_QUICK, features.BRANCH_FULL):
            self.assertEqual(features.feature_names(branch, with_fragment_length_ratio=True),
                             features.feature_names(branch) + [features.FRAGMENT_LENGTH_RATIO_FEATURE])

    def test_a_parquet_must_supply_the_fragment_length_the_ratio_is_computed_from(self):
        import tempfile
        names = features.feature_names(features.BRANCH_QUICK, with_fragment_length_ratio=True)
        stored = {c: [1.0] for c in features.QUICK_FEATURES if c not in features.ENGINEERED_FEATURES}
        with tempfile.TemporaryDirectory() as d:
            without, with_ = os.path.join(d, "without.parquet"), os.path.join(d, "with.parquet")
            pd.DataFrame(stored).to_parquet(without)
            pd.DataFrame(dict(stored, fragment_length=[500.0])).to_parquet(with_)
            self.assertEqual(features.missing_feature_columns([without, with_], names), {without: ["fragment_length"]})


if __name__ == "__main__":
    unittest.main()
