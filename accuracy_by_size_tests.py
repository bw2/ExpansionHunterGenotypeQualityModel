"""Tests for accuracy_by_size.py's prediction-sharing rule.

``categorize_parquet`` itself needs a real parquet plus a real exported model, so it is exercised
end to end by the report pipeline rather than here. What IS unit-tested is the rule that decides
which alleles get a model prediction: ExpansionHunter makes one prediction per allele it scores
(one for a whole homozygous call), while this report keeps both genotype copies for its truth join,
so a copy without its own quality metrics must inherit its twin's prediction rather than getting an
independently-scored one the deployed binary would never emit.
"""

import unittest

import numpy as np
import pandas as pd

import accuracy_by_size as A


class CorrectedCategoryTest(unittest.TestCase):
    def test_correcting_one_allele_changes_its_ungated_partners_category(self):
        # Reference 7 repeats, truth 9/9, raw call 5/7. Correcting only the 5 to 7 makes the locus
        # homozygous-reference, so the UNGATED 7 must move from Het Ref to Hom Ref with it; this is why
        # categorize_parquet takes the recomputed category for every applied row, not just gated ones.
        raw, ref, true, drr_truth = np.array([5.0, 7.0]), np.full(2, 7.0), np.full(2, 9.0), np.full(2, 2.0)
        out, _ = A._corrected_category(raw, np.array([7.0, 7.0]), np.array([True, False]), ref, true,
                                       drr_truth, np.array(["L", "L"]))
        self.assertEqual(list(out), ["Called Hom Ref", "Called Hom Ref"])
        raw_cat = A.classify(raw - true, raw, drr_truth, raw - ref, raw == ref, np.array([False, False]))
        self.assertEqual(raw_cat[1], "Called Het Ref")


class TruthOrderSourceTest(unittest.TestCase):
    def test_reversed_corrected_calls_are_repaired_by_size(self):
        # Raw 10/18, truth 7/10; correcting the 18 to 6 gives 10/6, which matches the truth unphased.
        locus = np.array(["L", "L", "M", "M"], dtype=object)
        calls = np.array([10.0, 6.0, 5.0, 9.0])
        true = np.array([7.0, 10.0, 5.0, 9.0])
        src = A._truth_order_source(calls, true, locus)
        np.testing.assert_array_equal(calls[src], [6.0, 10.0, 5.0, 9.0])
        out, src = A._corrected_category(np.array([10.0, 18.0]), np.array([10.0, 6.0]), np.array([False, True]),
                                         np.full(2, 10.0), np.array([7.0, 10.0]), np.array([-3.0, 0.0]),
                                         np.array(["L", "L"], dtype=object))
        self.assertEqual(list(out), ["Same", "Same"])
        pok = np.array([0.9, 0.1])  # each call's own pOk travels with it: the 6 (pOk 0.1) is now row 0
        np.testing.assert_array_equal(pok[src], [0.1, 0.9])

    def test_pok_strata_of_a_corrected_variant_use_its_own_pok_column(self):
        cat = pd.DataFrame({"motif": [3, 3], "purity": [1.0, 1.0], "xbin": [0, 0], "locus": ["L", "L"],
                            "category": ["Same", "Same"], "pok": [0.9, 0.1],
                            "category__p050": ["Same", "2"], "pok__p050": [0.1, 0.9]})
        lt = A.bin_counts(cat, "category__p050", False, pok_stratum=("lt", 0.5))
        self.assertEqual(lt["total"], 1)
        self.assertEqual(sum(lt["counts"]["Same"]), 1)  # the row whose re-paired call has pOk 0.1

    def test_pok_strata_use_their_own_threshold(self):
        # One pOk between each pair of thresholds, plus one exactly at 0.3 to check the strict "<".
        cat = pd.DataFrame({"motif": [3] * 6, "purity": [1.0] * 6, "xbin": [0] * 6, "locus": list("ABCDEF"),
                            "category": ["Same"] * 6, "pok": [0.15, 0.25, 0.3, 0.35, 0.45, 0.6]})
        totals = {key: A.bin_counts(cat, "category", False, pok_stratum=mode)["total"]
                  for key, _, mode in A.POK_VARIANTS}
        self.assertEqual(totals, {"all": 6, "lt020": 1, "lt030": 2, "lt040": 4, "lt050": 5, "ge050": 1})


class LociCohortTest(unittest.TestCase):
    def test_whole_loci_are_kept_together_within_the_cap(self):
        locus = np.array(["A", "A", "B", "B", "C", "C", "D"], dtype=object)
        scoreable = np.array([True, True, True, False, True, True, False])  # D is a no-call locus
        cohort = A._loci_cohort(locus, scoreable, 3)
        for loc in set(locus):  # a locus is in or out as a whole
            self.assertEqual(len(set(cohort[locus == loc])), 1)
        self.assertLessEqual(int((cohort & scoreable).sum()), 3)
        np.testing.assert_array_equal(cohort, A._loci_cohort(locus, scoreable, 3))

    def test_no_cap_keeps_everything(self):
        locus = np.array(["A", "B"], dtype=object)
        self.assertTrue(A._loci_cohort(locus, np.array([True, True]), None).all())
        self.assertTrue(A._loci_cohort(locus, np.array([True, True]), 5).all())


def _frame(locus_ids, ehs, own_metrics=None):
    data = {"locus_id": locus_ids, "eh": ehs}
    if own_metrics is not None:
        data["has_own_quality_metrics"] = own_metrics
    return pd.DataFrame(data)


class SharePredictionsWithinLocusTest(unittest.TestCase):
    def test_duplicate_copy_inherits_its_twins_prediction(self):
        df = _frame(["L1", "L1", "L2"], [20.0, 20.0, 15.0], [True, False, True])
        lcf, pok, ns = A._share_predictions_within_locus(
            df,
            scoreable=np.array([True, False, True]),
            called=np.array([True, True, True]),
            lcf=np.array([1.25, np.nan, 0.9]),
            pok=np.array([0.3, np.nan, 0.8]),
            non_spanning=np.array([True, False, False]))
        np.testing.assert_allclose(lcf, [1.25, 1.25, 0.9])
        np.testing.assert_allclose(pok, [0.3, 0.3, 0.8])
        np.testing.assert_array_equal(ns, [True, True, False])

    def test_a_different_call_at_the_same_locus_does_not_inherit(self):
        # Only genotype copies of the SAME call share a prediction; a het locus has two real alleles,
        # each of which ExpansionHunter scores on its own.
        df = _frame(["L1", "L1"], [20.0, 97.0], [True, True])
        lcf, pok, _ = A._share_predictions_within_locus(
            df,
            scoreable=np.array([True, True]),
            called=np.array([True, True]),
            lcf=np.array([1.25, np.nan]),
            pok=np.array([0.3, np.nan]),
            non_spanning=np.zeros(2, dtype=bool))
        self.assertTrue(np.isnan(pok[1]))
        self.assertTrue(np.isnan(lcf[1]))

    def test_parquet_without_the_column_is_unchanged(self):
        # Predating has_own_quality_metrics, every row is treated as scoreable, so nothing is shared.
        df = _frame(["L1", "L1"], [20.0, 20.0])
        lcf_in, pok_in = np.array([1.25, np.nan]), np.array([0.3, np.nan])
        lcf, pok, _ = A._share_predictions_within_locus(
            df,
            scoreable=np.ones(2, dtype=bool),
            called=np.array([True, True]),
            lcf=lcf_in,
            pok=pok_in,
            non_spanning=np.zeros(2, dtype=bool))
        self.assertTrue(np.isnan(pok[1]))

    def test_uncalled_row_never_inherits(self):
        # A no-call allele has no eh to correct; it must stay out of the corrected panels.
        df = _frame(["L1", "L1"], [20.0, np.nan], [True, False])
        _, pok, _ = A._share_predictions_within_locus(
            df,
            scoreable=np.array([True, False]),
            called=np.array([True, False]),
            lcf=np.array([1.25, np.nan]),
            pok=np.array([0.3, np.nan]),
            non_spanning=np.zeros(2, dtype=bool))
        self.assertTrue(np.isnan(pok[1]))

    def test_row_the_cap_left_unscored_is_not_filled_in(self):
        # corrected_cap can leave a scoreable row unscored; it must stay NaN (dropped from the
        # corrected panels) rather than borrowing a neighbour's prediction.
        df = _frame(["L1", "L1"], [20.0, 20.0], [True, True])
        _, pok, _ = A._share_predictions_within_locus(
            df,
            scoreable=np.array([True, True]),
            called=np.array([True, True]),
            lcf=np.array([1.25, np.nan]),
            pok=np.array([0.3, np.nan]),
            non_spanning=np.zeros(2, dtype=bool))
        self.assertTrue(np.isnan(pok[1]))


if __name__ == "__main__":
    unittest.main()
