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
