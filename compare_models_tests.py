"""Tests for compare_models' per-sample paired comparison helpers."""

import unittest

import numpy as np

import compare_models


PREV = {"S1": {"n": 10, "sum_gated_abs_error": 20.0}, "S2": {"n": 5, "sum_gated_abs_error": 5.0},
        "S3": {"n": 0, "sum_gated_abs_error": 0.0}}
NEW = {"S1": {"n": 10, "sum_gated_abs_error": 10.0}, "S2": {"n": 5, "sum_gated_abs_error": 10.0},
       "S3": {"n": 0, "sum_gated_abs_error": 0.0}}


class PairedComparisonTest(unittest.TestCase):
    def test_per_sample_maes_pair_the_same_samples_and_skip_empty_samples(self):
        prev, new = compare_models._per_sample_gated_maes(PREV, NEW, "n", "sum_gated_abs_error",
                                                          ["S1", "S2", "S3", "S4"])
        np.testing.assert_allclose(prev, [2.0, 1.0])
        np.testing.assert_allclose(new, [1.0, 2.0])

    def test_bootstrap_interval_is_deterministic_and_brackets_the_mean(self):
        values = np.array([-0.3, -0.1, -0.2, 0.05, -0.15])
        first = compare_models._bootstrap_mean_interval(values)
        np.testing.assert_array_equal(first, compare_models._bootstrap_mean_interval(values))
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())


class PairedGatedMaeComparisonTest(unittest.TestCase):
    """paired_gated_mae_comparison and verdict_of on run_eval-shaped results."""

    @staticmethod
    def _result(errors_by_regime, n_samples=12):
        samples = ["S%02d" % i for i in range(n_samples)]
        regimes = {}
        for regime, error in errors_by_regime.items():
            per_sample = {s: {"n": 10, "sum_gated_abs_error": 10 * error(i), "homopolymer_n": 0,
                              "homopolymer_sum_gated_abs_error": 0.0} for i, s in enumerate(samples)}
            regimes[regime] = {"n": 10 * n_samples, "mae_raw": 2.0, "per_sample": per_sample, "homopolymer": {"n": 0}}
        return {"genotyping_regimes": regimes}, samples

    def test_results_that_scored_different_alleles_are_not_paired(self):
        prev, samples = self._result({r: (lambda i: 1.0) for r in ("quick", "full_spanning", "full_nonspanning")})
        new, _ = self._result({r: (lambda i: 0.5) for r in ("quick", "full_spanning", "full_nonspanning")})
        new["genotyping_regimes"]["full_spanning"]["n"] += 1
        with self.assertRaisesRegex(ValueError, "full_spanning, non-homopolymer"):
            compare_models.paired_gated_mae_comparison(prev, new, samples)

    def test_a_stratum_is_won_only_when_the_interval_excludes_zero_and_needs_enough_samples(self):
        prev, samples = self._result({"quick": lambda i: 1.0, "full_spanning": lambda i: 1.0,
                                      "full_nonspanning": lambda i: 1.0})
        new, _ = self._result({"quick": lambda i: 0.5, "full_spanning": lambda i: 1.5,
                               "full_nonspanning": lambda i: 1.0 + (0.1 if i % 2 else -0.1)})
        rows = compare_models.paired_gated_mae_comparison(prev, new, samples)
        self.assertEqual([(r["regime"], r["stratum"], r["winner"]) for r in rows],
                         [("quick", "non-homopolymer", "new"), ("full_spanning", "non-homopolymer", "prev"),
                          ("full_nonspanning", "non-homopolymer", "tie")])
        self.assertEqual(compare_models.verdict_of(rows),
                         ("MIXED: each model is better in some strata", {"new": 1, "prev": 1, "tie": 1,
                                                                         "too few samples": 0}))
        self.assertEqual([r["new_mean_over_samples"] - r["prev_mean_over_samples"] for r in rows], [-0.5, 0.5, 0.0])
        # Without the diagnosis samples (none of S00-S11), the same per-sample statistic and its interval.
        self.assertEqual([r["new_without_diagnosis"] - r["prev_without_diagnosis"] for r in rows], [-0.5, 0.5, 0.0])
        self.assertEqual(rows[0]["interval_without_diagnosis"], (-0.5, -0.5))
        # A one-sided verdict says how many of the judged strata it rests on.
        self.assertEqual(compare_models.verdict_of(rows[:1] + rows[2:])[0],
                         "NEW is better in 1 of 2 judged strata, worse in none")
        # No categories in these results (scored before run_eval reported them): no category rows.
        self.assertIsNone(compare_models.paired_gated_mae_by_motif_size_category(prev, new, samples))
        few_prev, few = self._result({r: (lambda i: 1.0) for r in ("quick", "full_spanning", "full_nonspanning")}, 5)
        few_new, _ = self._result({r: (lambda i: 0.5) for r in ("quick", "full_spanning", "full_nonspanning")}, 5)
        self.assertEqual({r["winner"] for r in compare_models.paired_gated_mae_comparison(few_prev, few_new, few)},
                         {"too few samples"})


class MotifSizeCategoryComparisonTest(unittest.TestCase):
    @staticmethod
    def _result(error_by_category, n_samples=12, raw=2.0):
        samples = ["S%02d" % i for i in range(n_samples)]
        regime = {"by_motif_size_category": {
            category: {"n": 10 * n_samples, "mae_raw": raw, "mae_gated": error,
                       "per_sample": {s: {"n": 10, "sum_gated_abs_error": 10 * error} for s in samples}}
            for category, error in error_by_category.items()}}
        return {"genotyping_regimes": {r: regime for r in ("quick", "full_spanning", "full_nonspanning")}}, samples

    def test_each_category_is_paired_by_sample_and_categories_missing_from_prev_are_skipped(self):
        prev, samples = self._result({"1": 1.0, "3": 2.0})
        new, _ = self._result({"1": 1.5, "3": 1.0, "25+": 9.0})
        rows = compare_models.paired_gated_mae_by_motif_size_category(prev, new, samples)
        self.assertEqual([(r["regime"], r["category"], r["winner"]) for r in rows][:2],
                         [("quick", "1", "prev"), ("quick", "3", "new")])
        self.assertEqual(len(rows), 2 * 3)
        self.assertAlmostEqual(rows[1]["new_mean_over_samples"] - rows[1]["prev_mean_over_samples"], -1.0)

    def test_categories_that_scored_different_alleles_are_not_paired(self):
        prev, samples = self._result({"3": 2.0})
        new, _ = self._result({"3": 1.0}, raw=2.5)
        with self.assertRaisesRegex(ValueError, "quick, motif size 3"):
            compare_models.paired_gated_mae_by_motif_size_category(prev, new, samples)


if __name__ == "__main__":
    unittest.main()
