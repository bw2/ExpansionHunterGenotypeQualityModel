"""Unit tests for ``evaluate`` on synthetic arrays / DataFrames.

Synthetic-only (no fixtures, no model imports). Where randomness appears it is
drawn with ``numpy.random.default_rng(SEED)`` so the suite is deterministic.
Cases are constructed so the "model" is clearly better than chance / the
raw-EH baseline, and so quantities with a known closed form (exact-match rate,
interval coverage) can be checked exactly.

Run with:  python3 -m unittest evaluate_tests -v
"""

import unittest

import numpy as np
import pandas as pd

import evaluate as E


SEED = 20260616


def _make_direction_data(n=3000, seed=SEED):
    """Builds ``(y_true, proba)`` for an accurate, roughly calibrated classifier.

    Each row's truth class gets the bulk of the mass (0.8) plus a little
    Dirichlet noise, so argmax is almost always correct and the TOO_LONG/TOO_SHORT
    one-vs-rest scores separate cleanly.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    proba = np.full((n, 3), 0.1)
    proba[np.arange(n), y] = 0.8
    proba += rng.dirichlet([1.0, 1.0, 1.0], size=n) * 0.1
    return y, proba / proba.sum(axis=1, keepdims=True)


def _make_baseline_df(n=2000, seed=SEED, include_eh_q=True):
    """Builds a df where ``ci_width`` is wider and ``eh_q`` lower for miscalls."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    is_miscall = y != 0
    data = {
        "dir_code": y,
        "ci_width": rng.normal(loc=np.where(is_miscall, 8.0, 2.0), scale=1.0),
    }
    if include_eh_q:
        data["eh_q"] = rng.normal(loc=np.where(is_miscall, 5.0, 30.0), scale=3.0)
    return pd.DataFrame(data)


def _make_q_df(n=1500, seed=SEED):
    """Builds a stratifiable df of q-eval inputs with a good median predictor."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(10.0, 120.0, size=n)
    t_true = rng.normal(0.0, 0.2, size=n)
    t_pred = t_true + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame({
        "t_true": t_true,
        "q50": t_pred,
        "q10": t_pred - 0.3,
        "q90": t_pred + 0.3,
        "q05": t_pred - 0.5,
        "q95": t_pred + 0.5,
        "eh": np.round(true * np.exp(t_true)),
        "true": true,
        "source": rng.choice(["real", "sim"], size=n),
        "motif_size": rng.integers(1, 8, size=n),
        "coverage": rng.choice([10.0, 20.0, 31.0], size=n),
        "sample": rng.choice(["HG002", "CHM"], size=n),
    })


def _make_dir_df(n=2400, seed=SEED):
    """Builds a stratifiable df of direction proba + labels + baseline signals."""
    y, proba = _make_direction_data(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame({
        "dir_code": y,
        "p_ok": proba[:, 0],
        "p_too_long": proba[:, 1],
        "p_too_short": proba[:, 2],
        "ci_width": rng.normal(loc=np.where(y != 0, 8.0, 2.0), scale=1.0),
        "source": rng.choice(["real", "sim"], size=n),
        "motif_size": rng.integers(1, 8, size=n),
        "coverage": rng.choice([10.0, 20.0, 31.0], size=n),
        "sample": rng.choice(["HG002", "CHM"], size=n),
    })


class EvaluateQTest(unittest.TestCase):
    def test_exact_match_beats_eh_baseline(self):
        # round(true) = [10, 22, 31]. Raw EH = [10, 20, 30] matches only the
        # first -> 1/3. The model recovers true_pred = [10, 22, 31] exactly -> 1.
        eh = np.array([10.0, 20.0, 30.0])
        true = np.array([10.2, 22.0, 31.0])
        t_pred_median = np.log(eh) - np.log(np.array([10.0, 22.0, 31.0]))
        t_true = np.log(eh) - np.log(true)
        metrics = E.evaluate_q(t_true, t_pred_median, eh, true,
                               {a: t_true for a in E.QUANTILES})
        self.assertAlmostEqual(metrics["exact_match_rate"], 1.0)
        self.assertAlmostEqual(metrics["eh_exact_match_rate"], 1.0 / 3.0)
        self.assertGreater(metrics["exact_match_rate"], metrics["eh_exact_match_rate"])

    def test_perfect_median_gives_zero_point_error(self):
        eh = np.array([10.0, 50.0, 80.0])
        true = np.array([12.0, 47.0, 90.0])
        t_true = np.log(eh) - np.log(true)
        metrics = E.evaluate_q(t_true, t_true, eh, true,
                               {a: t_true for a in E.QUANTILES})
        self.assertAlmostEqual(metrics["mae_t"], 0.0)
        self.assertAlmostEqual(metrics["rmse_t"], 0.0)
        # true_pred = eh / exp(t_true) == true exactly.
        self.assertAlmostEqual(metrics["mae_true"], 0.0, places=9)
        self.assertAlmostEqual(metrics["rmse_true"], 0.0, places=9)

    def test_coverage_all_inside_is_one(self):
        # t_true == 0 everywhere, inside [-1, 1] and [-2, 2] for every row.
        t_true = np.zeros(5)
        quantile_preds = {
            0.05: -2.0 * np.ones(5), 0.1: -1.0 * np.ones(5), 0.5: np.zeros(5),
            0.9: np.ones(5), 0.95: 2.0 * np.ones(5),
        }
        metrics = E.evaluate_q(t_true, np.zeros(5), 10.0 * np.ones(5),
                               10.0 * np.ones(5), quantile_preds)
        self.assertEqual(metrics["coverage_80"], 1.0)
        self.assertEqual(metrics["coverage_90"], 1.0)

    def test_coverage_partial_is_exact_fraction(self):
        # Row 0 (t=0) inside [-1, 1]; row 1 (t=5) outside -> 0.5 coverage.
        t_true = np.array([0.0, 5.0])
        quantile_preds = {
            0.05: np.array([-2.0, -2.0]), 0.1: np.array([-1.0, -1.0]),
            0.5: np.array([0.0, 0.0]), 0.9: np.array([1.0, 1.0]),
            0.95: np.array([2.0, 2.0]),
        }
        metrics = E.evaluate_q(t_true, np.zeros(2), np.array([10.0, 10.0]),
                               np.array([10.0, 10.0]), quantile_preds)
        self.assertEqual(metrics["coverage_80"], 0.5)
        self.assertEqual(metrics["coverage_90"], 0.5)

    def test_empty_input_is_nan(self):
        metrics = E.evaluate_q([], [], [], [], {a: [] for a in E.QUANTILES})
        self.assertEqual(metrics["n"], 0)
        for key in ("mae_t", "rmse_t", "mae_true", "coverage_80", "exact_match_rate"):
            self.assertTrue(np.isnan(metrics[key]))


class EvaluateDirectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.proba = _make_direction_data()
        cls.metrics = E.evaluate_direction(cls.y, cls.proba, "full")

    def test_aucs_clearly_above_chance(self):
        self.assertGreater(self.metrics["too_long_auc"], 0.7)
        self.assertGreater(self.metrics["too_short_auc"], 0.7)
        self.assertGreater(self.metrics["too_long_ap"], 0.5)
        self.assertGreater(self.metrics["too_short_ap"], 0.5)

    def test_log_loss_beats_uniform(self):
        uniform = E.evaluate_direction(
            self.y, np.full_like(self.proba, 1.0 / 3.0), "full")
        self.assertLess(self.metrics["log_loss"], uniform["log_loss"])
        self.assertAlmostEqual(uniform["log_loss"], np.log(3.0), places=6)

    def test_confusion_is_3x3_and_diagonal_dominant(self):
        confusion = np.array(self.metrics["confusion"])
        self.assertEqual(confusion.shape, (3, 3))
        self.assertEqual(int(confusion.sum()), self.y.size)
        self.assertGreater(confusion.trace(), 0.6 * confusion.sum())

    def test_reliability_and_ece_in_range(self):
        for cls_key in ("too_long", "too_short"):
            reliability = self.metrics["reliability"][cls_key]
            self.assertEqual(len(reliability["bin_centers"]), 10)
            self.assertEqual(len(reliability["bin_freq"]), 10)
            self.assertGreaterEqual(reliability["ece"], 0.0)
            self.assertLessEqual(reliability["ece"], 0.5)
            self.assertEqual(self.metrics["ece"][cls_key], reliability["ece"])

    def test_renormalizes_unnormalized_proba(self):
        scaled = E.evaluate_direction(self.y, self.proba * 7.0, "full")
        self.assertAlmostEqual(scaled["log_loss"], self.metrics["log_loss"], places=9)
        self.assertAlmostEqual(scaled["too_long_auc"], self.metrics["too_long_auc"], places=9)

    def test_rejects_wrong_proba_shape(self):
        with self.assertRaises(ValueError):
            E.evaluate_direction(self.y, self.proba[:, :2], "full")

    def test_empty_input_is_nan(self):
        metrics = E.evaluate_direction([], np.empty((0, 3)), "fast")
        self.assertEqual(metrics["n"], 0)
        self.assertTrue(np.isnan(metrics["log_loss"]))
        self.assertEqual(metrics["confusion"], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])


class DirectionBaselineTest(unittest.TestCase):
    def test_full_uses_ci_width_and_eh_q(self):
        baselines = E.direction_baseline(_make_baseline_df(include_eh_q=True), "full")
        self.assertIn("ci_width", baselines)
        self.assertIn("neg_eh_q", baselines)
        # ci_width is wider for miscalls => ranks TOO_LONG above OK better than chance.
        self.assertGreater(baselines["ci_width"]["too_long_auc"], 0.5)
        self.assertGreater(baselines["neg_eh_q"]["too_long_auc"], 0.5)

    def test_fast_does_not_require_eh_q(self):
        # No eh_q column at all -> must not raise, ci_width baseline only.
        baselines = E.direction_baseline(_make_baseline_df(include_eh_q=False), "fast")
        self.assertIn("ci_width", baselines)
        self.assertNotIn("neg_eh_q", baselines)

    def test_fast_skips_eh_q_even_if_present(self):
        baselines = E.direction_baseline(_make_baseline_df(include_eh_q=True), "fast")
        self.assertIn("ci_width", baselines)
        self.assertNotIn("neg_eh_q", baselines)

    def test_all_nan_column_is_skipped(self):
        df = _make_baseline_df(include_eh_q=True)
        df["ci_width"] = np.nan
        baselines = E.direction_baseline(df, "full")
        self.assertNotIn("ci_width", baselines)
        self.assertIn("neg_eh_q", baselines)

    def test_baseline_cols_attached_in_evaluate_direction(self):
        y, proba = _make_direction_data(n=1500)
        # A perfect ranking score for TOO_LONG: 1 when truly TOO_LONG else 0.
        metrics = E.evaluate_direction(y, proba, "full",
                                       baseline_cols={"oracle": (y == 1).astype(float)})
        self.assertAlmostEqual(metrics["baselines"]["oracle"]["too_long_auc"], 1.0)


class StratifiedTest(unittest.TestCase):
    def test_stratified_q_structure(self):
        strata = E.stratified_q(_make_q_df())
        self.assertEqual(set(strata["source"]), {"pooled", "real", "sim"})
        self.assertEqual(set(strata["motif_size"]),
                         {"1", "2", "3", "4", "5", "6+"})
        self.assertIn("mae_t", strata["source"]["pooled"])
        # pooled n equals the sum of real + sim n.
        self.assertEqual(strata["source"]["pooled"]["n"],
                         strata["source"]["real"]["n"] + strata["source"]["sim"]["n"])

    def test_stratified_direction_structure(self):
        strata = E.stratified_direction(_make_dir_df(), "fast")
        self.assertEqual(set(strata["source"]), {"pooled", "real", "sim"})
        self.assertIn("log_loss", strata["source"]["pooled"])
        self.assertIn("baselines", strata["source"]["pooled"])
        self.assertIn("ci_width", strata["source"]["pooled"]["baselines"])
        # fast branch must never carry an eh_q baseline.
        self.assertNotIn("neg_eh_q", strata["source"]["pooled"]["baselines"])
        self.assertIn("6+", strata["motif_size"])

    def test_stratified_handles_missing_stratum_column(self):
        df = _make_q_df().drop(columns=["coverage", "sample"])
        strata = E.stratified_q(df)
        self.assertIn("source", strata)
        self.assertIn("motif_size", strata)
        self.assertNotIn("coverage", strata)
        self.assertNotIn("sample", strata)

    def test_size_bin_stratum_present_when_column_given(self):
        df = _make_q_df()
        rng = np.random.default_rng(SEED)
        df["size_bin"] = rng.integers(0, 27, size=len(df))
        strata = E.stratified_q(df)
        self.assertIn("size_bin", strata)
        # every occupied bin appears, keyed by its stringified integer index.
        self.assertEqual(set(strata["size_bin"]),
                         {str(b) for b in sorted(df["size_bin"].unique())})


class MacroOverBinsTest(unittest.TestCase):
    def test_equal_weight_per_bin_ignores_bin_size(self):
        # A huge bin at 0.9 and a tiny bin at 0.1 -> macro mean is 0.5, NOT
        # size-weighted (which would be ~0.9).
        bin_metrics = {"0": {"n": 100000, "x": 0.9}, "1": {"n": 3, "x": 0.1}}
        macro = E.macro_over_bins(bin_metrics, ["x"])
        self.assertAlmostEqual(macro["x"], 0.5, places=12)
        self.assertEqual(macro["n_bins"], 2)

    def test_skips_nan_and_below_min_n(self):
        bin_metrics = {"0": {"n": 50, "x": 0.4}, "1": {"n": 50, "x": float("nan")},
                       "2": {"n": 0, "x": 0.9}}
        macro = E.macro_over_bins(bin_metrics, ["x"], min_n=1)
        self.assertAlmostEqual(macro["x"], 0.4, places=12)  # only bin 0 contributes
        self.assertEqual(macro["n_bins"], 2)  # bins 0 and 1 meet min_n; bin 2 excluded

    def test_all_nan_or_empty_gives_nan(self):
        self.assertTrue(np.isnan(E.macro_over_bins({}, ["x"])["x"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
