"""Unit tests for ``explain`` on synthetic data with a known-informative feature.

A tiny real ``HistGradientBoostingRegressor`` is fit on synthetic data whose
target depends strongly on ``feat0`` (and weakly on ``feat1``); the remaining
columns are pure noise. The tests check that permutation / drop-column importance
both rank ``feat0`` first and that every plot is written as ``.svg`` + ``.png``.

Run with:  python3 -m unittest explain_tests -v
"""

import os
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import explain as E


FEATURE_NAMES = ["feat0", "feat1", "noise2", "noise3", "noise4"]


def _assert_svg_png(test, out_prefix):
    """Asserts both ``out_prefix.svg`` and ``out_prefix.png`` exist and are non-empty."""
    for ext in ("svg", "png"):
        path = f"{out_prefix}.{ext}"
        test.assertTrue(os.path.exists(path), f"missing {path}")
        test.assertGreater(os.path.getsize(path), 0, f"empty {path}")


def _make_dataset(n=2000, seed=E.SEED):
    """Returns ``(X_train, y_train, X_test, y_test)`` with feat0 dominant."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = 2.0 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


class ExplainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = _make_dataset()
        cls.estimator = HistGradientBoostingRegressor(
            max_iter=80, random_state=E.SEED).fit(cls.X_train, cls.y_train)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_permutation_importance_ranks_informative_first(self):
        out = os.path.join(self.tmp.name, "perm_imp")
        ranked = E.permutation_importance_plot(
            self.estimator, self.X_test, self.y_test, FEATURE_NAMES, out, n_repeats=5)
        self.assertEqual(len(ranked), len(FEATURE_NAMES))
        # Returned most->least; the strongest signal must be on top.
        self.assertEqual(ranked[0][0], "feat0")
        means = [r[1] for r in ranked]
        self.assertEqual(means, sorted(means, reverse=True))
        # Each entry is (name, mean, std) with a non-negative std.
        for _, _, std in ranked:
            self.assertGreaterEqual(std, 0.0)
        _assert_svg_png(self, out)

    def test_partial_dependence_plots_saved(self):
        out = os.path.join(self.tmp.name, "pdp")
        plotted = E.partial_dependence_plots(
            self.estimator, self.X_test, FEATURE_NAMES, ["feat0", "feat1"], out)
        self.assertEqual(plotted, ["feat0", "feat1"])
        _assert_svg_png(self, out)

    def test_partial_dependence_accepts_indices_and_skips_unknown(self):
        out = os.path.join(self.tmp.name, "pdp_idx")
        plotted = E.partial_dependence_plots(
            self.estimator, self.X_test, FEATURE_NAMES, [0, "nope"], out)
        self.assertEqual(plotted, ["feat0"])
        _assert_svg_png(self, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
