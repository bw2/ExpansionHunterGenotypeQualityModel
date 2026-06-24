"""Unit tests for ``ablation`` on synthetic data with known feature importance.

Real ``HistGradientBoostingRegressor`` fits drive the ``rank_fn`` / ``score_fn``
callables: the target depends strongly on ``feat0`` (weakly on ``feat1``); the
rest are noise. The tests check that ``add_one_curve`` returns a small near-peak
prefix containing ``feat0`` and that ``grouped_ablation`` reports one marginal per
family with the informative family on top. Every plot is written as ``.svg`` +
``.png``.

Run with:  python3 -m unittest ablation_tests -v
"""

import os
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import ablation as A


FEATURE_NAMES = ["feat0", "feat1", "noise2", "noise3", "noise4"]
FAMILIES = {"informative": ["feat0", "feat1"], "noise": ["noise2", "noise3", "noise4"]}


def _assert_svg_png(test, out_prefix):
    """Asserts both ``out_prefix.svg`` and ``out_prefix.png`` exist and are non-empty."""
    for ext in ("svg", "png"):
        path = f"{out_prefix}.{ext}"
        test.assertTrue(os.path.exists(path), f"missing {path}")
        test.assertGreater(os.path.getsize(path), 0, f"empty {path}")


def _make_dataset(n=2400, seed=A.SEED):
    """Returns ``(X_train, y_train, X_calib, y_calib)`` with feat0 dominant."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = 2.0 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


class AblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.X_calib, cls.y_calib = _make_dataset()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _score_fn(self, subset):
        # Train on train rows, score (R^2) on the calib rows -- never test.
        cols = [FEATURE_NAMES.index(f) for f in subset]
        model = HistGradientBoostingRegressor(max_iter=60, random_state=A.SEED)
        model.fit(self.X_train[:, cols], self.y_train)
        return model.score(self.X_calib[:, cols], self.y_calib)

    def _rank_fn(self):
        # Permutation importance on the calib rows -> feature names most->least.
        model = HistGradientBoostingRegressor(max_iter=60, random_state=A.SEED)
        model.fit(self.X_train, self.y_train)
        result = permutation_importance(
            model, self.X_calib, self.y_calib, n_repeats=5, random_state=A.SEED)
        return [FEATURE_NAMES[i] for i in np.argsort(result.importances_mean)[::-1]]

    def test_add_one_curve_returns_small_subset_with_feat0(self):
        out = os.path.join(self.tmp.name, "add_one")
        k, subset = A.add_one_curve(self._rank_fn, self._score_fn, FEATURE_NAMES, out)
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, len(FEATURE_NAMES))
        self.assertEqual(len(subset), k)
        self.assertIn("feat0", subset)
        # subset must be the top-k prefix of the importance ranking.
        self.assertEqual(subset, [f for f in self._rank_fn() if f in FEATURE_NAMES][:k])
        _assert_svg_png(self, out)

    def test_grouped_ablation_one_entry_per_family(self):
        out = os.path.join(self.tmp.name, "grouped")
        marginals = A.grouped_ablation(self._score_fn, FAMILIES, FEATURE_NAMES, out)
        self.assertEqual(set(marginals), set(FAMILIES))
        # Dropping the informative family hurts more than dropping noise.
        self.assertGreater(marginals["informative"], marginals["noise"])
        _assert_svg_png(self, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
