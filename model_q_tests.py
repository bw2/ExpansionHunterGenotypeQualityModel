"""Unit tests for ``model_q`` on synthetic data.

Synthetic-only (no fixtures): ``t`` is a noisy function of 3-4 features drawn
with ``numpy.random.default_rng(SEED)`` so the suite is deterministic and fast.
Iteration caps are kept small so all five quantile heads fit in a few seconds.

Run with:  python3 -m unittest model_q_tests -v
"""

import unittest

import numpy as np
from sklearn.metrics import mean_pinball_loss

import model_q as M


# Small early-stop caps to keep the suite fast while still exercising the loop.
_FAST_STOP = {"max_total_iter": 150, "step": 25, "patience": 3}


def _make_noisy_dataset(n=4000, seed=M.SEED):
    """Builds ``(X, t)`` where ``t`` is a noisy function of 3-4 features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    t = (0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.25 * X[:, 2]
         + 0.15 * X[:, 2] * X[:, 3] + rng.normal(scale=0.3, size=n))
    return X, t


class PinballLossTest(unittest.TestCase):
    def test_hand_computed_value_median(self):
        # y_true-y_pred = [-0.5, 0.5, 1.5]; q=0.5 => 0.5*MAE = 0.5*(2.5/3).
        loss = M.pinball_loss([1.0, 2.0, 3.0], [1.5, 1.5, 1.5], 0.5)
        self.assertAlmostEqual(loss, 1.25 / 3.0, places=10)

    def test_hand_computed_value_high_quantile(self):
        # q=0.9: (0.1*0.5 + 0.9*0.5 + 0.9*1.5)/3 = 1.85/3.
        loss = M.pinball_loss([1.0, 2.0, 3.0], [1.5, 1.5, 1.5], 0.9)
        self.assertAlmostEqual(loss, 1.85 / 3.0, places=10)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(size=200)
        y_pred = rng.normal(size=200)
        for q in M.QUANTILES:
            self.assertAlmostEqual(
                M.pinball_loss(y_true, y_pred, q),
                mean_pinball_loss(y_true, y_pred, alpha=q), places=12)

    def test_zero_when_exact(self):
        self.assertEqual(M.pinball_loss([1.0, 2.0], [1.0, 2.0], 0.3), 0.0)

    def test_weighted_matches_sklearn(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(size=200)
        y_pred = rng.normal(size=200)
        w = rng.uniform(0.1, 5.0, size=200)
        for q in M.QUANTILES:
            self.assertAlmostEqual(
                M.pinball_loss(y_true, y_pred, q, sample_weight=w),
                mean_pinball_loss(y_true, y_pred, alpha=q, sample_weight=w), places=12)

    def test_weights_change_the_fit(self):
        # Up-weighting one tail group should pull its predictions; the weighted and
        # unweighted median fits must differ on that group.
        X, t = _make_noisy_dataset(n=3000)
        w = np.where(X[:, 0] > 1.0, 50.0, 1.0)
        m_un = M.train_q(X[:2400], t[:2400], X[2400:], t[2400:], quantiles=[0.5],
                         early_stop_kwargs=_FAST_STOP)["models"][0.5]
        m_w = M.train_q(X[:2400], t[:2400], X[2400:], t[2400:], quantiles=[0.5],
                        early_stop_kwargs=_FAST_STOP,
                        sample_weight=w[:2400], sample_weight_calib=w[2400:])["models"][0.5]
        tail = X[2400:][:, 0] > 1.0
        self.assertGreater(
            float(np.mean(np.abs(m_un.predict(X[2400:][tail]) - m_w.predict(X[2400:][tail])))),
            1e-6)


class TrainPredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, t = _make_noisy_dataset()
        cls.X_train, cls.t_train = X[:3200], t[:3200]
        cls.X_calib, cls.t_calib = X[3200:], t[3200:]
        cls.fit = M.train_q(cls.X_train, cls.t_train, cls.X_calib, cls.t_calib,
                            early_stop_kwargs=_FAST_STOP)
        cls.models = cls.fit["models"]

    def test_predict_q_shapes(self):
        preds = M.predict_q(self.models, self.X_calib)
        self.assertEqual(sorted(preds.keys()), M.QUANTILES)
        for q in M.QUANTILES:
            self.assertEqual(preds[q].shape, (self.X_calib.shape[0],))

    def test_quantiles_non_crossing(self):
        preds = M.predict_q(self.models, self.X_calib)
        stacked = np.column_stack([preds[q] for q in M.QUANTILES])
        diffs = np.diff(stacked, axis=1)
        # q05 <= q10 <= q50 <= q90 <= q95 for every row (allow fp slack).
        self.assertTrue(np.all(diffs >= -1e-9))

    def test_interval_coverage_sane(self):
        lo, hi = M.predictive_interval(self.models, self.X_calib, 0.8)
        coverage = float(np.mean((self.t_calib >= lo) & (self.t_calib <= hi)))
        self.assertGreaterEqual(coverage, 0.6)
        self.assertLessEqual(coverage, 0.95)

    def test_predictive_interval_ordering(self):
        for level in (0.8, 0.9):
            lo, hi = M.predictive_interval(self.models, self.X_calib, level)
            self.assertTrue(np.all(hi >= lo))

    def test_predictive_interval_rejects_bad_level(self):
        with self.assertRaises(ValueError):
            M.predictive_interval(self.models, self.X_calib, 0.5)

    def test_histories_non_empty_and_improve(self):
        histories = self.fit["histories"]
        for q in M.QUANTILES:
            history = histories[q]
            self.assertGreater(len(history), 0)
            for step in history:
                self.assertEqual(set(step), {"n_iter", "train_loss", "calib_loss"})
            calib = [s["calib_loss"] for s in history]
            # Calib loss decreases from the first step then plateaus: the best
            # (minimum) is no worse than where it started.
            self.assertLessEqual(min(calib), calib[0] + 1e-9)

    def test_point_q_requires_median(self):
        with self.assertRaises(KeyError):
            M.point_q({0.1: self.models[0.1], 0.9: self.models[0.9]}, self.X_calib)


class RecoverTrueTest(unittest.TestCase):
    def test_roundtrip_on_noiseless_data(self):
        rng = np.random.default_rng(M.SEED + 1)
        X = rng.normal(size=(4000, 4))
        t_clean = 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
        true = rng.uniform(10.0, 100.0, size=4000)
        eh = np.exp(t_clean) * true
        models = M.train_q(X[:3200], t_clean[:3200], X[3200:], t_clean[3200:],
                           quantiles=[0.5],
                           early_stop_kwargs={"max_total_iter": 200, "step": 25,
                                              "patience": 3})["models"]
        recovered = M.recover_true(models, X[3200:], eh[3200:])
        self.assertEqual(recovered.shape, (800,))
        rel_err = np.abs(recovered - true[3200:]) / true[3200:]
        # Noiseless target: median recovery error should be small.
        self.assertLess(float(np.median(rel_err)), 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
