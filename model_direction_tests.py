"""Unit tests for ``model_direction`` on synthetic data.

Synthetic-only (no fixtures): a 3-class label is drawn as a noisy function of a
handful of features via ``numpy.random.default_rng(SEED)`` (Gumbel-noise argmax,
i.e. a multinomial logit), so the suite is deterministic and fast. Iteration caps
are kept small so the classifier and its calibrators fit in a few seconds.

Run with:  python3 -m unittest model_direction_tests -v
"""

import unittest

import numpy as np
from sklearn.metrics import log_loss

import model_direction as M


# Small early-stop caps to keep the suite fast while still exercising the loop.
_FAST_STOP = {"max_total_iter": 150, "step": 25, "patience": 3}


def _make_noisy_dataset(n=4000, seed=M.SEED):
    """Builds ``(X, y)`` with a 3-class ``y`` that is a noisy function of ``X``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logits = np.column_stack([
        np.zeros(n),                                       # OK baseline
        1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2],     # TOO_LONG
        -1.0 * X[:, 0] + 0.9 * X[:, 3] - 0.4 * X[:, 4],    # TOO_SHORT
    ])
    # Gumbel-noise argmax over logits == sampling from the softmax (multinomial).
    y = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)
    return X, y


class MulticlassLogLossTest(unittest.TestCase):
    def test_hand_computed_value(self):
        proba = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        expected = -(np.log(0.7) + np.log(0.8) + np.log(0.6)) / 3.0
        self.assertAlmostEqual(M.multiclass_log_loss([0, 1, 2], proba), expected, places=12)

    def test_matches_sklearn(self):
        rng = np.random.default_rng(1)
        proba = rng.dirichlet(np.ones(3), size=300)
        y = rng.integers(0, 3, size=300)
        self.assertAlmostEqual(
            M.multiclass_log_loss(y, proba),
            log_loss(y, proba, labels=[0, 1, 2]), places=10)

    def test_uniform_equals_log_three(self):
        proba = np.full((50, 3), 1.0 / 3.0)
        y = np.arange(50) % 3
        self.assertAlmostEqual(M.multiclass_log_loss(y, proba), np.log(3.0), places=12)

    def test_weighted_matches_sklearn(self):
        rng = np.random.default_rng(2)
        proba = rng.dirichlet(np.ones(3), size=300)
        y = rng.integers(0, 3, size=300)
        w = rng.uniform(0.1, 5.0, size=300)
        self.assertAlmostEqual(
            M.multiclass_log_loss(y, proba, sample_weight=w),
            log_loss(y, proba, labels=[0, 1, 2], sample_weight=w), places=10)


class TrainPredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = _make_noisy_dataset()
        cls.X_train, cls.y_train = X[:3200], y[:3200]
        cls.X_calib, cls.y_calib = X[3200:], y[3200:]
        cls.model = M.train_direction(cls.X_train, cls.y_train, cls.X_calib, cls.y_calib,
                                      early_stop_kwargs=_FAST_STOP)
        cls.proba = M.predict_proba(cls.model, cls.X_calib)

    def test_proba_shape(self):
        self.assertEqual(self.proba.shape, (self.X_calib.shape[0], M.N_CLASSES))

    def test_rows_sum_to_one(self):
        self.assertTrue(np.allclose(self.proba.sum(axis=1), 1.0, atol=1e-9))

    def test_probabilities_valid_range(self):
        self.assertTrue(np.all(self.proba >= 0.0))
        self.assertTrue(np.all(self.proba <= 1.0))

    def test_beats_uniform_baseline(self):
        loss = M.multiclass_log_loss(self.y_calib, self.proba)
        self.assertLess(loss, np.log(3.0))

    def test_history_non_empty_and_well_formed(self):
        history = self.model["history"]
        self.assertGreater(len(history), 0)
        for step in history:
            self.assertEqual(set(step), {"n_iter", "train_loss", "calib_loss"})
        calib = [s["calib_loss"] for s in history]
        # Best (minimum) calib loss is no worse than where it started.
        self.assertLessEqual(min(calib), calib[0] + 1e-9)

    def test_calibrators_present_per_class(self):
        self.assertEqual(set(self.model["calibrators"]), {M.OK, M.TOO_LONG, M.TOO_SHORT})

    def test_all_zero_row_falls_back_to_uniform(self):
        # Force every calibrator to map to 0 so the renormalization hits the
        # all-zero fallback path; rows must become uniform 1/3.
        zero_model = dict(self.model)
        zeroed = {idx: _ConstCalibrator(0.0) for idx in range(M.N_CLASSES)}
        zero_model["calibrators"] = zeroed
        out = M.predict_proba(zero_model, self.X_calib[:10])
        self.assertTrue(np.allclose(out, 1.0 / M.N_CLASSES))


class _ConstCalibrator:
    """Minimal stand-in for IsotonicRegression returning a constant prediction."""

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class AbsentCalibClassTest(unittest.TestCase):
    """A class missing from the calibration split must not be zeroed for every test row.

    Regression: fitting isotonic on an all-zero one-vs-rest target collapsed the class to
    a constant 0, which after renormalization gave P=0 for that class on EVERY row.
    """

    def test_absent_class_uses_passthrough_and_keeps_mass(self):
        X, y = _make_noisy_dataset(n=2000)
        # Calibration rows drawn only from {OK, TOO_LONG}; TOO_SHORT is absent from the calib split
        # (but present in train and test), so its calibrator must be the passthrough.
        keep = y != M.TOO_SHORT
        X_calib, y_calib = X[keep][:400], y[keep][:400]
        model = M.train_direction(X[:1600], y[:1600], X_calib, y_calib,
                                  early_stop_kwargs=_FAST_STOP)
        self.assertIs(model["calibrators"][M.TOO_SHORT], M._PassthroughCalibrator)
        proba = M.predict_proba(model, X)
        # TOO_SHORT occurs in the data, so some rows must retain nonzero P_TOO_SHORT (not all-zeroed).
        self.assertGreater(float(proba[:, M.TOO_SHORT].max()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
