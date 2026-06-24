"""Unit tests for run_cv's RunConfig threading.

Verifies the fit-controlling caps + bin-weight cap flow EXPLICITLY through
``run_fold`` / ``run_cross`` / ``_bin_weights`` from a ``RunConfig`` (no module
globals), and -- as a side check -- that ``run_cross`` holds out a whole
chromosome for calibration rather than a random row split.

The heavy model fit / scoring are stubbed (``unittest.mock``), so these tests run
instantly and assert only the wiring: which rows / weights each internal receives.

Run with:  python3 -m unittest run_cv_tests -v
"""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

import run_cv


def _toy_frame():
    """A 12-row toy frame over 3 chroms with the columns run_fold / run_cross touch."""
    chrom = ["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"]
    n = len(chrom)
    df = pd.DataFrame({
        "chrom": chrom,
        "eh": np.arange(n, dtype=float) + 5.0,
        "true": np.arange(n, dtype=float) + 5.0,
        "tol_repeats": np.ones(n),
    })
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})  # default RangeIndex == df row positions
    t = np.zeros(n)
    y = np.zeros(n, dtype=int)
    bins = np.zeros(n, dtype=int)
    return df, X, t, y, bins


def _stub_fit(record):
    """Returns a fake _fit_predict that records what it was handed and returns a minimal fit."""
    def fake(X_tr, t_tr, y_tr, X_ca, t_ca, y_ca, X_te, w_tr=None, w_ca=None):
        record["n_train"] = len(X_tr)
        record["tr_index"] = list(np.asarray(X_tr.index))
        record["ca_index"] = list(np.asarray(X_ca.index))
        record["w_tr"] = w_tr
        record["w_ca"] = w_ca
        n = len(X_te)
        return {"q_models": {0.5: None}, "q_hist": {0.5: None}, "dir_model": {"clf": None},
                "dir_hist": None, "qpreds": {0.5: np.zeros(n)}, "proba": np.zeros((n, 3))}
    return fake


class BinWeightsConfigTest(unittest.TestCase):
    def test_negative_cap_disables_weighting(self):
        bins = np.array([0, 0, 1, 13])
        self.assertIsNone(run_cv._bin_weights(bins, run_cv.RunConfig(None, None, -1.0)))

    def test_zero_cap_is_pure_inverse_frequency(self):
        bins = np.array([0] * 10 + [13] * 100)
        w = run_cv._bin_weights(bins, run_cv.RunConfig(None, None, 0.0))
        self.assertIsNotNone(w)
        self.assertAlmostEqual(float(np.mean(w)), 1.0, places=6)

    def test_positive_cap_is_a_hard_ceiling(self):
        bins = np.array([0] * 1000 + [5] * 2 + [13] * 50000)
        w = run_cv._bin_weights(bins, run_cv.RunConfig(None, None, 50.0))
        self.assertLessEqual(float(np.max(w)), 50.0 + 1e-9)


class RunFoldConfigTest(unittest.TestCase):
    def test_train_cap_and_disabled_weights_come_from_cfg(self):
        df, X, t, y, bins = _toy_frame()
        fold = {"train": ["1", "2"], "calib": ["3"], "test": ["3"]}  # train=8, calib=test=4
        record = {}
        cfg = run_cv.RunConfig(train_cap=3, extras_cap=None, bin_weight_cap=-1.0)
        with mock.patch.object(run_cv, "_fit_predict", _stub_fit(record)), \
             mock.patch.object(run_cv, "_score", lambda *a, **k: {}), \
             mock.patch.object(run_cv.MQ, "predict_q", lambda m, Xc: {0.5: np.zeros(len(Xc))}), \
             mock.patch.object(run_cv.evaluate, "evaluate_q", lambda *a, **k: {}):
            res, _fit = run_cv.run_fold(df, "fast", X, t, y, bins, fold, 0, cfg)
        self.assertEqual(record["n_train"], 3)     # cfg.train_cap applied by _cap_idx
        self.assertIsNone(record["w_tr"])          # cfg.bin_weight_cap < 0 -> no weights
        self.assertIsNone(record["w_ca"])
        self.assertEqual(res["fold_i"], 0)

    def test_weights_enabled_when_cfg_cap_nonnegative(self):
        df, X, t, y, bins = _toy_frame()
        fold = {"train": ["1", "2"], "calib": ["3"], "test": ["3"]}
        record = {}
        cfg = run_cv.RunConfig(train_cap=None, extras_cap=None, bin_weight_cap=0.0)
        with mock.patch.object(run_cv, "_fit_predict", _stub_fit(record)), \
             mock.patch.object(run_cv, "_score", lambda *a, **k: {}), \
             mock.patch.object(run_cv.MQ, "predict_q", lambda m, Xc: {0.5: np.zeros(len(Xc))}), \
             mock.patch.object(run_cv.evaluate, "evaluate_q", lambda *a, **k: {}):
            run_cv.run_fold(df, "fast", X, t, y, bins, fold, 0, cfg)
        self.assertIsNotNone(record["w_tr"])       # cfg.bin_weight_cap >= 0 -> weights present


class RunCrossConfigTest(unittest.TestCase):
    def test_extras_cap_from_cfg_and_calib_is_chromosome_held_out(self):
        df, X, t, y, bins = _toy_frame()
        train_mask = df["chrom"].isin(["1", "2"]).to_numpy()   # chroms 1 & 2 (8 rows)
        test_mask = df["chrom"].isin(["3"]).to_numpy()         # chrom 3 (4 rows)
        chrom = df["chrom"].to_numpy()
        record = {}
        cfg = run_cv.RunConfig(train_cap=None, extras_cap=2, bin_weight_cap=-1.0)
        with mock.patch.object(run_cv, "_fit_predict", _stub_fit(record)), \
             mock.patch.object(run_cv, "_score", lambda *a, **k: {"ok": True}):
            out = run_cv.run_cross(df, "fast", X, t, y, bins, "cov", train_mask, test_mask, cfg)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(record["n_train"], 2)     # cfg.extras_cap applied to the fit rows
        fit_chroms = set(chrom[record["tr_index"]])
        calib_chroms = set(chrom[record["ca_index"]])
        self.assertTrue(fit_chroms.isdisjoint(calib_chroms))   # whole-chromosome holdout, no leak
        self.assertTrue(calib_chroms.issubset({"1", "2"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
