"""Tests for the model heads and the serialize -> round-trip-verify invariant.

The headline guarantee is that inference re-implemented from the serialized JSON
(mirroring the C++ consumer) reproduces sklearn's predictions; ``verify_genotyping_regime``
raises if it does not, so a passing fit+serialize+verify is the test.
"""

import unittest

import numpy as np
import pandas as pd

import model as M


def _synthetic(n=1500, k=22, seed=0):
    """Builds a separable-ish synthetic (X, t, dir_code) with all 3 direction classes present."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=["f%d" % i for i in range(k)])
    t = 0.3 * X["f0"].to_numpy() - 0.2 * X["f1"].to_numpy() + rng.normal(scale=0.1, size=n)
    # three classes from a monotone signal so every fold/calib split sees all of them
    s = X["f0"].to_numpy() + 0.5 * rng.normal(size=n)
    y = np.where(s > 0.6, 1, np.where(s < -0.6, 2, 0))
    return X, t, y


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        X, t, y = _synthetic()
        # chrom-clean-style split by row index halves (content-irrelevant for this unit test)
        self.Xtr, self.Xca = X.iloc[:1000], X.iloc[1000:]
        self.ttr, self.tca = t[:1000], t[1000:]
        self.ytr, self.yca = y[:1000], y[1000:]

    def test_serialize_round_trip(self):
        qreg = M.train_q_median(self.Xtr, self.ttr, self.Xca, self.tca)
        dmodel = M.train_direction(self.Xtr, self.ytr, self.Xca, self.yca)
        genotyping_regime_json = M.serialize_genotyping_regime(qreg, dmodel)
        # raises on any mismatch > 1e-6
        M.verify_genotyping_regime("unit", qreg, dmodel, genotyping_regime_json, self.Xca.iloc[:200])

        self.assertEqual(genotyping_regime_json["direction"]["classes"], ["OK", "TOO_LONG", "TOO_SHORT"])
        self.assertEqual(len(genotyping_regime_json["direction"]["baseline"]), 3)
        self.assertTrue(genotyping_regime_json["q_median"]["trees"])

    def test_predict_proba_simplex(self):
        dmodel = M.train_direction(self.Xtr, self.ytr, self.Xca, self.yca)
        proba = M.predict_proba(dmodel, self.Xca)
        self.assertEqual(proba.shape, (len(self.Xca), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_lcf_positive(self):
        qreg = M.train_q_median(self.Xtr, self.ttr, self.Xca, self.tca)
        self.assertTrue(np.all(M.predict_lcf(qreg, self.Xca) > 0))

    def test_vectorized_json_inference_matches_sklearn(self):
        # The vectorized JSON evaluator (used to APPLY the exported model, e.g. on held-out
        # samples) must reproduce sklearn's predictions through the serialized dict.
        qreg = M.train_q_median(self.Xtr, self.ttr, self.Xca, self.tca)
        dmodel = M.train_direction(self.Xtr, self.ytr, self.Xca, self.yca)
        comp = M.compile_genotyping_regime(M.serialize_genotyping_regime(qreg, dmodel))
        np.testing.assert_allclose(M.predict_lcf_json(comp, self.Xca),
                                   M.predict_lcf(qreg, self.Xca), atol=1e-9)
        np.testing.assert_allclose(M.predict_proba_json(comp, self.Xca),
                                   M.predict_proba(dmodel, self.Xca), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
