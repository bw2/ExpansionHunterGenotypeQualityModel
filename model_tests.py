"""Tests for the model heads and the serialize -> round-trip-verify invariant.

The headline guarantee is that inference re-implemented from the serialized JSON
(mirroring the C++ consumer) reproduces sklearn's predictions; ``verify_genotyping_regime``
raises if it does not, so a passing fit+serialize+verify is the test.
"""

import unittest

import numpy as np
import pandas as pd

import features
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

    def test_fixed_n_iter_fits_exactly_that_many_iterations(self):
        qreg = M.train_q_median(self.Xtr, self.ttr, n_iter=37)
        dmodel = M.train_direction(self.Xtr, self.ytr, self.Xca, self.yca, n_iter=23)
        self.assertEqual(qreg.n_iter_, 37)
        self.assertEqual(dmodel["clf"].n_iter_, 23)
        proba = M.predict_proba(dmodel, self.Xca)  # still calibrated on the calib rows
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


class FeatureNamesOfTest(unittest.TestCase):
    """The compiled trees index features by POSITION, so every applier builds its matrix from the
    model's OWN declared list -- including a model exported under an older contract."""

    def _model(self, quick=None, full=None):
        return {"feature_names": {
            "quick": features.QUICK_FEATURES if quick is None else quick,
            "full": features.FULL_FEATURES if full is None else full}}

    def test_returns_the_declared_lists_verbatim(self):
        self.assertEqual(M.feature_names_of(self._model(), "m.json.gz"),
                         {"quick": features.QUICK_FEATURES, "full": features.FULL_FEATURES})

    def test_an_older_contract_is_returned_as_declared_not_rejected(self):
        # Applying a previously deployed model (compare_models) depends on this: its list is shorter
        # than today's, and its trees index THAT list, so it must come back unchanged.
        older = [c for c in features.QUICK_FEATURES if c != "n_alleles"]
        self.assertEqual(M.feature_names_of(self._model(quick=older), "old.json.gz")["quick"], older)

    def test_order_is_preserved(self):
        reordered = list(features.FULL_FEATURES)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        self.assertEqual(M.feature_names_of(self._model(full=reordered), "m.json.gz")["full"],
                         reordered)

    def test_model_without_feature_names_is_rejected(self):
        # Nothing can be built safely: the positional tree indices have no list to resolve against.
        with self.assertRaises(SystemExit) as ctx:
            M.feature_names_of({}, "m.json.gz")
        self.assertIn("m.json.gz", str(ctx.exception))


class FingerprintTest(unittest.TestCase):
    def test_same_name_different_content_differs(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.20261006.json.gz")
            with open(path, "wb") as f:
                f.write(b"first")
            first = M.fingerprint(path)
            with open(path, "wb") as f:
                f.write(b"second")
            self.assertTrue(first.startswith("m.20261006.json.gz@"))
            self.assertNotEqual(first, M.fingerprint(path))


if __name__ == "__main__":
    unittest.main()
