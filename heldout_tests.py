"""Tests for heldout.py: per-sample build wiring, gated-correction accumulation, run_eval wiring.

The direction/q-median model heads are mocked out everywhere here (``model.predict_lcf_json`` /
``model.predict_proba_json``) so the gate-and-accumulate arithmetic is verified against known inputs
without needing a trained model -- see ``model_tests.py`` for the actual head-fitting tests.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import dataset
import dataset_tests  # for write_contract_parquet, so both suites build the same fixture parquet
import features
import heldout
import model as M


def _raw_feature_row(**overrides):
    """One row with every raw column ``features.build_matrix`` needs (full branch), plus the
    join/labeling columns ``dataset.label_and_filter`` needs; override any field by keyword."""
    row = {c: 1.0 for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")}
    row.update({"ci_start": 1.0, "ci_end": 3.0, "locus_id": "1-1-2-AAA", "is_negative_locus": False,
               "allele_rank": 0, "genotyping_branch": "full", "spanning_at_called": 5,
               # Not a feature, but part of the extractor's contract: run_eval refuses a parquet
               # without it, since label_and_filter could not then drop unscoreable hom copies.
               "has_own_quality_metrics": True})
    row.update(overrides)
    return row


class CatalogBaseTest(unittest.TestCase):
    def test_path_shape(self):
        base = heldout._catalog_base("HG00438", "30x")
        self.assertEqual(base, "%s/HG00438/illumina/%s/30x_coverage/%s/"
                         % (heldout.GCS_ROOT, heldout.VARIANT, heldout.CATALOG))


class DiscoverCovTest(unittest.TestCase):
    def test_parses_coverage_dir(self):
        with mock.patch.object(heldout.subprocess, "run",
                               return_value=SimpleNamespace(stdout="gs://x/illumina/32x_coverage/\n")):
            self.assertEqual(heldout._discover_cov("HG00438"), "32x")

    def test_missing_coverage_dir_raises(self):
        with mock.patch.object(heldout.subprocess, "run", return_value=SimpleNamespace(stdout="")):
            with self.assertRaises(RuntimeError):
                heldout._discover_cov("HG00438")


class NewAccTest(unittest.TestCase):
    def test_zero_defaults(self):
        acc = heldout._new_acc()
        self.assertEqual(acc["n"], 0)
        self.assertEqual(acc["sum_db"], 0.0)
        self.assertEqual(acc["err_raw"], [])
        self.assertEqual(acc["h_n"], 0)


class StridedTest(unittest.TestCase):
    def test_passthrough_when_small(self):
        a = np.array([1.0, 2.0, 3.0])
        out = heldout._strided(a, slice(None))
        np.testing.assert_array_equal(out, a)
        self.assertEqual(out.dtype, np.float32)

    def test_strided_slice(self):
        a = np.arange(10, dtype=float)
        out = heldout._strided(a, slice(None, None, 2))
        np.testing.assert_array_equal(out, [0, 2, 4, 6, 8])


class AccumulateTest(unittest.TestCase):
    def _sub(self):
        return pd.DataFrame([
            _raw_feature_row(eh=12.0, true=10.0, motif_size=1, dir_code=features.TOO_LONG),   # homopolymer
            _raw_feature_row(eh=14.0, true=10.0, motif_size=3, dir_code=features.TOO_LONG),   # non-homopolymer
        ])

    def _run(self):
        acc = heldout._new_acc()
        with mock.patch.object(M, "predict_lcf_json", return_value=np.array([1.2, 1.4])), \
             mock.patch.object(M, "predict_proba_json",
                               return_value=np.array([[0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])):
            heldout._accumulate(acc, self._sub(), comp=None, branch="full",
                                names=features.FULL_FEATURES)
        return acc

    def test_non_homopolymer_scalars(self):
        acc = self._run()
        self.assertEqual(acc["n"], 1)
        self.assertAlmostEqual(acc["sum_db"], 4.0)   # |10-14|
        self.assertAlmostEqual(acc["sum_da"], 0.0)   # gate applies (p_ok=0.1): corrected = 14/1.4 = 10
        self.assertEqual(acc["ex_eh"], 0)            # round(14) != round(10)
        self.assertEqual(acc["ex_gated"], 1)         # round(10) == round(10)
        self.assertEqual(acc["pok_correct"], 1)      # argmax([.1,.8,.1])=1=TOO_LONG=dir_code
        self.assertEqual((acc["n_lt"], acc["helped_lt"], acc["hurt_lt"]), (1, 1, 0))
        self.assertEqual((acc["n_ge"], acc["helped_ge"], acc["hurt_ge"]), (0, 0, 0))
        np.testing.assert_allclose(acc["red_all"][0], [4.0])
        np.testing.assert_allclose(acc["pdiff_all"][0], [0.7])  # p_long - p_short = 0.8 - 0.1

    def test_homopolymer_scalars_and_by_motif_sample_keeps_all(self):
        acc = self._run()
        self.assertEqual(acc["h_n"], 1)
        self.assertAlmostEqual(acc["h_sum_db"], 2.0)  # |10-12|
        self.assertAlmostEqual(acc["h_sum_da"], 0.0)  # gate applies (p_ok=0.2): corrected = 12/1.2 = 10
        self.assertEqual((acc["h_n_lt"], acc["h_helped_lt"], acc["h_hurt_lt"]), (1, 1, 0))
        np.testing.assert_allclose(acc["mred_all"][0], [2.0, 4.0])   # both rows, homopolymer first
        np.testing.assert_allclose(acc["motif_all"][0], [1, 3])

    def test_all_homopolymer_rows_skip_non_homopolymer_accumulation(self):
        acc = heldout._new_acc()
        sub = pd.DataFrame([_raw_feature_row(eh=10.0, true=10.0, motif_size=1, dir_code=features.OK)])
        with mock.patch.object(M, "predict_lcf_json", return_value=np.array([1.0])), \
             mock.patch.object(M, "predict_proba_json", return_value=np.array([[0.9, 0.05, 0.05]])):
            heldout._accumulate(acc, sub, comp=None, branch="full", names=features.FULL_FEATURES)
        self.assertEqual(acc["n"], 0)
        self.assertEqual(acc["h_n"], 1)


class HomopolymerSummaryTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(heldout._homopolymer_summary(heldout._new_acc()), {"n": 0})

    def test_nonempty(self):
        acc = heldout._new_acc()
        acc.update(h_n=2, h_sum_db=6.0, h_sum_da=2.0, h_err_raw=[np.array([2.0, 4.0], np.float32)],
                  h_err_gated=[np.array([1.0, 1.0], np.float32)], h_n_lt=2, h_helped_lt=2, h_hurt_lt=0,
                  h_n_ge=0, h_helped_ge=0, h_hurt_ge=0)
        summary = heldout._homopolymer_summary(acc)
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mae_raw"], 3.0)
        self.assertAlmostEqual(summary["mae_gated"], 1.0)
        self.assertAlmostEqual(summary["median_raw"], 3.0)


class FinalizeTest(unittest.TestCase):
    def test_empty(self):
        result = heldout._finalize(heldout._new_acc(), n_samples=2)
        self.assertEqual(result, {"n": 0, "homopolymer": {"n": 0}})

    def test_nonempty(self):
        acc = heldout._new_acc()
        acc.update(n=1, sum_db=4.0, sum_da=0.0, err_raw=[np.array([4.0], np.float32)],
                  err_gated=[np.array([0.0], np.float32)], ex_eh=0, ex_gated=1, pok_correct=1,
                  n_lt=1, helped_lt=1, hurt_lt=0, n_ge=0, helped_ge=0, hurt_ge=0)
        result = heldout._finalize(acc, n_samples=3)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["n_samples"], 3)
        self.assertAlmostEqual(result["mae_raw"], 4.0)
        self.assertAlmostEqual(result["mae_gated"], 0.0)
        self.assertAlmostEqual(result["dist_reduction"], 1.0)
        self.assertAlmostEqual(result["exact_eh"], 0.0)
        self.assertAlmostEqual(result["exact_gated"], 1.0)

    def test_zero_raw_mae_gives_nan_dist_reduction(self):
        acc = heldout._new_acc()
        acc.update(n=1, sum_db=0.0, sum_da=0.0, err_raw=[np.array([0.0], np.float32)],
                  err_gated=[np.array([0.0], np.float32)], ex_eh=1, ex_gated=1, pok_correct=1,
                  n_lt=0, helped_lt=0, hurt_lt=0, n_ge=1, helped_ge=0, hurt_ge=0)
        self.assertTrue(np.isnan(heldout._finalize(acc, n_samples=1)["dist_reduction"]))


class BuildSampleTest(unittest.TestCase):
    def test_cache_hit_skips_download(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "real_43", "HG00438.parquet")
            os.makedirs(os.path.dirname(out_path))
            # Upstream local sources must exist and be OLDER than the parquet for it to be reused.
            dl_dir = os.path.join(d, "real_43", "_downloads", "HG00438")
            os.makedirs(dl_dir)
            json_local = os.path.join(dl_dir, "a.json.gz")
            tsv_local = os.path.join(dl_dir, "HG00438.tandem_repeat_genotypes.tsv.gz")
            open(json_local, "w").close()
            open(tsv_local, "w").close()
            dataset_tests.write_contract_parquet(out_path, n_rows=2)
            newer = max(os.path.getmtime(json_local), os.path.getmtime(tsv_local)) + 100
            os.utime(out_path, (newer, newer))
            with mock.patch.object(heldout, "_discover_cov", return_value="30x"), \
                 mock.patch.object(heldout.subprocess, "run",
                                   return_value=SimpleNamespace(stdout="gs://x/json/a.json.gz")), \
                 mock.patch.object(dataset, "_check_freshness", return_value=0), \
                 mock.patch.object(dataset, "_download", side_effect=AssertionError("should not download")):
                out = heldout.build_sample("HG00438", d, force=False)
            self.assertEqual(out, out_path)

    def test_fresh_build_joins_and_writes_parquet(self):
        fake_rows = [{"locus_id": "1-1-2-A", "allele_rank": 0, "eh": 10.0,
                     "genotyping_branch": "quick", "sample_id": "HG00438"}]
        fake_tsv_df = pd.DataFrame({"LocusId": ["1-1-2-A"], "allele_rank": [0], "true": [9.0],
                                   "purity": [1.0], "is_negative_locus": [False]})
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(heldout, "_discover_cov", return_value="30x"), \
                 mock.patch.object(heldout.subprocess, "run",
                                   return_value=SimpleNamespace(stdout="gs://x/json/a.json.gz")), \
                 mock.patch.object(dataset, "_download",
                                   side_effect=lambda remote, dest: [os.path.join(dest, os.path.basename(p))
                                                                     for p in remote]), \
                 mock.patch.object(dataset, "assert_eh_build_matches"), \
                 mock.patch.object(heldout.eh_json, "extract_rows", return_value=fake_rows), \
                 mock.patch.object(dataset, "_load_truth_from_genotypes_tsv", return_value=fake_tsv_df):
                out_path = heldout.build_sample("HG00438", d, force=True)
            merged = pd.read_parquet(out_path)
            self.assertEqual(len(merged), 1)
            self.assertNotIn("sample_id", merged.columns)
            self.assertEqual(merged["true"].iloc[0], 9.0)


class AssertParquetsCarryContractTest(unittest.TestCase):
    """The eval-path guard must enforce the SAME contract as the training path, or a parquet the
    training path would rebuild slips into a benchmark."""

    def _pair(self, d, damage):
        """Writes a contract-clean NEW.parquet and an OLD.parquet damaged the given way."""
        fresh, stale = os.path.join(d, "NEW.parquet"), os.path.join(d, "OLD.parquet")
        dataset_tests.write_contract_parquet(fresh)
        df = dataset_tests.write_contract_parquet(stale)
        if damage == "missing_column":
            df = df.drop(columns=["coverage"])
        elif damage == "float32":
            df["coverage"] = df["coverage"].astype(np.float32)
        else:
            df = df.drop(columns=["has_own_quality_metrics"])
        df.to_parquet(stale, index=False)
        return fresh, stale

    def test_names_only_the_offending_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            fresh, stale = self._pair(d, "missing_column")
            heldout.assert_parquets_carry_contract([fresh])  # must not raise
            with self.assertRaises(RuntimeError) as ctx:
                heldout.assert_parquets_carry_contract([fresh, stale])
            self.assertIn("OLD.parquet", str(ctx.exception))
            self.assertIn("coverage", str(ctx.exception))
            self.assertNotIn("NEW.parquet", str(ctx.exception))

    def test_float32_and_missing_quality_metrics_column_are_caught_too(self):
        # Neither is a missing feature NAME, so a name-only check let both through: float32 died
        # later inside build_matrix, and the missing column silently kept unscoreable alleles.
        for damage in ("float32", "missing_quality_metrics"):
            with tempfile.TemporaryDirectory() as d:
                _, stale = self._pair(d, damage)
                with self.assertRaises(RuntimeError, msg=damage):
                    heldout.assert_parquets_carry_contract([stale])

    def test_error_names_the_command_that_can_actually_rebuild_the_path(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "real_43"))
            os.makedirs(os.path.join(d, "real_quick"))
            sample = os.path.join(d, "real_43", "HG00438.parquet")
            combo = os.path.join(d, "real_quick", "HG002_31x.parquet")
            for path in (sample, combo):
                dataset_tests.write_contract_parquet(path).drop(
                    columns=["coverage"]).to_parquet(path, index=False)
            with self.assertRaises(RuntimeError) as ctx:
                heldout.assert_parquets_carry_contract([sample, combo])
            msg = str(ctx.exception)
            # heldout.py cannot rebuild a training-combo parquet and dataset.py cannot rebuild a
            # held-out sample's, so one blanket command would send the reader to a no-op.
            self.assertIn("heldout.py --build-only --force --samples HG00438", msg)
            self.assertIn("dataset.py --force", msg)


class RunEvalTest(unittest.TestCase):
    def test_writes_metrics_and_violin_npz(self):
        rows = [
            _raw_feature_row(eh=10.0, true=9.0, motif_size=3, genotyping_branch="quick",
                             spanning_at_called=5, locus_id="1-1-2-AAA"),
            _raw_feature_row(eh=10.0, true=8.0, motif_size=3, genotyping_branch="full",
                             spanning_at_called=5, locus_id="1-3-4-AAA"),
            _raw_feature_row(eh=20.0, true=10.0, motif_size=3, genotyping_branch="full",
                             spanning_at_called=0, locus_id="1-5-6-AAA"),
        ]
        with tempfile.TemporaryDirectory() as d:
            parquet_path = os.path.join(d, "S.parquet")
            pd.DataFrame(rows).to_parquet(parquet_path)
            out_json = os.path.join(d, "eval.json")
            # feature_names must match the current contract -- run_eval refuses a model exported
            # under a different one, since the compiled trees index features positionally.
            fake_model = {"feature_names": {b: features.feature_names(b)
                                            for b in (features.BRANCH_QUICK, features.BRANCH_FULL)},
                          "genotyping_regimes": {r: {} for r in features.GENOTYPING_REGIMES}}
            with mock.patch.object(M, "load", return_value=fake_model), \
                 mock.patch.object(M, "compile_genotyping_regime", side_effect=lambda regime_json: regime_json), \
                 mock.patch.object(M, "predict_lcf_json", side_effect=lambda comp, X: np.ones(len(X))), \
                 mock.patch.object(M, "predict_proba_json",
                                   side_effect=lambda comp, X: np.tile([0.9, 0.05, 0.05], (len(X), 1))):
                out = heldout.run_eval([parquet_path], "fake_model.json.gz", out_json, max_alleles=0)

            self.assertEqual(out["n_samples"], 1)
            regimes = out["genotyping_regimes"]
            self.assertEqual(regimes[features.GENOTYPING_REGIME_QUICK]["n"], 1)
            self.assertAlmostEqual(regimes[features.GENOTYPING_REGIME_QUICK]["mae_raw"], 1.0)
            self.assertEqual(regimes[features.GENOTYPING_REGIME_FULL_SPANNING]["n"], 1)
            self.assertAlmostEqual(regimes[features.GENOTYPING_REGIME_FULL_SPANNING]["mae_raw"], 2.0)
            self.assertEqual(regimes[features.GENOTYPING_REGIME_FULL_NONSPANNING]["n"], 1)
            self.assertAlmostEqual(regimes[features.GENOTYPING_REGIME_FULL_NONSPANNING]["mae_raw"], 10.0)
            # gate never applies (p_ok=0.9 fixed): gated MAE == raw MAE everywhere
            self.assertAlmostEqual(regimes[features.GENOTYPING_REGIME_FULL_NONSPANNING]["mae_gated"], 10.0)

            self.assertTrue(os.path.exists(out_json))
            with open(out_json) as f:
                self.assertEqual(json.load(f), out)
            self.assertTrue(os.path.exists(os.path.join(d, "eval_violin.npz")))


if __name__ == "__main__":
    unittest.main()
