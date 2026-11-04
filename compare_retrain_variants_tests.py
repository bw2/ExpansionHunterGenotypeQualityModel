"""Tests for compare_retrain_variants.py. The held-out scoring itself (heldout.run_eval and
heldout.all_loci_early_warning) is mocked; heldout_tests covers it."""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import compare_retrain_variants as C
import features
import heldout

SAMPLES = ["S%02d" % i for i in range(12)]


def fake_variant_locus_result(gated_error_per_allele):
    """A heldout.run_eval-shaped result: every regime's non-homopolymer stratum has 10 alleles per sample with the
    given mean gated error, and no homopolymer alleles."""
    per_sample = {s: {"n": 10, "sum_gated_abs_error": 10 * gated_error_per_allele, "homopolymer_n": 0,
                      "homopolymer_sum_gated_abs_error": 0.0} for s in SAMPLES}
    regime = {"n": 10 * len(SAMPLES), "mae_raw": 2.0, "mae_gated": gated_error_per_allele, "per_sample": per_sample,
              "homopolymer": {"n": 0}, "gate_decision_accuracy": 0.9, "p_ok_accuracy": 0.8, "median_gated": 1.0,
              "helped_lt": 5, "hurt_lt": 1,
              "by_motif_size_category": {"3": {
                  "n": 10 * len(SAMPLES), "mae_raw": 2.0, "mae_gated": gated_error_per_allele,
                  "per_sample": {s: {"n": 10, "sum_gated_abs_error": 10 * gated_error_per_allele} for s in SAMPLES}}}}
    return {"model": "m@0", "genotyping_regimes": {r: regime for r in features.GENOTYPING_REGIMES}}


FAKE_EARLY_WARNING = {"per_sample": {"S00": {"sex": "female"}}, "auc_miss_mean": 0.9, "share_below_half_mean": 0.01,
                      "recall": {"all": {rate: 0.5 for rate in heldout.EARLY_WARNING_FLAG_RATES}}}


class ArgumentParsingTest(unittest.TestCase):
    def test_labeled_models_keep_their_order_and_reject_bad_values(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.json.gz"), os.path.join(d, "b.json.gz")
            for path in (a, b):
                open(path, "w").close()
            self.assertEqual(list(C.parse_labeled_models(["B=%s" % b, "A=%s" % a])), ["B", "A"])
            for bad in (["A%s" % a], ["A=%s" % a, "A=%s" % b], ["A=%s" % os.path.join(d, "missing.json.gz")]):
                with self.assertRaises(ValueError):
                    C.parse_labeled_models(bad)

    def test_report_is_dated_by_the_newest_model_file(self):
        self.assertEqual(C.report_file_name(["/m/x_plus50.20261017.json.gz", "/m/y_relative_tolerance_10pct.20261025.json.gz",
                                             "/m/z.20261024.json.gz"]), "retrain_variants.2026-10-25_model.html")
        self.assertEqual(C.report_file_name(["/m/undated.json.gz"]), "retrain_variants.html")

    def test_pairs_default_to_every_model_against_the_first(self):
        self.assertEqual(C.parse_pairs(None, ["deployed", "A", "B"]), [("A", "deployed"), ("B", "deployed")])
        self.assertEqual(C.parse_pairs(["B:A"], ["deployed", "A", "B"]), [("B", "A")])
        for bad in (["B-A"], ["C:A"], ["A:A"]):
            with self.assertRaises(ValueError):
                C.parse_pairs(bad, ["deployed", "A", "B"])


class ScoredResultsTest(unittest.TestCase):
    def _score(self, out_dir, model_path, paths):
        with mock.patch.object(heldout, "run_eval", return_value=fake_variant_locus_result(1.0)) as run_eval, \
             mock.patch.object(heldout, "all_loci_early_warning", return_value=FAKE_EARLY_WARNING):
            results = C.scored_results("A + B 10%", model_path, paths, out_dir, 30000)
        return results, run_eval.call_count

    @staticmethod
    def _parquets(d):
        """Empty stand-ins for the held-out parquets, written an hour ago."""
        paths = [os.path.join(d, "%s.parquet" % s) for s in SAMPLES]
        for path in paths:
            open(path, "w").close()
            os.utime(path, (os.path.getmtime(path) - 3600,) * 2)
        return paths

    def test_a_model_is_scored_once_and_scored_again_only_when_its_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = os.path.join(d, "m.json.gz")
            with open(model_path, "w") as f:
                f.write("first")
            paths = self._parquets(d)
            results, calls = self._score(d, model_path, paths)
            self.assertEqual(calls, 1)
            self.assertTrue(os.path.exists(os.path.join(d, "A_B_10.results.json")))
            # Saved JSON has string keys; the flag rates come back as floats, so they index like a fresh result.
            self.assertEqual(results["early_warning"]["recall"]["all"][heldout.EARLY_WARNING_FLAG_RATES[0]], 0.5)
            self.assertEqual(self._score(d, model_path, paths)[1], 0)
            with open(model_path, "w") as f:
                f.write("second")
            self.assertEqual(self._score(d, model_path, paths)[1], 1)
            self.assertEqual(self._score(d, model_path, paths[:-1])[1], 1)   # another sample set

    def test_results_of_other_scoring_code_are_stale_even_when_newer_than_the_edit(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = os.path.join(d, "m.json.gz")
            open(model_path, "w").close()
            paths = self._parquets(d)
            with mock.patch.object(C, "SCORING_CODE_HASH", "code as imported by run 1"):
                self.assertEqual(self._score(d, model_path, paths)[1], 1)
                self.assertEqual(self._score(d, model_path, paths)[1], 0)
            # A later run imports edited scoring code; the saved results are newer than any file but still stale.
            with mock.patch.object(C, "SCORING_CODE_HASH", "edited code"):
                self.assertEqual(self._score(d, model_path, paths)[1], 1)

    def test_results_saved_without_a_code_hash_are_judged_by_the_scoring_modules_file_times(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = os.path.join(d, "m.json.gz")
            open(model_path, "w").close()
            paths = self._parquets(d)
            self.assertEqual(self._score(d, model_path, paths)[1], 1)
            results_path = os.path.join(d, "A_B_10.results.json")
            with open(results_path) as f:
                saved = json.load(f)
            del saved["scoring_code_hash"]
            with open(results_path, "w") as f:
                json.dump(saved, f)
            scoring_module = os.path.join(d, "scoring_module.py")
            open(scoring_module, "w").close()
            os.utime(scoring_module, (os.path.getmtime(results_path) - 3600,) * 2)
            with mock.patch.object(C, "SCORING_MODULES", (types.SimpleNamespace(__file__=scoring_module),)):
                self.assertEqual(self._score(d, model_path, paths)[1], 0)
                os.utime(scoring_module, (os.path.getmtime(results_path) + 60,) * 2)   # edited after they were saved
                self.assertEqual(self._score(d, model_path, paths)[1], 1)

    def test_the_scoring_code_hash_follows_the_module_sources(self):
        with tempfile.TemporaryDirectory() as d:
            source = os.path.join(d, "scoring_module.py")
            with open(source, "w") as f:
                f.write("x = 1\n")
            module = types.SimpleNamespace(__file__=source)
            first = C.scoring_code_hash([module])
            self.assertEqual(first, C.scoring_code_hash([module]))
            with open(source, "w") as f:
                f.write("x = 2\n")
            self.assertNotEqual(first, C.scoring_code_hash([module]))

    def test_a_rebuilt_held_out_parquet_makes_saved_results_stale(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = os.path.join(d, "m.json.gz")
            open(model_path, "w").close()
            paths = self._parquets(d)
            self.assertEqual(self._score(d, model_path, paths)[1], 1)
            self.assertEqual(self._score(d, model_path, paths)[1], 0)
            results_mtime = os.path.getmtime(os.path.join(d, "A_B_10.results.json"))
            os.utime(paths[3], (results_mtime + 60,) * 2)   # rebuilt after the results were saved
            self.assertEqual(self._score(d, model_path, paths)[1], 1)

    def test_an_interrupted_scoring_leaves_no_results_file(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = os.path.join(d, "m.json.gz")
            open(model_path, "w").close()
            with mock.patch.object(heldout, "run_eval", return_value=fake_variant_locus_result(1.0)), \
                 mock.patch.object(heldout, "all_loci_early_warning", side_effect=KeyboardInterrupt), \
                 self.assertRaises(KeyboardInterrupt):
                C.scored_results("A", model_path, [os.path.join(d, "S00.parquet")], d, 30000)
            self.assertFalse(os.path.exists(os.path.join(d, "A.results.json")))


class ReportHtmlTest(unittest.TestCase):
    def test_page_lists_each_model_marks_the_best_and_states_each_pair_verdict(self):
        results = {}
        for label, error in (("deployed", 1.0), ("A", 0.5)):
            saved = {"model_path": "/m/%s.json.gz" % label, "fingerprint": "%s.json.gz@abc" % label,
                     "max_alleles_per_sample": 30000, "samples": SAMPLES,
                     "variant_locus": fake_variant_locus_result(error), "early_warning": FAKE_EARLY_WARNING}
            results[label] = json.loads(json.dumps(saved, default=float))
            results[label]["early_warning"] = C._early_warning_from_json(results[label]["early_warning"])
        comparison = C.compare_models.paired_gated_mae_comparison(
            results["deployed"]["variant_locus"], results["A"]["variant_locus"], SAMPLES)
        page = C.report_html(results, [("A", "deployed")], {("A", "deployed"): comparison})
        self.assertIn("deployed.json.gz", page)
        self.assertIn("<b>0.5000</b>", page)
        self.assertIn("A vs deployed: NEW is better", page)
        # Once in the models table, then as the winner of each stratum.
        self.assertEqual(page.count("<td>A</td>"), 1 + len(features.GENOTYPING_REGIMES))
        self.assertTrue(np.all([row["winner"] == "new" for row in comparison]))
        # The motif-size category table, pooled over alleles: one row per regime for the 3 bp category.
        self.assertIn("gated MAE by motif size category", page)
        self.assertEqual(page.count("<td>3</td>"), len(features.GENOTYPING_REGIMES))
        by_category = C.compare_models.paired_gated_mae_by_motif_size_category(
            results["deployed"]["variant_locus"], results["A"]["variant_locus"], SAMPLES)
        page = C.report_html(results, [("A", "deployed")], {("A", "deployed"): comparison},
                             {("A", "deployed"): by_category})
        self.assertIn("A vs deployed by motif size category", page)
        self.assertEqual(page.count("<td>3</td>"), 2 * len(features.GENOTYPING_REGIMES))


if __name__ == "__main__":
    unittest.main()
