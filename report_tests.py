"""Tests for report.py's non-plotting logic: fold partitioning, table/HTML string builders, pill
switchers, and the fully-wired ``render_html`` output.

Plot-rendering functions (matplotlib figures) are mostly not covered here -- they have no meaningful
assertions beyond "a PNG got written", which would just re-test matplotlib itself. The one exception is
``plot_violins``, which has an edge case (a genotyping regime with zero alleles) that previously raised.
"""

import base64
import os
import tempfile
import unittest

import numpy as np

import eh_json
import features
import report as R


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


class MakeFoldsTest(unittest.TestCase):
    def test_partitions_all_chroms_exactly_once(self):
        folds = R.make_folds(n_folds=5, n_calib=2, seed=R.SEED)
        self.assertEqual(len(folds), 5)
        seen_test = []
        for f in folds:
            self.assertEqual(set(), set(f["train"]) & set(f["calib"]))
            self.assertEqual(set(), set(f["train"]) & set(f["test"]))
            self.assertEqual(set(), set(f["calib"]) & set(f["test"]))
            self.assertEqual(len(f["train"]) + len(f["calib"]) + len(f["test"]), len(R.ALL_CHROMS))
            seen_test += f["test"]
        self.assertEqual(sorted(seen_test), sorted(R.ALL_CHROMS))

    def test_deterministic_given_seed(self):
        self.assertEqual(R.make_folds(seed=42), R.make_folds(seed=42))

    def test_different_seeds_differ(self):
        self.assertNotEqual(R.make_folds(seed=1), R.make_folds(seed=2))


class CapRowsTest(unittest.TestCase):
    def test_caps_to_requested_size(self):
        idx = np.arange(100)
        out = R._cap_rows(idx, 10, seed=1)
        self.assertEqual(out.size, 10)
        self.assertTrue(set(out).issubset(set(idx)))
        self.assertTrue(np.array_equal(out, np.sort(out)))  # sorted

    def test_deterministic_given_seed(self):
        idx = np.arange(100)
        np.testing.assert_array_equal(R._cap_rows(idx, 10, seed=7), R._cap_rows(idx, 10, seed=7))

    def test_no_cap_or_cap_above_size_is_a_no_op(self):
        idx = np.arange(5)
        np.testing.assert_array_equal(R._cap_rows(idx, None, seed=1), idx)
        np.testing.assert_array_equal(R._cap_rows(idx, 0, seed=1), idx)
        np.testing.assert_array_equal(R._cap_rows(idx, 100, seed=1), idx)


class ReliabilityPointsTest(unittest.TestCase):
    def test_bins_by_predicted_probability(self):
        y = [0, 0, 1, 1, 1]
        p = [0.05, 0.15, 0.85, 0.95, 0.9]
        xs, ys, ns = R._reliability_points(y, p, n_bins=10)
        # bins: 0 -> [0.05], 1 -> [0.15], 8 -> [0.85], 9 -> [0.95, 0.9]
        np.testing.assert_array_equal(ns, [1, 1, 1, 2])
        self.assertAlmostEqual(xs[-1], (0.95 + 0.9) / 2)
        self.assertAlmostEqual(ys[-1], 1.0)

    def test_empty_input_returns_empty_arrays(self):
        xs, ys, ns = R._reliability_points([], [])
        self.assertEqual(xs.size, 0)


class MotifBinTest(unittest.TestCase):
    def test_bin_edges(self):
        motif = [1, 2, 3, 4, 5, 6, 7, 24, 25, 100, 0, np.nan]
        self.assertEqual(list(R._motif_bin(motif)), [0, 1, 2, 3, 4, 5, 6, 6, 7, 7, -1, -1])


class HoldoutResultsTest(unittest.TestCase):
    def _holdout(self):
        return {"genotyping_regimes": {
            features.GENOTYPING_REGIME_QUICK: {"n": 10, "mae_raw": 1.0, "mae_gated": 0.5,
                                               "homopolymer": {"n": 0}},
            features.GENOTYPING_REGIME_FULL_SPANNING: {"n": 0, "mae_raw": 0.0, "mae_gated": 0.0,
                                                        "homopolymer": {"n": 4, "mae_raw": 2.0, "mae_gated": 1.0}},
            features.GENOTYPING_REGIME_FULL_NONSPANNING: {"n": 5, "mae_raw": 3.0, "mae_gated": 2.0,
                                                           "homopolymer": {"n": 0}},
        }}

    def test_only_regimes_with_n_are_included(self):
        results = R._holdout_results(self._holdout())
        regimes = [r["genotyping_regime"] for r in results]
        self.assertEqual(regimes, [features.GENOTYPING_REGIME_QUICK, features.GENOTYPING_REGIME_FULL_NONSPANNING])

    def test_homopolymer_variant_reads_the_sub_dict(self):
        results = R._holdout_homopolymer_results(self._holdout())
        self.assertEqual([r["genotyping_regime"] for r in results], [features.GENOTYPING_REGIME_FULL_SPANNING])
        self.assertEqual(results[0]["gated"]["mae_raw"], 2.0)


def _fake_results():
    """One metrics bundle per genotyping regime, shaped like ``evaluate_genotyping_regime``'s output."""
    out = []
    for i, regime in enumerate(features.GENOTYPING_REGIMES):
        out.append({
            "genotyping_regime": regime,
            "q": {"n": 100 + i, "mae_eh": 2.0, "mae_true": 1.0, "dist_reduction": 0.5,
                 "median_ae_eh": 1.5, "median_ae_true": 0.5, "eh_exact_match_rate": 0.4,
                 "exact_match_rate": 0.7},
            "direction": {"log_loss": 0.3, "too_long_auc": 0.9, "too_short_auc": 0.85, "ece": 0.02,
                         "p_ok_accuracy": 0.88, "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "importance": [("eh", 0.9, 0.01), ("depth", 0.5, 0.02)],
        })
    return out


class TableBuildersTest(unittest.TestCase):
    def test_q_table_has_one_row_per_regime(self):
        html = R._q_table(_fake_results())
        self.assertEqual(html.count("<tr>"), 1 + len(features.GENOTYPING_REGIMES))
        self.assertIn("quick", html)

    def test_dir_table_has_one_row_per_regime(self):
        html = R._dir_table(_fake_results())
        self.assertEqual(html.count("<tr>"), 1 + len(features.GENOTYPING_REGIMES))

    def test_feature_glossary_ranked_by_full_nonspanning_importance(self):
        html = R._feature_glossary(_fake_results())
        self.assertIn("#1", html)
        self.assertIn("eh", html)
        self.assertLess(html.index(">eh<"), html.index(">depth<"))  # #1 (eh) listed before #2 (depth)

    def test_eh_output_glossary_lists_every_field(self):
        html = R._eh_output_glossary()
        for field, _ in eh_json.EH_OUTPUT_FIELDS:
            self.assertIn(field, html)

    def test_model_outputs_table_lists_all_four_outputs(self):
        html = R._model_outputs_table()
        for name in ("pOk", "pTooLong", "pTooShort", "LCF"):
            self.assertIn(name, html)

    def test_holdout_table_skips_empty_regimes(self):
        holdout = {"genotyping_regimes": {
            features.GENOTYPING_REGIME_QUICK: {"n": 0},
            features.GENOTYPING_REGIME_FULL_SPANNING: {"n": 3, "mae_raw": 1.0, "mae_gated": 0.5,
                                                        "dist_reduction": 0.5, "median_raw": 1.0,
                                                        "median_gated": 0.5, "exact_eh": 0.3,
                                                        "exact_gated": 0.6, "p_ok_accuracy": 0.9},
            features.GENOTYPING_REGIME_FULL_NONSPANNING: {"n": 0},
        }}
        html = R._holdout_table(holdout)
        self.assertEqual(html.count("<tr>"), 2)  # header + the one nonempty regime


class ImgTest(unittest.TestCase):
    def test_embeds_base64_data_uri(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            with open(path, "wb") as f:
                f.write(b"fake-png-bytes")
            html = R._img(path)
            self.assertIn(base64.b64encode(b"fake-png-bytes").decode(), html)
            self.assertTrue(html.startswith('<img src="data:image/png;base64,'))


class ImgToggleTest(unittest.TestCase):
    def test_both_missing_is_empty(self):
        self.assertEqual(R._img_toggle(None, None, "g"), "")

    def test_only_excluded_present_has_no_toggle_markup(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            _write(path, b"a")
            html = R._img_toggle(path, None, "g")
            self.assertNotIn("htgroup", html)

    def test_both_present_renders_toggle(self):
        with tempfile.TemporaryDirectory() as d:
            ex, ho = os.path.join(d, "ex.png"), os.path.join(d, "ho.png")
            _write(ex, b"a")
            _write(ho, b"b")
            html = R._img_toggle(ex, ho, "mygroup")
            self.assertIn("htgroup", html)
            self.assertIn("mygroup", html)


class PillsTest(unittest.TestCase):
    def test_renders_one_radio_per_value_and_one_content_per_present_combo(self):
        dims = [("ds", "Dataset", [("a", "A"), ("b", "B")])]
        contents = {("a",): "<p>content-a</p>", ("b",): "<p>content-b</p>"}
        html = R._pills("g1", dims, contents)
        self.assertIn("content-a", html)
        self.assertIn("content-b", html)
        self.assertEqual(html.count("type='radio'"), 2)

    def test_missing_combo_is_skipped(self):
        dims = [("ds", "Dataset", [("a", "A"), ("b", "B")])]
        html = R._pills("g1", dims, {("a",): "<p>only-a</p>"})
        self.assertIn("only-a", html)
        self.assertNotIn("class='pc c-b'", html)


class DsDimForTest(unittest.TestCase):
    def test_restricts_to_datasets_present_in_contents(self):
        present = [("hg002_genome", "HG002 genome"), ("heldout43", "43 held-out")]
        dim = R._ds_dim_for({("hg002_genome", "nh"): "x"}, present)
        self.assertEqual(dim, ("ds", "Dataset", [("hg002_genome", "HG002 genome")]))


class PlotViolinsTest(unittest.TestCase):
    def test_all_empty_regime_does_not_raise(self):
        # every threshold bin is empty for "quick" -- previously raised ValueError in _violin_ylim
        # because it ran before the zero-fill instead of after.
        violin = {"quick__red": np.zeros(0, dtype=np.float32), "quick__pok": np.zeros(0, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "violins.png")
            R.plot_violins(violin, out_png)  # must not raise
            self.assertTrue(os.path.exists(out_png))


class RenderHtmlTest(unittest.TestCase):
    def test_writes_html_and_skips_missing_optional_sections(self):
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "p.png")
            _write(png, b"x")
            out_html = os.path.join(d, "report.html")
            R.render_html(_fake_results(), png, png, png, "some_model.20260707.json.gz", out_html,
                          confusion_png=png)  # calib/pr/roc/prob_violins left as None (optional)
            with open(out_html) as f:
                html = f.read()
            self.assertTrue(html.startswith("<!doctype html>"))
            self.assertIn("some_model.20260707.json.gz", html)
            self.assertIn("Feature definitions", html)
            self.assertNotIn("Calibration (reliability)", html)  # calib_png=None -> section skipped
            self.assertNotIn("PR-ROC curves", html)               # pr_png=None -> section skipped


if __name__ == "__main__":
    unittest.main()
