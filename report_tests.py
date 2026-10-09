"""Tests for report.py's non-plotting logic: fold partitioning, table/HTML string builders, pill
switchers, and the fully-wired ``render_html`` output.

Plot-rendering functions (matplotlib figures) are mostly not covered here -- they have no meaningful
assertions beyond "a PNG got written", which would just re-test matplotlib itself. The one exception is
``plot_violins``, which has an edge case (a genotyping regime with zero alleles) that previously raised.
"""

import base64
import io
import os
import tempfile
import unittest

import numpy as np

import accuracy_by_size as ABS
import eh_json
import features
import report as R


def _write_png(path):
    """Writes a tiny real PNG; ``_img`` decodes + re-encodes it, so a fake-bytes file no longer works."""
    from PIL import Image
    Image.new("RGB", (4, 3), (10, 20, 30)).save(path, format="PNG")


class MakeFoldsTest(unittest.TestCase):
    def test_partitions_all_chroms_exactly_once(self):
        folds = R.make_folds(n_folds=5, n_calib=2, seed=R.SEED)
        self.assertEqual(len(folds), 5)
        seen_test = []
        for f in folds:
            groups = [set(f[k]) for k in ("train", "calib", "test")]
            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    self.assertEqual(set(), groups[a] & groups[b])
            self.assertEqual(len(f["calib"]), 2)
            self.assertEqual(sum(len(g) for g in groups), len(R.ALL_CHROMS))
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
            "importance_direction": [("depth", 0.7, 0.03), ("eh", 0.2, 0.01)],
            "ablation": [{"k": 0, "feature": "(raw EH)", "mae": 2.0},
                         {"k": 1, "feature": "eh", "mae": 1.4}],
            "dir_ablation": [{"k": 0, "feature": "(class prior)", "log_loss": 1.0},
                             {"k": 1, "feature": "depth", "log_loss": 0.6}],
        })
    return out


class TableBuildersTest(unittest.TestCase):
    def test_feature_glossary_ranked_by_full_nonspanning_importance(self):
        html = R._feature_glossary(_fake_results())
        self.assertIn("#1", html)
        self.assertIn("eh", html)
        self.assertLess(html.index(">eh<"), html.index(">depth<"))  # #1 (eh) listed before #2 (depth)

    def test_feature_glossary_lists_every_current_feature_even_if_the_cache_is_short(self):
        # A ranking cached before a feature was added used to silently omit that feature's row --
        # the published report shipped 24 rows while the contract already had 29.
        html = R._feature_glossary(_fake_results())   # its ranking covers only eh + depth
        for feat in features.FULL_FEATURES:
            self.assertIn(">%s<" % feat, html, feat)
        self.assertIn("#-", html)                     # unranked features are marked, not dropped
        self.assertLess(html.index(">eh<"), html.index(">coverage<"))  # ranked ones still sort first

    def test_stale_cache_warning_names_the_missing_features(self):
        note = R._stale_contract_warning(_fake_results())
        self.assertIn("Stale cache", note)
        self.assertIn("coverage", note)

    def test_no_stale_warning_when_the_ranking_covers_the_contract(self):
        full = [(f, 1.0, 0.0) for f in features.FULL_FEATURES]
        quick = [(f, 1.0, 0.0) for f in features.QUICK_FEATURES]
        results = [dict(r, importance=(quick if r["genotyping_regime"] == features.GENOTYPING_REGIME_QUICK
                                       else full))
                   for r in _fake_results()]
        self.assertEqual(R._stale_contract_warning(results), "")

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
            _write_png(path)
            html = R._img(path)
            self.assertTrue(html.startswith('<img src="data:image/png;base64,'))
            embedded = base64.b64decode(html.split("base64,", 1)[1].split('"', 1)[0])
            self.assertTrue(embedded.startswith(b"\x89PNG\r\n\x1a\n"))  # re-encoded to a valid PNG

    def test_fixed_palette_keeps_every_category_color_exactly(self):
        # One 1-pixel column per accuracy-by-size category plus a column of the plot's gray text color:
        # the categories come back unchanged, and the gray maps to a nearby gray, not a category color.
        from PIL import Image
        cats = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in ABS.CATEGORY_COLORS.values()]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            src = Image.new("RGB", (len(cats) + 1, 1))
            src.putdata(cats + [(0x77, 0x77, 0x77)])
            src.save(path, format="PNG")
            html = R._img(path, palette=R._fixed_palette(ABS.CATEGORY_COLORS.values()))
            out = list(Image.open(io.BytesIO(base64.b64decode(html.split("base64,", 1)[1].split('"', 1)[0])))
                       .convert("RGB").getdata())
        self.assertEqual(out[:-1], cats)
        r, g, b = out[-1]
        self.assertTrue(r == g == b and abs(r - 0x77) <= 8)


class ReportHtmlNameTest(unittest.TestCase):
    def test_dated_model_gets_its_date(self):
        self.assertEqual(
            R.report_html_name("/m/genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.20261008.json.gz"),
            "model_report.2026-10-08_model.html")

    def test_model_name_not_ending_in_a_date_is_used_whole(self):
        self.assertEqual(R.report_html_name("model/q_plus50.20261006.quick500.json.gz"),
                         "model_report.q_plus50.20261006.quick500_model.html")

    def test_no_model_keeps_the_plain_name(self):
        self.assertEqual(R.report_html_name(""), "model_report.html")


class StackedHasEveryPillOptionTest(unittest.TestCase):
    @staticmethod
    def _stacked(pok_keys):
        return {h: {vk: {pk: {kk: {} for kk in pok_keys} for pk, _, _ in ABS.PURITY_VARIANTS}
                    for vk, _, _, _ in ABS.CORRECTION_VARIANTS} for h in ("nonhomo", "homo")}

    def test_current_layout_passes(self):
        self.assertTrue(R._stacked_has_every_pill_option(self._stacked([k for k, _, _ in ABS.POK_VARIANTS])))

    def test_json_written_before_the_lower_pok_thresholds_is_out_of_date(self):
        self.assertFalse(R._stacked_has_every_pill_option(self._stacked(["all", "lt050", "ge050"])))


class ImgToggleTest(unittest.TestCase):
    def test_both_missing_is_empty(self):
        self.assertEqual(R._img_toggle(None, None, "g"), "")

    def test_only_excluded_present_has_no_toggle_markup(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            _write_png(path)
            html = R._img_toggle(path, None, "g")
            self.assertNotIn("htgroup", html)

    def test_both_present_renders_toggle(self):
        with tempfile.TemporaryDirectory() as d:
            ex, ho = os.path.join(d, "ex.png"), os.path.join(d, "ho.png")
            _write_png(ex)
            _write_png(ho)
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
        present = [("hg002_genome", "HG002 genome"), ("heldout_hprc", "held-out HPRC")]
        dim = R._ds_dim_for({("hg002_genome", "nh"): "x"}, present)
        self.assertEqual(dim, ("ds", "Dataset", [("hg002_genome", "HG002 genome")]))


class ViolinsSkippingEmptyBinsTest(unittest.TestCase):
    def test_empty_bins_get_no_violin_and_keep_their_slot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            data = [np.array([1.0, 2.0]), np.array([]), np.array([-1.0, 0.5, 3.0])]
            vp = R._violins_skipping_empty_bins(ax, data, ["#111111", "#222222", "#333333"], 0.5)
            self.assertEqual(len(vp["bodies"]), 2)
            # Medians sit at x positions 1 and 3; nothing is drawn at the empty bin's position 2.
            xs = sorted(seg[0][0] for seg in vp["cmedians"].get_segments())
            np.testing.assert_allclose(xs, [1 - 0.85 / 4, 3 - 0.85 / 4])
            self.assertIsNone(R._violins_skipping_empty_bins(ax, [np.array([])], ["#111111"], 0.5))
        finally:
            plt.close(fig)


class PlotViolinsTest(unittest.TestCase):
    def test_all_empty_regime_does_not_raise(self):
        # every threshold bin is empty for "quick" -- previously raised ValueError in _violin_ylim
        # because it ran before the zero-fill instead of after.
        violin = {"quick__red": np.zeros(0, dtype=np.float32), "quick__pok": np.zeros(0, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "violins.png")
            R.plot_violins(violin, out_png)  # must not raise
            self.assertTrue(os.path.exists(out_png))


class DirAblationTest(unittest.TestCase):
    """The direction-head ablation's pure pieces (the fitting loop itself needs a real pool)."""

    def test_scores_drop_the_non_plottable_entries(self):
        y = np.array([0, 0, 1, 2, 1, 0, 2, 0])
        proba = np.full((y.size, 3), 1.0 / 3)
        scored = R._dir_ablation_scores(y, proba)
        self.assertIn("log_loss", scored)
        self.assertIn("too_short_ap", scored)
        # "n" and the 3x3 confusion matrix are per-k noise in a curve, not something to plot vs k.
        self.assertNotIn("confusion", scored)
        self.assertNotIn("n", scored)

    def test_uniform_proba_log_loss_is_log_three(self):
        y = np.array([0, 1, 2])
        scored = R._dir_ablation_scores(y, np.full((3, 3), 1.0 / 3))
        self.assertAlmostEqual(scored["log_loss"], float(np.log(3)), places=6)

    def test_plot_returns_false_and_writes_nothing_without_a_curve(self):
        cached_without_curve = [{"genotyping_regime": r} for r in features.GENOTYPING_REGIMES]
        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "dir_ablation.png")
            self.assertFalse(R.plot_dir_ablation(cached_without_curve, out_png))
            self.assertFalse(os.path.exists(out_png))

    def test_plot_draws_when_every_regime_has_a_curve(self):
        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "dir_ablation.png")
            self.assertTrue(R.plot_dir_ablation(_fake_results(), out_png))
            self.assertTrue(os.path.exists(out_png))


class DirImportanceTest(unittest.TestCase):
    """Direct coverage of the in-place column shuffle in ``_dir_importance``."""

    def _fitted_two_signals(self):
        """Fits a head where ``depth`` and ``coverage`` BOTH carry signal, ``noise`` none.

        A column that is shuffled and not restored would corrupt every feature measured after it, so
        the later informative feature would score near zero. One informative feature cannot show
        that; two, with the noise column last, can.
        """
        import pandas as pd
        import model as M
        rng = np.random.default_rng(1)
        n = 4000
        X = pd.DataFrame({"depth": rng.normal(size=n), "coverage": rng.normal(size=n),
                          "noise": rng.normal(size=n)})
        signal = X["depth"] + X["coverage"]
        y = np.where(signal > 0.7, features.TOO_LONG,
                     np.where(signal < -0.7, features.TOO_SHORT, features.OK)).astype(int)
        half = n // 2
        dmodel = M.train_direction(X.iloc[:half], y[:half], X.iloc[half:], y[half:])
        return dmodel, X.iloc[half:].reset_index(drop=True), y[half:]

    def test_a_shuffled_column_is_restored_before_the_next_feature_is_measured(self):
        dmodel, X, y = self._fitted_two_signals()
        ranked = dict((f, m) for f, m, _ in R._dir_importance(dmodel, X, y, list(X.columns)))
        # Both informative features must score well above the noise one. If the shuffle of an
        # earlier column leaked, whichever informative column came after it would collapse.
        self.assertGreater(ranked["depth"], 10 * abs(ranked["noise"]) + 0.01)
        self.assertGreater(ranked["coverage"], 10 * abs(ranked["noise"]) + 0.01)

    def _fitted(self):
        """Fits a small direction head where only ``depth`` carries signal.

        ``model._GBM_KWARGS`` sets ``min_samples_leaf=300``, so a fixture of a few hundred rows
        cannot split at all and fits a CONSTANT predictor -- against which permuting any feature
        is a genuine no-op and the test would assert nothing. 4000 rows (2000 train) is the
        smallest size tried that actually learns the rule.
        """
        import pandas as pd
        import model as M
        rng = np.random.default_rng(0)
        n = 4000
        X = pd.DataFrame({"depth": rng.normal(size=n), "noise": rng.normal(size=n)})
        y = np.where(X["depth"] > 0.4, features.TOO_LONG,
                     np.where(X["depth"] < -0.4, features.TOO_SHORT, features.OK)).astype(int)
        half = n // 2
        dmodel = M.train_direction(X.iloc[:half], y[:half], X.iloc[half:], y[half:])
        return dmodel, X.iloc[half:].reset_index(drop=True), y[half:]

    def test_ranks_the_informative_feature_first_and_leaves_the_caller_frame_untouched(self):
        dmodel, X, y = self._fitted()
        before = X.copy()
        ranked = R._dir_importance(dmodel, X, y, list(X.columns))
        self.assertEqual([f for f, _, _ in ranked], ["depth", "noise"])
        self.assertGreater(ranked[0][1], 0.0)  # permuting the signal raises log-loss
        # The permutation is applied in place on an internal copy; the caller's frame and dtypes
        # must come back exactly as they went in.
        self.assertTrue(before.equals(X))
        self.assertEqual(list(before.dtypes), list(X.dtypes))

    def test_is_deterministic(self):
        dmodel, X, y = self._fitted()
        first = R._dir_importance(dmodel, X, y, list(X.columns))
        second = R._dir_importance(dmodel, X, y, list(X.columns))
        self.assertEqual(first, second)


class ImportancePanelTest(unittest.TestCase):
    def test_missing_ranking_returns_false_and_writes_nothing(self):
        # a results.json cached before the direction ranking existed must skip the panel, not crash
        stale = [{"genotyping_regime": r, "importance": [("eh", 1.0, 0.0)]}
                 for r in features.GENOTYPING_REGIMES]
        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "imp.png")
            self.assertFalse(R.plot_importance_panel(stale, out_png, key="importance_direction"))
            self.assertFalse(os.path.exists(out_png))

    def test_panel_rows_read_the_requested_ranking(self):
        # _fake_results gives the two keys OPPOSITE orders (importance: eh > depth,
        # importance_direction: depth > eh), so the first row identifies which one was read.
        # Asserting on _panel_rows rather than on "a PNG appeared" is what makes this catch a panel
        # that plots the q-head bars under a direction-head title.
        results = _fake_results()
        for key, expected_first in (("importance", "eh"), ("importance_direction", "depth")):
            rank, by_reg = R._panel_rows(results, key, top_n=R.TOP_N)
            self.assertEqual(rank[expected_first], 1, key)
            for regime, ranked in by_reg.items():
                self.assertEqual(ranked[0][0], expected_first, "%s / %s" % (key, regime))

    def test_panel_rows_returns_nothing_when_a_regime_lacks_the_ranking(self):
        partial = _fake_results()
        del partial[1]["importance_direction"]
        self.assertEqual(R._panel_rows(partial, "importance_direction", top_n=5), ({}, {}))

    def test_direction_ranking_orders_bars_by_its_own_key(self):
        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "imp.png")
            self.assertTrue(R.plot_importance_panel(_fake_results(), out_png,
                                                    key="importance_direction"))
            self.assertTrue(os.path.exists(out_png))


class RenderHtmlTest(unittest.TestCase):
    def test_writes_html_and_skips_missing_optional_sections(self):
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "p.png")
            _write_png(png)
            out_html = os.path.join(d, "report.html")
            R.render_html(_fake_results(), png, png, png, "some_model.20260707.json.gz", out_html,
                          confusion_png=png)  # pr/roc/prob_violins left as None (optional)
            with open(out_html) as f:
                html = f.read()
            self.assertTrue(html.startswith("<!doctype html>"))
            self.assertIn("some_model.20260707.json.gz", html)
            self.assertIn("Feature definitions", html)
            self.assertNotIn("PR-ROC curves", html)               # pr_png=None -> section skipped
            # the direction-head panels are optional too
            self.assertNotIn("Add-one-feature ablation (direction prediction)", html)

    def test_direction_ablation_sections_render_when_their_pngs_are_supplied(self):
        with tempfile.TemporaryDirectory() as d:
            png = os.path.join(d, "p.png")
            _write_png(png)
            out_html = os.path.join(d, "report.html")
            R.render_html(_fake_results(), png, png, png, "m.json.gz", out_html,
                          dir_importance_png=png, dir_ablation_png=png)
            with open(out_html) as f:
                html = f.read()
            self.assertIn("Add-one-feature ablation (direction prediction)", html)
            self.assertIn("Relative feature importance (per allele size bucket, direction prediction)",
                          html)


if __name__ == "__main__":
    unittest.main()
