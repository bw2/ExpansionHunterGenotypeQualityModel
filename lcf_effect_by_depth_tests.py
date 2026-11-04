"""Tests for lcf_effect_by_depth.py. The model heads are mocked, as in heldout_tests.py."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import features
import heldout
import heldout_tests
import lcf_effect_by_depth as L
import model as M


class BinTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(L.bin_labels(L.DEPTH_BIN_EDGES, whole_numbers=False),
                         ["0-2", "2-3", "3-4", "4-6", "6-8", "8-12", "12-20", "20-40", "40+"])
        self.assertEqual(L.bin_labels(L.SPANNING_TOTAL_BIN_EDGES, whole_numbers=True),
                         ["0", "1", "2", "3", "4-5", "6-9", "10-19", "20-39", "40+"])

    def test_indices_use_lower_edges_and_mark_missing(self):
        np.testing.assert_array_equal(
            L.bin_indices([0, 1.99, 2, 39.9, 40, 1000, np.nan, -1], L.DEPTH_BIN_EDGES),
            [0, 0, 1, 7, 8, 8, -1, -1])


class DistanceReductionIntervalTest(unittest.TestCase):
    def test_one_cluster_has_no_interval(self):
        self.assertTrue(np.isnan(L.distance_reduction_interval([1, 2], [0, 1], ["a", "a"])).all())

    def test_same_reduction_in_every_cluster_gives_a_point(self):
        low, high = L.distance_reduction_interval([2, 4, 6], [1, 2, 3], ["a", "b", "c"], n_resamples=250)
        self.assertAlmostEqual(low, 0.5)
        self.assertAlmostEqual(high, 0.5)

    def test_seeded_and_spans_the_point_estimate(self):
        rng = np.random.default_rng(0)
        before = rng.uniform(1, 5, 300)
        after = before * rng.uniform(0.2, 1.2, 300)
        clusters = np.repeat(np.arange(30), 10)
        first = L.distance_reduction_interval(before, after, clusters)
        self.assertEqual(first, L.distance_reduction_interval(before, after, clusters))
        self.assertLess(first[0], 1 - after.sum() / before.sum())
        self.assertGreater(first[1], 1 - after.sum() / before.sum())


class SummarizeCallsTest(unittest.TestCase):
    def test_measures(self):
        with mock.patch.object(L, "distance_reduction_interval", return_value=(0.1, 0.2)):
            s = L.summarize_calls([2, 2, 0, 4], [0, 3, 0, 4], ["a", "b", "c", "d"])
        self.assertEqual(s, {"calls": 4, "share_closer": 0.25, "share_further": 0.25, "mae_before": 2.0,
                             "mae_after": 1.75, "distance_reduction": 0.125, "interval_low": 0.1,
                             "interval_high": 0.2})


def _gated_rows(**columns):
    """A gated-rows frame with one row per value of the given columns; the rest get defaults."""
    n = len(next(iter(columns.values())))
    defaults = {"data_set": "set", "sample": "s1", "locus_id": ["1-%d-%d-A" % (i, i + 1) for i in range(n)],
                "genotyping_regime": features.GENOTYPING_REGIME_QUICK, "motif_size": 3.0, "depth": 5.0,
                "spanning_total": 5.0, "eh": 10.0, "true": 12.0, "corrected": 11.0}
    defaults.update(columns)
    return pd.DataFrame({c: v if isinstance(v, list) else [v] * n for c, v in defaults.items()})


class SummarizeTest(unittest.TestCase):
    def test_bins_strata_and_clusters(self):
        rows = pd.concat([
            _gated_rows(data_set=["held-out"] * 4, sample=["s1", "s1", "s2", "s2"], depth=[1.0, 2.5, 2.5, 50.0],
                        motif_size=[3.0, 3.0, 3.0, 1.0]),
            _gated_rows(data_set=["HG002"] * 2, sample=["h"] * 2, locus_id=["7-1-2-A", "7-5-6-A"])],
            ignore_index=True)
        clusters_seen = []
        with mock.patch.object(L, "distance_reduction_interval",
                               side_effect=lambda before, after, ids: clusters_seen.append(list(ids)) or (0.0, 0.0)):
            summary = L.summarize(rows)
        depth = summary[(summary["binning"] == "depth") & (summary["motif_stratum"] == "non-homopolymer")]
        self.assertEqual(depth[depth["data_set"] == "held-out"][["bin", "calls"]].values.tolist(),
                         [["all", 3], ["0-2", 1], ["2-3", 2]])
        homopolymer = summary[(summary["data_set"] == "held-out") & (summary["motif_stratum"] == "homopolymer")
                              & (summary["binning"] == "depth")]
        self.assertEqual(homopolymer["bin"].tolist(), ["all", "40+"])
        # The held-out set resamples samples, the single-sample set loci.
        self.assertIn(["s1", "s1", "s2"], clusters_seen)
        self.assertIn(["7-1-2-A", "7-5-6-A"], clusters_seen)
        all_hg002 = summary[(summary["data_set"] == "HG002") & (summary["bin"] == "all")].iloc[0]
        self.assertEqual((all_hg002["mae_before"], all_hg002["mae_after"]), (2.0, 1.0))


class HarmfulBinsTest(unittest.TestCase):
    def test_harm_needs_a_negative_interval_and_share_flips_are_listed_apart(self):
        summary = pd.DataFrame([
            {"bin": "0-2", "calls": 500, "interval_high": -0.01, "share_closer": 0.6, "share_further": 0.3},
            {"bin": "2-3", "calls": 500, "interval_high": 0.1, "share_closer": 0.3, "share_further": 0.4},
            {"bin": "3-4", "calls": 500, "interval_high": 0.1, "share_closer": 0.6, "share_further": 0.3},
            {"bin": "4-6", "calls": 99, "interval_high": -0.5, "share_closer": 0.1, "share_further": 0.9},
            {"bin": "6-8", "calls": 500, "interval_high": -0.2, "share_closer": 0.1, "share_further": 0.9},
            {"bin": "all", "calls": 500, "interval_high": -0.5, "share_closer": 0.1, "share_further": 0.9}])
        self.assertEqual(L.harmful_bins(summary)["bin"].tolist(), ["0-2", "6-8"])
        self.assertEqual(L.bins_moving_more_calls_further(summary)["bin"].tolist(), ["2-3"])


class TablesTest(unittest.TestCase):
    def test_tables_to_score_lists_hg002_chromosome_then_heldout_samples(self):
        with tempfile.TemporaryDirectory() as td:
            heldout_dir, hg002_dir = os.path.join(td, "real_43.eh_b1fbc23"), os.path.join(td, "real_quick.eh_b1fbc23")
            os.makedirs(heldout_dir)
            os.makedirs(hg002_dir)
            for sample in heldout.SAMPLES:
                open(os.path.join(heldout_dir, "%s.parquet" % sample), "w").close()
            for coverage in L.HG002_COVERAGES:
                open(os.path.join(hg002_dir, "HG002_%s.parquet" % coverage), "w").close()
            tables = L.tables_to_score(heldout_dir, hg002_dir)
            self.assertEqual(len(tables), 3 + len(heldout.SAMPLES))
            self.assertEqual(tables[0][:3],
                             ("HG002 10x, chr7", "HG002_10x", os.path.join(hg002_dir, "HG002_10x.parquet")))
            self.assertEqual(tables[0][3], "7")
            self.assertEqual(tables[3][::3], (L.HELDOUT_DATA_SET, None))
            os.remove(os.path.join(heldout_dir, "%s.parquet" % heldout.SAMPLES[5]))
            with self.assertRaises(SystemExit):
                L.tables_to_score(heldout_dir, hg002_dir)

    def test_default_table_dirs_are_the_b1fbc23_copies(self):
        self.assertEqual(os.path.basename(L.DEFAULT_HELDOUT_TABLES_DIR), "real_43.eh_b1fbc23")
        self.assertEqual(os.path.basename(L.DEFAULT_HG002_TABLES_DIR), "real_quick.eh_b1fbc23")
        self.assertEqual(L.TRAINING_DATA_EH_BUILD, "b1fbc23")

    def test_eh_build_check(self):
        with tempfile.TemporaryDirectory() as td:
            def table(name, builds):
                path = os.path.join(td, name)
                with open(path + ".sources.json", "w") as f:
                    json.dump({"eh_build_by_json": builds} if builds is not None else {}, f)
                return path
            right = table("right.parquet", {"gs://a.json.gz": "b1fbc23"})
            L.assert_tables_come_from_eh_build([right], "b1fbc23")
            for wrong in (table("other.parquet", {"gs://a.json.gz": "053dcd8"}),
                          table("mixed.parquet", {"gs://a.json.gz": "b1fbc23", "gs://b.json.gz": "053dcd8"}),
                          table("unrecorded.parquet", None)):
                with self.assertRaises(SystemExit):
                    L.assert_tables_come_from_eh_build([right, wrong], "b1fbc23")


class ReportTest(unittest.TestCase):
    def test_output_prefix_follows_the_model_report_name(self):
        model_path = "/m/genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.20261017.json.gz"
        self.assertEqual(L.output_prefix("/r", model_path, "b1fbc23"), "/r/lcf_effect_by_depth.2026-10-17_plus50_model")
        # Tables from another build never share the training build's saved calls or report.
        self.assertEqual(L.output_prefix("/r", model_path, "e90307c"),
                         "/r/lcf_effect_by_depth_on_eh_e90307c_tables.2026-10-17_plus50_model")

    def test_renders_with_a_bin_below_the_floor_and_a_bin_without_interval(self):
        rng = np.random.default_rng(1)
        n = 400
        rows = _gated_rows(data_set=[L.HELDOUT_DATA_SET] * n, sample=["s%d" % (i % 5) for i in range(n)],
                           depth=list(rng.uniform(0, 50, n)), spanning_total=list(rng.integers(0, 50, n).astype(float)),
                           motif_size=list(rng.choice([1.0, 3.0], n)), eh=list(rng.integers(5, 15, n).astype(float)),
                           corrected=list(rng.integers(5, 15, n).astype(float)))
        # One HG002 locus far off after correction: its bin has one cluster (no interval) and a large negative
        # distance reduction.
        rows = pd.concat([rows, _gated_rows(data_set=[L.HG002_DATA_SETS[0]] * 25, sample=["HG002_10x"] * 25,
                                            locus_id=["7-1-2-A"] * 25, true=12.0, eh=12.5, corrected=40.0)],
                         ignore_index=True)
        summary = L.summarize(rows)
        hg002 = summary[(summary["data_set"] == L.HG002_DATA_SETS[0]) & (summary["bin"] == "4-6")].iloc[0]
        self.assertTrue(np.isnan(hg002["interval_low"]))
        self.assertLess(100 * hg002["distance_reduction"], L.PLOT_FLOOR_PERCENT)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "r.html")
            L.render_html(summary, "m.20261017.json.gz", "b1fbc23", "g.parquet", out)
            with open(out) as f:
                page = f.read()
        self.assertIn("<h2>Verdict</h2>", page)
        self.assertEqual(page.count("<img src='data:image/png;base64,"), 2)
        self.assertIn(L.HG002_DATA_SETS[0], page)


class WriteGatedRowsTest(unittest.TestCase):
    def test_a_stopped_run_resumes_from_the_saved_tables(self):
        tables = [("set", "a", "a.parquet", None), ("set", "b", "b.parquet", None)]
        computed, stop_at_b = [], [True]

        def compute(path, data_set, sample, chromosome, compiled, declared):
            if sample == "b" and stop_at_b[0]:
                raise KeyboardInterrupt("stopped")
            computed.append(sample)
            return _gated_rows(sample=[sample])

        with tempfile.TemporaryDirectory() as td:
            model_path, out_path = os.path.join(td, "m.json.gz"), os.path.join(td, "out", "g.parquet")
            open(model_path, "w").close()
            with mock.patch.object(L, "tables_to_score", return_value=tables), \
                 mock.patch.object(L, "assert_tables_come_from_eh_build"), \
                 mock.patch.object(heldout, "assert_parquets_carry_contract"), \
                 mock.patch.object(heldout, "assert_parquets_supply_features"), \
                 mock.patch.object(M, "load",
                                   return_value={"genotyping_regimes": {r: None for r in features.GENOTYPING_REGIMES}}), \
                 mock.patch.object(M, "feature_names_of", return_value={"quick": ["eh"]}), \
                 mock.patch.object(M, "compile_genotyping_regime", return_value=None), \
                 mock.patch.object(L, "gated_rows_of_table", side_effect=compute):
                with self.assertRaises(KeyboardInterrupt):
                    L.write_gated_rows(model_path, "h", "t", "b1fbc23", out_path)
                self.assertEqual(computed, ["a"])
                stop_at_b[0] = False
                L.write_gated_rows(model_path, "h", "t", "b1fbc23", out_path)
            self.assertEqual(computed, ["a", "b"])  # the rerun computed only b
            self.assertEqual(pd.read_parquet(out_path)["sample"].tolist(), ["a", "b"])
            self.assertEqual(os.listdir(os.path.dirname(out_path)), ["g.parquet"])  # the parts folder is gone


class GatedRowsOfTableTest(unittest.TestCase):
    def test_filters_chromosome_gates_rounded_p_ok_and_corrects(self):
        # The mocked heads read pOk from the coverage column and the LCF from hq_unambiguous_reads.
        rows = [heldout_tests._raw_feature_row(locus_id=locus_id, genotyping_branch=branch, eh=10.0, true=8.0,
                                               coverage=p_ok, hq_unambiguous_reads=lcf)
                for locus_id, branch, p_ok, lcf in (
                    ("7-1-2-AAA", "quick", 0.2, 1.2504),     # gated; LCF rounds to 1.25 -> corrected 8
                    ("7-3-4-AAA", "quick", 0.4996, 1.2),     # pOk rounds to 0.5: not gated
                    ("7-5-6-AAA", "full", 0.4994, 0.9),      # gated, full_spanning; corrected round(11.1) = 11
                    ("1-1-2-AAA", "quick", 0.1, 1.2))]       # another chromosome
        declared = {features.BRANCH_QUICK: features.QUICK_FEATURES, features.BRANCH_FULL: features.FULL_FEATURES}

        def proba(comp, X):
            p = X[:, features.QUICK_FEATURES.index("coverage")]
            return np.column_stack([p, (1 - p) / 2, (1 - p) / 2])

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.parquet")
            pd.DataFrame(rows).to_parquet(path, index=False)
            with mock.patch.object(M, "predict_proba_json", side_effect=proba), \
                 mock.patch.object(M, "predict_lcf_json",
                                   side_effect=lambda comp, X: X[:, features.QUICK_FEATURES.index("hq_unambiguous_reads")]):
                gated = L.gated_rows_of_table(path, "set", "s", "7", {r: None for r in features.GENOTYPING_REGIMES},
                                              declared)
        self.assertEqual(gated["locus_id"].tolist(), ["7-1-2-AAA", "7-5-6-AAA"])
        self.assertEqual(gated["genotyping_regime"].tolist(),
                         [features.GENOTYPING_REGIME_QUICK, features.GENOTYPING_REGIME_FULL_SPANNING])
        self.assertEqual(gated["lcf"].tolist(), [1.25, 0.9])
        self.assertEqual(gated["corrected"].tolist(), [8.0, 11.0])
        self.assertEqual(gated["p_ok"].tolist(), [0.2, 0.499])


if __name__ == "__main__":
    unittest.main()
