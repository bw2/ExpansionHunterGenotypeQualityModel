"""Tests for build_per_source_parquets_with_homopolymer_handoff_fix.py, on small hand-made per-source tables."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import build_per_source_parquets_with_homopolymer_handoff_fix as H
import dataset

INDIVIDUAL = "P"


def locus_id_with_hash_fraction(below, prefix):
    """A locus id whose hash fraction for INDIVIDUAL is below (or not below) the lowest fast-path keep rate."""
    for i in range(10000):
        locus_id = "%s%d" % (prefix, i)
        if (dataset._locus_hash_fraction(locus_id, INDIVIDUAL) < H.LOWEST_FAST_PATH_KEEP_RATE) == below:
            return locus_id
    raise AssertionError("no such locus id")


def rows(locus_id, motif_size, branch, variant, eh, n_alleles=2):
    return [{"locus_id": locus_id, "allele_rank": rank, "motif_size": motif_size, "genotyping_branch": branch,
             "variant_locus": variant, "eh": eh + rank, "fragment_length": 500} for rank in range(n_alleles)]


DROPPED = locus_id_with_hash_fraction(False, "dropped")
UNCALLED = locus_id_with_hash_fraction(True, "uncalled")
CURRENT = pd.DataFrame(
    rows("variant_moved", 1, "full", True, 250) + rows("homref_moved", 1, "full", False, 260)
    + rows(DROPPED, 1, "full", False, 270) + rows(UNCALLED, 1, "full", False, 15)
    + rows("full_in_both", 1, "full", True, 30) + rows("other_motif_moved", 3, "full", True, 40)
    + rows("quick_now_full_before", 1, "quick", True, 12) + rows("quick_in_both", 1, "quick", False, 20))
OLD = pd.DataFrame(
    rows("quick_in_both", 1, "quick", False, 20) + rows("other_motif_moved", 3, "quick", True, 41)
    + rows("homref_moved", 1, "quick", False, 20) + rows("variant_moved", 1, "quick", True, 18)
    + rows("old_only", 1, "quick", False, 22)
    + rows("full_in_both", 1, "full", True, 31) + rows("quick_now_full_before", 1, "full", True, 13))


class BranchOfEachLocusTest(unittest.TestCase):
    def test_a_locus_on_both_branches_is_an_error(self):
        df = pd.DataFrame(rows("A", 1, "quick", True, 10, n_alleles=1) + rows("A", 1, "full", True, 10, n_alleles=1))
        with self.assertRaises(ValueError):
            H.branch_of_each_locus(df)


class HomopolymerLociToReplaceTest(unittest.TestCase):
    def setUp(self):
        self.to_replace, self.counts = H.homopolymer_loci_to_replace(CURRENT, OLD, INDIVIDUAL)

    def test_homopolymers_the_old_run_genotyped_on_the_fast_path_or_left_out_are_replaced(self):
        self.assertEqual(self.to_replace, {"variant_moved", "homref_moved", "old_only", DROPPED, UNCALLED})

    def test_every_case_is_counted(self):
        self.assertEqual(list(self.counts.values()), [1, 1, 1, 2, 1, 1, 1])
        self.assertEqual(self.counts[H.COUNT_UNCALLED_IN_B1FBC23], 1)

    def test_a_homopolymer_variant_locus_missing_from_either_parquet_is_an_error(self):
        with self.assertRaises(ValueError):
            H.homopolymer_loci_to_replace(CURRENT, OLD[OLD["locus_id"] != "variant_moved"], INDIVIDUAL)
        with self.assertRaises(ValueError):
            H.homopolymer_loci_to_replace(CURRENT[CURRENT["locus_id"] != "variant_moved"], OLD, INDIVIDUAL)


class RowsWithReplacedLociTest(unittest.TestCase):
    def test_replaced_loci_take_the_old_rows_in_the_current_order_and_added_loci_come_last(self):
        to_replace, _ = H.homopolymer_loci_to_replace(CURRENT, OLD, INDIVIDUAL)
        out = H.rows_with_replaced_loci(CURRENT, OLD, to_replace)
        self.assertEqual(list(out["locus_id"].drop_duplicates()),
                         ["variant_moved", "homref_moved", "full_in_both", "other_motif_moved",
                          "quick_now_full_before", "quick_in_both", "old_only"])
        self.assertEqual(list(out.loc[out["locus_id"] == "variant_moved", "eh"]), [18, 19])
        self.assertEqual(list(out.loc[out["locus_id"] == "old_only", "allele_rank"]), [0, 1])
        from_old = ["variant_moved", "homref_moved", "old_only"]
        self.assertEqual(set(out.loc[out["locus_id"].isin(from_old), "genotyping_branch"]), {"quick"})
        kept = out[~out["locus_id"].isin(from_old)].reset_index(drop=True)
        expected = CURRENT[~CURRENT["locus_id"].isin(["variant_moved", "homref_moved", DROPPED, UNCALLED])
                           ].reset_index(drop=True)
        pd.testing.assert_frame_equal(kept, expected)
        self.assertEqual(list(out.columns), list(CURRENT.columns))


class DifferencesAtHomopolymerLociOnTheFastPathInBothTest(unittest.TestCase):
    def test_identical_rows_and_a_changed_column(self):
        self.assertEqual(H.differences_at_homopolymer_loci_on_the_fast_path_in_both(CURRENT, OLD), (2, 0, {}))
        changed = OLD.copy()
        changed.loc[changed["locus_id"] == "quick_in_both", "fragment_length"] = np.nan
        self.assertEqual(H.differences_at_homopolymer_loci_on_the_fast_path_in_both(CURRENT, changed),
                         (2, 2, {"fragment_length": 2}))


    def test_replaced_variant_loci_against_the_b1fbc23_table(self):
        with tempfile.TemporaryDirectory() as d:
            # The old tables hold variant loci only, and fewer columns.
            table_path = os.path.join(d, "b1fbc23.parquet")
            OLD[OLD["variant_locus"]].drop(columns=["fragment_length"]).to_parquet(table_path, index=False)
            to_replace, _ = H.homopolymer_loci_to_replace(CURRENT, OLD, INDIVIDUAL)
            out = H.rows_with_replaced_loci(CURRENT, OLD, to_replace)
            self.assertEqual(H.differences_from_the_b1fbc23_tables(out, {"variant_moved"}, table_path), (2, 0, {}))
            changed = out.copy()
            changed.loc[changed["locus_id"] == "variant_moved", "eh"] += 1
            self.assertEqual(H.differences_from_the_b1fbc23_tables(changed, {"variant_moved"}, table_path),
                             (2, 2, {"eh": 2}))


def write_with_record(df, path, eh_build):
    df.to_parquet(path, index=False)
    with open(dataset._sources_record_path(path), "w") as f:
        json.dump({"cloud_versions": {"gs://run/%s.json.gz" % eh_build: ["md5", 1.0]},
                   "eh_build_by_json": {"gs://run/%s.json.gz" % eh_build: eh_build}}, f)


class BuildOneSourceTest(unittest.TestCase):
    def test_writes_the_parquet_and_summary_and_reuses_them_while_the_inputs_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            current_path, old_path = os.path.join(d, "current.parquet"), os.path.join(d, "old.parquet")
            out_path, table_path = os.path.join(d, "out", "P.parquet"), os.path.join(d, "b1fbc23.parquet")
            write_with_record(CURRENT, current_path, "e90307c")
            write_with_record(OLD, old_path, "b1fbc23")
            OLD[OLD["variant_locus"]].to_parquet(table_path, index=False)
            summary = H.build_one_source(current_path, old_path, out_path, INDIVIDUAL, table_path)
            self.assertEqual(summary["rows"], {"current": len(CURRENT), "out": len(CURRENT) - 2})
            self.assertEqual((summary["replaced variant-locus rows compared with the b1fbc23 table"],
                              summary["replaced variant-locus rows that differ from the b1fbc23 table"]), (2, 0))
            with open(out_path + H.SUMMARY_SUFFIX) as f:
                self.assertEqual(json.load(f), summary)
            with open(dataset._sources_record_path(out_path)) as f:
                record = json.load(f)
            self.assertEqual(record["eh_build_by_json"], {"gs://run/e90307c.json.gz": "e90307c"})
            self.assertEqual(record["homopolymer_handoff_fix"]["record of the b1fbc23 extraction"]["eh_build_by_json"],
                             {"gs://run/b1fbc23.json.gz": "b1fbc23"})
            self.assertEqual(len(pd.read_parquet(out_path)), len(CURRENT) - 2)
            written = os.path.getmtime(out_path)
            H.build_one_source(current_path, old_path, out_path, INDIVIDUAL, table_path)
            self.assertEqual(os.path.getmtime(out_path), written)
            os.utime(old_path, (written + 10, written + 10))
            H.build_one_source(current_path, old_path, out_path, INDIVIDUAL, table_path)
            rewritten = os.path.getmtime(out_path)
            self.assertNotEqual(rewritten, written)
            with mock.patch.object(H, "_code_signature", return_value={"changed": "code"}):
                H.build_one_source(current_path, old_path, out_path, INDIVIDUAL, table_path)
            self.assertNotEqual(os.path.getmtime(out_path), rewritten)

    def test_different_column_types_are_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            current_path, old_path = os.path.join(d, "current.parquet"), os.path.join(d, "old.parquet")
            write_with_record(CURRENT, current_path, "e90307c")
            write_with_record(OLD.astype({"fragment_length": float}), old_path, "b1fbc23")
            OLD.to_parquet(os.path.join(d, "b1fbc23.parquet"), index=False)
            with self.assertRaises(ValueError):
                H.build_one_source(current_path, old_path, os.path.join(d, "out.parquet"), INDIVIDUAL,
                                   os.path.join(d, "b1fbc23.parquet"))


if __name__ == "__main__":
    unittest.main()
