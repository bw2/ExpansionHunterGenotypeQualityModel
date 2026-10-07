"""Tests for dataset.py: truth loading/joining, filtering, freshness checks, combo assembly.

Network-touching helpers (``_gcs_stat``, ``_list_json_inputs``, ``_download``, ``_bw2_head_sha``) are
exercised by mocking ``subprocess.run`` / ``os.path`` rather than hitting GCS or git.
"""

import glob
import gzip
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import dataset
import features
import heldout


def write_contract_parquet(path, n_rows=1):
    """Writes a minimal parquet that satisfies the current feature contract (all columns, float64).

    ``dataset._parquet_reusable`` / ``assert_parquets_up_to_date`` reject a parquet that is missing a
    contract column or holds float32 feature columns, so a cache-hit fixture has to be contract-clean
    or it is (correctly) treated as needing a rebuild. Shared with ``heldout_tests``.
    """
    stored = [c for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")]
    df = pd.DataFrame({c: np.ones(n_rows, dtype=np.float64) for c in stored})
    df["has_own_quality_metrics"] = True  # not a feature, but part of what the extractor produces
    df.to_parquet(path, index=False)
    return df


def _gz_tsv(path, rows):
    """Writes ``rows`` (list of dicts) as a gzip tab-separated file with a header row."""
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _write_truth_tsv_and_bed(path, rows, bed_lines=("chr1\t0\t1000", "chrM\t0\t1000")):
    """Writes a truth TSV whose Start0Based/End/Motif come from each LocusId ("1-3-4-A" -> 3, 4, A), plus
    a high-confidence BED next to it, and returns the BED's path."""
    for row in rows:
        start, end, motif = row["LocusId"].split("-")[1:4]
        row.setdefault("Start0Based", int(start))
        row.setdefault("End", int(end))
        row.setdefault("Motif", motif)
    _gz_tsv(path, rows)
    bed_path = path + ".bed"
    with open(bed_path, "w") as f:
        f.write("\n".join(bed_lines) + "\n")
    return bed_path


class SizeBinLabelTest(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(dataset._size_bin_label(0), "0-20")
        self.assertEqual(dataset._size_bin_label(19), "0-20")
        self.assertEqual(dataset._size_bin_label(20), "20-50")
        self.assertEqual(dataset._size_bin_label(99), "50-100")
        self.assertEqual(dataset._size_bin_label(199), "100-200")
        self.assertEqual(dataset._size_bin_label(200), "200+")
        self.assertEqual(dataset._size_bin_label(10000), "200+")

    def test_negative_is_unmatched(self):
        self.assertEqual(dataset._size_bin_label(-1), "?")


class QuotaSampleTest(unittest.TestCase):
    def test_small_cells_are_kept_whole_and_large_cells_share_the_rest(self):
        # 900 homopolymer rows called at 20 bp, 100 non-homopolymer rows called at 70 x 3 = 210 bp.
        motif = np.array([1] * 900 + [3] * 100, dtype=float)
        eh = np.array([20] * 900 + [70] * 100, dtype=float)
        idx = np.arange(1000)
        out = dataset.quota_sample(motif, eh, idx, 300, 1)
        self.assertEqual((out >= 900).sum(), 100)    # the rare cell keeps all of its rows
        self.assertEqual((out < 900).sum(), 200)     # the common cell gets the remaining quota
        self.assertTrue(np.all(np.diff(out) > 0))   # sorted, no duplicates
        np.testing.assert_array_equal(out, dataset.quota_sample(motif, eh, idx, 300, 1))

    def test_no_cap_or_small_pool_returns_the_pool(self):
        motif, eh, idx = np.ones(10), np.full(10, 5.0), np.arange(10)
        np.testing.assert_array_equal(dataset.quota_sample(motif, eh, idx, None, 1), idx)
        np.testing.assert_array_equal(dataset.quota_sample(motif, eh, idx, 50, 1), idx)

    def test_per_source_cap_keeps_a_uniform_sample_plus_a_training_only_top_up(self):
        df = pd.DataFrame({"genotyping_regime": ["quick"] * 1000 + ["full_spanning"] * 5,
                           "motif_size": [1] * 950 + [3] * 50 + [1] * 5,
                           "eh": [20] * 950 + [70] * 50 + [20] * 5,
                           "row": range(1005)})
        out = dataset._cap_rows_per_genotyping_regime(df, 200, 1)
        quick, rep = out["genotyping_regime"] == "quick", out["representative"]
        self.assertEqual((quick & rep).sum(), 200)                      # uniform sample, the cap
        self.assertEqual((quick & (out["motif_size"] == 3)).sum(), 50)  # the rare cell is whole ...
        self.assertEqual((quick & (out["motif_size"] == 3) & rep).sum() + (quick & ~rep).sum(), 50)  # ... via the top-up
        self.assertFalse(out["row"].duplicated().any())
        self.assertTrue(out["row"].is_monotonic_increasing)
        self.assertEqual(((~quick) & rep).sum(), 5)                     # a small regime stays whole

    def test_top_up_counts_rows_already_chosen(self):
        motif, eh = np.array([1.0] * 90 + [3.0] * 10), np.array([20.0] * 90 + [70.0] * 10)
        idx = np.arange(100)
        chosen = np.array([0, 1, 95])                                   # one rare row already chosen
        extra = dataset._quota_top_up(motif, eh, idx, chosen, 20, 1)
        self.assertFalse(np.isin(extra, chosen).any())
        self.assertEqual(int((extra >= 90).sum()), 9)                   # the rare cell reaches all 10


class IndividualOfPartTest(unittest.TestCase):
    def test_coverage_variants_of_one_person_share_an_individual(self):
        self.assertEqual({dataset.individual_of_part("/d/HG002_%s.parquet" % c) for c in ("10x", "20x", "31x")},
                         {"HG002"})
        self.assertEqual(dataset.individual_of_part("/d/CHM1_CHM13_46x.parquet"), "CHM1_CHM13")
        self.assertEqual(dataset.individual_of_part("/d/NA12878.parquet"), "NA12878")


class ChromFromLocusTest(unittest.TestCase):
    def test_valid_and_invalid(self):
        locus = pd.Series(["1-100-200-ATG", "chr2-5-10-A", "chrX-1-2-A", "chrM-1-2-A", "chrUn-1-2-A"])
        out = dataset._chrom_from_locus(locus)
        self.assertEqual(list(out[:3]), ["1", "2", "X"])
        self.assertTrue(out[3:].isna().all())


class LoadTruthFromGenotypesTsvTest(unittest.TestCase):
    def test_reshape_wide_to_long_and_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truth.tsv.gz")
            bed = _write_truth_tsv_and_bed(path, [
                {"LocusId": "1-1-2-A", "Chrom": "chr1", "NumRepeatsInReference": 1,
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 9,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 0.9},
                {"LocusId": "1-3-4-A", "Chrom": "chr1", "NumRepeatsInReference": 1,
                 "NumRepeatsShortAllele": 2, "NumRepeatsLongAllele": 2,
                 "RepeatPurityShortAllele": 0.8, "RepeatPurityLongAllele": 0.8},
                # exact duplicate row -- must be collapsed by drop_duplicates("LocusId")
                {"LocusId": "1-3-4-A", "Chrom": "chr1", "NumRepeatsInReference": 1,
                 "NumRepeatsShortAllele": 2, "NumRepeatsLongAllele": 2,
                 "RepeatPurityShortAllele": 0.8, "RepeatPurityLongAllele": 0.8},
            ])
            out, _ = dataset._load_truth_from_genotypes_tsv(path, bed)
            self.assertEqual(len(out), 4)  # 2 unique loci x 2 alleles, not 3 x 2
            self.assertEqual(list(out.columns), ["LocusId", "allele_rank", "true", "purity", "is_negative_locus"])
            row = out[(out["LocusId"] == "1-1-2-A") & (out["allele_rank"] == 0)].iloc[0]
            self.assertEqual(row["true"], 5)
            self.assertEqual(row["purity"], 1.0)
            self.assertFalse(row["is_negative_locus"])
            row = out[(out["LocusId"] == "1-1-2-A") & (out["allele_rank"] == 1)].iloc[0]
            self.assertEqual(row["true"], 9)

    def test_filters_drop_hom_ref_nonprimary_and_unparseable(self):
        # Only the one variant primary-contig locus is trained on; the hom-ref one still counts as a
        # truth locus for the catalog-agreement check. chr prefixes are stripped.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truth.tsv.gz")
            bed = _write_truth_tsv_and_bed(path, [
                {"LocusId": "1-1-2-A", "Chrom": "chr1", "NumRepeatsInReference": 5,  # hom-ref -> drop
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 5,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
                {"LocusId": "chr1-3-4-A", "Chrom": "chr1", "NumRepeatsInReference": 5,  # variant -> keep
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 9,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 0.9},
                {"LocusId": "M-1-2-A", "Chrom": "chrM", "NumRepeatsInReference": 1,  # non-primary -> drop
                 "NumRepeatsShortAllele": 2, "NumRepeatsLongAllele": 4,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
                {"LocusId": "1-9-9-A", "Chrom": "chr1", "NumRepeatsInReference": 1,  # unparseable -> drop
                 "NumRepeatsShortAllele": "NA", "NumRepeatsLongAllele": 4,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
            ])
            out, truth_locus_ids = dataset._load_truth_from_genotypes_tsv(path, bed)
            self.assertEqual(set(out["LocusId"]), {"1-3-4-A"})
            self.assertEqual(len(out), 2)  # the one kept locus -> Short + Long allele rows
            self.assertEqual(truth_locus_ids, {"1-1-2-A", "1-3-4-A"})

    def test_drops_loci_not_wholly_inside_one_high_confidence_region(self):
        row = {"Chrom": "chr1", "NumRepeatsInReference": 5, "NumRepeatsShortAllele": 5,
               "NumRepeatsLongAllele": 9, "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truth.tsv.gz")
            bed = _write_truth_tsv_and_bed(path, [
                dict(row, LocusId="1-10-20-A"),    # inside the first region -> keep
                dict(row, LocusId="1-90-110-A"),   # straddles the first region's end -> drop
                dict(row, LocusId="1-150-160-A"),  # in the gap between regions -> drop
                dict(row, LocusId="1-200-300-A"),  # exactly the second region -> keep
                dict(row, LocusId="1-95-205-A"),   # spans both regions and the gap -> drop
                dict(row, LocusId="2-10-20-A", Chrom="chr2"),  # chromosome absent from the BED -> drop
            ], bed_lines=("chr1\t200\t300", "chr1\t0\t100"))  # unsorted on purpose
            out, truth_locus_ids = dataset._load_truth_from_genotypes_tsv(path, bed)
            self.assertEqual(truth_locus_ids, {"1-10-20-A", "1-200-300-A"})
            self.assertEqual(set(out["LocusId"]), {"1-10-20-A", "1-200-300-A"})

    def test_loci_eh_skips_for_the_read_length_are_not_truth_loci(self):
        # With 40 bp reads EH skips a reference region wider than 80 bp or a motif longer than 20 bp.
        row = {"Chrom": "chr1", "NumRepeatsInReference": 5, "NumRepeatsShortAllele": 5,
               "NumRepeatsLongAllele": 9, "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truth.tsv.gz")
            bed = _write_truth_tsv_and_bed(path, [
                dict(row, LocusId="1-10-90-A"),                       # 80 bp wide -> genotyped
                dict(row, LocusId="1-100-181-A"),                     # 81 bp wide -> skipped
                dict(row, LocusId="1-200-242-%s" % ("C" * 21)),       # 21 bp motif -> skipped
            ])
            out, truth_locus_ids = dataset._load_truth_from_genotypes_tsv(path, bed, eh_read_length=40)
            self.assertEqual(truth_locus_ids, {"1-10-90-A"})
            self.assertEqual(len(set(out["LocusId"])), 3)  # still in the truth used for the join


class JoinTruthTest(unittest.TestCase):
    def test_joins_on_locus_and_rank(self):
        json_df = pd.DataFrame({"locus_id": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 1], "eh": [10, 20]})
        tsv_df = pd.DataFrame({"LocusId": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 1], "true": [9, 21],
                              "purity": [1.0, 1.0], "is_negative_locus": [False, False]})
        merged = dataset._join_truth(json_df, tsv_df)
        self.assertEqual(list(merged["true"]), [9, 21])
        self.assertNotIn("LocusId", merged.columns)

    def test_unmatched_json_row_gets_nan_truth(self):
        json_df = pd.DataFrame({"locus_id": ["1-1-2-A"], "allele_rank": [0], "eh": [10]})
        tsv_df = pd.DataFrame({"LocusId": [], "allele_rank": [], "true": [], "purity": [],
                              "is_negative_locus": []})
        merged = dataset._join_truth(json_df, tsv_df)
        self.assertTrue(pd.isna(merged["true"].iloc[0]))

    def test_duplicate_json_key_raises(self):
        json_df = pd.DataFrame({"locus_id": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 0], "eh": [10, 11]})
        tsv_df = pd.DataFrame({"LocusId": ["1-1-2-A"], "allele_rank": [0], "true": [9], "purity": [1.0],
                              "is_negative_locus": [False]})
        with self.assertRaises(AssertionError):
            dataset._join_truth(json_df, tsv_df)


class ExtractRowsAndJoinTruthTest(unittest.TestCase):
    TRUTH_DF = pd.DataFrame({"LocusId": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 1], "true": [9.0, 21.0],
                             "purity": [1.0, 1.0], "is_negative_locus": [False, False]})

    def _run(self, rows):
        with mock.patch.object(dataset.eh_json, "extract_rows", return_value=rows), \
             mock.patch.object(dataset.eh_json, "typical_read_length_in_file", return_value=150), \
             mock.patch.object(dataset, "_load_truth_from_genotypes_tsv",
                               return_value=(self.TRUTH_DF.copy(), {"1-1-2-A", "1-5-6-A"})) as load, \
             mock.patch.object(dataset, "_assert_catalog_agreement") as check:
            merged = dataset.extract_rows_and_join_truth(["a.json.gz"], "t.tsv.gz", "t.bed.gz", "S", "unit")
            self.assertEqual(load.call_args[0][2], 150)  # the JSON's read length reaches the truth loader
            return merged, check

    def test_rows_spanning_several_chunks_are_all_kept(self):
        rows = [{"locus_id": "1-1-2-A", "allele_rank": i % 2, "eh": 10.0 + i} for i in range(2)]
        with mock.patch.object(dataset, "_ROWS_PER_CHUNK", 1):
            merged, _ = self._run(rows)
        self.assertEqual(list(merged["eh"]), [10.0, 11.0])

    def test_keeps_only_rows_at_variant_truth_loci_and_strips_chr(self):
        rows = [{"locus_id": "chr1-1-2-A", "allele_rank": 0, "eh": 10.0},
                {"locus_id": "chr1-1-2-A", "allele_rank": 1, "eh": 20.0},
                {"locus_id": "chr1-5-6-A", "allele_rank": 0, "eh": 5.0},   # hom-ref in truth -> dropped
                {"locus_id": "chr1-9-9-A", "allele_rank": 0, "eh": 5.0}]   # no truth -> dropped
        merged, check = self._run(rows)
        self.assertEqual(list(merged["locus_id"]), ["1-1-2-A", "1-1-2-A"])
        self.assertEqual(list(merged["true"]), [9.0, 21.0])
        # The agreement check sees every JSON locus, not just the kept ones.
        self.assertEqual(check.call_args[0][0], {"1-1-2-A", "1-5-6-A", "1-9-9-A"})

    def test_no_row_at_a_variant_truth_locus_raises(self):
        with self.assertRaises(RuntimeError):
            self._run([{"locus_id": "X-1-2-A", "allele_rank": 0, "eh": 10.0}])


class AssertCatalogAgreementTest(unittest.TestCase):
    def test_agreement_within_tolerance_does_not_raise(self):
        loci = {"1-%d-2-A" % i for i in range(100)}
        truth_df = pd.DataFrame({"LocusId": sorted(loci), "true": [10] * 100})
        dataset._assert_catalog_agreement(loci, loci, truth_df, "unit")  # must not raise

    def test_size_bins_only_count_truth_loci(self):
        # 10 large-allele variant loci, 9 of them left out of truth_locus_ids (EH skipped them); the one
        # truth locus is missing from the JSON, so its bin is 100% missing, not 1 in 10.
        truth_df = pd.DataFrame({"LocusId": ["1-%d-2-A" % i for i in range(10)], "true": [300] * 10})
        json_loci = {"1-%d-9-A" % i for i in range(1000)}
        truth_loci = {"1-0-2-A"} | {"1-%d-9-A" % i for i in range(1000)}
        with self.assertRaisesRegex(RuntimeError, "100.0% missing"):
            dataset._assert_catalog_agreement(json_loci, truth_loci, truth_df, "unit")

    def test_loci_only_in_json_are_not_counted(self):
        truth_df = pd.DataFrame({"LocusId": ["1-0-2-A"], "true": [10]})
        json_loci = {"1-0-2-A"} | {"1-%d-9-A" % i for i in range(100)}
        dataset._assert_catalog_agreement(json_loci, {"1-0-2-A"}, truth_df, "unit")  # must not raise

    def test_mismatch_beyond_tolerance_raises(self):
        truth_df = pd.DataFrame({"LocusId": ["1-0-2-A", "1-1-2-A", "1-2-2-A"], "true": [10, 300, 300]})
        with self.assertRaises(RuntimeError):
            dataset._assert_catalog_agreement({"1-0-2-A"}, set(truth_df["LocusId"]), truth_df, "unit")

    def test_missing_hom_ref_truth_loci_count_toward_the_global_gap(self):
        # Every variant locus is present, but most hom-ref truth loci are missing from the JSON.
        truth_df = pd.DataFrame({"LocusId": ["1-0-2-A"], "true": [10]})
        truth_loci = {"1-0-2-A"} | {"1-%d-5-A" % i for i in range(10)}
        with self.assertRaises(RuntimeError):
            dataset._assert_catalog_agreement({"1-0-2-A"}, truth_loci, truth_df, "unit")

    def test_no_truth_loci_is_a_no_op(self):
        dataset._assert_catalog_agreement({"1-0-2-A"}, set(), pd.DataFrame({"LocusId": [], "true": []}),
                                          "unit")


class LabelAndFilterTest(unittest.TestCase):
    def test_drops_and_labels(self):
        df = pd.DataFrame([
            {"locus_id": "chrM-1-2-A", "is_negative_locus": False, "eh": 10.0, "true": 10.0,
             "motif_size": 1, "genotyping_branch": "quick", "spanning_at_called": 5},
            {"locus_id": "1-1-2-A", "is_negative_locus": True, "eh": 10.0, "true": 10.0,
             "motif_size": 1, "genotyping_branch": "quick", "spanning_at_called": 5},
            {"locus_id": "1-3-4-A", "is_negative_locus": False, "eh": np.nan, "true": 10.0,
             "motif_size": 1, "genotyping_branch": "quick", "spanning_at_called": 5},
            {"locus_id": "1-5-6-A", "is_negative_locus": False, "eh": -5.0, "true": 10.0,
             "motif_size": 1, "genotyping_branch": "quick", "spanning_at_called": 5},
            {"locus_id": "1-7-8-A", "is_negative_locus": False, "eh": 10.0, "true": 10.0,
             "motif_size": 0, "genotyping_branch": "quick", "spanning_at_called": 5},
            {"locus_id": "1-9-10-A", "is_negative_locus": False, "eh": 10.0, "true": 10.0,
             "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5},
            {"locus_id": "2-1-2-A", "is_negative_locus": False, "eh": 14.0, "true": 10.0,
             "motif_size": 3, "genotyping_branch": "full", "spanning_at_called": 0},
        ])
        kept, drops = dataset.label_and_filter(df)
        self.assertEqual(drops, {"chrM_or_unknown_contig": 1, "negative_control_locus": 1,
                                 "missing_eh_or_true": 1, "nonpositive_eh_or_true": 1,
                                 "missing_motif_size": 1, "no_own_quality_metrics": 0})
        self.assertEqual(len(kept), 2)
        self.assertNotIn("locus_id", kept.columns)
        self.assertNotIn("is_negative_locus", kept.columns)
        quick_row = kept[kept["genotyping_branch"] == "quick"].iloc[0]
        self.assertEqual(quick_row["chrom"], "1")
        self.assertEqual(quick_row["genotyping_regime"], "quick")
        full_row = kept[kept["genotyping_branch"] == "full"].iloc[0]
        self.assertEqual(full_row["chrom"], "2")
        self.assertEqual(full_row["genotyping_regime"], "full_nonspanning")  # spanning_at_called == 0

    def _hom_pair(self, has_own):
        """A homozygous call's two rows: identical features, different truth alleles."""
        return pd.DataFrame([
            {"locus_id": "1-9-10-A", "is_negative_locus": False, "eh": 20.0, "true": 18.0,
             "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5,
             "allele_rank": 0, "has_own_quality_metrics": has_own[0]},
            {"locus_id": "1-9-10-A", "is_negative_locus": False, "eh": 20.0, "true": 25.0,
             "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5,
             "allele_rank": 1, "has_own_quality_metrics": has_own[1]},
        ])

    def test_drops_the_second_copy_of_a_homozygous_call(self):
        # EH scores one allele for a hom call, so the rank-1 row is unreachable at inference: same
        # features as rank 0 but joined to the LONG truth allele, i.e. a contradictory target.
        kept, drops = dataset.label_and_filter(self._hom_pair([True, False]))
        self.assertEqual(drops["no_own_quality_metrics"], 1)
        self.assertEqual(list(kept["allele_rank"]), [0])
        self.assertEqual(list(kept["true"]), [18.0])

    def test_keeps_both_copies_when_each_has_its_own_metrics(self):
        kept, _ = dataset.label_and_filter(self._hom_pair([True, True]))
        self.assertEqual(list(kept["allele_rank"]), [0, 1])

    def test_parquet_without_the_column_is_left_alone(self):
        # A parquet predating the column must not be half-filtered; the freshness guards catch it.
        old = self._hom_pair([True, False]).drop(columns=["has_own_quality_metrics"])
        kept, drops = dataset.label_and_filter(old)
        self.assertEqual(drops["no_own_quality_metrics"], 0)
        self.assertEqual(len(kept), 2)


class LocalMd5Test(unittest.TestCase):
    def test_matches_hashlib_base64(self):
        import base64
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.bin")
            with open(path, "wb") as f:
                f.write(b"some file content")
            expected = base64.b64encode(hashlib.md5(b"some file content").digest()).decode()
            self.assertEqual(dataset._local_md5(path), expected)


class GcsStatTest(unittest.TestCase):
    def test_parses_hash_and_prefers_update_over_creation(self):
        import email.utils
        stdout = ("gs://x/y:\n    Creation time:    Wed, 02 Jul 2026 10:00:00 GMT\n"
                  "    Update time:      Wed, 02 Jul 2026 22:46:00 GMT\n"
                  "    Hash (md5):          abcd1234==\n")
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout, returncode=0, stderr="")) as run:
            md5, mtime = dataset._gcs_stat("gs://x/y")
        self.assertEqual(md5, "abcd1234==")
        self.assertAlmostEqual(
            mtime, email.utils.parsedate_to_datetime("Wed, 02 Jul 2026 22:46:00 GMT").timestamp())
        self.assertEqual(run.call_count, 1)  # both signals from ONE stat, not two

    def test_falls_back_to_creation_time(self):
        import email.utils
        stdout = "gs://x/y:\n    Creation time:    Wed, 02 Jul 2026 10:00:00 GMT\n"
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout, returncode=0, stderr="")):
            _, mtime = dataset._gcs_stat("gs://x/y")
        self.assertAlmostEqual(
            mtime, email.utils.parsedate_to_datetime("Wed, 02 Jul 2026 10:00:00 GMT").timestamp())

    def test_failed_stat_returns_none_and_warns(self):
        # (None, None) reads as "unchanged" downstream, which is the right default (an unreachable
        # bucket must not delete local data) but silent -- so the failure has to be announced.
        failed = SimpleNamespace(stdout="", returncode=1, stderr="AccessDeniedException: 403\n")
        with mock.patch.object(dataset.subprocess, "run", return_value=failed):
            self.assertEqual(dataset._gcs_stat("gs://x/y"), (None, None))

    def test_missing_lines_return_none(self):
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout="nope\n", returncode=0, stderr="")):
            self.assertEqual(dataset._gcs_stat("gs://x/y"), (None, None))


class Bw2HeadShaTest(unittest.TestCase):
    def test_missing_checkout_returns_none(self):
        with mock.patch.object(dataset.os.path, "isdir", return_value=False):
            self.assertIsNone(dataset._bw2_head_sha())

    def test_returns_stripped_sha(self):
        with mock.patch.object(dataset.os.path, "isdir", return_value=True), \
             mock.patch.object(dataset.subprocess, "run",
                               return_value=SimpleNamespace(stdout="abc1234\n", returncode=0)):
            self.assertEqual(dataset._bw2_head_sha(), "abc1234")

    def test_nonzero_returncode_returns_none(self):
        with mock.patch.object(dataset.os.path, "isdir", return_value=True), \
             mock.patch.object(dataset.subprocess, "run",
                               return_value=SimpleNamespace(stdout="", returncode=1)):
            self.assertIsNone(dataset._bw2_head_sha())


class JsonEhVersionTest(unittest.TestCase):
    def test_plain_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            with open(path, "w") as f:
                json.dump({"SampleParameters": {"Version": "deadbeef"}}, f)
            self.assertEqual(dataset._json_eh_version(path), "deadbeef")

    def test_gzip_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json.gz")
            with gzip.open(path, "wt") as f:
                json.dump({"SampleParameters": {"Version": "deadbeef"}}, f)
            self.assertEqual(dataset._json_eh_version(path), "deadbeef")

    def test_missing_version_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            with open(path, "w") as f:
                json.dump({"SampleParameters": {}}, f)
            self.assertIsNone(dataset._json_eh_version(path))


class ListJsonInputsTest(unittest.TestCase):
    def test_prefers_combined_over_shards(self):
        stdout = ("gs://x/json/a.shard000_of_002.json.gz gs://x/json/a.shard001_of_002.json.gz "
                  "gs://x/json/a.json.gz")
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout, returncode=0, stderr="")):
            self.assertEqual(dataset._list_json_inputs("S"), ["gs://x/json/a.json.gz"])

    def test_falls_back_to_shards(self):
        stdout = "gs://x/json/a.shard000_of_002.json.gz gs://x/json/a.shard001_of_002.json.gz"
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout, returncode=0, stderr="")):
            self.assertEqual(dataset._list_json_inputs("S"),
                             ["gs://x/json/a.shard000_of_002.json.gz", "gs://x/json/a.shard001_of_002.json.gz"])


class DownloadTest(unittest.TestCase):
    def test_skips_download_when_already_present(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.json.gz"), "w").close()
            with mock.patch.object(dataset.subprocess, "run", side_effect=AssertionError("should not be called")):
                out = dataset._download(["gs://x/a.json.gz"], d)
            self.assertEqual(out, [os.path.join(d, "a.json.gz")])

    def test_raises_after_exhausting_retries(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(dataset.subprocess, "run", return_value=None):
                with self.assertRaises(RuntimeError):
                    dataset._download(["gs://x/missing.json.gz"], d)


class CheckFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.local = os.path.join(self.tmp.name, "a.json")
        with open(self.local, "w") as f:
            json.dump({"SampleParameters": {}}, f)
        self.lm = os.path.getmtime(self.local)

    def test_no_local_file_skips_entirely(self):
        with mock.patch.object(dataset, "_gcs_stat", side_effect=AssertionError("should not be called")):
            n = dataset._check_freshness("d", [("json shard", "gs://r", "/nonexistent/path")])
        self.assertEqual(n, 0)

    def test_up_to_date_is_a_no_op(self):
        # md5 matches AND the bucket is not newer than the local file -> keep it, nothing removed.
        with mock.patch.object(dataset, "_gcs_stat", return_value=("samehash", self.lm - 100)), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 0)
        self.assertTrue(os.path.exists(self.local))

    def test_md5_mismatch_deletes_for_redownload(self):
        with mock.patch.object(dataset, "_gcs_stat", return_value=("new", self.lm - 100)), \
             mock.patch.object(dataset, "_local_md5", return_value="old"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 1)
        self.assertFalse(os.path.exists(self.local))  # deleted so _download re-fetches

    def test_bucket_newer_mtime_deletes_for_redownload(self):
        with mock.patch.object(dataset, "_gcs_stat", return_value=("samehash", self.lm + 100)), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 1)
        self.assertFalse(os.path.exists(self.local))

    def test_eh_build_staleness_refuses_for_kept_json(self):
        # up-to-date content but produced by an older EH build -> hard refusal (can't fix by redownload).
        with mock.patch.object(dataset, "_gcs_stat", return_value=("samehash", self.lm - 100)), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_json_eh_version", return_value="old_sha"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="new_sha"):
            with self.assertRaises(SystemExit):
                dataset._check_freshness("d", [("json shard", "gs://r", self.local)])

    def test_eh_build_staleness_ignored_for_non_json_labels(self):
        with mock.patch.object(dataset, "_gcs_stat", return_value=("samehash", self.lm - 100)), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_json_eh_version", return_value="old_sha"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="new_sha"):
            n = dataset._check_freshness("d", [("truth-genotypes TSV", "gs://r", self.local)])
        self.assertEqual(n, 0)  # build check only applies to json-labeled sources

    def test_cache_stale_json_is_redownloaded_without_build_refusal(self):
        # A json that is BOTH bucket-newer AND from an older build is re-downloaded (not refused): the
        # fresh copy replaces it, so its stale build sha is not judged here.
        with mock.patch.object(dataset, "_gcs_stat", return_value=("samehash", self.lm + 100)), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_json_eh_version", return_value="old_sha"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="new_sha"):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 1)
        self.assertFalse(os.path.exists(self.local))


class BuildComboTest(unittest.TestCase):
    def test_cache_hit_skips_download(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "sub", "HG002_10x.parquet")
            os.makedirs(os.path.dirname(out_path))
            # Upstream local sources must exist and be OLDER than the parquet for it to be reused.
            dl_dir = os.path.join(d, "sub", "_downloads", "HG002_10x")
            gt_dir = os.path.join(d, "sub", "_downloads", "HG002")
            os.makedirs(dl_dir)
            os.makedirs(gt_dir)
            json_local = os.path.join(dl_dir, "a.json.gz")
            tsv_local = os.path.join(gt_dir, "HG002.tandem_repeat_genotypes.tsv.gz")
            open(json_local, "w").close()
            open(tsv_local, "w").close()
            open(os.path.join(gt_dir, "HG002.dip.bed.gz"), "w").close()  # basename from high_confidence_beds.tsv
            write_contract_parquet(out_path, n_rows=3)
            newer = max(os.path.getmtime(json_local), os.path.getmtime(tsv_local)) + 100
            os.utime(out_path, (newer, newer))
            with mock.patch.object(dataset, "_list_json_inputs", return_value=["gs://x/a.json.gz"]), \
                 mock.patch.object(dataset, "_check_freshness", return_value=0), \
                 mock.patch.object(dataset, "_download", side_effect=AssertionError("should not download")):
                n = dataset.build_combo("sub", "HG002", "10x", "HG002_10x", d, force=False)
            self.assertEqual(n, 3)

    def test_fresh_build_joins_and_writes_parquet(self):
        fake_rows = [
            {"locus_id": "1-1-2-A", "allele_rank": 0, "eh": 10.0, "genotyping_branch": "quick",
             "sample_id": "HG002_10x"},
            {"locus_id": "1-1-2-A", "allele_rank": 1, "eh": 20.0, "genotyping_branch": "quick",
             "sample_id": "HG002_10x"},
            {"locus_id": "2-1-2-A", "allele_rank": 0, "eh": 5.0, "genotyping_branch": "full",
             "sample_id": "HG002_10x"},
        ]
        fake_tsv_df = pd.DataFrame({
            "LocusId": ["1-1-2-A", "1-1-2-A", "2-1-2-A"], "allele_rank": [0, 1, 0],
            "true": [9.0, 21.0, 5.0], "purity": [1.0, 1.0, 1.0], "is_negative_locus": [False, False, False]})
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(dataset, "_list_json_inputs", return_value=["gs://x/a.json.gz"]), \
                 mock.patch.object(dataset, "_download",
                                   side_effect=lambda remote, dest: [os.path.join(dest, os.path.basename(p))
                                                                     for p in remote]), \
                 mock.patch.object(dataset, "assert_eh_build_matches"), \
                 mock.patch.object(dataset, "_gcs_stat", return_value=("md5", 1.0)), \
                 mock.patch.object(dataset, "_json_eh_version", return_value="b1fbc23"), \
                 mock.patch.object(dataset.eh_json, "typical_read_length_in_file", return_value=150), \
                 mock.patch.object(dataset.eh_json, "extract_rows", return_value=fake_rows), \
                 mock.patch.object(dataset, "_load_truth_from_genotypes_tsv",
                                   return_value=(fake_tsv_df, {"1-1-2-A", "2-1-2-A"})):
                n = dataset.build_combo("sub", "HG002", "10x", "HG002_10x", d, force=True)
            self.assertEqual(n, 3)
            out_path = os.path.join(d, "sub", "HG002_10x.parquet")
            merged = pd.read_parquet(out_path)
            self.assertNotIn("sample_id", merged.columns)
            # Float columns stay float64: the C++ scorer feeds the model full doubles, so the
            # training data must not be quantized (see features.build_matrix).
            self.assertEqual(merged["true"].dtype, np.float64)
            row = merged[(merged["locus_id"] == "1-1-2-A") & (merged["allele_rank"] == 1)].iloc[0]
            self.assertEqual(row["true"], 21.0)


class SourcesRecordTest(unittest.TestCase):
    """A parquet whose downloads were deleted after the build is reused only while the cloud sources
    still match the versions recorded when it was built, and its EH build still counts as current."""

    def _built_parquet_without_downloads(self, d, cloud_version=("md5", 1.0)):
        out_path = os.path.join(d, "S.parquet")
        local = os.path.join(d, "_downloads", "a.json.gz")
        os.makedirs(os.path.dirname(local))
        open(local, "w").close()
        write_contract_parquet(out_path, n_rows=2)
        sources = [("json shard 0", "gs://x/a.json.gz", local)]
        with mock.patch.object(dataset, "_json_eh_version", return_value="b1fbc23"):
            dataset._record_sources_and_remove_downloads(
                out_path, sources, {"gs://x/a.json.gz": list(cloud_version)})
        return out_path, local, sources

    def test_record_written_and_downloads_removed(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, local, _ = self._built_parquet_without_downloads(d)
            self.assertFalse(os.path.exists(local))
            with open(dataset._sources_record_path(out_path)) as f:
                self.assertEqual(json.load(f), {"cloud_versions": {"gs://x/a.json.gz": ["md5", 1.0]},
                                                "eh_build_by_json": {"gs://x/a.json.gz": "b1fbc23"}})

    def test_unknown_cloud_version_keeps_the_downloads_and_writes_no_record(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, local, _ = self._built_parquet_without_downloads(d, cloud_version=(None, None))
            self.assertTrue(os.path.exists(local))
            self.assertFalse(os.path.exists(dataset._sources_record_path(out_path)))

    def test_reused_while_cloud_versions_match(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, sources = self._built_parquet_without_downloads(d)
            with mock.patch.object(dataset, "_gcs_stat", return_value=("md5", 1.0)), \
                 mock.patch.object(dataset, "_bw2_head_sha", return_value="b1fbc23"):
                self.assertTrue(dataset._parquet_reusable(out_path, sources, force=False))

    def test_rebuilt_when_the_recorded_eh_build_is_no_longer_current(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, sources = self._built_parquet_without_downloads(d)
            with mock.patch.object(dataset, "_gcs_stat", return_value=("md5", 1.0)), \
                 mock.patch.object(dataset, "_bw2_head_sha", return_value="c0ffee1"), \
                 mock.patch.object(dataset, "_bw2_files_changed_since", return_value=["ehunter/app/Main.cpp"]):
                self.assertFalse(dataset._parquet_reusable(out_path, sources, force=False))

    def test_rebuilt_from_a_record_in_the_older_format(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, sources = self._built_parquet_without_downloads(d)
            with open(dataset._sources_record_path(out_path), "w") as f:
                json.dump({"gs://x/a.json.gz": ["md5", 1.0]}, f)
            with mock.patch.object(dataset, "_gcs_stat", return_value=("md5", 1.0)):
                self.assertFalse(dataset._parquet_reusable(out_path, sources, force=False))

    def test_rebuilt_when_cloud_version_changed_or_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, sources = self._built_parquet_without_downloads(d)
            for changed in (("other", 1.0), ("md5", 2.0), (None, None)):
                with mock.patch.object(dataset, "_gcs_stat", return_value=changed):
                    self.assertFalse(dataset._parquet_reusable(out_path, sources, force=False))

    def test_rebuilt_without_record(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, sources = self._built_parquet_without_downloads(d)
            os.remove(dataset._sources_record_path(out_path))
            with mock.patch.object(dataset, "_gcs_stat", return_value=("md5", 1.0)):
                self.assertFalse(dataset._parquet_reusable(out_path, sources, force=False))


class AssembleBranchTest(unittest.TestCase):
    def test_filters_branch_and_writes_labeled_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = os.path.join(d, "src")
            os.makedirs(src_dir)
            # Every feature column must be present -- assemble_branch refuses parts that predate
            # the current contract (they would concat to all-NaN for the missing feature).
            base = {c: 1.0 for c in features.FULL_FEATURES
                    if c not in ("ci_asymmetry", "ci_over_eh")}
            base["has_own_quality_metrics"] = True  # part of the contract, though not a feature
            pd.DataFrame([
                dict(base, **{"locus_id": "1-1-2-A", "is_negative_locus": False, "eh": 10.0,
                              "true": 10.0, "motif_size": 3, "genotyping_branch": "quick",
                              "spanning_at_called": 5}),
                dict(base, **{"locus_id": "1-3-4-A", "is_negative_locus": False, "eh": np.nan,
                              "true": 10.0, "motif_size": 3, "genotyping_branch": "quick",
                              "spanning_at_called": 5}),
                dict(base, **{"locus_id": "2-1-2-A", "is_negative_locus": False, "eh": 14.0,
                              "true": 10.0, "motif_size": 3, "genotyping_branch": "full",
                              "spanning_at_called": 0}),
            ]).to_parquet(os.path.join(src_dir, "combo1.parquet"))

            dataset.assemble_branch(d, "quick", "src")
            quick = pd.read_parquet(os.path.join(d, "parquet", "quick.parquet"))
            self.assertEqual(len(quick), 1)  # the NaN-eh row is dropped, the full row excluded

            dataset.assemble_branch(d, "full", "src")
            full = pd.read_parquet(os.path.join(d, "parquet", "full.parquet"))
            self.assertEqual(len(full), 1)
            self.assertEqual(full.iloc[0]["genotyping_regime"], "full_nonspanning")

    def test_each_source_keeps_at_most_the_cap_of_representative_rows_per_genotyping_regime(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = os.path.join(d, "src")
            os.makedirs(src_dir)
            base = {c: 1.0 for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")}
            base.update({"has_own_quality_metrics": True, "is_negative_locus": False, "true": 10.0,
                         "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5})
            for name in ("a", "b"):
                pd.DataFrame([dict(base, locus_id="1-%d-%d-A" % (i, i + 1), eh=10.0 + i) for i in range(5)]
                             ).to_parquet(os.path.join(src_dir, "%s.parquet" % name))
            with mock.patch.object(dataset, "MAX_ALLELES_PER_SOURCE_PER_GENOTYPING_REGIME", 3):
                dataset.assemble_branch(d, "quick", "src")
                first = pd.read_parquet(os.path.join(d, "parquet", "quick.parquet"))
                dataset.assemble_branch(d, "quick", "src")
                second = pd.read_parquet(os.path.join(d, "parquet", "quick.parquet"))
            # The cap bounds each source's REPRESENTATIVE rows; the training-only quota top-up comes on
            # top and is tested in QuotaSampleTest.
            self.assertEqual(first[first["representative"]].groupby("individual").size().to_dict(),
                             {"a": 3, "b": 3})
            self.assertTrue(first["individual"].isin(["a", "b"]).all())
            pd.testing.assert_frame_equal(first, second)  # the subsample is deterministic

    def test_no_parquets_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            dataset.assemble_branch(d, "quick", "src")  # must not raise
            self.assertFalse(os.path.exists(os.path.join(d, "parquet", "quick.parquet")))


class LinkPromotedHeldoutSamplesTest(unittest.TestCase):
    def test_symlinks_each_promoted_sample_once(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.parquet")
            pd.DataFrame({"eh": [1.0]}).to_parquet(src)
            with mock.patch.object(heldout, "build_sample", return_value=src) as m:
                dataset._link_promoted_heldout_samples(d, force=False)
                self.assertEqual(m.call_count, len(dataset.PROMOTED_HELDOUT_SAMPLES))
                for sample in dataset.PROMOTED_HELDOUT_SAMPLES:
                    dst = os.path.join(d, dataset.SOURCE_SUBDIR, "%s.parquet" % sample)
                    self.assertTrue(os.path.islink(dst))
                # a second pass must not re-symlink or crash on an already-existing dst
                dataset._link_promoted_heldout_samples(d, force=False)


class PromotedHeldoutSamplesTest(unittest.TestCase):
    def test_disjoint_from_remaining_heldout_samples(self):
        self.assertEqual(set(dataset.PROMOTED_HELDOUT_SAMPLES) & set(heldout.SAMPLES), set())


class AssertParquetsUpToDateTest(unittest.TestCase):
    def _setup(self, d):
        """Builds a minimal data_dir with one combo's JSON + truth TSV under _downloads; returns them."""
        os.makedirs(os.path.join(d, "parquet"))
        combo_dl = os.path.join(d, dataset.SOURCE_SUBDIR, "_downloads", "HG002_31x")
        os.makedirs(combo_dl)
        json_p = os.path.join(combo_dl, "HG002.EHv5.shard000.json.gz")
        open(json_p, "w").close()
        tsv_p = os.path.join(d, dataset.SOURCE_SUBDIR, "_downloads", "HG002",
                             "HG002.tandem_repeat_genotypes.tsv.gz")
        os.makedirs(os.path.dirname(tsv_p))
        open(tsv_p, "w").close()
        return json_p, tsv_p

    def _write_parquets(self, d):
        for b in ("quick", "full"):
            write_contract_parquet(os.path.join(d, "parquet", "%s.parquet" % b))

    def test_missing_parquet_exits(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ()):
            self._setup(d)  # upstream present, but no parquet written
            with self.assertRaises(SystemExit):
                dataset.assert_parquets_up_to_date(d)

    def test_stale_parquet_exits(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ()):
            json_p, _ = self._setup(d)
            self._write_parquets(d)
            future = os.path.getmtime(os.path.join(d, "parquet", "quick.parquet")) + 100
            os.utime(json_p, (future, future))  # an upstream JSON now newer than the parquet
            with self.assertRaises(SystemExit):
                dataset.assert_parquets_up_to_date(d)

    def test_fresh_parquet_ok(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ()):
            json_p, tsv_p = self._setup(d)
            self._write_parquets(d)
            newer = max(os.path.getmtime(json_p), os.path.getmtime(tsv_p)) + 100
            for b in ("quick", "full"):
                os.utime(os.path.join(d, "parquet", "%s.parquet" % b), (newer, newer))
            dataset.assert_parquets_up_to_date(d)  # must not raise

    def test_off_contract_parquet_exits_even_when_mtimes_are_fine(self):
        # float32 feature columns and a missing feature column both leave every upstream mtime
        # untouched, so only the contract check can catch them.
        for damage in ("float32", "missing_column"):
            with tempfile.TemporaryDirectory() as d, \
                 mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ()):
                json_p, tsv_p = self._setup(d)
                for b in ("quick", "full"):
                    path = os.path.join(d, "parquet", "%s.parquet" % b)
                    df = write_contract_parquet(path)
                    if damage == "float32":
                        df["coverage"] = df["coverage"].astype(np.float32)
                    else:
                        df = df.drop(columns=["coverage"])
                    df.to_parquet(path, index=False)
                newer = max(os.path.getmtime(json_p), os.path.getmtime(tsv_p)) + 100
                for b in ("quick", "full"):
                    os.utime(os.path.join(d, "parquet", "%s.parquet" % b), (newer, newer))
                with self.assertRaises(SystemExit, msg=damage):
                    dataset.assert_parquets_up_to_date(d)


class DePromotedSymlinkCleanupTest(unittest.TestCase):
    """De-promoting a sample must remove its training-pool symlink, or it leaks: assemble_branch
    globs *.parquet for training while the sample is back in heldout.SAMPLES for evaluation.

    Named apart from ``LinkPromotedHeldoutSamplesTest`` above -- reusing that name rebound it and
    silently dropped that class's test from the suite.
    """

    def _setup(self, d):
        src_dir = os.path.join(d, "src")
        subdir = os.path.join(d, "data", dataset.SOURCE_SUBDIR)
        os.makedirs(src_dir)
        os.makedirs(subdir)
        for sample in ("KEPT", "DROPPED"):
            src = os.path.join(src_dir, "%s.parquet" % sample)
            open(src, "w").close()
            os.symlink(src, os.path.join(subdir, "%s.parquet" % sample))
        open(os.path.join(subdir, "HG002_31x.parquet"), "w").close()  # a real combo parquet
        return subdir, src_dir

    def test_symlink_for_a_de_promoted_sample_is_removed(self):
        with tempfile.TemporaryDirectory() as d:
            subdir, src_dir = self._setup(d)
            import heldout as heldout_module
            with mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ("KEPT",)), \
                 mock.patch.object(heldout_module, "build_sample",
                                   side_effect=lambda s, dd, force: os.path.join(src_dir, "%s.parquet" % s)):
                dataset._link_promoted_heldout_samples(os.path.join(d, "data"), force=False)
            remaining = sorted(os.path.basename(p) for p in glob.glob(os.path.join(subdir, "*.parquet")))
            self.assertEqual(remaining, ["HG002_31x.parquet", "KEPT.parquet"])

    def test_real_combo_parquets_are_never_removed(self):
        # Only symlinks are pruned -- a real per-combo parquet is expensive to rebuild.
        with tempfile.TemporaryDirectory() as d:
            subdir, src_dir = self._setup(d)
            import heldout as heldout_module
            with mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ()), \
                 mock.patch.object(heldout_module, "build_sample",
                                   side_effect=AssertionError("nothing to build")):
                dataset._link_promoted_heldout_samples(os.path.join(d, "data"), force=False)
            remaining = sorted(os.path.basename(p) for p in glob.glob(os.path.join(subdir, "*.parquet")))
            self.assertEqual(remaining, ["HG002_31x.parquet"])


class AssertEhBuildMatchesTest(unittest.TestCase):
    def _json(self, path, version):
        with gzip.open(path, "wt") as f:
            json.dump({"RunInfo": {"Version": version}}, f)
        return path

    def test_mismatched_build_sha_exits(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="7ee80de"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=["ehunter/app/Main.cpp"]):
            p = self._json(os.path.join(d, "a.json.gz"), "3789ba4")
            with self.assertRaises(SystemExit):
                dataset.assert_eh_build_matches("combo", [("json shard 0", p)])

    def test_matching_build_sha_passes(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="7ee80de"):
            p = self._json(os.path.join(d, "a.json.gz"), "7ee80de")
            dataset.assert_eh_build_matches("combo", [("json shard 0", p)])  # must not raise

    def test_older_build_with_only_doc_and_digest_changes_since_passes(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="7ee80de"), \
             mock.patch.object(dataset, "_bw2_files_changed_since",
                               return_value=["README.md", "docker/sha256.txt", ".github/workflows/docker.yml"]):
            p = self._json(os.path.join(d, "a.json.gz"), "3789ba4")
            dataset.assert_eh_build_matches("combo", [("json shard 0", p)])  # must not raise

    def test_older_build_with_only_vcf_writer_changes_since_passes(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="b5ef91d"), \
             mock.patch.object(dataset, "_bw2_files_changed_since",
                               return_value=["ehunter/io/VcfWriter.cpp", "example/output/repeats.vcf"]):
            p = self._json(os.path.join(d, "a.json.gz"), "b1fbc23")
            dataset.assert_eh_build_matches("combo", [("json shard 0", p)])  # must not raise

    def test_older_build_with_only_an_embedded_model_swap_since_passes(self):
        changed = ["ehunter/data/genotype_quality_model_from_HG002_and_CHM1_CHM13.20261004.json.gz",
                   "ehunter/data/README.md", "ehunter/CMakeLists.txt"]
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="feed123"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=changed), \
             mock.patch.object(dataset, "_cmake_change_only_swaps_the_embedded_model", return_value=True):
            p = self._json(os.path.join(d, "a.json.gz"), "b1fbc23")
            dataset.assert_eh_build_matches("combo", [("json shard 0", p)])  # must not raise

    def test_older_build_with_other_cmake_changes_since_exits(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="feed123"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=["ehunter/CMakeLists.txt"]), \
             mock.patch.object(dataset, "_cmake_change_only_swaps_the_embedded_model", return_value=False):
            p = self._json(os.path.join(d, "a.json.gz"), "b1fbc23")
            with self.assertRaises(SystemExit):
                dataset.assert_eh_build_matches("combo", [("json shard 0", p)])

    def test_cmake_change_is_judged_by_its_changed_lines(self):
        diff = ("--- a/ehunter/CMakeLists.txt\n+++ b/ehunter/CMakeLists.txt\n@@ -74 +74 @@\n"
                "-set(GQ_MODEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data/genotype_quality_model_a.json.gz)\n"
                "+set(GQ_MODEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data/genotype_quality_model_b.json.gz)\n")
        for extra, expected in (("", True), ("+add_compile_options(-O3)\n", False)):
            result = mock.Mock(returncode=0, stdout=diff + extra)
            with mock.patch.object(dataset.subprocess, "run", return_value=result):
                self.assertEqual(dataset._cmake_change_only_swaps_the_embedded_model("b1fbc23"), expected)

    def test_older_build_with_only_the_inrepeat_feature_wiring_since_passes(self):
        changed = ["ehunter/genotype_quality/GenotypeQualityFeatures.cpp",
                   "ehunter/genotype_quality/GenotypeQualityFeatures.hh", "ehunter/io/JsonWriter.cpp",
                   "ehunter/tests/GenotypeQualityFeaturesTest.cpp"]
        diff = ("--- a/ehunter/io/JsonWriter.cpp\n+++ b/ehunter/io/JsonWriter.cpp\n@@ -324 +324,2 @@\n"
                "-                    .numDistinctAlleles = numDistinctAlleles};\n"
                "+                    .numDistinctAlleles = numDistinctAlleles,\n"
                "+                    .inrepeatReads = &repeatFindings.countsOfInrepeatReads()};\n")
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="feed123"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=changed), \
             mock.patch.object(dataset.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=diff)):
            p = self._json(os.path.join(d, "a.json.gz"), "b1fbc23")
            dataset.assert_eh_build_matches("combo", [("json shard 0", p)])  # must not raise

    def test_json_writer_change_beyond_the_model_wiring_counts(self):
        diff = ("--- a/ehunter/io/JsonWriter.cpp\n+++ b/ehunter/io/JsonWriter.cpp\n@@ -224 +224 @@\n"
                '+    record_["NewField"] = 1;\n')
        with mock.patch.object(dataset.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=diff)):
            self.assertFalse(dataset._json_writer_change_only_feeds_the_model("b1fbc23"))

    def test_older_build_with_source_changes_since_exits(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="7ee80de"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=["README.md", "ehunter/app/Main.cpp"]):
            p = self._json(os.path.join(d, "a.json.gz"), "3789ba4")
            with self.assertRaises(SystemExit):
                dataset.assert_eh_build_matches("combo", [("json shard 0", p)])

    def test_build_unknown_to_the_local_checkout_exits(self):
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="7ee80de"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=None):
            p = self._json(os.path.join(d, "a.json.gz"), "unknown")
            with self.assertRaises(SystemExit):
                dataset.assert_eh_build_matches("combo", [("json shard 0", p)])

    def test_no_local_checkout_skips_the_check(self):
        # Without the checkout there is no HEAD to compare against, so the JSON is never even opened.
        with mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            dataset.assert_eh_build_matches("combo", [("json shard 0", "/nonexistent.json.gz")])


class AssertPartsShareFeatureContractTest(unittest.TestCase):
    def _write(self, path, columns):
        df = pd.DataFrame({c: np.ones(1, dtype=np.float64) for c in columns})
        df["has_own_quality_metrics"] = True   # part of the contract, though not a feature
        df.to_parquet(path, index=False)

    def test_accepts_parts_carrying_the_full_contract(self):
        cols = [c for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.parquet")
            self._write(p, cols)
            dataset._assert_parts_share_feature_contract([p])  # no raise

    def test_rejects_a_part_missing_a_new_feature_column(self):
        cols = [c for c in features.FULL_FEATURES
                if c not in ("ci_asymmetry", "ci_over_eh", "coverage")]
        with tempfile.TemporaryDirectory() as d:
            good = os.path.join(d, "new.parquet")
            stale = os.path.join(d, "old.parquet")
            self._write(good, cols + ["coverage"])
            self._write(stale, cols)
            with self.assertRaises(RuntimeError) as ctx:
                dataset._assert_parts_share_feature_contract([good, stale])
            self.assertIn("old.parquet", str(ctx.exception))
            self.assertIn("coverage", str(ctx.exception))
            self.assertNotIn("new.parquet", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
