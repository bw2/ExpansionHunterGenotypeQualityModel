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


class GenotypeLevelLabelTest(unittest.TestCase):
    """``dir_code_genotype``: a homozygous call is OK only if it is OK against both truth alleles."""

    def _call(self, eh, true_short, true_long, n_alleles=2, n_distinct=1, motif_size=3):
        rows = [{"locus_id": "1-9-10-A", "is_negative_locus": False, "eh": eh, "true": true_short,
                 "motif_size": motif_size, "genotyping_branch": "quick", "spanning_at_called": 5,
                 "allele_rank": 0, "n_alleles": n_alleles, "n_distinct_alleles": n_distinct,
                 "has_own_quality_metrics": True}]
        if n_alleles == 2:
            rows.append(dict(rows[0], allele_rank=1, true=true_long, has_own_quality_metrics=n_distinct == 2))
        kept, _ = dataset.label_and_filter(pd.DataFrame(rows))
        return kept

    def test_reference_call_that_misses_an_expansion_is_too_short(self):
        kept = self._call(eh=10.0, true_short=10.0, true_long=15.0)
        self.assertEqual(len(kept), 1)  # the rank-1 twin is still dropped
        self.assertEqual(kept["dir_code"].iloc[0], features.OK)  # per allele: matched to the short truth
        self.assertEqual(kept["true_other_allele"].iloc[0], 15.0)
        self.assertEqual(kept["dir_code_genotype"].iloc[0], features.TOO_SHORT)

    def test_reference_call_that_misses_a_contraction_stays_too_long(self):
        kept = self._call(eh=10.0, true_short=5.0, true_long=10.0)
        self.assertEqual(kept["dir_code"].iloc[0], features.TOO_LONG)
        self.assertEqual(kept["dir_code_genotype"].iloc[0], features.TOO_LONG)

    def test_call_right_for_both_alleles_is_ok(self):
        kept = self._call(eh=10.0, true_short=10.0, true_long=10.0)
        self.assertEqual(kept["dir_code_genotype"].iloc[0], features.OK)

    def test_the_larger_miss_wins_and_a_tie_goes_to_the_longer_allele(self):
        larger_long_miss = self._call(eh=10.0, true_short=8.0, true_long=15.0)
        self.assertEqual(larger_long_miss["dir_code_genotype"].iloc[0], features.TOO_SHORT)
        larger_short_miss = self._call(eh=10.0, true_short=4.0, true_long=12.0)
        self.assertEqual(larger_short_miss["dir_code_genotype"].iloc[0], features.TOO_LONG)
        tie = self._call(eh=10.0, true_short=7.0, true_long=13.0)
        self.assertEqual(tie["dir_code_genotype"].iloc[0], features.TOO_SHORT)

    def test_when_both_alleles_are_missed_the_larger_miss_in_repeats_wins(self):
        # HG002 10x, locus 2-59086030-59086114 (36 bp motif): call 9/9, truth 5/14. Tolerances are 2 repeats
        # (180 bp) and 4 repeats (504 bp); the misses are 4 and 5 repeats, so the long allele's miss is larger
        # although it is smaller once each tolerance is subtracted.
        kept = self._call(eh=9.0, true_short=5.0, true_long=14.0, motif_size=36)
        self.assertEqual(kept["dir_code"].iloc[0], features.TOO_LONG)
        self.assertEqual(kept["dir_code_genotype"].iloc[0], features.TOO_SHORT)

    def test_tolerance_comes_from_each_truth_allele_size(self):
        # 20 repeats x 3 bp = 60 bp -> +-1 repeat; the 10-repeat (30 bp) short allele needs an exact match.
        kept = self._call(eh=10.0, true_short=10.0, true_long=21.0)
        self.assertEqual(kept["dir_code_genotype"].iloc[0], features.TOO_SHORT)
        within = self._call(eh=20.0, true_short=20.0, true_long=21.0)
        self.assertEqual(within["dir_code_genotype"].iloc[0], features.OK)

    def test_heterozygous_and_one_allele_calls_keep_their_per_allele_label(self):
        het = self._call(eh=10.0, true_short=10.0, true_long=15.0, n_distinct=2)
        self.assertEqual(list(het["dir_code_genotype"]), list(het["dir_code"]))
        self.assertTrue(het["true_other_allele"].isna().all())
        one_allele = self._call(eh=10.0, true_short=15.0, true_long=15.0, n_alleles=1)
        self.assertEqual(list(one_allele["dir_code_genotype"]), [features.TOO_SHORT])
        self.assertTrue(one_allele["true_other_allele"].isna().all())

    def test_two_sources_in_one_frame_are_refused(self):
        rows = [{"locus_id": "1-9-10-A", "is_negative_locus": False, "eh": 10.0, "true": 10.0, "motif_size": 3,
                 "genotyping_branch": "quick", "spanning_at_called": 5, "allele_rank": rank, "n_alleles": 2,
                 "n_distinct_alleles": 1, "has_own_quality_metrics": rank == 0} for rank in (0, 1, 0, 1)]
        with self.assertRaises(ValueError):
            dataset.label_and_filter(pd.DataFrame(rows))


class KeepHomRefLociTest(unittest.TestCase):
    """``extract_rows_and_join_truth(..., keep_hom_ref_loci=True)`` and its helpers."""

    TRUTH_DF = ExtractRowsAndJoinTruthTest.TRUTH_DF
    TRUTH_BY_LOCUS = pd.DataFrame({"reference_repeats": [8.0, 4.0, 6.0], "is_hom_ref": [False, True, True],
                                   "purity": [1.0, 1.0, 0.9]},
                                  index=pd.Index(["1-1-2-A", "1-5-6-A", "1-7-8-A"], name="LocusId"))

    @staticmethod
    def _row(locus_id, rank, eh, branch="quick", motif_size=3):
        return {"locus_id": locus_id, "allele_rank": rank, "eh": eh, "motif_size": motif_size,
                "genotyping_branch": branch}

    def _run(self, rows, hash_fractions, keep_hom_ref_loci=True, row_sample_id="HG002_10x"):
        individuals = []

        def hash_fraction(locus, individual):
            individuals.append(individual)
            return hash_fractions[locus]

        with mock.patch.object(dataset.eh_json, "extract_rows", return_value=rows), \
             mock.patch.object(dataset.eh_json, "typical_read_length_in_file", return_value=150), \
             mock.patch.object(dataset, "_load_truth_from_genotypes_tsv",
                               return_value=(self.TRUTH_DF.copy(), {"1-1-2-A", "1-5-6-A", "1-7-8-A"})), \
             mock.patch.object(dataset, "_load_reference_repeats_and_hom_ref_truth",
                               return_value=self.TRUTH_BY_LOCUS) as load_hom_ref, \
             mock.patch.object(dataset, "_locus_hash_fraction", side_effect=hash_fraction), \
             mock.patch.object(dataset, "_assert_catalog_agreement"):
            merged = dataset.extract_rows_and_join_truth(["a.json.gz"], "t.tsv.gz", "t.bed.gz", row_sample_id,
                                                         "unit", keep_hom_ref_loci=keep_hom_ref_loci)
            self.assertEqual(load_hom_ref.called, keep_hom_ref_loci)
            # Loci are sampled per individual: HG002_10x's coverage suffix is dropped.
            self.assertEqual(set(individuals), {"HG002"} if individuals else set())
            return merged

    def test_keeps_sampled_hom_ref_loci_whole_with_their_truth(self):
        rows = [self._row("chr1-1-2-A", 0, 10.0), self._row("chr1-1-2-A", 1, 20.0),   # variant locus
                self._row("chr1-5-6-A", 0, 4.0), self._row("chr1-5-6-A", 1, 4.0),     # hom-ref, sampled
                self._row("chr1-7-8-A", 0, 6.0), self._row("chr1-7-8-A", 1, 6.0),     # hom-ref, not sampled
                self._row("chr1-9-9-A", 0, 5.0)]                                       # no truth
        merged = self._run(rows, {"1-1-2-A": 0.5, "1-5-6-A": 0.01, "1-7-8-A": 0.99})
        self.assertEqual(list(merged["locus_id"]), ["1-1-2-A", "1-1-2-A", "1-5-6-A", "1-5-6-A"])
        self.assertEqual(list(merged["true"]), [9.0, 21.0, 4.0, 4.0])
        self.assertEqual(list(merged["purity"]), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(list(merged["variant_locus"]), [True, True, False, False])
        self.assertEqual(list(merged["truth_reference_repeats"]), [8.0, 8.0, 4.0, 4.0])
        # 20 repeats x 3 bp = 60 bp puts the variant locus in the 60-100 bp bin (rate 0.5); the hom-ref locus
        # (12 bp calls) is in the lowest bin (0.02).
        self.assertEqual(list(merged["all_loci_keep_probability"]), [0.5, 0.5, 0.02, 0.02])
        self.assertEqual(list(merged["in_all_loci_pool"]), [False, False, True, True])
        self.assertEqual(list(merged["all_loci_representative"]), [False, False, True, True])

    def test_default_extraction_is_unchanged(self):
        rows = [self._row("chr1-1-2-A", 0, 10.0), self._row("chr1-1-2-A", 1, 20.0),
                self._row("chr1-5-6-A", 0, 4.0)]
        merged = self._run(rows, {}, keep_hom_ref_loci=False)
        self.assertEqual(list(merged["locus_id"]), ["1-1-2-A", "1-1-2-A"])
        self.assertNotIn("variant_locus", merged.columns)

    def test_keep_rates_follow_the_called_size_and_the_branch(self):
        rates = dataset._locus_keep_and_representative_rates
        quick = dataset.HOM_REF_KEEP_RATE_BY_CALLED_SIZE_BIN[dataset.eh_json.BRANCH_QUICK]
        self.assertEqual(rates([self._row("x", 0, 5.0)]), (quick[0], quick[0]))           # 15 bp
        self.assertEqual(rates([self._row("x", 0, 5.0), self._row("x", 1, 40.0)]),
                         (quick[4], quick[0]))                                            # 120 bp allele wins
        self.assertEqual(rates([self._row("x", 0, None), self._row("x", 1, None)]), (0.0, 0.0))  # no-call locus
        self.assertEqual(rates([self._row("x", 0, None, branch=dataset.eh_json.BRANCH_FULL)]), (0.0, 0.0))
        self.assertEqual(rates([self._row("x", 0, 5.0, branch=dataset.eh_json.BRANCH_FULL)]), (1.0, 1.0))
        for branch_rates in dataset.HOM_REF_KEEP_RATE_BY_CALLED_SIZE_BIN.values():
            self.assertEqual(list(branch_rates), sorted(branch_rates))  # the lowest bin's rate is the lowest

    def test_locus_hash_fraction_is_fixed_in_range_and_differs_between_individuals(self):
        loci = ["1-%d-%d-CAG" % (i, i + 30) for i in range(1000)]
        values = [dataset._locus_hash_fraction(locus, "HG002") for locus in loci]
        self.assertEqual(values[:3], [dataset._locus_hash_fraction(locus, "HG002") for locus in loci[:3]])
        self.assertTrue(all(0.0 <= v < 1.0 for v in values))
        self.assertAlmostEqual(np.mean(values), 0.5, delta=0.05)
        other = [dataset._locus_hash_fraction(locus, "HG00738") for locus in loci]
        kept_by_both = sum(a < 0.1 and b < 0.1 for a, b in zip(values, other))
        self.assertLess(kept_by_both, 30)  # about 10 expected for independent 10% samples of 1,000 loci

    def test_no_call_loci_are_left_out_of_the_all_loci_sample(self):
        rows = [self._row("chr1-1-2-A", 0, None, branch="full"), self._row("chr1-1-2-A", 1, None, branch="full"),
                self._row("chr1-5-6-A", 0, None, branch="full"), self._row("chr1-5-6-A", 1, None, branch="full")]
        merged = self._run(rows, {"1-1-2-A": 0.0, "1-5-6-A": 0.0})
        # The variant no-call locus stays for the variant-only views; the hom-ref one is not kept.
        self.assertEqual(list(merged["locus_id"]), ["1-1-2-A", "1-1-2-A"])
        self.assertEqual(list(merged["all_loci_keep_probability"]), [0.0, 0.0])
        self.assertFalse(merged["in_all_loci_pool"].any() or merged["all_loci_representative"].any())

    def test_keep_hom_ref_loci_needs_a_sample_id(self):
        with self.assertRaises(ValueError):
            self._run([self._row("chr1-1-2-A", 0, 10.0)], {"1-1-2-A": 0.5}, row_sample_id=None)

    def test_reference_repeats_and_hom_ref_truth_cover_only_truth_loci(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truth.tsv.gz")
            _write_truth_tsv_and_bed(path, [
                {"LocusId": "chr1-1-2-A", "Chrom": "chr1", "NumRepeatsInReference": 5,      # hom-ref
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 5,
                 "RepeatPurityShortAllele": 0.9, "RepeatPurityLongAllele": 0.9},
                {"LocusId": "1-3-4-A", "Chrom": "chr1", "NumRepeatsInReference": 5,         # variant
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 9,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
                {"LocusId": "1-5-6-A", "Chrom": "chr1", "NumRepeatsInReference": 2,         # not a truth locus
                 "NumRepeatsShortAllele": 2, "NumRepeatsLongAllele": 2,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
            ])
            out = dataset._load_reference_repeats_and_hom_ref_truth(path, {"1-1-2-A", "1-3-4-A"})
        self.assertEqual(sorted(out.index), ["1-1-2-A", "1-3-4-A"])
        self.assertEqual(list(out.loc[["1-1-2-A", "1-3-4-A"], "reference_repeats"]), [5.0, 5.0])
        self.assertEqual(list(out.loc[["1-1-2-A", "1-3-4-A"], "is_hom_ref"]), [True, False])
        self.assertEqual(out.at["1-1-2-A", "purity"], 0.9)

    def test_variant_only_readers_and_rebuild_check(self):
        df = pd.DataFrame({"eh": [1.0, 2.0, 3.0], "variant_locus": [True, False, True]})
        self.assertEqual(list(dataset.variant_locus_rows(df)["eh"]), [1.0, 3.0])
        old = df.drop(columns=["variant_locus"])
        self.assertIs(dataset.variant_locus_rows(old), old)
        with tempfile.TemporaryDirectory() as d:
            with_columns, without_columns = os.path.join(d, "a.parquet"), os.path.join(d, "b.parquet")
            df.to_parquet(with_columns)
            old.to_parquet(without_columns)
            self.assertTrue(dataset.lacks_all_loci_columns(without_columns, keep_hom_ref_loci=True))
            self.assertFalse(dataset.lacks_all_loci_columns(with_columns, keep_hom_ref_loci=True))
            self.assertFalse(dataset.lacks_all_loci_columns(without_columns, keep_hom_ref_loci=False))
            self.assertFalse(dataset.lacks_all_loci_columns(os.path.join(d, "missing.parquet"), True))


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
                 mock.patch.object(dataset, "_past_build_only_commits", side_effect=lambda sha: sha), \
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

    @staticmethod
    def _all_loci_part(path):
        """An all-loci source: 2 representative hom-ref loci, 1 pool-only hom-ref locus, 1 variant locus in the
        pool and 1 variant locus outside it."""
        base = {c: 1.0 for c in features.FULL_FEATURES if c not in ("ci_asymmetry", "ci_over_eh")}
        base.update({"has_own_quality_metrics": True, "is_negative_locus": False, "true": 10.0, "eh": 10.0,
                     "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5,
                     "all_loci_keep_probability": 0.5})
        rows = [dict(base, locus_id="1-1-2-A", variant_locus=False, in_all_loci_pool=True, all_loci_representative=True),
                dict(base, locus_id="1-3-4-A", variant_locus=False, in_all_loci_pool=True, all_loci_representative=True),
                dict(base, locus_id="1-5-6-A", variant_locus=False, in_all_loci_pool=True, all_loci_representative=False),
                dict(base, locus_id="1-7-8-A", variant_locus=True, in_all_loci_pool=True, all_loci_representative=False,
                     true=12.0),
                dict(base, locus_id="1-9-10-A", variant_locus=True, in_all_loci_pool=False,
                     all_loci_representative=False, true=12.0)]
        pd.DataFrame(rows).to_parquet(path)

    def test_all_loci_population_uses_the_representative_rows_and_the_pool(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            self._all_loci_part(os.path.join(d, "src", "a.parquet"))
            dataset.assemble_branch(d, "quick", "src", training_population="all_loci")
            out = pd.read_parquet(os.path.join(d, "parquet_all_loci", "quick.parquet"))
            # Within the cap the whole pool is kept: 2 representative rows plus the 2 other pool rows; the
            # variant locus outside the pool is left out.
            self.assertEqual(len(out), 4)
            self.assertEqual(int(out["representative"].sum()), 2)
            self.assertTrue(out.loc[out["representative"], "all_loci_representative"].all())
            self.assertTrue(out["in_all_loci_pool"].all())
            # The variant-only recipe on the same part keeps the 2 variant-locus rows and writes data/parquet/.
            dataset.assemble_branch(d, "quick", "src")
            self.assertEqual(len(pd.read_parquet(os.path.join(d, "parquet", "quick.parquet"))), 2)

    def test_all_loci_cap_bounds_the_representative_rows_and_tops_up_only_from_the_pool(self):
        df = pd.DataFrame({"genotyping_regime": ["quick"] * 6, "motif_size": [3.0] * 6,
                           "eh": [10.0, 10.0, 10.0, 10.0, 40.0, 40.0],
                           "in_all_loci_pool": [True, True, True, True, True, False],
                           "all_loci_representative": [True, True, True, False, False, False]})
        out = dataset._cap_all_loci_rows_per_genotyping_regime(df, cap=2, seed=1)
        self.assertEqual(int(out["representative"].sum()), 2)            # 3 representative rows, capped to 2
        self.assertTrue(out["in_all_loci_pool"].all())                   # never the row outside the pool
        self.assertIn(40.0, list(out.loc[~out["representative"], "eh"]))  # the rare 120 bp cell is topped up

    def test_all_loci_population_refuses_variant_only_parts(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            write_contract_parquet(os.path.join(d, "src", "old.parquet"))
            with self.assertRaises(RuntimeError):
                dataset.assemble_branch(d, "quick", "src", training_population="all_loci")


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

    def test_keep_hom_ref_loci_reaches_every_promoted_sample(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.parquet")
            pd.DataFrame({"eh": [1.0]}).to_parquet(src)
            with mock.patch.object(heldout, "build_sample", return_value=src) as m:
                dataset._link_promoted_heldout_samples(d, force=False, keep_hom_ref_loci=True)
            self.assertEqual(len(m.call_args_list), len(dataset.PROMOTED_HELDOUT_SAMPLES))
            self.assertTrue(all(c.kwargs["keep_hom_ref_loci"] for c in m.call_args_list))

    def test_main_passes_keep_hom_ref_loci_to_combos_and_promoted_samples(self):
        with mock.patch.object(dataset, "build_combo") as build_combo, \
             mock.patch.object(dataset, "_link_promoted_heldout_samples") as link, \
             mock.patch.object(dataset, "assemble_branch"), \
             mock.patch("sys.argv", ["dataset.py", "--keep-hom-ref-loci"]):
            dataset.main()
        self.assertEqual(len(build_combo.call_args_list), len(dataset.COMBOS))
        self.assertTrue(all(c.kwargs["keep_hom_ref_loci"] for c in build_combo.call_args_list))
        self.assertTrue(link.call_args.kwargs["keep_hom_ref_loci"])


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

    def test_the_suggested_command_rebuilds_the_checked_population_and_keeps_the_hom_ref_loci(self):
        # No --force (it would rebuild every per-source parquet), and --keep-hom-ref-loci wherever a rebuild
        # without it would replace per-source parquets that keep the hom-ref loci with variant-only ones.
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "PROMOTED_HELDOUT_SAMPLES", ()):
            self._setup(d)
            self.assertEqual(dataset.command_that_builds_branch_parquets(d, "parquet"),
                             "`python3 dataset.py --data-dir %s`" % d)
            with self.assertRaises(SystemExit) as raised:
                dataset.assert_parquets_up_to_date(d, parquet_subdir="parquet_all_loci")
            self.assertIn("`python3 dataset.py --data-dir %s --keep-hom-ref-loci --training-population all_loci`" % d,
                          str(raised.exception))
            part = os.path.join(d, dataset.SOURCE_SUBDIR, "HG002_31x.parquet")
            write_contract_parquet(part).assign(variant_locus=True).to_parquet(part, index=False)
            self.assertEqual(dataset.command_that_builds_branch_parquets(d, "parquet"),
                             "`python3 dataset.py --data-dir %s --keep-hom-ref-loci`" % d)

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
                                   side_effect=lambda s, dd, force, keep_hom_ref_loci:
                                   os.path.join(src_dir, "%s.parquet" % s)):
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
    def setUp(self):
        # Keep these tests off the local ExpansionHunter-bw2 checkout (PastBuildOnlyCommitsTest covers it).
        patcher = mock.patch.object(dataset, "_past_build_only_commits", side_effect=lambda sha: sha)
        self.past_build_only_commits = patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_changed_lines_keep_source_lines_that_look_like_file_headers(self):
        stdout = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -3 +3 @@\n---i;\n+++i;\n"
        with mock.patch.object(dataset.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=stdout)):
            self.assertEqual(dataset._changed_lines_since("b1fbc23", "ehunter/io/JsonWriter.cpp"), ["--i;", "++i;"])

    def test_build_is_judged_from_past_the_build_only_commits(self):
        self.past_build_only_commits.side_effect = lambda sha: "ba9ad21" if sha == "b1fbc23" else sha
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="feed123"), \
             mock.patch.object(dataset, "_bw2_files_changed_since", return_value=["docker/sha256.txt"]) as changed:
            dataset.assert_eh_build_matches("combo", [("json shard 0", self._json(os.path.join(d, "a.json.gz"),
                                                                                    "b1fbc23"))])  # must not raise
        changed.assert_called_once_with("ba9ad21")

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


class PastBuildOnlyCommitsTest(unittest.TestCase):
    FIRST, SECOND = "45c7c07" + "1" * 33, "ba9ad21" + "2" * 33  # a chain: parent -> FIRST -> SECOND
    PARENT, OTHER = "b1fbc23" + "3" * 33, "c0ffee1" + "4" * 33

    def _past(self, stamped_commit, git_ok=True):
        def run(cmd, **kwargs):
            if "log" in cmd:  # each build-only commit with its parents
                return mock.Mock(returncode=0, stdout="%s %s\n%s %s\n" % (self.FIRST, self.PARENT,
                                                                         self.SECOND, self.FIRST))
            return mock.Mock(returncode=0 if git_ok else 128, stdout=stamped_commit + "\n")  # rev-parse
        with mock.patch.object(dataset.subprocess, "run", side_effect=run):
            return dataset._past_build_only_commits(stamped_commit[:7])

    def test_follows_the_chain_from_its_parent(self):
        self.assertEqual(self._past(self.PARENT), self.SECOND)

    def test_follows_the_rest_of_the_chain_from_inside_it(self):
        self.assertEqual(self._past(self.FIRST), self.SECOND)

    def test_does_not_move_a_commit_the_chain_does_not_start_from(self):
        self.assertEqual(self._past(self.OTHER), self.OTHER)

    def test_unknown_commit_is_returned_unchanged(self):
        self.assertEqual(self._past(self.OTHER, git_ok=False), self.OTHER[:7])


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


class AllLociPopulationNeedsHomRefLociTest(unittest.TestCase):
    def test_refused_before_any_per_source_parquet_is_built(self):
        argv = ["dataset.py", "--data-dir", "D", "--training-population", "all_loci"]
        with mock.patch("sys.argv", argv), mock.patch.object(dataset, "build_combo") as build_combo, \
             mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            dataset.main()
        build_combo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
