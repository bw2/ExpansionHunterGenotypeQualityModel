"""Tests for dataset.py: truth loading/joining, filtering, freshness checks, combo assembly.

Network-touching helpers (``_gcs_md5``, ``_list_json_inputs``, ``_download``, ``_bw2_head_sha``) are
exercised by mocking ``subprocess.run`` / ``os.path`` rather than hitting GCS or git.
"""

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
import heldout


def _gz_tsv(path, rows):
    """Writes ``rows`` (list of dicts) as a gzip tab-separated file with a header row."""
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


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
            _gz_tsv(path, [
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
            out = dataset._load_truth_from_genotypes_tsv(path)
            self.assertEqual(len(out), 4)  # 2 unique loci x 2 alleles, not 3 x 2
            self.assertEqual(list(out.columns), ["LocusId", "allele_rank", "true", "purity", "is_negative_locus"])
            row = out[(out["LocusId"] == "1-1-2-A") & (out["allele_rank"] == 0)].iloc[0]
            self.assertEqual(row["true"], 5)
            self.assertEqual(row["purity"], 1.0)
            self.assertFalse(row["is_negative_locus"])
            row = out[(out["LocusId"] == "1-1-2-A") & (out["allele_rank"] == 1)].iloc[0]
            self.assertEqual(row["true"], 9)

    def test_eh_catalog_filters_drop_hom_ref_nonprimary_and_unparseable(self):
        # Mirrors convert_truth_set_to_variant_catalogs.py: keep only primary-contig, variant loci
        # with parseable repeat counts. Only the one variant primary-contig locus survives.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truth.tsv.gz")
            _gz_tsv(path, [
                {"LocusId": "1-1-2-A", "Chrom": "chr1", "NumRepeatsInReference": 5,  # hom-ref -> drop
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 5,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
                {"LocusId": "1-3-4-A", "Chrom": "chr1", "NumRepeatsInReference": 5,  # variant -> keep
                 "NumRepeatsShortAllele": 5, "NumRepeatsLongAllele": 9,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 0.9},
                {"LocusId": "M-1-2-A", "Chrom": "chrM", "NumRepeatsInReference": 1,  # non-primary -> drop
                 "NumRepeatsShortAllele": 2, "NumRepeatsLongAllele": 4,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
                {"LocusId": "1-9-9-A", "Chrom": "chr1", "NumRepeatsInReference": 1,  # unparseable -> drop
                 "NumRepeatsShortAllele": "NA", "NumRepeatsLongAllele": 4,
                 "RepeatPurityShortAllele": 1.0, "RepeatPurityLongAllele": 1.0},
            ])
            out = dataset._load_truth_from_genotypes_tsv(path)
            self.assertEqual(set(out["LocusId"]), {"1-3-4-A"})
            self.assertEqual(len(out), 2)  # the one kept locus -> Short + Long allele rows


class JoinTruthTest(unittest.TestCase):
    def test_strips_chr_prefix_and_joins(self):
        json_df = pd.DataFrame({"locus_id": ["chr1-1-2-A", "chr1-1-2-A"], "allele_rank": [0, 1], "eh": [10, 20]})
        tsv_df = pd.DataFrame({"LocusId": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 1], "true": [9, 21],
                              "purity": [1.0, 1.0], "is_negative_locus": [False, False]})
        merged = dataset._join_truth(json_df, tsv_df, "unit")
        self.assertEqual(list(merged["true"]), [9, 21])
        self.assertNotIn("LocusId", merged.columns)

    def test_unmatched_json_row_gets_nan_truth(self):
        json_df = pd.DataFrame({"locus_id": ["1-1-2-A"], "allele_rank": [0], "eh": [10]})
        tsv_df = pd.DataFrame({"LocusId": [], "allele_rank": [], "true": [], "purity": [],
                              "is_negative_locus": []})
        merged = dataset._join_truth(json_df, tsv_df, "unit")
        self.assertTrue(pd.isna(merged["true"].iloc[0]))

    def test_duplicate_json_key_raises(self):
        json_df = pd.DataFrame({"locus_id": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 0], "eh": [10, 11]})
        tsv_df = pd.DataFrame({"LocusId": ["1-1-2-A"], "allele_rank": [0], "true": [9], "purity": [1.0],
                              "is_negative_locus": [False]})
        with self.assertRaises(AssertionError):
            dataset._join_truth(json_df, tsv_df, "unit")


class AssertCatalogAgreementTest(unittest.TestCase):
    def test_agreement_within_tolerance_does_not_raise(self):
        json_df = pd.DataFrame({"locus_id": ["1-%d-2-A" % i for i in range(100)]})
        tsv_df = pd.DataFrame({"LocusId": ["1-%d-2-A" % i for i in range(100)], "true": [10] * 100})
        dataset._assert_catalog_agreement(json_df, tsv_df, "unit", fatal=True)  # must not raise

    def test_mismatch_beyond_tolerance_raises_when_fatal(self):
        json_df = pd.DataFrame({"locus_id": ["1-0-2-A"]})
        tsv_df = pd.DataFrame({"LocusId": ["1-0-2-A", "1-1-2-A", "1-2-2-A"], "true": [10, 300, 300]})
        with self.assertRaises(RuntimeError):
            dataset._assert_catalog_agreement(json_df, tsv_df, "unit", fatal=True)

    def test_mismatch_beyond_tolerance_only_warns_when_not_fatal(self):
        json_df = pd.DataFrame({"locus_id": ["1-0-2-A"]})
        tsv_df = pd.DataFrame({"LocusId": ["1-0-2-A", "1-1-2-A", "1-2-2-A"], "true": [10, 300, 300]})
        dataset._assert_catalog_agreement(json_df, tsv_df, "unit", fatal=False)  # must not raise

    def test_empty_union_is_a_no_op(self):
        dataset._assert_catalog_agreement(pd.DataFrame({"locus_id": []}), pd.DataFrame({"LocusId": [], "true": []}),
                                          "unit", fatal=True)


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
                                 "missing_motif_size": 1})
        self.assertEqual(len(kept), 2)
        self.assertNotIn("locus_id", kept.columns)
        self.assertNotIn("is_negative_locus", kept.columns)
        quick_row = kept[kept["genotyping_branch"] == "quick"].iloc[0]
        self.assertEqual(quick_row["chrom"], "1")
        self.assertEqual(quick_row["genotyping_regime"], "quick")
        full_row = kept[kept["genotyping_branch"] == "full"].iloc[0]
        self.assertEqual(full_row["chrom"], "2")
        self.assertEqual(full_row["genotyping_regime"], "full_nonspanning")  # spanning_at_called == 0


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


class GcsMd5Test(unittest.TestCase):
    def test_parses_hash_line(self):
        stdout = "gs://x/y.json.gz:\n    Creation time:  ...\n    Hash (md5):          abcd1234==\n"
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
            self.assertEqual(dataset._gcs_md5("gs://x/y.json.gz"), "abcd1234==")

    def test_missing_hash_line_returns_none(self):
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout="not found\n")):
            self.assertIsNone(dataset._gcs_md5("gs://x/y.json.gz"))


class GcsMtimeTest(unittest.TestCase):
    def test_prefers_update_over_creation(self):
        import email.utils
        stdout = ("gs://x/y:\n    Creation time:    Wed, 02 Jul 2026 10:00:00 GMT\n"
                  "    Update time:      Wed, 02 Jul 2026 22:46:00 GMT\n    Hash (md5): abc==\n")
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
            got = dataset._gcs_mtime("gs://x/y")
        self.assertAlmostEqual(
            got, email.utils.parsedate_to_datetime("Wed, 02 Jul 2026 22:46:00 GMT").timestamp())

    def test_falls_back_to_creation_time(self):
        import email.utils
        stdout = "gs://x/y:\n    Creation time:    Wed, 02 Jul 2026 10:00:00 GMT\n"
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
            self.assertAlmostEqual(
                dataset._gcs_mtime("gs://x/y"),
                email.utils.parsedate_to_datetime("Wed, 02 Jul 2026 10:00:00 GMT").timestamp())

    def test_no_time_line_returns_none(self):
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout="nope\n")):
            self.assertIsNone(dataset._gcs_mtime("gs://x/y"))


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
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
            self.assertEqual(dataset._list_json_inputs("S", "V", "10x"), ["gs://x/json/a.json.gz"])

    def test_falls_back_to_shards(self):
        stdout = "gs://x/json/a.shard000_of_002.json.gz gs://x/json/a.shard001_of_002.json.gz"
        with mock.patch.object(dataset.subprocess, "run", return_value=SimpleNamespace(stdout=stdout)):
            self.assertEqual(dataset._list_json_inputs("S", "V", "10x"),
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
        with mock.patch.object(dataset, "_gcs_md5", side_effect=AssertionError("should not be called")):
            n = dataset._check_freshness("d", [("json shard", "gs://r", "/nonexistent/path")])
        self.assertEqual(n, 0)

    def test_up_to_date_is_a_no_op(self):
        # md5 matches AND the bucket is not newer than the local file -> keep it, nothing removed.
        with mock.patch.object(dataset, "_gcs_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_gcs_mtime", return_value=self.lm - 100), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 0)
        self.assertTrue(os.path.exists(self.local))

    def test_md5_mismatch_deletes_for_redownload(self):
        with mock.patch.object(dataset, "_gcs_md5", return_value="new"), \
             mock.patch.object(dataset, "_local_md5", return_value="old"), \
             mock.patch.object(dataset, "_gcs_mtime", return_value=self.lm - 100), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 1)
        self.assertFalse(os.path.exists(self.local))  # deleted so _download re-fetches

    def test_bucket_newer_mtime_deletes_for_redownload(self):
        with mock.patch.object(dataset, "_gcs_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_gcs_mtime", return_value=self.lm + 100), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value=None):
            n = dataset._check_freshness("d", [("json shard", "gs://r", self.local)])
        self.assertEqual(n, 1)
        self.assertFalse(os.path.exists(self.local))

    def test_eh_build_staleness_refuses_for_kept_json(self):
        # up-to-date content but produced by an older EH build -> hard refusal (can't fix by redownload).
        with mock.patch.object(dataset, "_gcs_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_gcs_mtime", return_value=self.lm - 100), \
             mock.patch.object(dataset, "_json_eh_version", return_value="old_sha"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="new_sha"):
            with self.assertRaises(SystemExit):
                dataset._check_freshness("d", [("json shard", "gs://r", self.local)])

    def test_eh_build_staleness_ignored_for_non_json_labels(self):
        with mock.patch.object(dataset, "_gcs_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_gcs_mtime", return_value=self.lm - 100), \
             mock.patch.object(dataset, "_json_eh_version", return_value="old_sha"), \
             mock.patch.object(dataset, "_bw2_head_sha", return_value="new_sha"):
            n = dataset._check_freshness("d", [("truth-genotypes TSV", "gs://r", self.local)])
        self.assertEqual(n, 0)  # build check only applies to json-labeled sources

    def test_cache_stale_json_is_redownloaded_without_build_refusal(self):
        # A json that is BOTH bucket-newer AND from an older build is re-downloaded (not refused): the
        # fresh copy replaces it, so its stale build sha is not judged here.
        with mock.patch.object(dataset, "_gcs_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_local_md5", return_value="samehash"), \
             mock.patch.object(dataset, "_gcs_mtime", return_value=self.lm + 100), \
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
            pd.DataFrame({"eh": [1.0, 2.0, 3.0]}).to_parquet(out_path)
            newer = max(os.path.getmtime(json_local), os.path.getmtime(tsv_local)) + 100
            os.utime(out_path, (newer, newer))
            with mock.patch.object(dataset, "_list_json_inputs", return_value=["gs://x/a.json.gz"]), \
                 mock.patch.object(dataset, "_check_freshness", return_value=0), \
                 mock.patch.object(dataset, "_download", side_effect=AssertionError("should not download")):
                n = dataset.build_combo("V", "sub", "HG002", "10x", d, force=False)
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
                 mock.patch.object(dataset.eh_json, "extract_rows", return_value=fake_rows), \
                 mock.patch.object(dataset, "_load_truth_from_genotypes_tsv", return_value=fake_tsv_df):
                n = dataset.build_combo("V", "sub", "HG002", "10x", d, force=True)
            self.assertEqual(n, 3)
            out_path = os.path.join(d, "sub", "HG002_10x.parquet")
            merged = pd.read_parquet(out_path)
            self.assertNotIn("sample_id", merged.columns)
            self.assertEqual(merged["true"].dtype, np.float32)
            row = merged[(merged["locus_id"] == "1-1-2-A") & (merged["allele_rank"] == 1)].iloc[0]
            self.assertEqual(row["true"], 21.0)


class AssembleBranchTest(unittest.TestCase):
    def test_filters_branch_and_writes_labeled_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = os.path.join(d, "src")
            os.makedirs(src_dir)
            pd.DataFrame([
                {"locus_id": "1-1-2-A", "is_negative_locus": False, "eh": 10.0, "true": 10.0,
                 "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5},
                {"locus_id": "1-3-4-A", "is_negative_locus": False, "eh": np.nan, "true": 10.0,
                 "motif_size": 3, "genotyping_branch": "quick", "spanning_at_called": 5},
                {"locus_id": "2-1-2-A", "is_negative_locus": False, "eh": 14.0, "true": 10.0,
                 "motif_size": 3, "genotyping_branch": "full", "spanning_at_called": 0},
            ]).to_parquet(os.path.join(src_dir, "combo1.parquet"))

            dataset.assemble_branch(d, "quick", "src")
            quick = pd.read_parquet(os.path.join(d, "parquet", "quick.parquet"))
            self.assertEqual(len(quick), 1)  # the NaN-eh row is dropped, the full row excluded

            dataset.assemble_branch(d, "full", "src")
            full = pd.read_parquet(os.path.join(d, "parquet", "full.parquet"))
            self.assertEqual(len(full), 1)
            self.assertEqual(full.iloc[0]["genotyping_regime"], "full_nonspanning")

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
            open(os.path.join(d, "parquet", "%s.parquet" % b), "w").close()

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


if __name__ == "__main__":
    unittest.main()
