"""Tests for check_truth_genotypes_against_short_reads.py: the judging rule, the loci each person is checked at, and
marking the per-source parquets."""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import check_truth_genotypes_against_short_reads as check
import dataset


def reads(*groups):
    """Builds a measured_reads string from (count, kind, length_bp[, strand[, base_quality[, mapping_quality]]])."""
    out = []
    for group in groups:
        count, kind, length_bp = group[:3]
        strand = group[3] if len(group) > 3 else "-"
        base_quality = group[4] if len(group) > 4 else 30
        mapping_quality = group[5] if len(group) > 5 else 60
        out += ["%s:%d:%d:%d:%s" % (kind, length_bp, mapping_quality, base_quality, strand)] * count
    return ",".join(out)


def judge(measured_reads, truth_short, truth_long, motif_size):
    return check.judge_truth_genotype_against_reads(
        check.parse_measured_reads(measured_reads, motif_size), truth_short, truth_long, motif_size)


class ParseMeasuredReadsTest(unittest.TestCase):
    def test_converts_to_repeats_and_drops_low_mapping_quality(self):
        parsed = check.parse_measured_reads("s:21:60:30:+,l:13:60:12:+,r:13:60:30:-,s:21:19:30:+", 3)
        self.assertEqual(parsed, [("s", 7, 30, False), ("l", 4, 12, True), ("r", 4, 30, True)])
        self.assertEqual(check.parse_measured_reads("", 3), [])


class JudgeTruthGenotypeTest(unittest.TestCase):
    def test_missed_heterozygous_allele_is_contradicted(self):
        self.assertEqual(judge(reads((12, "s", 20), (6, "s", 28)), 10, 10, 2),
                         check.SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS)

    def test_reads_matching_the_truth_are_not_contradicted(self):
        self.assertEqual(judge(reads((12, "s", 20), (6, "s", 28)), 10, 14, 2), check.TRUTH_ALLELES_BOTH_SUPPORTED)
        self.assertEqual(judge(reads((18, "s", 20)), 10, 10, 2), check.NOT_CONTRADICTED)

    def test_fewer_than_10_reads_are_never_judged(self):
        self.assertEqual(judge(reads((6, "s", 20), (3, "s", 28)), 10, 10, 2), check.TOO_FEW_READS)

    def test_a_third_allele_next_to_two_supported_truth_alleles_is_not_contradicted(self):
        # Mosaicism or a mapping artifact can add an allele; the assembly is not clearly wrong.
        self.assertEqual(judge(reads((8, "s", 20), (6, "s", 28), (5, "s", 36)), 10, 14, 2),
                         check.TRUTH_ALLELES_BOTH_SUPPORTED)

    def test_both_truth_alleles_count_as_supported_only_when_spanning_reads_could_show_them(self):
        # No spanning reads: support was never checked.
        self.assertEqual(judge(reads((12, "l", 80)), 10, 12, 2), check.NOT_CONTRADICTED)
        # The long allele is too long to span, so its support can't be checked.
        self.assertEqual(judge(reads((10, "s", 20), (2, "l", 30)), 10, 60, 2), check.NOT_CONTRADICTED)

    def test_spanning_reads_showing_a_deleted_repeat_count_as_clean(self):
        # A 0 bp spanning read has no base quality (-1): truth 5/5, reads 5 and 0 repeats is a missed deletion.
        self.assertEqual(judge(reads((10, "s", 10), (9, "s", 0, "+", -1)), 5, 5, 2),
                         check.SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS)

    def test_reads_count_toward_the_nearest_truth_allele_only(self):
        # Truth 8/9, reads 8 and 11: the reads at 8 must not count as support for 9.
        self.assertEqual(judge(reads((16, "s", 16), (9, "s", 22)), 8, 9, 2),
                         check.SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS)

    def test_a_2_bp_homopolymer_difference_is_not_enough(self):
        self.assertEqual(judge(reads((10, "s", 20)), 18, 18, 1), check.NOT_CONTRADICTED)
        self.assertEqual(judge(reads((10, "s", 21)), 18, 18, 1), check.SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS)

    def test_a_1_repeat_difference_is_not_enough(self):
        self.assertEqual(judge(reads((10, "s", 22)), 10, 10, 2), check.NOT_CONTRADICTED)

    def test_too_few_or_too_rare_reads_with_an_allele_the_truth_lacks(self):
        self.assertEqual(judge(reads((14, "s", 20), (3, "s", 28)), 10, 10, 2), check.NOT_CONTRADICTED)
        self.assertEqual(judge(reads((16, "s", 20), (5, "s", 28)), 10, 10, 2), check.NOT_CONTRADICTED)  # 24%

    def test_reads_longer_than_an_allele_too_long_to_span(self):
        measured = reads((8, "s", 40), (6, "l", 100), (2, "r", 120))
        self.assertEqual(judge(measured, 20, 20, 2), check.READS_LONGER_THAN_THE_TRUTH)
        # The truth's long allele is itself too long to be spanned: the longer reads contradict it directly.
        self.assertEqual(judge(reads((8, "s", 40), (6, "l", 130), (2, "r", 140)), 20, 50, 2),
                         check.READS_LONGER_THAN_THE_TRUTH)

    def test_reads_longer_than_the_truth_need_robust_reads(self):
        # One-sided reads with the motif run at their sequencing 3' end (l on +) can be poly-G or low-quality tails.
        self.assertEqual(judge(reads((8, "s", 40), (8, "l", 100, "+")), 20, 20, 2), check.NOT_CONTRADICTED)
        self.assertEqual(judge(reads((8, "s", 40), (8, "l", 100, "-", 12)), 20, 20, 2), check.NOT_CONTRADICTED)

    def test_an_unreliable_long_read_does_not_lift_the_spanning_requirement(self):
        # Truth A 11/11, all spanning reads at 11, robust longer reads of 17-23 bp, and one 105 bp read tail of base
        # quality 4 at the read's 3' end.
        # The 17 and 23 bp reads are robust (their motif run is at the sequencing 5' end), so 4 longer reads are robust
        # and only the spanning requirement keeps the locus unmarked.
        measured = reads((11, "s", 11), (2, "l", 17), (2, "r", 23, "+"), (1, "l", 105, "+", 4))
        self.assertEqual(judge(measured, 11, 11, 1), check.NOT_CONTRADICTED)

    def test_reads_below_the_mapping_quality_floor_are_ignored(self):
        self.assertEqual(judge(reads((12, "s", 20), (6, "s", 28, "-", 30, 30)), 10, 10, 2), check.NOT_CONTRADICTED)

    def test_one_sided_reads_of_a_spannable_allele_need_spanning_support(self):
        # Truth T 7/7, 11 spanning reads at 7, one-sided reads at 10, 25, 25, 26.
        measured = reads((11, "s", 7), (1, "l", 10), (2, "l", 25), (1, "r", 26))
        self.assertEqual(judge(measured, 7, 7, 1), check.NOT_CONTRADICTED)

    def test_a_cluster_of_spanning_reads_can_center_on_a_length_no_read_shows(self):
        # Truth AT 72/72; clean spanning reads at 6 (x2), 8 (x3) and 11 (x3) repeats, so five are within 1 repeat of 7.
        measured = reads((2, "s", 12), (3, "s", 16), (3, "s", 22), (4, "l", 10))
        self.assertEqual(judge(measured, 72, 72, 2), check.SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS)


class LociOfEachPersonTest(unittest.TestCase):
    @staticmethod
    def write_source(path, eh_of_second_locus=10.0):
        """Two loci: 1-100-110-AC (truth 10/10, called 10/10) and 2-50-60-AC (truth 10/12, called 10 and
        ``eh_of_second_locus``)."""
        rows = []
        for locus_id, truths, ehs in (("1-100-110-AC", (10.0, 10.0), (10.0, 10.0)),
                                      ("2-50-60-AC", (10.0, 12.0), (10.0, eh_of_second_locus))):
            for rank, (true, eh) in enumerate(zip(truths, ehs)):
                rows.append({"locus_id": locus_id, "allele_rank": rank, "eh": eh, "true": true, "motif_size": 2,
                             "is_negative_locus": False, "has_own_quality_metrics": True,
                             "genotyping_branch": "quick", "spanning_at_called": 5, "depth": 30.0})
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_disagreeing_loci_and_truth(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_source(os.path.join(d, "a.parquet"))
            disagreeing, truth = check.disagreeing_loci_and_truth_of_source(os.path.join(d, "a.parquet"))
        self.assertEqual(disagreeing, {"2-50-60-AC"})
        self.assertEqual(truth.loc["2-50-60-AC"].tolist(), [10, 12, 2])
        self.assertEqual(truth.loc["1-100-110-AC"].tolist(), [10, 10, 2])

    def test_rows_training_leaves_out_still_count_as_disagreeing(self):
        # A homozygous call 10/10 whose far truth allele sits on the unscored second copy, and a 0-repeat truth allele.
        rows = [{"locus_id": "1-5-25-AC", "eh": 10.0, "true": 10.0, "motif_size": 2},
                {"locus_id": "1-5-25-AC", "eh": 10.0, "true": 14.0, "motif_size": 2},
                {"locus_id": "2-5-11-TCC", "eh": 3.0, "true": 0.0, "motif_size": 3},
                {"locus_id": "3-5-11-TCC", "eh": np.nan, "true": 4.0, "motif_size": 3}]
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(rows).to_parquet(os.path.join(d, "a.parquet"), index=False)
            disagreeing, truth = check.disagreeing_loci_and_truth_of_source(os.path.join(d, "a.parquet"))
        self.assertEqual(disagreeing, {"1-5-25-AC", "2-5-11-TCC"})
        self.assertEqual(truth.loc["2-5-11-TCC"].tolist(), [0, 0, 3])

    def test_loci_table_is_sorted_by_position_and_keyed_by_content(self):
        table = check.loci_table({"X-5-10-A", "1-100-110-AC", "2-3-9-CAG", "1-20-30-T"})
        self.assertEqual(table["locus_id"].tolist(), ["1-20-30-T", "1-100-110-AC", "2-3-9-CAG", "X-5-10-A"])
        self.assertEqual(table.iloc[1].tolist(), ["1-100-110-AC", "chr1", 100, 110, "AC"])
        text = check.loci_table_text(table)
        same_loci_listed_in_another_order = check.loci_table(["1-20-30-T", "X-5-10-A", "2-3-9-CAG", "1-100-110-AC"])
        self.assertEqual(check.measurement_key(text),
                         check.measurement_key(check.loci_table_text(same_loci_listed_in_another_order)))
        self.assertNotEqual(check.measurement_key(text), check.measurement_key(text.replace("110", "111")))

    def test_measurement_key_ignores_the_measuring_scripts_comments_and_docstrings_but_not_its_code(self):
        script = '"""Measures."""\n\ndef f(x):\n    """Doubles x."""\n    return 2 * x  # twice\n'
        edited_comments = '"""Measures reads."""\n\n# a new comment\ndef f(x):\n    """Twice x."""\n    return 2 * x\n'
        edited_code = '"""Measures."""\n\ndef f(x):\n    """Doubles x."""\n    return 3 * x  # twice\n'
        keys = []
        with tempfile.TemporaryDirectory() as d:
            for i, text in enumerate((script, edited_comments, edited_code)):
                path = os.path.join(d, "script_%d.py" % i)
                with open(path, "w") as f:
                    f.write(text)
                with mock.patch.object(check.measure, "__file__", path):
                    keys.append(check.measurement_key("1-20-30-T\tchr1\t20\t30\tT\n"))
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])

    def test_a_person_is_checked_at_the_union_of_the_disagreeing_loci_of_their_sources(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_source(os.path.join(d, "P_10x.parquet"), eh_of_second_locus=12.0)
            self.write_source(os.path.join(d, "P_30x.parquet"))
            table, truth = check.person_loci_and_truth("P", [os.path.join(d, "P_10x.parquet"),
                                                             os.path.join(d, "P_30x.parquet")])
        self.assertEqual(table["locus_id"].tolist(), ["2-50-60-AC"])


class MarkPersonTest(unittest.TestCase):
    def test_marks_the_rows_of_contradicted_loci_once(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, "P_10x.parquet"), os.path.join(d, "P_30x.parquet")]
            for path in paths:
                LociOfEachPersonTest.write_source(path, eh_of_second_locus=18.0)
            table, _ = check.person_loci_and_truth("P", paths)
            key = check.measurement_key(check.loci_table_text(table))
            # Truth 10/12 at 2-50-60-AC, reads 10 and 18: the 12 has no read, the 18 is missing from the truth.
            pd.DataFrame({"locus_id": ["2-50-60-AC"], "has_unique_flanks_next_to_locus": [True],
                          "n_reads_overlapping": [20], "n_reads_mapq_below_20": [0],
                          "measured_reads": [reads((8, "s", 20), (7, "s", 36))]}).to_csv(
                os.path.join(d, "P.%s%s" % (key, check.MEASUREMENTS_SUFFIX)), sep="\t", index=False)
            before = pd.read_parquet(paths[0])

            summary = check.mark_person("P", paths, d, "gs://unused", "rule1")
            self.assertEqual((summary["status"], summary["n_contradicted"]), ("marked", 1))
            for path in paths:
                marked = pd.read_parquet(path)
                self.assertEqual(marked.loc[marked[check.TRUTH_CONTRADICTED_COLUMN], "locus_id"].tolist(),
                                 ["2-50-60-AC", "2-50-60-AC"])
                pd.testing.assert_frame_equal(marked.drop(columns=[check.TRUTH_CONTRADICTED_COLUMN]), before)
                self.assertEqual(check.parquet_check_record(path),
                                 {"measurements": "P.%s%s" % (key, check.MEASUREMENTS_SUFFIX), "judging_rule": "rule1"})
            self.assertEqual(dataset.parquets_without_the_short_read_truth_check(paths), [])
            self.assertEqual(check.mark_person("P", paths, d, "gs://unused", "rule1")["status"], "already marked")
            judgments = pd.read_csv(os.path.join(d, "P.%s.judgments.tsv.gz" % key), sep="\t")
            self.assertEqual(judgments["judgment"].tolist(), [check.SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS])

    def test_a_missing_measurement_file_is_not_measured_but_a_failed_download_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "P.parquet")
            LociOfEachPersonTest.write_source(path, eh_of_second_locus=18.0)
            missing = subprocess.CompletedProcess([], 1, "", "CommandException: No URLs matched: gs://x/P.y")
            with mock.patch.object(check, "gsutil", return_value=missing):
                self.assertEqual(check.mark_person("P", [path], d, "gs://x", "rule1")["status"], "not measured")
            denied = subprocess.CompletedProcess([], 1, "", "AccessDeniedException: 403 credentials expired")
            with mock.patch.object(check, "gsutil", return_value=denied), \
                    self.assertRaisesRegex(RuntimeError, "credentials expired"):
                check.mark_person("P", [path], d, "gs://x", "rule1")

    def test_measurements_must_cover_exactly_the_persons_loci(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "P.parquet")
            LociOfEachPersonTest.write_source(path, eh_of_second_locus=18.0)
            table, _ = check.person_loci_and_truth("P", [path])
            key = check.measurement_key(check.loci_table_text(table))
            pd.DataFrame({"locus_id": ["9-1-2-A"], "has_unique_flanks_next_to_locus": [True],
                          "n_reads_overlapping": [0], "n_reads_mapq_below_20": [0], "measured_reads": [""]}).to_csv(
                os.path.join(d, "P.%s%s" % (key, check.MEASUREMENTS_SUFFIX)), sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "does not cover"):
                check.mark_person("P", [path], d, "gs://unused", "rule1")


if __name__ == "__main__":
    unittest.main()
