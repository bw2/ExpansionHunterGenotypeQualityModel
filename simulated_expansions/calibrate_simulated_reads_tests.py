import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd
import pysam

import dataset
from simulated_expansions import calibrate_simulated_reads as calibrate
from simulated_expansions import expanded_allele_reads as sim
from simulated_expansions import spike_in_expansions as spike


def training_rows(rows):
    return pd.DataFrame(rows, columns=["locus_id", "motif_size", "true", "purity", "genotyping_branch",
                                       dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN])


class CandidateLociTest(unittest.TestCase):
    def test_long_pure_full_genotyper_autosomal_loci_of_two_to_six_bp_motifs(self):
        rows = training_rows([
            ["1-100-130-AAAG", 4, 10, 1.0, "full", False], ["1-100-130-AAAG", 4, 40, 0.95, "full", False],  # kept
            ["1-900-930-AAAG", 4, 10, 1.0, "full", False], ["1-900-930-AAAG", 4, 30, 1.0, "full", False],  # 120 bp
            ["2-100-130-CAG", 3, 60, 0.95, "full", False], ["2-100-130-CAG", 3, 60, 0.85, "full", False],  # impure
            ["3-100-130-CAG", 3, 60, 1.0, "full", True], ["3-100-130-CAG", 3, 60, 1.0, "full", True],  # contradicted
            ["4-100-130-CAG", 3, 60, 1.0, "quick", False],  # quick branch
            ["5-100-130-A", 1, 200, 1.0, "full", False],  # homopolymer
            ["6-100-160-ACGTACGTAC", 10, 20, 1.0, "full", False],  # 10 bp motif
            ["X-100-130-CAG", 3, 60, 1.0, "full", False],  # chrX
            ["7-100-130-CAG", 3, np.nan, np.nan, "full", False], ["7-100-130-CAG", 3, 55, 0.99, "full", False]])
        with tempfile.TemporaryDirectory() as d:
            rows.to_parquet(os.path.join(d, "P.parquet"))
            self.assertEqual(calibrate.candidate_loci("P", d), ["1-100-130-AAAG", "7-100-130-CAG"])


def truth_with_sequences(rows):
    return pd.DataFrame(rows, columns=["LocusId"] + spike.TRUTH_COLUMNS[1:] + calibrate.ALLELE_SEQUENCE_COLUMNS
                        + ["confident"]).set_index("LocusId")


class LociWithAlleleSequencesTest(unittest.TestCase):
    def test_confident_loci_with_their_sequences_keeping_the_first_of_nearby_ones(self):
        truth = truth_with_sequences([
            ["1-100000-100012-CAG", "chr1", 100000, 100012, "CAG", 4, 4, 60, 1.0, 1.0, "CAG" * 4, "CAG" * 60, True],
            ["1-102000-102012-CAG", "chr1", 102000, 102012, "CAG", 4, 4, 70, 1.0, 1.0, "CAG" * 70, "CAG" * 4, True],
            ["2-100000-100012-CAG", "chr2", 100000, 100012, "CAG", 4, 0, 60, 1.0, 1.0, np.nan, "CAG" * 60, True],
            ["3-100000-100012-CAG", "chr3", 100000, 100012, "CAG", 4, 60, 60, 1.0, 1.0, "CAG" * 60, "CAG" * 60,
             False]])
        loci = calibrate.loci_with_allele_sequences("P", "female", list(truth.index) + ["4-1-13-CAG"], truth)
        self.assertEqual([locus["locus_id"] for locus in loci], ["1-100000-100012-CAG", "2-100000-100012-CAG"])
        self.assertEqual(loci[0]["allele2_sequence"], "CAG" * 60)
        self.assertEqual(loci[1]["allele1_sequence"], "")
        self.assertEqual((loci[0]["chrom"], loci[0]["start_0based"], loci[0]["reference_motif"]),
                         ("chr1", 100000, "CAG"))


class CompareTest(unittest.TestCase):
    LOCI = pd.DataFrame([{"person": "P", "locus_id": "1-100-130-CAG", "reference_motif": "CAG",
                          "short_allele_repeats": 10, "long_allele_repeats": 60}])

    @staticmethod
    def row(eh, purity, own_metrics=True):
        return {"eh": eh, "read_repeat_purity": purity, "has_own_quality_metrics": own_metrics}

    def test_pairs_each_alleles_rows_and_skips_rows_without_their_own_metrics(self):
        real = {"P": {("1-100-130-CAG", 0): self.row(10, 0.99), ("1-100-130-CAG", 1): self.row(40, 0.96)}}
        simulated = {"P": {("1-100-130-CAG", 0): self.row(10, 0.99, own_metrics=False),
                           ("1-100-130-CAG", 1): self.row(45, 0.995)}}
        pairs = calibrate.paired_rows(self.LOCI, real, simulated)
        self.assertEqual(len(pairs), 1)
        row = pairs.iloc[0]
        self.assertEqual((row["allele_rank"], row["true_repeats"], row["true_bp"]), (1, 60, 180))
        self.assertEqual((row["real_eh"], row["simulated_eh"]), (40, 45))
        self.assertAlmostEqual(row["real_read_repeat_purity"], 0.96)

    def test_summary_by_true_size_with_paired_differences_and_exact_calls(self):
        columns = {"true_repeats": [60, 60, 10], "true_bp": [180, 180, 30],
                   "real_read_repeat_purity": [0.96, 0.98, 0.99], "simulated_read_repeat_purity": [0.99, 0.99, 0.99],
                   "real_eh": [60, 50, 10], "simulated_eh": [60, 60, 10]}
        for field in calibrate.COMPARED_FIELDS:
            for run in ("real", "simulated"):
                columns.setdefault("%s_%s" % (run, field), [np.nan] * 3)
        pairs = pd.DataFrame(columns)
        table = calibrate.summary(pairs).set_index(["true_bp", "field"])
        purity = table.loc[("[150.0, 300.0)", "read_repeat_purity")]
        self.assertEqual(purity["alleles"], 2)
        self.assertAlmostEqual(purity["median_difference"], 0.02)
        exact = table.loc[("[150.0, 300.0)", "called exactly")]
        self.assertEqual((exact["mean_real"], exact["mean_simulated"]), (0.5, 1.0))


class RepeatReadErrorsTest(unittest.TestCase):
    CONTIGS = {"L_allele1": ("A" * 10 + "CAG" * 10 + "T" * 10, sim.Repeat(10, 40, "CAG"))}

    def write_aligned_bam(self, path, reads):
        header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "L_allele1", "LN": 50}]}
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for name, start, cigar, sequence, qualities in reads:
                read = pysam.AlignedSegment(out.header)
                read.query_name, read.flag, read.reference_name = name, 0, "L_allele1"
                read.reference_start, read.cigarstring, read.mapping_quality = start, cigar, 60
                read.query_sequence, read.query_qualities = sequence, pysam.qualitystring_to_array(qualities)
                read.set_tag("MD", self.md_tag(self.CONTIGS["L_allele1"][0], start, cigar, sequence))
                out.write(read)

    @staticmethod
    def md_tag(contig, start, cigar, sequence):
        """The MD tag of a read with only M and D operations, from the contig."""
        md, run, position, read_position = "", 0, start, 0
        for length, operation in ((int(n), op) for n, op in re.findall(r"(\d+)([MD])", cigar)):
            if operation == "M":
                for _ in range(length):
                    if sequence[read_position] == contig[position]:
                        run += 1
                    else:
                        md, run = md + "%d%s" % (run, contig[position]), 0
                    position, read_position = position + 1, read_position + 1
            else:
                md, run = md + "%d^%s" % (run, contig[position:position + length]), 0
                position += length
        return md + str(run)

    def test_counts_in_repeat_bases_mismatches_low_qualities_and_indels(self):
        contig = self.CONTIGS["L_allele1"][0]
        mismatched = contig[5:20] + "T" + contig[21:25]  # one mismatch at contig position 20, in the repeat
        deleted = contig[0:15] + contig[18:30]  # a 3 bp (one motif copy) deletion at contig position 15
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aligned.bam")
            self.write_aligned_bam(path, [("m", 5, "20M", mismatched, "I" * 15 + "#" + "I" * 4),
                                          ("d", 0, "15M3D12M", deleted, "I" * 27),
                                          ("flank_only", 0, "8M", contig[:8], "I" * 8)])
            counts = calibrate.new_error_counts()
            calibrate.tally_in_repeat_errors(path, self.CONTIGS, counts)
        # Read "m" has contig positions 10-24 in the repeat (15 bases, the mismatch at Q2 among them); "d" has 10-14
        # and 18-29 (17 bases).
        self.assertEqual(sum(counts["bases"].values()), 32)
        self.assertEqual(counts["mismatches"][("<=2/3", 0, "<20")], 1)
        self.assertEqual(counts["low_quality_values"], {2: 1})
        self.assertEqual(counts["indels"], {("<=2/3", "deletion whole-motif"): 1})
        self.assertEqual(counts["indel_sizes"], {("deletion whole-motif", 1): 1})

    def test_counts_the_fragments_of_pairs_aligned_to_one_contig_by_their_share_in_the_repeat(self):
        contig = self.CONTIGS["L_allele1"][0]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aligned.bam")
            # The pair's fragment spans contig positions 0-30: 20 of its 30 bases are in the repeat [10, 40).
            self.write_aligned_bam(path, [("pair", 0, "10M", contig[0:10], "I" * 10),
                                          ("pair", 20, "10M", contig[20:30], "I" * 10),
                                          ("single", 30, "10M", contig[30:40], "I" * 10)])
            counts = calibrate.new_error_counts()
            calibrate.tally_in_repeat_errors(path, self.CONTIGS, counts)
        self.assertEqual(counts["fragments"], {("<=2/3", 0.8): 1})

    @staticmethod
    def counts_in_every_bin(indel_events=6, fragments=None):
        counts = calibrate.new_error_counts()
        for gc in sim.GC_BINS:
            for cycle in range(0, 121, 30):
                for quality in sim.QUALITY_BINS:
                    counts["bases"][(gc, cycle, quality)] += 100
                    counts["mismatches"][(gc, cycle, quality)] += {"<20": 20, "20-29": 5, "30+": 1}[quality]
            counts["indels"][(gc, "deletion whole-motif")] += indel_events
            for share, n in (fragments or {}).items():
                counts["fragments"][(gc, share)] += n
        counts["indel_sizes"][("deletion whole-motif", 1)] += 3
        counts["indel_sizes"][("deletion whole-motif", 2)] += 1
        counts["low_quality_values"][2] += 3
        counts["low_quality_values"][12] += 1
        return counts

    def test_simulated_indel_rates_and_fragment_loss_are_corrected_by_how_the_simulated_reads_came_out(self):
        # Real reads: half as many fragments wholly in the repeat as clear of it; the simulated reads (made with a
        # keep base of 0.8): as many. So the fragment loss needs a keep base 0.8 * 0.5. Simulated deletions came out
        # at 4 of the real 6 events, so the rate to simulate is 1.5 times the one used.
        real = self.counts_in_every_bin(6, {0.0: 1000, 1.0: 500})
        simulated = self.counts_in_every_bin(4, {0.0: 2000, 1.0: 2000})
        used = sim.RepeatReadErrors({}, {}, {(gc, k): 0.002 for gc in sim.GC_BINS for k in sim.INDEL_KINDS}, {},
                                    ([], []), {gc: 0.8 for gc in sim.GC_BINS})
        table = calibrate.calibrated_against_simulation(calibrate.repeat_read_error_table(real), real, simulated, used)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "errors.tsv")
            table.to_csv(path, sep="\t", index=False)
            errors = sim.load_repeat_read_errors(path)
        # The fit takes the bin's middle (0.9) as the fragments' share in the repeat: 0.5 = c ** 0.9.
        self.assertAlmostEqual(errors.fragment_keep_base["0"], 0.8 * 0.5 ** (1 / 0.9))
        self.assertAlmostEqual(errors.indel_rate[("0", "deletion whole-motif")], 0.002 * 6 / 4)
        # Kinds with no simulated events keep the real rate.
        self.assertEqual(errors.indel_rate[("0", "insertion whole-motif")], 0)

    def test_the_table_reads_back_as_the_simulation_uses_it_and_needs_every_bin(self):
        counts = self.counts_in_every_bin()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "errors.tsv")
            calibrate.repeat_read_error_table(counts).to_csv(path, sep="\t", index=False)
            errors = sim.load_repeat_read_errors(path)
        self.assertAlmostEqual(errors.low_quality_share[(">2/3", 120)], 1 / 3)
        self.assertAlmostEqual(errors.mismatch_rate[("0", "<20")], 0.2)
        self.assertAlmostEqual(errors.indel_rate[("<=1/3", "deletion whole-motif")], 6 / 1500)
        self.assertEqual(errors.indel_rate[("<=1/3", "insertion partial-motif")], 0)
        self.assertEqual(errors.indel_sizes["deletion whole-motif"], ([1, 2], [0.75, 0.25]))
        self.assertEqual(errors.low_quality_values, ([2, 12], [0.75, 0.25]))
        self.assertEqual(errors.fragment_keep_base, {"0": 1.0, "<=1/3": 1.0, "<=2/3": 1.0, ">2/3": 1.0})
        del counts["bases"][(">2/3", 0, "30+")]
        del counts["bases"][(">2/3", 0, "20-29")]
        del counts["bases"][(">2/3", 0, "<20")]
        with self.assertRaises(RuntimeError):
            calibrate.repeat_read_error_table(counts)


if __name__ == "__main__":
    unittest.main()
