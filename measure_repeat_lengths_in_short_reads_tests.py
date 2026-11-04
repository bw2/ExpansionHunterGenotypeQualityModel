"""Tests for measure_repeat_lengths_in_short_reads.py."""

import gzip
import os
import random
import tempfile
import unittest

import measure_repeat_lengths_in_short_reads as measure

# A reference with a (CAG)5 repeat at [200, 215) between 200 bp of random sequence on each side.
_RNG = random.Random(20261101)
LEFT_CONTEXT = "".join(_RNG.choice("ACGT") for _ in range(200))
RIGHT_CONTEXT = "".join(_RNG.choice("ACGT") for _ in range(200))
REFERENCE = LEFT_CONTEXT + "CAG" * 5 + RIGHT_CONTEXT
START, END = 200, 215
LEFT_FLANK, RIGHT_FLANK = REFERENCE[START - 20:START], REFERENCE[END:END + 20]


def mutate(sequence, position):
    return sequence[:position] + ("A" if sequence[position] != "A" else "C") + sequence[position + 1:]


def write_fasta(directory, line_bases=10):
    """Writes REFERENCE as contig chrT with a .fai index and returns the FASTA path."""
    path = os.path.join(directory, "ref.fa")
    with open(path, "w") as f:
        f.write(">chrT\n")
        for i in range(0, len(REFERENCE), line_bases):
            f.write(REFERENCE[i:i + line_bases] + "\n")
    with open(path + ".fai", "w") as f:
        f.write("chrT\t%d\t6\t%d\t%d\n" % (len(REFERENCE), line_bases, line_bases + 1))
    return path


class FlankMatchPositionsTest(unittest.TestCase):
    def test_one_mismatch_is_allowed_only_away_from_the_repeat(self):
        read = "TT" + mutate(LEFT_FLANK, 0) + "CAGCAG"
        self.assertEqual(measure.flank_match_positions(read, LEFT_FLANK, is_left_flank=True), [2])
        read = "TT" + mutate(LEFT_FLANK, 19) + "CAGCAG"  # the base next to the repeat
        self.assertEqual(measure.flank_match_positions(read, LEFT_FLANK, is_left_flank=True), [])
        read = "CAGCAG" + mutate(RIGHT_FLANK, 0) + "TT"
        self.assertEqual(measure.flank_match_positions(read, RIGHT_FLANK, is_left_flank=False), [])
        read = "CAGCAG" + mutate(RIGHT_FLANK, 19) + "TT"
        self.assertEqual(measure.flank_match_positions(read, RIGHT_FLANK, is_left_flank=False), [6])


class MeasureReadTest(unittest.TestCase):
    def test_spanning_read(self):
        read = LEFT_CONTEXT[-30:] + "CAG" * 7 + RIGHT_CONTEXT[:30]
        kind, start, end = measure.measure_read(read, LEFT_FLANK, RIGHT_FLANK, "CAG")
        self.assertEqual((kind, end - start), ("s", 21))

    def test_reads_ending_inside_the_repeat_give_lower_bounds(self):
        read = LEFT_CONTEXT[-40:] + "CAG" * 30
        kind, start, end = measure.measure_read(read, LEFT_FLANK, RIGHT_FLANK, "CAG")
        self.assertEqual((kind, end - start), ("l", 90))
        read = "AG" + "CAG" * 30 + RIGHT_CONTEXT[:40]
        kind, start, end = measure.measure_read(read, LEFT_FLANK, RIGHT_FLANK, "CAG")
        self.assertEqual((kind, end - start), ("r", 92))

    def test_one_sided_read_not_ending_in_the_motif_is_not_measured(self):
        read = LEFT_CONTEXT[-40:] + "CAG" * 5 + "TTTTTTTTTTTTTTTT"
        self.assertIsNone(measure.measure_read(read, LEFT_FLANK, RIGHT_FLANK, "CAG"))

    def test_a_flank_found_twice_makes_the_read_unmeasurable(self):
        read = LEFT_FLANK + "CAG" * 3 + LEFT_FLANK + "CAG" * 3 + RIGHT_FLANK
        self.assertIsNone(measure.measure_read(read, LEFT_FLANK, RIGHT_FLANK, "CAG"))

    def test_a_flank_whose_base_next_to_the_repeat_differs_is_not_matched(self):
        # A deletion that removes the flank's last base lets a 1-mismatch match sit inside
        # the person's longer repeat, so such a read must not be measured from that flank.
        read = LEFT_CONTEXT[-40:-1] + mutate(LEFT_CONTEXT, 199)[-1] + "CAG" * 7 + RIGHT_CONTEXT[:30]
        self.assertIsNone(measure.measure_read(read, LEFT_FLANK, RIGHT_FLANK, "CAG"))

    def test_alignment_check_drops_a_match_far_from_where_the_read_aligns(self):
        read = LEFT_CONTEXT[-30:] + "CAG" * 7 + RIGHT_CONTEXT[:30]
        positions = measure.aligned_reference_positions(START - 30 + 1 + 50, "%dM" % len(read), len(read))
        result = measure.measure_read(
            read, LEFT_FLANK, RIGHT_FLANK, "CAG",
            lambda p: measure.is_where_the_read_aligns(positions, p, START - 20),
            lambda p: measure.is_where_the_read_aligns(positions, p, END))
        self.assertIsNone(result)


class ReferenceAndFlanksTest(unittest.TestCase):
    def test_fetch_reference_sequence_across_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_fasta(d)
            index = measure.read_fasta_index(path + ".fai")
            with open(path, "rb") as f:
                self.assertEqual(measure.fetch_reference_sequence(f, index, "chrT", 195, 223), REFERENCE[195:223])
                self.assertEqual(measure.fetch_reference_sequence(f, index, "chrT", -5, 3), REFERENCE[:3])

    def test_flanks_must_be_unique_in_the_window(self):
        self.assertEqual(measure.flanks_next_to_locus(REFERENCE, 0, START, END), (LEFT_FLANK, RIGHT_FLANK))
        repeated = REFERENCE[:END] + LEFT_FLANK + REFERENCE[END + 20:]
        self.assertEqual(measure.flanks_next_to_locus(repeated, 0, START, END)[0], None)

    def test_cigar_helpers(self):
        self.assertEqual(measure.reference_span(11, "5S10M2I3M4D6M"), (10, 33))
        positions = measure.aligned_reference_positions(11, "2S3M1I2M", 8)
        self.assertEqual(positions, [None, None, 10, 11, 12, None, 13, 14])


class LociByChromosomeTest(unittest.TestCase):
    def test_finds_loci_within_a_flank_of_the_read(self):
        loci = [{"chrom": "chr1", "start_0based": 100, "end": 120}, {"chrom": "chr1", "start_0based": 130, "end": 135},
                {"chrom": "chr1", "start_0based": 500, "end": 2000}, {"chrom": "chr2", "start_0based": 100, "end": 120}]
        finder = measure.LociByChromosome(loci)
        self.assertEqual(finder.overlapping("chr1", 140, 290), [1])
        self.assertEqual(finder.overlapping("chr1", 1000, 1100), [2])
        self.assertEqual(finder.overlapping("chr1", 2100, 2200), [])
        self.assertEqual(finder.overlapping("chr3", 0, 10 ** 9), [])


class MeasureLociTest(unittest.TestCase):
    def test_end_to_end_from_sam_records(self):
        spanning = REFERENCE[170:200] + "CAG" * 7 + REFERENCE[215:250]
        left_bound = REFERENCE[140:200] + "CAG" * 20
        sam = ["@HD\tVN:1.6",
               "\t".join(["r1", "0", "chrT", "171", "60", "45M6I35M", "*", "0", "0", spanning, "I" * len(spanning)]),
               "\t".join(["r2", "16", "chrT", "141", "60", "75M45S", "*", "0", "0", left_bound, "I" * len(left_bound)]),
               "\t".join(["r3", "0", "chrT", "171", "5", "45M6I35M", "*", "0", "0", spanning, "I" * len(spanning)]),
               "\t".join(["r4", "0", "chrT", "301", "60", "30M", "*", "0", "0", REFERENCE[300:330], "I" * 30])]
        loci = [{"locus_id": "T-200-215-CAG", "chrom": "chrT", "start_0based": START, "end": END, "motif": "CAG"}]
        with tempfile.TemporaryDirectory() as d:
            rows = measure.measure_loci(loci, sam, write_fasta(d))
            out = os.path.join(d, "out.tsv.gz")
            measure.write_measurements(rows, out)
            with gzip.open(out, "rt") as f:
                self.assertEqual(f.readline().rstrip("\n").split("\t"), list(measure.OUTPUT_COLUMNS))
        self.assertEqual(rows[0]["n_reads_overlapping"], 3)
        self.assertEqual(rows[0]["n_reads_mapq_below_20"], 1)
        self.assertTrue(rows[0]["has_unique_flanks_next_to_locus"])
        self.assertEqual(rows[0]["measured_reads"].split(","), ["s:21:60:40:+", "l:60:60:40:-", "s:21:5:40:+"])

    def test_one_sided_read_aligned_through_a_variant_opposite_flank_is_not_measured(self):
        # The right flank's base next to the repeat differs, so the flank isn't matched; the read aligns on through it
        # and ends in motif copies further downstream, which must not become a long lower bound.
        through = REFERENCE[170:200] + "CAG" * 5 + mutate(RIGHT_CONTEXT, 0)[:35] + "CAGCAGCAG"
        sam = ["\t".join(["r1", "0", "chrT", "171", "60", "80M9S", "*", "0", "0", through, "I" * len(through)])]
        loci = [{"locus_id": "T-200-215-CAG", "chrom": "chrT", "start_0based": START, "end": END, "motif": "CAG"}]
        with tempfile.TemporaryDirectory() as d:
            rows = measure.measure_loci(loci, sam, write_fasta(d))
        self.assertEqual((rows[0]["n_reads_overlapping"], rows[0]["measured_reads"]), (1, ""))

    def test_regions_bed_covers_the_flanks(self):
        loci = [{"locus_id": "T-200-215-CAG", "chrom": "chrT", "start_0based": 10, "end": 215, "motif": "CAG"}]
        with tempfile.TemporaryDirectory() as d:
            measure.write_regions_bed(loci, os.path.join(d, "r.bed"))
            with open(os.path.join(d, "r.bed")) as f:
                self.assertEqual(f.read(), "chrT\t0\t240\n")


if __name__ == "__main__":
    unittest.main()
