import io
import os
import random
import tempfile
import unittest

import pyfaidx
import pysam

from simulated_expansions import expanded_allele_reads as sim

LEFT = "ACGTTGCAAC" * 5
RIGHT = "TTGACCGGTA" * 5
REFERENCE = LEFT + "CAG" * 4 + RIGHT  # the repeat is [50, 62)
NO_ERROR_QUALITY = "~"  # Phred 93: an error probability of 5e-10


def write_fasta(directory):
    path = os.path.join(directory, "ref.fa")
    with open(path, "w") as f:
        f.write(">chrT\n%s\n" % REFERENCE)
    pysam.faidx(path)
    return path


class ExpandedAlleleSequenceTest(unittest.TestCase):
    def test_replaces_the_repeat_between_reference_flanks(self):
        with tempfile.TemporaryDirectory() as d:
            fasta = pyfaidx.Fasta(write_fasta(d), one_based_attributes=False, as_raw=True)
            sequence = sim.expanded_allele_sequence(fasta, "chrT", 50, 62, "CTG", 10, 20)
        self.assertEqual(sequence, LEFT[-20:] + "CTG" * 10 + RIGHT[:20])

    def test_n_in_the_motif_cycles_through_the_four_bases(self):
        with tempfile.TemporaryDirectory() as d:
            fasta = pyfaidx.Fasta(write_fasta(d), one_based_attributes=False, as_raw=True)
            sequence = sim.expanded_allele_sequence(fasta, "chrT", 50, 62, "GCN", 4, 0)
        self.assertEqual(sequence, "GCAGCCGCGGCT")

    def test_a_given_repeat_sequence_between_reference_flanks(self):
        with tempfile.TemporaryDirectory() as d:
            fasta = pyfaidx.Fasta(write_fasta(d), one_based_attributes=False, as_raw=True)
            sequence = sim.allele_sequence_with_reference_flanks(fasta, "chrT", 50, 62, "cagcaacag", 20)
        self.assertEqual(sequence, LEFT[-20:] + "CAGCAACAG" + RIGHT[:20])


def repeat_errors(low_quality_share=0.0, mismatch_rate=0.0, indel_rates=None, fragment_keep_base=1.0):
    """``RepeatReadErrors`` with the same values in every GC, cycle and quality bin."""
    cycles = [k * sim.CYCLE_BIN_BP for k in range(sim.LAST_CYCLE_BIN + 1)]
    return sim.RepeatReadErrors(
        {(gc, cycle): low_quality_share for gc in sim.GC_BINS for cycle in cycles},
        {(gc, q): mismatch_rate for gc in sim.GC_BINS for q in sim.QUALITY_BINS},
        {(gc, kind): (indel_rates or {}).get(kind, 0.0) for gc in sim.GC_BINS for kind in sim.INDEL_KINDS},
        {kind: ([1], [1.0]) for kind in sim.INDEL_KINDS}, ([2], [1.0]),
        {gc: fragment_keep_base for gc in sim.GC_BINS})


class MotifGcBinTest(unittest.TestCase):
    def test_gc_share_bins_with_n_as_half_a_g_or_c(self):
        self.assertEqual([sim.motif_gc_bin(m) for m in ("AAAAT", "AAAAG", "CAG", "GGGGCC", "GCN", "ngc")],
                         ["0", "<=1/3", "<=2/3", ">2/3", ">2/3", ">2/3"])


class FragmentLossTest(unittest.TestCase):
    def test_fragments_over_the_repeat_are_kept_with_the_base_to_the_power_of_their_share_in_it(self):
        repeat, errors = sim.Repeat(100, 200, "CAG"), repeat_errors(fragment_keep_base=0.25)
        rng = random.Random(1)
        self.assertTrue(all(sim.fragment_kept(0, 100, repeat, errors, 1.0, rng) for _ in range(100)))
        kept = sum(sim.fragment_kept(100, 200, repeat, errors, 1.0, rng) for _ in range(20000)) / 20000
        self.assertAlmostEqual(kept, 0.25, delta=0.02)
        kept = sum(sim.fragment_kept(150, 250, repeat, errors, 1.0, rng) for _ in range(20000)) / 20000
        self.assertAlmostEqual(kept, 0.5, delta=0.02)  # half the fragment in the repeat: 0.25 ** 0.5
        kept = sum(sim.fragment_kept(100, 200, repeat, errors, 0.5, rng) for _ in range(20000)) / 20000
        self.assertAlmostEqual(kept, 0.5, delta=0.02)  # a weaker loss

    def test_over_a_repeat_of_blocks_each_block_loses_fragments_by_its_own_gc_bin(self):
        # An AT-only block (bin "0") beside a CAGG block (bin ">2/3"), each 50 bp of the 100 bp repeat.
        errors = repeat_errors()._replace(fragment_keep_base={"0": 0.81, "<=1/3": 1.0, "<=2/3": 1.0, ">2/3": 0.04})
        repeat = sim.Repeat(100, 200, "CAGG", [(100, 150, "AAAA"), (150, 200, "CAGG")])
        rng = random.Random(1)
        kept = sum(sim.fragment_kept(100, 150, repeat, errors, 1.0, rng) for _ in range(20000)) / 20000
        self.assertAlmostEqual(kept, 0.81, delta=0.02)
        # Half the fragment in each block: 0.81 ** 0.5 * 0.04 ** 0.5.
        kept = sum(sim.fragment_kept(100, 200, repeat, errors, 1.0, rng) for _ in range(20000)) / 20000
        self.assertAlmostEqual(kept, 0.9 * 0.2, delta=0.02)

    def test_pairs_of_lost_fragments_are_left_out(self):
        profile = sim.make_read_profile([40], [NO_ERROR_QUALITY * 20], [NO_ERROR_QUALITY * 20])
        sequence = "ACGTTGCAAC" * 5 + "CAG" * 100 + "ACGTTGCAAC" * 5
        pairs = list(sim.simulate_read_pairs(sequence, 1000, profile, random.Random(2), "x", sim.Repeat(50, 350, "CAG"),
                                             repeat_errors(fragment_keep_base=0.0)))
        # Only fragments clear of the repeat remain: those of the 40 bp fragments starting in the first or last 10 bp.
        self.assertLess(len(pairs), 100)
        self.assertTrue(all("CAGCAG" not in read1 + read2 for _, read1, _, read2, _ in pairs))


class SimulatedReadTest(unittest.TestCase):
    FLANKS = ("ACGTTGCAAC", "TTGACCGGTA")
    SEQUENCE = FLANKS[0] + "CAG" * 10 + FLANKS[1]
    REPEAT = sim.Repeat(10, 40, "CAG")

    def read(self, errors, step=1, rng=None):
        first = 0 if step == 1 else len(self.SEQUENCE) - 1
        return sim.simulated_read(self.SEQUENCE, first, step, 0, len(self.SEQUENCE), NO_ERROR_QUALITY * 60,
                                  rng or random.Random(1), self.REPEAT, errors, {c: 0.0 for c in range(0, 121, 30)})

    def test_high_qualities_keep_the_bases_and_the_lowest_changes_most_of_them(self):
        sequence = "ACGT" * 25
        read, quality = sim.simulated_read(sequence, 0, 1, 0, 100, NO_ERROR_QUALITY * 100, random.Random(1))
        self.assertEqual((read, quality), (sequence, NO_ERROR_QUALITY * 100))
        read, _ = sim.simulated_read("A" * 1000, 0, 1, 0, 1000, "!" * 1000, random.Random(1))  # Phred 0: always wrong
        self.assertNotIn("A", read)

    def test_minus_strand_reads_are_complemented_and_stop_at_the_fragment_end(self):
        read, quality = sim.simulated_read("AACCG", 4, -1, 1, 5, NO_ERROR_QUALITY * 10, random.Random(1))
        self.assertEqual((read, len(quality)), ("CGGT", 4))

    def test_without_in_repeat_errors_the_read_is_the_allele(self):
        self.assertEqual(self.read(repeat_errors()), (self.SEQUENCE, NO_ERROR_QUALITY * 50))

    def test_deletions_at_every_in_repeat_base_remove_the_repeat_a_motif_copy_at_a_time(self):
        read, _ = self.read(repeat_errors(indel_rates={"deletion whole-motif": 1.0}))
        self.assertEqual(read, "".join(self.FLANKS))
        read, _ = self.read(repeat_errors(indel_rates={"deletion whole-motif": 1.0}), step=-1)
        self.assertEqual(read, sim.reverse_complement("".join(self.FLANKS)))

    def test_insertions_repeat_the_reads_last_bases(self):
        read, _ = self.read(repeat_errors(indel_rates={"insertion partial-motif": 1.0}))
        # Each in-repeat base is preceded by a copy of the base before it, until the read is full.
        repeat = "CAG" * 10
        before = self.FLANKS[0][-1] + repeat[:-1]
        self.assertEqual(read, (self.FLANKS[0] + "".join(p + b for p, b in zip(before, repeat)))[:60])

    def test_motifs_longer_than_those_measured_get_no_insertions_or_deletions(self):
        sequence = self.FLANKS[0] + "ACGTTACGGT" * 3 + self.FLANKS[1]
        read, _ = sim.simulated_read(sequence, 0, 1, 0, len(sequence), NO_ERROR_QUALITY * 60, random.Random(1),
                                     sim.Repeat(10, 40, "ACGTTACGGT"),
                                     repeat_errors(indel_rates={"deletion whole-motif": 1.0}),
                                     {c: 0.0 for c in range(0, 121, 30)})
        self.assertEqual(read, sequence)

    def test_partial_motif_sizes_that_are_whole_copies_of_the_motif_are_not_drawn(self):
        sizes = {kind: ([1, 3, 6], [0.2, 0.4, 0.4]) for kind in sim.INDEL_KINDS}
        drawn = {sim.in_repeat_indel(random.Random(seed), [("deletion partial-motif", 1.0)], sizes, 3)[1]
                 for seed in range(50)}
        self.assertEqual(drawn, {1})

    def test_in_repeat_bases_get_the_measured_low_qualities_and_mismatches(self):
        read, quality = self.read(repeat_errors(low_quality_share=1.0, mismatch_rate=1.0))
        self.assertEqual(quality, NO_ERROR_QUALITY * 10 + "#" * 30 + NO_ERROR_QUALITY * 10)  # "#" is Phred 2
        self.assertEqual(read[:10] + read[40:], "".join(self.FLANKS))
        self.assertTrue(all(a != b for a, b in zip(read[10:40], "CAG" * 10)))


class SimulateReadPairsTest(unittest.TestCase):
    PROFILE = sim.make_read_profile([40, 60], [NO_ERROR_QUALITY * 20], [NO_ERROR_QUALITY * 20])

    def test_mates_face_each_other_on_a_fragment_of_the_sequence(self):
        sequence = "".join(random.Random(2).choices("ACGT", k=300))
        pairs = list(sim.simulate_read_pairs(sequence, 50, self.PROFILE, random.Random(3), "sim"))
        self.assertEqual(len(pairs), 50)
        both_strands = sequence + "|" + sim.reverse_complement(sequence)
        for name, read1, quality1, read2, quality2 in pairs:
            self.assertEqual((len(read1), len(read2), len(quality1)), (20, 20, 20))
            self.assertIn(read1, both_strands)
            self.assertIn(read2, both_strands)
            for fragment_length in (40, 60):
                fragment = None
                for strand in (sequence, sim.reverse_complement(sequence)):
                    start = strand.find(read1)
                    if start >= 0 and strand[start:start + fragment_length].endswith(sim.reverse_complement(read2)):
                        fragment = strand[start:start + fragment_length]
                if fragment:
                    break
            self.assertIsNotNone(fragment, name)

    def test_the_same_seed_gives_the_same_reads(self):
        sequence = "ACGTTGCAAC" * 30
        first = list(sim.simulate_read_pairs(sequence, 5, self.PROFILE, sim.seeded_rng("P", "L", 3), "x"))
        second = list(sim.simulate_read_pairs(sequence, 5, self.PROFILE, sim.seeded_rng("P", "L", 3), "x"))
        self.assertEqual(first, second)

    def test_writes_paired_fastq_records(self):
        out1, out2 = io.StringIO(), io.StringIO()
        sim.write_fastq_pairs([("p:0", "ACGT", "IIII", "TTTT", "####")], out1, out2)
        self.assertEqual(out1.getvalue(), "@p:0/1\nACGT\n+\nIIII\n")
        self.assertEqual(out2.getvalue(), "@p:0/2\nTTTT\n+\n####\n")


class ReadProfileTest(unittest.TestCase):
    def test_profile_of_a_small_bam(self):
        with tempfile.TemporaryDirectory() as d:
            fasta = write_fasta(d)
            path = os.path.join(d, "reads.bam")
            header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chrT", "LN": len(REFERENCE)}]}
            with pysam.AlignmentFile(path, "wb", header=header) as out:
                for i, (start, mate_start, is_read1, is_reverse, duplicate) in enumerate(
                        [(0, 40, True, False, False), (5, 45, True, False, True), (10, 70, True, False, False),
                         (40, 0, False, True, False), (45, 5, False, True, True), (70, 10, False, True, False)]):
                    read = pysam.AlignedSegment()
                    read.query_name = "dup" if duplicate else "r%d" % (i % 3)
                    read.query_sequence = REFERENCE[start:start + 20]
                    read.flag = (1 | 2 | (64 if is_read1 else 128) | (16 if is_reverse else 32)
                                 | (0x400 if duplicate else 0))
                    read.reference_id, read.reference_start, read.cigarstring = 0, start, "20M"
                    read.next_reference_id, read.next_reference_start = 0, mate_start
                    read.template_length = 60 if is_read1 else -60
                    read.mapping_quality = 60
                    read.query_qualities = pysam.qualitystring_to_array("A" * 19 + "B")
                    out.write(read)
            pysam.index(path)
            profile = sim.read_profile(path, fasta, [("chrT", 0, len(REFERENCE))])
            rate = sim.first_mates_starting_per_bp(path, fasta, "chrT", [(0, 20)])
        self.assertEqual(profile.fragment_lengths, [60, 60])
        self.assertEqual(profile.first_mate_qualities, ["A" * 19 + "B"] * 2)
        # Second mates are reverse reads, so their qualities come back in sequencing order.
        self.assertEqual(profile.second_mate_qualities, ["B" + "A" * 19] * 2)
        # The duplicate pair is left out of the profile but counts toward the fragment start rate.
        self.assertAlmostEqual(rate, 3 / 20)


if __name__ == "__main__":
    unittest.main()
