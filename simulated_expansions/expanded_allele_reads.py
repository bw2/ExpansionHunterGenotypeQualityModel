"""Simulates the read pairs of one expanded allele so that they look like a given person's reads.

The allele is the reference with its repeat replaced by copies of a motif, plus reference flanks
(str_analysis.simulate_str_expansions.generate_synthetic_reference_sequence). Each read pair comes from a fragment whose
length is drawn from the person's own read pairs, starting at a uniformly random position on a random strand. Each read
takes the base qualities of a randomly drawn real read of the same mate (first or second), in sequencing order, and
each base becomes a different random base with the error probability its quality gives (10^(-Q/10)), so the reads
carry the person's quality profile rather than a constant one.

Inside the repeat, real reads carry more errors than their qualities give, and more low-quality bases, the more so the
richer in G and C the motif is and the further into the read; they also gain and lose bases (whole motif copies and
partial ones). Given the errors measured in real reads of long repeats (``RepeatReadErrors``, from
``repeat_read_errors.tsv``), the read bases inside the repeat get those instead (``simulated_read``).
"""

import collections
import os
import random

import pandas as pd
import pysam
from str_analysis.simulate_str_expansions import generate_synthetic_reference_sequence

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
# Reads flagged as unmapped, secondary, QC fail, duplicate or supplementary are left out of the profile.
FLAGS_LEFT_OUT_OF_THE_PROFILE = 0xF04
# The fragment start rate counts duplicates too (it leaves out unmapped, secondary, QC fail and supplementary reads):
# ExpansionHunter counts them, and the person's read pairs the simulated ones replace include them.
FLAGS_LEFT_OUT_OF_THE_FRAGMENT_START_RATE = 0xB04
# The motif sizes, 2 to 6 bp inclusive, of the alleles the in-repeat errors are measured at
# (calibrate_simulated_reads.py); the measured insertions and deletions, sized in motif copies or a few bp, are given
# only to repeats of these motif sizes.
MEASURED_MOTIF_SIZE_RANGE_BP = (2, 6)
MAX_FRAGMENT_LENGTH = 1500
REPEAT_READ_ERRORS_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repeat_read_errors.tsv")
# Base qualities below this count as low; the measured in-repeat errors are kept per bin of these (QUALITY_BINS), of
# read cycles (CYCLE_BIN_BP wide, the last one open-ended) and of the motif's GC share (GC_BINS, motif_gc_bin).
LOW_QUALITY = 20
QUALITY_BINS = ("<20", "20-29", "30+")
GC_BINS = ("0", "<=1/3", "<=2/3", ">2/3")
CYCLE_BIN_BP = 30
LAST_CYCLE_BIN = 4
INDEL_KINDS = ("deletion whole-motif", "deletion partial-motif", "insertion whole-motif", "insertion partial-motif")
# The error probability of each Phred quality, 10^(-Q/10), for the qualities a FASTQ can hold.
ERROR_RATE_OF_QUALITY = [10 ** (-quality / 10) for quality in range(94)]

# A person's reads as the simulation copies them (make_read_profile); the low quality shares are
# low_quality_share_by_cycle_bin of each mate's quality strings.
ReadProfile = collections.namedtuple("ReadProfile", ["fragment_lengths", "first_mate_qualities",
                                                     "second_mate_qualities", "first_mate_low_quality_shares",
                                                     "second_mate_low_quality_shares"])
# The repeat of a simulated allele: [start, end) on the allele's sequence, and its motif, which sets the reads' errors
# inside it. An allele of blocks of several motifs also lists them as (start, end, motif), each of which sets the loss
# of the fragments over it (fragment_kept).
Repeat = collections.namedtuple("Repeat", ["start", "end", "motif", "blocks"], defaults=[None])
# Errors of real reads inside long repeats, as repeat_read_errors.tsv holds them:
#   low_quality_share: {(GC bin, cycle bin): share of in-repeat bases with a quality below LOW_QUALITY}
#   mismatch_rate: {(GC bin, quality bin): share of in-repeat bases that differ from the allele}
#   indel_rate: {(GC bin, kind of INDEL_KINDS): events per in-repeat base to simulate}
#   indel_sizes: {kind: (sizes, weights)}, whole-motif sizes in motif copies, partial ones in bp
#   low_quality_values: (qualities below LOW_QUALITY, weights)
#   fragment_keep_base: {GC bin: c}, a fragment with a share f of its bases inside the repeat is kept with
#       probability c^f (real reads cover long repeats less than their flanks, the more so the more extreme the GC)
RepeatReadErrors = collections.namedtuple("RepeatReadErrors", ["low_quality_share", "mismatch_rate", "indel_rate",
                                                               "indel_sizes", "low_quality_values",
                                                               "fragment_keep_base"])


def motif_gc_bin(motif):
    """The bin of ``GC_BINS`` of the motif's share of G and C bases: none, up to 1/3, up to 2/3, or more. An N counts
    as half a G or C, as a simulated allele cycles it through A, C, G and T (``expanded_allele_sequence``), so GCN is
    5/6 G or C."""
    gc = sum(1.0 if base in "GC" else 0.5 if base == "N" else 0.0 for base in motif.upper()) / len(motif)
    return GC_BINS[0] if gc == 0 else GC_BINS[1] if gc <= 1 / 3 else GC_BINS[2] if gc <= 2 / 3 else GC_BINS[3]


def quality_bin(quality):
    return QUALITY_BINS[0] if quality < LOW_QUALITY else QUALITY_BINS[1] if quality < 30 else QUALITY_BINS[2]


def cycle_bin(cycle):
    """The bin of a 0-based read cycle (position in sequencing order), as its first cycle."""
    return min(cycle // CYCLE_BIN_BP, LAST_CYCLE_BIN) * CYCLE_BIN_BP


def load_repeat_read_errors(tsv_path=REPEAT_READ_ERRORS_TSV):
    """Reads ``RepeatReadErrors`` from the table ``calibrate_simulated_reads.py measure-read-errors`` writes (columns
    quantity, group, key, value)."""
    table = pd.read_csv(tsv_path, sep="\t", dtype={"group": str, "key": str}, keep_default_na=False)
    by_quantity = {quantity: rows for quantity, rows in table.groupby("quantity")}

    def keyed(quantity, key_type):
        return {(row["group"], key_type(row["key"])): row["value"] for row in by_quantity[quantity].to_dict("records")}

    indel_sizes = {}
    for row in by_quantity["indel_size_share"].to_dict("records"):
        sizes, weights = indel_sizes.setdefault(row["group"], ([], []))
        sizes.append(int(row["key"]))
        weights.append(row["value"])
    low_quality_values = by_quantity["low_quality_value_share"]
    return RepeatReadErrors(keyed("low_quality_share", int), keyed("mismatch_rate", str), keyed("indel_rate", str),
                            indel_sizes, (low_quality_values["key"].astype(int).tolist(),
                                          low_quality_values["value"].tolist()),
                            {gc: c for (gc, _), c in keyed("fragment_keep_base", str).items()})


def fragment_kept(fragment_start, fragment_end, repeat, repeat_errors, keep_exponent, rng):
    """Whether a simulated fragment [fragment_start, fragment_end) survives: with probability c^(f * keep_exponent),
    c being ``repeat_errors.fragment_keep_base`` of the motif's GC bin and f the share of the fragment's bases inside
    the repeat (``keep_exponent`` 1 for the measured loss, larger for more, smaller for less); over a repeat of
    ``blocks``, the product of each block's c^(f * keep_exponent), f the share of the fragment inside the block."""
    keep, overlaps = 1.0, False
    for start, end, motif in repeat.blocks or [(repeat.start, repeat.end, repeat.motif)]:
        inside = min(fragment_end, end) - max(fragment_start, start)
        if inside > 0:
            overlaps = True
            keep *= repeat_errors.fragment_keep_base[motif_gc_bin(motif)] ** (
                keep_exponent * inside / (fragment_end - fragment_start))
    return not overlaps or rng.random() < keep


def low_quality_share_by_cycle_bin(quality_strings):
    """``{cycle bin: share of the bases below LOW_QUALITY}`` over the quality strings (sequencing order)."""
    low, total = collections.Counter(), collections.Counter()
    for quality_string in quality_strings:
        for cycle, character in enumerate(quality_string):
            total[cycle_bin(cycle)] += 1
            low[cycle_bin(cycle)] += ord(character) - 33 < LOW_QUALITY
    return {key: low[key] / total[key] for key in total}


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def expanded_allele_sequence(fasta, chrom, start_0based, end, motif, n_copies, flank_bp):
    """The allele with the reference repeat [start_0based, end) replaced by ``n_copies`` of ``motif`` (N in the motif
    cycles through A, C, G and T), between ``flank_bp`` reference bases on either side.

    Args:
        fasta: ``pyfaidx.Fasta(path, one_based_attributes=False, as_raw=True)``, as the str_analysis function needs.
    """
    return generate_synthetic_reference_sequence(fasta, chrom, start_0based + 1, end, flank_bp, motif, n_copies).upper()


def allele_sequence_with_reference_flanks(fasta, chrom, start_0based, end, repeat_sequence, flank_bp):
    """An allele whose repeat [start_0based, end) reads ``repeat_sequence`` (e.g. a person's allele from the truth
    table), between ``flank_bp`` reference bases on either side.

    Args:
        fasta: ``pyfaidx.Fasta(path, one_based_attributes=False, as_raw=True)``.
    """
    return (fasta[chrom][start_0based - flank_bp:start_0based] + repeat_sequence
            + fasta[chrom][end:end + flank_bp]).upper()


def read_profile(bam_path, reference_fasta, regions, max_reads=20000):
    """Draws the person's read length, fragment lengths and per-mate base-quality strings from their reads.

    Args:
        bam_path: The person's reads (BAM or CRAM, indexed).
        reference_fasta: The reference for decoding a CRAM.
        regions: ``[(chrom, start_0based, end), ...]`` to read from.
        max_reads: Stop after this many reads.

    Returns:
        A ``ReadProfile``: the fragment lengths of properly paired first mates (up to ``MAX_FRAGMENT_LENGTH``, and at
        least the most common read length), and the base-quality strings (in sequencing order) of first and second
        mates of the most common read length.
    """
    lengths, fragments, qualities = collections.Counter(), [], {True: [], False: []}
    with pysam.AlignmentFile(bam_path, reference_filename=reference_fasta) as bam:
        n = 0
        for chrom, start, end in regions:
            for read in bam.fetch(chrom, start, end):
                if read.flag & FLAGS_LEFT_OUT_OF_THE_PROFILE or not read.is_paired or read.query_qualities is None:
                    continue
                n += 1
                lengths[read.query_length] += 1
                if read.is_read1 and read.is_proper_pair and 0 < abs(read.template_length) <= MAX_FRAGMENT_LENGTH:
                    fragments.append(abs(read.template_length))
                quality = pysam.qualities_to_qualitystring(read.query_qualities)
                qualities[read.is_read1].append(quality[::-1] if read.is_reverse else quality)
                if n >= max_reads:
                    break
            if n >= max_reads:
                break
    read_length = lengths.most_common(1)[0][0]
    return make_read_profile([f for f in fragments if f >= read_length],
                             [q for q in qualities[True] if len(q) == read_length],
                             [q for q in qualities[False] if len(q) == read_length])


def make_read_profile(fragment_lengths, first_mate_qualities, second_mate_qualities):
    """A ``ReadProfile`` of these reads, with each mate's share of low qualities by read cycle, computed once here
    because every simulated allele uses them. A read's length is that of its quality string."""
    return ReadProfile(fragment_lengths, first_mate_qualities, second_mate_qualities,
                       low_quality_share_by_cycle_bin(first_mate_qualities),
                       low_quality_share_by_cycle_bin(second_mate_qualities))


def first_mates_starting_per_bp(bam_path, reference_fasta, chrom, windows):
    """The number of primary first mates of the person's read pairs that start in ``windows`` ([(start_0based, end)])
    per bp, i.e. the rate at which the person's fragments start, for setting the simulated allele's coverage."""
    count = 0
    with pysam.AlignmentFile(bam_path, reference_filename=reference_fasta) as bam:
        for start, end in windows:
            count += sum(1 for read in bam.fetch(chrom, start, end)
                         if read.is_read1 and not read.flag & FLAGS_LEFT_OUT_OF_THE_FRAGMENT_START_RATE
                         and start <= read.reference_start < end)
    return count / sum(end - start for start, end in windows)


def in_repeat_indel(rng, indel_rates, indel_sizes, motif_size):
    """``(kind, size in bp)`` of the insertion or deletion a read gets at one in-repeat base, or None (most bases).

    Args:
        indel_rates: ``[(kind, events per in-repeat base), ...]`` for the motif's GC bin.
        indel_sizes: ``RepeatReadErrors.indel_sizes``.
    """
    draw = rng.random()
    for kind, rate in indel_rates:
        if draw < rate:
            sizes, weights = indel_sizes[kind]
            if "whole-motif" in kind:
                return kind, rng.choices(sizes, weights)[0] * motif_size
            # The partial sizes are measured over motifs of several sizes: those that are whole copies of this motif
            # are left out here.
            partial = [(size, weight) for size, weight in zip(sizes, weights) if size % motif_size]
            return (kind, rng.choices(*zip(*partial))[0]) if partial else None
        draw -= rate
    return None


def simulated_read(sequence, first, step, fragment_start, fragment_end, quality_string, rng, repeat=None,
                   repeat_errors=None, baseline_low_quality_share=None):
    """One read of ``sequence`` with its base qualities and sequencing errors.

    The read starts at position ``first`` and reads the + strand (``step`` 1) or the complement of the - strand
    (``step`` -1), within the fragment [fragment_start, fragment_end), for up to ``len(quality_string)`` bases. Outside
    the repeat (or without ``repeat_errors``) each base becomes a different random base with the probability its
    quality gives (10^(-Q/10)). Inside ``repeat``, with ``repeat_errors`` for the motif's GC bin:

    - at each base the read may lose or gain bases (``in_repeat_indel``, for motif sizes within
      ``MEASURED_MOTIF_SIZE_RANGE_BP``); a gain repeats the read's last bases, as slippage on a repeat does;
    - a base whose quality is not low becomes low with the probability that brings the read cycle's share of low
      qualities from the person's (``baseline_low_quality_share``, ``low_quality_share_by_cycle_bin``) to the measured
      one, its value drawn from the measured low ones;
    - each base becomes a different random base with the measured mismatch rate of its quality bin.

    Returns:
        ``(read, quality string)``, shorter than the quality string only when the fragment ends first.
    """
    gc = motif_gc_bin(repeat.motif) if repeat is not None and repeat_errors is not None else None
    indel_rates = ([(kind, repeat_errors.indel_rate[(gc, kind)]) for kind in INDEL_KINDS]
                   if gc and MEASURED_MOTIF_SIZE_RANGE_BP[0] <= len(repeat.motif) <= MEASURED_MOTIF_SIZE_RANGE_BP[1]
                   else [])
    last = first + step * (len(quality_string) - 1)
    if gc is None or max(first, last) < repeat.start or min(first, last) >= repeat.end:
        # A read clear of the repeat is a slice of the fragment (the loop below would give the same bases).
        low, high = (first, min(first + len(quality_string), fragment_end)) if step == 1 else (
            max(first - len(quality_string) + 1, fragment_start), first + 1)
        read = sequence[low:high] if step == 1 else reverse_complement(sequence[low:high])
        bases, in_repeat = list(read), [False] * len(read)
    else:
        bases, in_repeat = in_repeat_read_bases(sequence, first, step, fragment_start, fragment_end,
                                                len(quality_string), rng, repeat, repeat_errors, indel_rates)
    qualities = [ord(character) - 33 for character in quality_string[:len(bases)]]
    for cycle, inside in enumerate(in_repeat):
        if inside and qualities[cycle] >= LOW_QUALITY:
            target = repeat_errors.low_quality_share[(gc, cycle_bin(cycle))]
            baseline = baseline_low_quality_share[cycle_bin(cycle)]
            if rng.random() < (target - baseline) / (1 - baseline):
                qualities[cycle] = rng.choices(*repeat_errors.low_quality_values)[0]
        error_rate = (repeat_errors.mismatch_rate[(gc, quality_bin(qualities[cycle]))] if inside
                      else ERROR_RATE_OF_QUALITY[qualities[cycle]])
        if rng.random() < error_rate:
            bases[cycle] = rng.choice([b for b in "ACGT" if b != bases[cycle]])
    return "".join(bases), "".join(chr(q + 33) for q in qualities)


def in_repeat_read_bases(sequence, first, step, fragment_start, fragment_end, read_length, rng, repeat, repeat_errors,
                         indel_rates):
    """The bases of a read that crosses the repeat, as ``simulated_read`` describes, and whether each lies in it."""
    bases, in_repeat = [], []
    position = first
    while len(bases) < read_length and fragment_start <= position < fragment_end:
        inside = repeat.start <= position < repeat.end
        if inside:
            indel = in_repeat_indel(rng, indel_rates, repeat_errors.indel_sizes, len(repeat.motif))
            if indel and indel[0].startswith("deletion"):
                position += step * indel[1]
                continue
            if indel:
                copied = bases[-indel[1]:]
                bases += copied
                in_repeat += [True] * len(copied)
        base = sequence[position]
        bases.append(base if step == 1 else base.translate(COMPLEMENT))
        in_repeat.append(inside)
        position += step
    return bases[:read_length], in_repeat[:read_length]


def simulate_read_pairs(sequence, n_pairs, profile, rng, name_prefix, repeat=None, repeat_errors=None,
                        fragment_keep_exponent=1.0):
    """Yields ``(name, read1, quality1, read2, quality2)`` for the read pairs of ``n_pairs`` fragments of
    ``sequence`` (sequencing order, as in a FASTQ), with fragment lengths, read length and base qualities from
    ``profile``. With ``repeat_errors``, the reads get the measured errors of real reads inside ``repeat``
    (``simulated_read``), and fragments over the repeat are lost as real ones are (``fragment_kept``)."""
    for i in range(n_pairs):
        fragment_length = min(rng.choice(profile.fragment_lengths), len(sequence))
        start = rng.randrange(len(sequence) - fragment_length + 1)
        end = start + fragment_length
        if repeat_errors is not None and not fragment_kept(start, end, repeat, repeat_errors, fragment_keep_exponent,
                                                           rng):
            continue
        # The first mate reads the fragment's + strand from its start, or its - strand from its end; the second mate
        # reads the other strand.
        plus_strand_first = rng.random() >= 0.5
        starts = ((start, 1), (end - 1, -1)) if plus_strand_first else ((end - 1, -1), (start, 1))
        quality1, quality2 = rng.choice(profile.first_mate_qualities), rng.choice(profile.second_mate_qualities)
        read1, quality1 = simulated_read(sequence, *starts[0], start, end, quality1, rng, repeat, repeat_errors,
                                         profile.first_mate_low_quality_shares)
        read2, quality2 = simulated_read(sequence, *starts[1], start, end, quality2, rng, repeat, repeat_errors,
                                         profile.second_mate_low_quality_shares)
        yield "%s:%d" % (name_prefix, i), read1, quality1, read2, quality2


def write_fastq_pairs(pairs, fastq1, fastq2):
    """Writes ``simulate_read_pairs`` output to two open FASTQ text files."""
    for name, read1, quality1, read2, quality2 in pairs:
        fastq1.write("@%s/1\n%s\n+\n%s\n" % (name, read1, quality1))
        fastq2.write("@%s/2\n%s\n+\n%s\n" % (name, read2, quality2))


def seeded_rng(*keys):
    """A random number generator seeded from ``keys``, so each simulated allele gets the same reads on every run."""
    return random.Random("\t".join(str(k) for k in keys))
