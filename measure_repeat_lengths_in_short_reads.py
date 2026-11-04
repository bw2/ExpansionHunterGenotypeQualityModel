"""Measures the repeat length each short read shows at selected loci, without any genotyper, so a truth genotype can be
checked against the reads (``check_truth_genotypes_against_short_reads.py`` launches it in one Hail Batch job per
person and judges its output).

Two subcommands:

    measure_repeat_lengths_in_short_reads.py write-regions-bed LOCI_TSV OUT_BED
    samtools view -M -F 0xF04 -T REFERENCE_FASTA -L OUT_BED READS | \\
        measure_repeat_lengths_in_short_reads.py measure LOCI_TSV REFERENCE_FASTA OUT_TSV_GZ

LOCI_TSV (optionally gzipped) has the columns locus_id, chrom, start_0based, end and motif. ``measure`` reads the SAM
records of the reads overlapping the loci from standard input and writes one row per locus to OUT_TSV_GZ.

How a read is measured: the 20 bp of reference just left of the repeat (left flank) and just right of it (right flank)
are looked up in the whole read sequence, soft-clipped bases included, allowing 1 mismatch but not in the 5 bases next
to the repeat (``flank_match_positions``). A locus is measured only when each flank occurs once within 150 bp of the
locus even allowing 1 mismatch anywhere (``has_unique_flanks_next_to_locus``). A read is
  spanning     (s) both flanks found: the repeat length is the number of bases between them, exactly
  left-bound   (l) only the left flank found and the read's last bases are copies of the motif, so the read ends inside
               the repeat: the bases after the flank are a lower bound on the repeat length
  right-bound  (r) the same from the right flank
A one-sided read whose alignment covers the whole opposite flank is not measured: it does not end inside the repeat
(the person has a variant in that flank, and the motif copies at the read's end lie beyond it).
A flank match counts only where the read's alignment (position and CIGAR) puts it, within 10 bp of the flank's
reference position, because a person's haplotype can carry a near-copy of a flank inside the repeat. The bases next to
the repeat must match exactly because a mismatch there can come from a deletion that removes the flank's last bases.

Each measured read is stored as "kind:length_bp:mapping_quality:mean_base_quality:strand", the mean base quality over
the measured part of the read, so the judging thresholds can change without fetching reads again. A one-sided read's
motif run is at its sequencing 3' end when it is l and + or r and -, which is where two-color poly-G artifacts appear.

Standard library only, so it runs in any image with Python 3 and samtools.
"""

import bisect
import csv
import gzip
import os
import re
import sys

FLANK_BP = 20
INNER_FLANK_BP_THAT_MUST_MATCH = 5
FLANK_UNIQUENESS_WINDOW_BP = 150
MAX_FLANK_ALIGNMENT_OFFSET_BP = 10
# Extra bases fetched around each locus's flanks, so samtools returns every read that reaches a flank.
REGION_PADDING_BP = 5
MIN_PURE_END_BP = 8
OUTPUT_COLUMNS = ("locus_id", "has_unique_flanks_next_to_locus", "n_reads_overlapping", "n_reads_mapq_below_20",
                  "measured_reads")
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def read_loci(loci_tsv):
    """Returns the loci as a list of dicts with locus_id, chrom, start_0based (int), end (int) and motif."""
    with (gzip.open if loci_tsv.endswith(".gz") else open)(loci_tsv, "rt") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["start_0based"], row["end"] = int(row["start_0based"]), int(row["end"])
    return rows


def write_regions_bed(loci, out_bed):
    """Writes the region samtools must fetch for each locus: the locus, its flanks and a few bases of padding."""
    padding = FLANK_BP + REGION_PADDING_BP
    with open(out_bed, "w") as f:
        for locus in loci:
            f.write("%s\t%d\t%d\n" % (locus["chrom"], max(0, locus["start_0based"] - padding), locus["end"] + padding))


def read_fasta_index(fai_path):
    """Returns {contig: (length, byte offset, bases per line, bytes per line)} from a samtools .fai index."""
    index = {}
    with open(fai_path) as f:
        for line in f:
            name, length, offset, line_bases, line_bytes = line.split("\t")[:5]
            index[name] = (int(length), int(offset), int(line_bases), int(line_bytes))
    return index


def fetch_reference_sequence(fasta_handle, index, chrom, start_0based, end):
    """Returns the uppercase reference bases [start_0based, end) of ``chrom``, clipped to the contig."""
    length, offset, line_bases, line_bytes = index[chrom]
    start_0based, end = max(0, start_0based), min(length, end)
    if end <= start_0based:
        return ""

    def byte_position(position):
        return offset + (position // line_bases) * line_bytes + position % line_bases

    fasta_handle.seek(byte_position(start_0based))
    raw = fasta_handle.read(byte_position(end) - byte_position(start_0based)).decode()
    return raw.replace("\n", "").replace("\r", "").upper()


def hamming_at_most_one(sequence, position, pattern):
    mismatches = 0
    for a, b in zip(sequence[position:position + len(pattern)], pattern):
        if a != b:
            mismatches += 1
            if mismatches > 1:
                return False
    return True


def approximate_match_positions(sequence, pattern):
    """Returns the sorted positions where ``pattern`` occurs in ``sequence`` with at most 1 mismatch.

    Any such occurrence contains an exact copy of one of the pattern's two halves, so those are used as seeds.
    """
    half = len(pattern) // 2
    candidates = set()
    for seed, offset in ((pattern[:half], 0), (pattern[half:], half)):
        position = sequence.find(seed)
        while position != -1:
            start = position - offset
            if 0 <= start <= len(sequence) - len(pattern):
                candidates.add(start)
            position = sequence.find(seed, position + 1)
    return sorted(p for p in candidates if hamming_at_most_one(sequence, p, pattern))


def flank_match_positions(sequence, flank, is_left_flank):
    """Returns the positions where ``flank`` occurs in ``sequence`` with at most 1 mismatch, none of it in the
    ``INNER_FLANK_BP_THAT_MUST_MATCH`` bases next to the repeat (the left flank's last bases, the right flank's first).

    One mismatch farther out tolerates a sequencing error or a SNV in the flank, which would otherwise hide every read
    of that haplotype. Next to the repeat it is not tolerated, because there it usually means the read's haplotype
    differs at the repeat's edge, e.g. a deletion that removes the flank's last base, and the match then sits inside
    the repeat.
    """
    inner = (slice(FLANK_BP - INNER_FLANK_BP_THAT_MUST_MATCH, FLANK_BP) if is_left_flank
             else slice(0, INNER_FLANK_BP_THAT_MUST_MATCH))
    return [p for p in approximate_match_positions(sequence, flank) if sequence[p:p + FLANK_BP][inner] == flank[inner]]


def flanks_next_to_locus(reference_window, window_start, start_0based, end):
    """Returns (left flank, right flank): the 20 bp just left and just right of the locus, or None for a flank that
    contains an N or occurs more than once (allowing 1 mismatch) in ``reference_window``, the reference around the
    locus starting at ``window_start``."""
    flanks = []
    for flank_start in (start_0based - FLANK_BP, end):
        flank = reference_window[flank_start - window_start:flank_start - window_start + FLANK_BP]
        unique = (len(flank) == FLANK_BP and "N" not in flank
                  and len(approximate_match_positions(reference_window, flank)) == 1)
        flanks.append(flank if unique else None)
    return tuple(flanks)


def motif_rotations(motif):
    return {motif[i:] + motif[:i] for i in range(len(motif))}


def is_motif_run(sequence, motif):
    """True if ``sequence`` is a stretch of consecutive motif copies in any phase, e.g. GCAGCAG for CAG."""
    for rotation in motif_rotations(motif):
        if (rotation * (len(sequence) // len(motif) + 1))[:len(sequence)] == sequence:
            return True
    return False


def pure_end_length(motif):
    """How many of a one-sided read's end bases must be motif copies for the read to count as ending in the repeat."""
    return max(MIN_PURE_END_BP, 2 * len(motif))


def aligned_reference_positions(position_1based, cigar, read_length):
    """Returns, for each base of the read sequence, the 0-based reference position it is aligned to, or None for
    inserted and soft-clipped bases."""
    positions = [None] * read_length
    reference_position, read_index = position_1based - 1, 0
    for length, operation in _CIGAR_RE.findall(cigar):
        length = int(length)
        if operation in "M=X":
            positions[read_index:read_index + length] = range(reference_position, reference_position + length)
            read_index += length
            reference_position += length
        elif operation in "IS":
            read_index += length
        elif operation in "DN":
            reference_position += length
    return positions


def is_where_the_read_aligns(positions, match_start, flank_reference_start):
    """True if the read's alignment puts the flank match within 10 bp of the flank's reference position (or aligns
    none of its bases, so it can't be checked)."""
    for j in range(FLANK_BP):
        position = positions[match_start + j]
        if position is not None:
            return abs(position - (flank_reference_start + j)) <= MAX_FLANK_ALIGNMENT_OFFSET_BP
    return True


def longest_suffix_matching_prefix(first, second):
    """Returns the length of the longest suffix of ``first`` that equals a prefix of ``second``."""
    for length in range(min(len(first), len(second)), 0, -1):
        if first[-length:] == second[:length]:
            return length
    return 0


def measure_read(sequence, left_flank, right_flank, motif, is_left_match_plausible=None,
                 is_right_match_plausible=None):
    """Returns (kind, start, end) for one read, or None if the read can't be measured.

    ``kind`` is "s" (spanning: sequence[start:end] is the whole repeat between the flanks), "l" or "r" (anchored on the
    left or right flank only and ending inside the repeat: sequence[start:end] is the part of the repeat the read
    covers, a lower bound on its length). The optional is_*_match_plausible(match position) functions drop flank
    matches the read's alignment doesn't support.
    """
    left = [p for p in flank_match_positions(sequence, left_flank, is_left_flank=True)
            if is_left_match_plausible is None or is_left_match_plausible(p)]
    right = [p for p in flank_match_positions(sequence, right_flank, is_left_flank=False)
             if is_right_match_plausible is None or is_right_match_plausible(p)]
    if len(left) > 1 or len(right) > 1:
        return None
    end_bp = pure_end_length(motif)
    if left and right:
        start, end = left[0] + FLANK_BP, right[0]
        return ("s", start, end) if end >= start else None
    if left:
        # A read that stops partway into the right flank isn't inside the repeat at its end, so the bases that match
        # the start of the right flank are cut off first (otherwise flank bases that resemble the motif would count).
        start = left[0] + FLANK_BP
        end = len(sequence) - longest_suffix_matching_prefix(sequence[start:], right_flank)
        if end - start >= end_bp and is_motif_run(sequence[end - end_bp:end], motif):
            return "l", start, end
    if right:
        start = longest_suffix_matching_prefix(left_flank, sequence[:right[0]])
        if right[0] - start >= end_bp and is_motif_run(sequence[start:start + end_bp], motif):
            return "r", start, right[0]
    return None


def mean_base_quality(qualities, start, end):
    """Returns the mean Phred base quality of qualities[start:end] (SAM QUAL string), or -1 if empty or absent."""
    if end <= start or qualities == "*":
        return -1
    return round(sum(ord(q) - 33 for q in qualities[start:end]) / (end - start))


def reference_span(position_1based, cigar):
    """Returns the 0-based [start, end) reference interval of an alignment."""
    return position_1based - 1, position_1based - 1 + sum(
        int(length) for length, operation in _CIGAR_RE.findall(cigar) if operation in "MDN=X")


class LociByChromosome:
    """Finds the loci a read overlaps (within a flank's length), for reads arriving in any order."""

    def __init__(self, loci):
        self._by_chrom = {}
        for i, locus in enumerate(loci):
            self._by_chrom.setdefault(locus["chrom"], []).append((locus["start_0based"], locus["end"], i))
        self._starts, self._longest = {}, {}
        for chrom, entries in self._by_chrom.items():
            entries.sort()
            self._starts[chrom] = [start for start, _, _ in entries]
            self._longest[chrom] = max(end - start for start, end, _ in entries)

    def overlapping(self, chrom, read_start, read_end):
        """Returns the indices of the loci whose [start - 20, end + 20) overlaps [read_start, read_end)."""
        entries = self._by_chrom.get(chrom)
        if not entries:
            return []
        starts = self._starts[chrom]
        low = bisect.bisect_left(starts, read_start - FLANK_BP - self._longest[chrom])
        high = bisect.bisect_left(starts, read_end + FLANK_BP)
        return [i for start, end, i in entries[low:high] if end + FLANK_BP > read_start]


def measure_loci(loci, sam_lines, fasta_path):
    """Measures every read of ``sam_lines`` (SAM records, header lines skipped) at the loci it overlaps.

    Returns:
        One output row (a dict with ``OUTPUT_COLUMNS``) per locus, in the order of ``loci``.
    """
    index = read_fasta_index(fasta_path + ".fai")
    margin = FLANK_BP + FLANK_UNIQUENESS_WINDOW_BP
    flanks = []
    with open(fasta_path, "rb") as fasta:
        for locus in loci:
            window_start = max(0, locus["start_0based"] - margin)
            window = fetch_reference_sequence(fasta, index, locus["chrom"], window_start, locus["end"] + margin)
            flanks.append(flanks_next_to_locus(window, window_start, locus["start_0based"], locus["end"]))
    finder = LociByChromosome(loci)
    reads = [[] for _ in loci]
    n_overlapping = [0] * len(loci)
    n_low_mapq = [0] * len(loci)
    for line in sam_lines:
        if line.startswith("@"):
            continue
        fields = line.rstrip("\n").split("\t", 11)
        flag, chrom, position, mapq, cigar = int(fields[1]), fields[2], int(fields[3]), int(fields[4]), fields[5]
        sequence, qualities = fields[9], fields[10]
        if cigar == "*" or sequence == "*":
            continue
        read_start, read_end = reference_span(position, cigar)
        positions = None
        for i in finder.overlapping(chrom, read_start, read_end):
            n_overlapping[i] += 1
            n_low_mapq[i] += mapq < 20
            left_flank, right_flank = flanks[i]
            if left_flank is None or right_flank is None:
                continue
            if positions is None:
                positions = aligned_reference_positions(position, cigar, len(sequence))
            locus = loci[i]
            measured = measure_read(
                sequence, left_flank, right_flank, locus["motif"],
                lambda p: is_where_the_read_aligns(positions, p, locus["start_0based"] - FLANK_BP),
                lambda p: is_where_the_read_aligns(positions, p, locus["end"]))
            # A one-sided read whose alignment covers the whole opposite flank does not end inside the repeat: the person
            # has a variant in that flank, and the motif copies at the read's end come from sequence beyond it.
            if measured and ((measured[0] == "l" and read_end >= locus["end"] + FLANK_BP)
                             or (measured[0] == "r" and read_start <= locus["start_0based"] - FLANK_BP)):
                measured = None
            if measured:
                kind, start, end = measured
                reads[i].append("%s:%d:%d:%d:%s" % (kind, end - start, mapq, mean_base_quality(qualities, start, end),
                                                    "-" if flag & 0x10 else "+"))
    return [{"locus_id": locus["locus_id"],
             "has_unique_flanks_next_to_locus": flanks[i][0] is not None and flanks[i][1] is not None,
             "n_reads_overlapping": n_overlapping[i], "n_reads_mapq_below_20": n_low_mapq[i],
             "measured_reads": ",".join(reads[i])} for i, locus in enumerate(loci)]


def write_measurements(rows, out_path):
    """Writes ``measure_loci``'s rows to a TSV (gzipped if ``out_path`` ends in .gz), via a temporary file."""
    temporary_path = out_path + ".tmp"
    with (gzip.open if out_path.endswith(".gz") else open)(temporary_path, "wt") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, out_path)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == "write-regions-bed" and len(sys.argv) == 4:
        write_regions_bed(read_loci(sys.argv[2]), sys.argv[3])
    elif command == "measure" and len(sys.argv) == 5:
        loci = read_loci(sys.argv[2])
        rows = measure_loci(loci, sys.stdin, sys.argv[3])
        write_measurements(rows, sys.argv[4])
        n_measured = sum(bool(row["measured_reads"]) for row in rows)
        print("measured reads at %d of %d loci -> %s" % (n_measured, len(rows), sys.argv[4]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
