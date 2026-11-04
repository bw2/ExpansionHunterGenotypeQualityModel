"""Measures how simulated reads of a long repeat differ from real ones, on the pilot people's own long alleles.

At loci where a pilot person has a long allele (at least ``MIN_LONG_ALLELE_BP`` in the truth table) of a 2-6 bp motif,
the person's read pairs that lie near the locus are replaced by simulated pairs of the person's own two alleles: their
sequences in the truth table, interruptions included, made the way ``spike_in_expansions.py spike`` makes an expanded
allele's. Pairs with a mate elsewhere or unmapped stay, as in the spiked samples
(``spike_in_expansions.read_pairs_within_windows``). The person's real reads (their minicram) and the simulated version
are genotyped side by side in the training runs' ExpansionHunter image, and each allele's row is compared between the
two. The person, locus and allele sequence are the same, so a difference comes from how the reads are simulated (and
from the few reads of the locus itself that pair with a mate elsewhere, which both versions hold). This measures what
the simulation leaves out on alleles of known sequence, without using the confirmed expansions that
``compare_simulated_and_real_expansions.py`` judges the simulated alleles against.

Subcommands, run in this order:

    select-loci     Each pilot person's loci (``candidate_loci``) whose truth lies inside the person's high-confidence
                    regions, at least ``2 * FLANK_BP`` apart, written with the person's allele sequences to
                    ``loci.tsv``, and an ExpansionHunter catalog of all of them to ``catalog.json``.
    make-minicrams  Each person's reads at the catalog's loci (``spike_in_expansions.make_minicram``).
    measure-read-errors
                    Realigns each person's real reads near their loci to their two alleles and overwrites
                    ``--repeat-read-errors`` (``repeat_read_errors.tsv``, which the spike step and ``simulate`` use)
                    with the errors of the read bases inside the repeats (``repeat_read_error_table``), which the
                    simulation then gives the bases of its reads inside a repeat
                    (``expanded_allele_reads.simulated_read``). Before any ``simulate``, it writes a fragment keep
                    base of 1 (no fragments lost over the repeat) and the real reads' insertion and deletion rates.
                    Once ``simulate`` has run, it also measures the simulated reads the same way, prints both, and
                    writes those two kinds of values corrected by how the simulated reads came out
                    (``calibrated_against_simulation``).
    simulate        Each person's calibration sample: the minicram with the read pairs that lie near the person's loci
                    replaced by simulated pairs of the person's two alleles, at the person's fragment start rate there.
    genotype        Genotypes each person's minicram ("real") and calibration sample ("simulated").
    compare         Pairs each allele's row in the two runs and summarizes how they differ, by the allele's true size.

Usage:
    python3 simulated_expansions/calibrate_simulated_reads.py select-loci --training-tables-dir TABLES
    python3 simulated_expansions/calibrate_simulated_reads.py make-minicrams --reference-fasta HG38_FA \\
        --billing-project PROJECT
    python3 simulated_expansions/calibrate_simulated_reads.py measure-read-errors --reference-fasta HG38_FA --bwa BWA
    python3 simulated_expansions/calibrate_simulated_reads.py simulate --reference-fasta HG38_FA --bwa BWA
    python3 simulated_expansions/calibrate_simulated_reads.py genotype --billing-project PROJECT --gcs-dir gs://...
    python3 simulated_expansions/calibrate_simulated_reads.py compare
"""

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyfaidx
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dataset  # noqa: E402
from simulated_expansions import compare_simulated_and_real_expansions as compare_with_real  # noqa: E402
from simulated_expansions import expanded_allele_reads as sim  # noqa: E402
from simulated_expansions import spike_in_expansions as spike  # noqa: E402
from simulated_expansions.build_disease_loci_table import catalog_entries  # noqa: E402

CALIBRATION_DIR = os.path.join(spike.WORK_DIR, "calibration")
MIN_LONG_ALLELE_BP = 150
# Both truth alleles at least this pure, as most known pathogenic expansions are.
MIN_TRUTH_PURITY = 0.9
ALLELE_SEQUENCE_COLUMNS = ["Allele1Sequence", "Allele2Sequence"]
TRUE_BP_BINS = (0, 150, 300, 600, np.inf)
# Fragments are counted by the share of their bases inside the repeat, in fifths, if they lie within this many bp of
# it (the minicrams hold the reads within 1 kb of each locus).
FRAGMENT_SHARE_BINS = 5
FRAGMENT_COUNTING_FLANK_BP = 900
COMPARED_FIELDS = ("eh",) + compare_with_real.COMPARED_INPUTS


def loci_tsv(work_dir):
    return os.path.join(work_dir, "loci.tsv")


def catalog_json(work_dir):
    return os.path.join(work_dir, "catalog.json")


def genotyped_json(work_dir, name):
    return os.path.join(work_dir, "expansion_hunter", "%s.json.gz" % name)


def simulated_bam(work_dir, person):
    """The person's calibration sample, which ``simulate`` writes."""
    return os.path.join(work_dir, "simulated", "%s.bam" % calibration_label(person))


def calibration_label(person):
    return "%s_calibration" % person


def candidate_loci(person, training_tables_dir):
    """The autosomal loci of the person's training table where the full genotyper called a repeat whose motif size is
    within ``expanded_allele_reads.MEASURED_MOTIF_SIZE_RANGE_BP``, a truth allele is at least ``MIN_LONG_ALLELE_BP``
    long, every truth allele is at least ``MIN_TRUTH_PURITY`` pure, and the person's short reads don't contradict the
    truth."""
    df = pd.read_parquet(os.path.join(training_tables_dir, "%s.parquet" % person),
                         columns=["locus_id", "motif_size", "true", "purity", "genotyping_branch",
                                  dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN])
    df = df[(df["genotyping_branch"] == "full") & df["motif_size"].between(*sim.MEASURED_MOTIF_SIZE_RANGE_BP)
            & ~df["locus_id"].str.match(r"^[XY]-")]
    by_locus = df.assign(true_bp=df["true"] * df["motif_size"]).groupby("locus_id").agg(
        longest_true_bp=("true_bp", "max"), lowest_purity=("purity", "min"),
        contradicted=(dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN, "max"))
    keep = ((by_locus["longest_true_bp"] >= MIN_LONG_ALLELE_BP) & (by_locus["lowest_purity"] >= MIN_TRUTH_PURITY)
            & ~by_locus["contradicted"].astype(bool))
    return sorted(by_locus.index[keep])


def loci_with_allele_sequences(person, sex, locus_ids, truth):
    """The loci of ``locus_ids`` inside the person's high-confidence regions (``truth``: ``person_truth_at_loci`` with
    ``ALLELE_SEQUENCE_COLUMNS``), as dicts with the person's two allele sequences, keeping only the first of loci closer
    than ``2 * FLANK_BP`` (``spike_in_expansions.clusters_of_nearby_loci``) so no read pair is near two of them."""
    loci = []
    for locus_id in locus_ids:
        if locus_id not in truth.index or not truth.at[locus_id, "confident"]:
            continue
        row = truth.loc[locus_id]
        # ANALYSIS_OK[imputation]: an empty allele sequence (a deleted repeat) is read as NaN.
        allele_sequences = [row[c] if isinstance(row[c], str) else "" for c in ALLELE_SEQUENCE_COLUMNS]
        loci.append({"person": person, "sex": sex, "locus_id": locus_id, "chrom": row["Chrom"],
                     "start_0based": int(row["Start0Based"]), "end": int(row["End"]),
                     "reference_motif": locus_id.split("-")[-1],
                     "short_allele_repeats": row["NumRepeatsShortAllele"],
                     "long_allele_repeats": row["NumRepeatsLongAllele"],
                     "allele1_sequence": allele_sequences[0], "allele2_sequence": allele_sequences[1]})
    return [cluster[0] for cluster in spike.clusters_of_nearby_loci(loci)]


def select_loci(people, training_tables_dir, work_dir):
    """Writes ``loci.tsv`` and ``catalog.json`` (see the module docstring)."""
    selected = []
    for person in people:
        locus_ids = candidate_loci(person, training_tables_dir)
        # The spike step's work folder, so both use one download of each person's truth.
        truth = spike.person_truth_at_loci(person, set(locus_ids), spike.WORK_DIR,
                                           columns=spike.TRUTH_COLUMNS + ALLELE_SEQUENCE_COLUMNS)
        loci = loci_with_allele_sequences(person, spike.sex_of(person, training_tables_dir), locus_ids, truth)
        print("%s: %d candidate loci, %d selected" % (person, len(locus_ids), len(loci)))
        selected += loci
    selected = pd.DataFrame(selected)
    os.makedirs(work_dir, exist_ok=True)
    selected.to_csv(loci_tsv(work_dir), sep="\t", index=False)
    with open(catalog_json(work_dir), "w") as f:
        json.dump(catalog_entries(selected.drop_duplicates("locus_id")), f, indent=1)


def allele_repeat(allele_sequence, motif):
    """The repeat (``expanded_allele_reads.Repeat``) of an allele sequence placed between ``FLANK_BP`` flanks."""
    return sim.Repeat(spike.FLANK_BP, spike.FLANK_BP + len(allele_sequence), motif)


def simulate_calibration_sample(person, loci, work_dir, reference_fasta, repeat_errors, fastq1, fastq2):
    """Writes the simulated read pairs of the person's calibration sample to the two FASTQs: each of the person's two
    alleles at each locus (``allele_contigs``, which ``measure_read_errors`` realigns the reads to) gets half the
    person's fragment start rate near the locus, and its in-repeat bases the measured errors ``repeat_errors``.
    Returns the names of the person's read pairs that lie near the loci
    (``spike_in_expansions.read_pairs_within_windows``), which the simulated ones replace."""
    minicram = spike.minicram_path(work_dir, person)
    windows, profile, mean_fragment, fasta = spike.simulation_inputs(minicram, reference_fasta, loci)
    label = calibration_label(person)
    for locus in loci:
        rate = spike.fragment_start_rate_near_locus(minicram, reference_fasta, locus["chrom"], locus["start_0based"],
                                                    locus["end"])
        for allele_number, (sequence, repeat) in enumerate(allele_contigs([locus], fasta).values(), start=1):
            spike.write_simulated_allele_pairs(
                sequence, rate, 0.5, profile, mean_fragment, "%s:%s:%d" % (label, locus["locus_id"], allele_number),
                fastq1, fastq2, repeat, repeat_errors)
    return set().union(*spike.read_pairs_within_windows(minicram, reference_fasta, windows).values())


def simulate(work_dir, reference_fasta, bwa, bwa_index_prefix, repeat_read_errors_tsv):
    """Simulates, aligns and merges every person's calibration sample into ``simulated/<label>.bam``, keeping a copy
    of the in-repeat read errors table it used (``used_repeat_read_errors_tsv``) for ``measure_read_errors``."""
    loci = pd.read_csv(loci_tsv(work_dir), sep="\t", keep_default_na=False)
    repeat_errors = sim.load_repeat_read_errors(repeat_read_errors_tsv)
    out_dir = os.path.join(work_dir, "simulated")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(repeat_read_errors_tsv, used_repeat_read_errors_tsv(work_dir))
    fastq1_path, fastq2_path = os.path.join(out_dir, "simulated_1.fq"), os.path.join(out_dir, "simulated_2.fq")
    dropped = {}
    with open(fastq1_path, "w") as fastq1, open(fastq2_path, "w") as fastq2:
        for person, person_loci in loci.groupby("person", sort=False):
            dropped[person] = simulate_calibration_sample(person, person_loci.to_dict("records"), work_dir,
                                                          reference_fasta, repeat_errors, fastq1, fastq2)
    simulated_reads = spike.write_samples_with_simulated_reads(
        fastq1_path, fastq2_path, bwa, bwa_index_prefix, os.path.join(out_dir, "simulated.bam"), reference_fasta,
        [(calibration_label(person), spike.minicram_path(work_dir, person), dropped[person],
          simulated_bam(work_dir, person)) for person in dropped])
    for person in dropped:
        print("%s: %d real pairs replaced by %d simulated reads" % (calibration_label(person), len(dropped[person]),
                                                                     simulated_reads[calibration_label(person)]))


def allele_contigs(loci, fasta):
    """``{contig name: (sequence, repeat)}``: each locus's two alleles between ``FLANK_BP`` reference flanks, to realign
    the reads near the locus to (``fasta``: ``pyfaidx.Fasta(path, one_based_attributes=False, as_raw=True)``)."""
    contigs = {}
    for locus in loci:
        for allele_number, column in enumerate(("allele1_sequence", "allele2_sequence"), start=1):
            contigs["%s_allele%d" % (locus["locus_id"], allele_number)] = (
                sim.allele_sequence_with_reference_flanks(fasta, locus["chrom"], locus["start_0based"], locus["end"],
                                                          locus[column], spike.FLANK_BP),
                allele_repeat(locus[column], locus["reference_motif"]))
    return contigs


def write_reads_as_sequenced(bam_path, reference_fasta, read_names, out_fastq):
    """Writes every primary read of the named pairs, wherever it mapped, as it was sequenced (a reverse read turned
    back), to a FASTQ."""
    with pysam.AlignmentFile(bam_path, reference_filename=reference_fasta) as bam, open(out_fastq, "w") as out:
        for read in bam.fetch(until_eof=True):
            if read.query_name not in read_names or read.is_secondary or read.is_supplementary:
                continue
            sequence, quality = read.query_sequence, pysam.qualities_to_qualitystring(read.query_qualities)
            if read.is_reverse:
                sequence, quality = sim.reverse_complement(sequence), quality[::-1]
            out.write("@%s/%d\n%s\n+\n%s\n" % (read.query_name, 1 if read.is_read1 else 2, sequence, quality))


def realign_to_alleles(fastq, contigs, bwa, out_dir, out_bam):
    """Aligns the reads of ``fastq`` one by one (bwa mem) to the allele contigs (``allele_contigs``), written and
    indexed in ``out_dir``, into an unsorted BAM."""
    contigs_fasta = os.path.join(out_dir, "alleles.fa")
    with open(contigs_fasta, "w") as f:
        for name, (sequence, _) in contigs.items():
            f.write(">%s\n%s\n" % (name, sequence))
    subprocess.run([bwa, "index", contigs_fasta], check=True, capture_output=True)
    spike.align([fastq], bwa, contigs_fasta, out_bam, threads=8)


def new_error_counts():
    """Empty counts for ``tally_in_repeat_errors``."""
    return {name: collections.Counter() for name in ("bases", "mismatches", "indels", "indel_sizes",
                                                     "low_quality_values", "fragments")}


def fragment_share_bin(fragment_start, fragment_end, repeat):
    """The share of the fragment's bases inside the repeat, rounded up to a fifth (0 for none)."""
    inside = max(0, min(fragment_end, repeat.end) - max(fragment_start, repeat.start))
    return -(-inside * FRAGMENT_SHARE_BINS // (fragment_end - fragment_start)) / FRAGMENT_SHARE_BINS


def tally_in_repeat_errors(aligned_bam, contigs, counts):
    """Adds the read bases that the primary alignments in ``aligned_bam`` place inside a repeat of ``contigs`` to
    ``counts`` (``new_error_counts``): bases and mismatches by (motif GC bin, read cycle bin, quality bin), insertion
    and deletion events by (GC bin, kind of ``expanded_allele_reads.INDEL_KINDS``), their sizes by (kind, size: motif
    copies for whole-motif ones, else bp), and the low quality values. Also counts the fragments whose two reads align
    to one contig within ``FRAGMENT_COUNTING_FLANK_BP`` of its repeat, by (GC bin, ``fragment_share_bin``)."""
    spans = collections.defaultdict(list)
    with pysam.AlignmentFile(aligned_bam) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            spans[read.query_name].append((read.reference_name, read.reference_start, read.reference_end))
            repeat = contigs[read.reference_name][1]
            if read.reference_end <= repeat.start or read.reference_start >= repeat.end:
                continue
            gc, qualities = sim.motif_gc_bin(repeat.motif), read.query_qualities
            for read_position, contig_position, contig_base in read.get_aligned_pairs(matches_only=True,
                                                                                      with_seq=True):
                if repeat.start <= contig_position < repeat.end:
                    cycle = read.query_length - 1 - read_position if read.is_reverse else read_position
                    key = (gc, sim.cycle_bin(cycle), sim.quality_bin(qualities[read_position]))
                    counts["bases"][key] += 1
                    counts["mismatches"][key] += contig_base.islower()
                    if qualities[read_position] < sim.LOW_QUALITY:
                        counts["low_quality_values"][qualities[read_position]] += 1
            contig_position = read.reference_start
            for operation, length in read.cigartuples:
                if operation in (pysam.CINS, pysam.CDEL) and repeat.start <= contig_position < repeat.end:
                    whole_motif = length % len(repeat.motif) == 0
                    kind = "%s %s-motif" % ("insertion" if operation == pysam.CINS else "deletion",
                                            "whole" if whole_motif else "partial")
                    counts["indels"][(gc, kind)] += 1
                    counts["indel_sizes"][(kind, length // len(repeat.motif) if whole_motif else length)] += 1
                if operation in (pysam.CMATCH, pysam.CDEL, pysam.CREF_SKIP, pysam.CEQUAL, pysam.CDIFF):
                    contig_position += length
    for mates in spans.values():
        if len(mates) == 2 and mates[0][0] == mates[1][0]:
            repeat = contigs[mates[0][0]][1]
            start, end = min(m[1] for m in mates), max(m[2] for m in mates)
            if repeat.start - FRAGMENT_COUNTING_FLANK_BP <= start and end <= repeat.end + FRAGMENT_COUNTING_FLANK_BP:
                counts["fragments"][(sim.motif_gc_bin(repeat.motif), fragment_share_bin(start, end, repeat))] += 1


def repeat_read_error_table(counts):
    """The quantities of ``expanded_allele_reads.RepeatReadErrors`` from ``counts`` (``tally_in_repeat_errors``), as
    rows of (quantity, group, key, value). Raises if a GC, cycle or quality bin the simulation looks up has no bases."""
    bases_by_gc_and_cycle, low_by_gc_and_cycle = collections.Counter(), collections.Counter()
    bases_by_gc_and_quality, mismatches_by_gc_and_quality = collections.Counter(), collections.Counter()
    for (gc, cycle, quality), n in counts["bases"].items():
        bases_by_gc_and_cycle[(gc, cycle)] += n
        low_by_gc_and_cycle[(gc, cycle)] += n if quality == sim.QUALITY_BINS[0] else 0
        bases_by_gc_and_quality[(gc, quality)] += n
        mismatches_by_gc_and_quality[(gc, quality)] += counts["mismatches"][(gc, cycle, quality)]
    cycles = [k * sim.CYCLE_BIN_BP for k in range(sim.LAST_CYCLE_BIN + 1)]
    missing = ([(gc, cycle) for gc in sim.GC_BINS for cycle in cycles if not bases_by_gc_and_cycle[(gc, cycle)]]
               + [(gc, q) for gc in sim.GC_BINS for q in sim.QUALITY_BINS if not bases_by_gc_and_quality[(gc, q)]])
    if missing:
        raise RuntimeError("no in-repeat bases measured in bins %s" % missing)
    rows = [("low_quality_share", gc, cycle, low_by_gc_and_cycle[(gc, cycle)] / bases_by_gc_and_cycle[(gc, cycle)])
            for gc in sim.GC_BINS for cycle in cycles]
    rows += [("mismatch_rate", gc, q, mismatches_by_gc_and_quality[(gc, q)] / bases_by_gc_and_quality[(gc, q)])
             for gc in sim.GC_BINS for q in sim.QUALITY_BINS]
    indel_rates = indel_rates_by_gc_and_kind(counts)
    rows += [("indel_rate", gc, kind, indel_rates[(gc, kind)]) for gc in sim.GC_BINS for kind in sim.INDEL_KINDS]
    for kind in sim.INDEL_KINDS:
        sizes = {size: n for (k, size), n in counts["indel_sizes"].items() if k == kind}
        # A kind never seen gets size 1 with no events (its rate is 0), so the simulation can still look it up.
        rows += [("indel_size_share", kind, size, n / sum(sizes.values())) for size, n in sorted(sizes.items())] or [
            ("indel_size_share", kind, 1, 1.0)]
    total_low = sum(counts["low_quality_values"].values())
    rows += [("low_quality_value_share", "", q, n / total_low) for q, n in sorted(counts["low_quality_values"].items())]
    # Real reads alone don't tell how many fragments over the repeat were lost; calibrated_against_simulation sets it.
    rows += [("fragment_keep_base", gc, "", 1.0) for gc in sim.GC_BINS]
    return pd.DataFrame(rows, columns=["quantity", "group", "key", "value"])


def indel_rates_by_gc_and_kind(counts):
    """``{(GC bin, kind): insertion or deletion events per in-repeat base}`` of ``counts`` (``tally_in_repeat_errors``),
    for every GC bin with in-repeat bases and every kind of ``expanded_allele_reads.INDEL_KINDS``."""
    bases_by_gc = collections.Counter()
    for (gc, _, _), n in counts["bases"].items():
        bases_by_gc[gc] += n
    return {(gc, kind): counts["indels"][(gc, kind)] / bases_by_gc[gc]
            for gc in bases_by_gc for kind in sim.INDEL_KINDS if bases_by_gc[gc]}


def fitted_fragment_keep_base(real_fragments, simulated_fragments, gc):
    """The c for which real over simulated fragment counts go as c^f in the GC bin, f being the fragments' share of
    bases inside the repeat (the bin's middle) and the counts taken relative to those of fragments clear of it: a least
    squares fit of the log ratios through the origin, weighted by the fragments, over the bins both runs have. 1 when
    there is nothing to fit."""
    def ratio(share):
        return real_fragments[(gc, share)] / simulated_fragments[(gc, share)]

    if not (real_fragments[(gc, 0.0)] and simulated_fragments[(gc, 0.0)]):
        return 1.0
    weighted_products = weighted_squares = 0.0
    for k in range(1, FRAGMENT_SHARE_BINS + 1):
        share = k / FRAGMENT_SHARE_BINS
        if real_fragments[(gc, share)] and simulated_fragments[(gc, share)]:
            weight = real_fragments[(gc, share)] + simulated_fragments[(gc, share)]
            middle = share - 0.5 / FRAGMENT_SHARE_BINS
            weighted_products += weight * middle * np.log(ratio(share) / ratio(0.0))
            weighted_squares += weight * middle ** 2
    return float(np.exp(weighted_products / weighted_squares)) if weighted_squares else 1.0


def calibrated_against_simulation(table, real_counts, simulated_counts, used_errors):
    """``table`` (``repeat_read_error_table`` of the real reads) with the two values a simulation can't take from real
    reads directly set so that its reads come out like the real ones, given the values ``used_errors`` it used: each
    insertion and deletion rate is the used one times the real reads' rate over the simulated reads' (realigned, some
    simulated ones near a read's end come out as mismatches or clipped bases), and each GC bin's fragment keep base
    is the used one times ``fitted_fragment_keep_base`` of the real over the simulated fragments."""
    simulated_rates = indel_rates_by_gc_and_kind(simulated_counts)
    table = table.copy()
    for i, row in table.iterrows():
        key = (row["group"], row["key"])
        if row["quantity"] == "indel_rate" and simulated_rates.get(key):
            # The table's value is the real reads' rate.
            table.at[i, "value"] = used_errors.indel_rate[key] * row["value"] / simulated_rates[key]
        elif row["quantity"] == "fragment_keep_base":
            table.at[i, "value"] = used_errors.fragment_keep_base[row["group"]] * fitted_fragment_keep_base(
                real_counts["fragments"], simulated_counts["fragments"], row["group"])
    return table


def error_summary(counts):
    """Per motif GC bin: the in-repeat bases, the share below LOW_QUALITY, and mismatches and insertion or deletion
    events per 1000 bases."""
    rows = {}
    for (gc, _, quality), n in counts["bases"].items():
        row = rows.setdefault(gc, collections.Counter())
        row["bases"] += n
        row["low"] += n if quality == sim.QUALITY_BINS[0] else 0
    for (gc, _, _), n in counts["mismatches"].items():
        rows[gc]["mismatches"] += n
    for (gc, _), n in counts["indels"].items():
        rows[gc]["indels"] += n
    return pd.DataFrame([{"motif GC": gc, "in-repeat bases": row["bases"], "share below Q20": row["low"] / row["bases"],
                          "mismatches /1000": 1000 * row["mismatches"] / row["bases"],
                          "insertions and deletions /1000": 1000 * row["indels"] / row["bases"]}
                         for gc, row in sorted(rows.items())])


def used_repeat_read_errors_tsv(work_dir):
    """Where ``simulate`` keeps a copy of the in-repeat read errors table it simulated with."""
    return os.path.join(work_dir, "simulated", "repeat_read_errors.used.tsv")


def measure_read_errors(work_dir, reference_fasta, bwa, out_tsv):
    """Measures the in-repeat errors of each person's real reads that lie near their loci (and of their calibration
    sample's simulated reads, once ``simulate`` has made them) by realigning them to the person's alleles, and writes
    the real reads' ``repeat_read_error_table`` to ``out_tsv``, its insertion and deletion rates and fragment keep
    bases ``calibrated_against_simulation`` when there are simulated reads. Prints ``error_summary`` of each."""
    loci = pd.read_csv(loci_tsv(work_dir), sep="\t", keep_default_na=False)
    fasta = pyfaidx.Fasta(reference_fasta, one_based_attributes=False, as_raw=True)
    counts = {"real": new_error_counts(), "simulated": new_error_counts()}
    for person, person_loci in loci.groupby("person"):
        records = person_loci.to_dict("records")
        out_dir = os.path.join(work_dir, "read_errors", person)
        os.makedirs(out_dir, exist_ok=True)
        contigs = allele_contigs(records, fasta)
        windows = spike.windows_around_loci(records)
        label = calibration_label(person)
        reads = [("real", spike.minicram_path(work_dir, person), "")]
        if os.path.exists(simulated_bam(work_dir, person)):
            reads.append(("simulated", simulated_bam(work_dir, person), label + ":"))
        for run, bam_path, name_prefix in reads:
            names = {name for names in spike.read_pairs_within_windows(bam_path, reference_fasta, windows).values()
                     for name in names if name.startswith(name_prefix)}
            fastq, aligned = os.path.join(out_dir, "%s.fq" % run), os.path.join(out_dir, "%s.bam" % run)
            write_reads_as_sequenced(bam_path, reference_fasta, names, fastq)
            realign_to_alleles(fastq, contigs, bwa, out_dir, aligned)
            tally_in_repeat_errors(aligned, contigs, counts[run])
        print("%s: measured" % person, flush=True)
    table = repeat_read_error_table(counts["real"])
    if counts["simulated"]["bases"]:
        table = calibrated_against_simulation(table, counts["real"], counts["simulated"],
                                              sim.load_repeat_read_errors(used_repeat_read_errors_tsv(work_dir)))
    table.to_csv(out_tsv, sep="\t", index=False)
    pd.set_option("display.width", 200)
    for run in ("real", "simulated"):
        if counts[run]["bases"]:
            print("\n%s reads inside the repeats:\n%s" % (run, error_summary(counts[run]).to_string(
                index=False, float_format=lambda v: "%.4f" % v)))
    print("\nfragment keep bases written: %s" % {row["group"]: round(row["value"], 3) for row in table[
        table["quantity"] == "fragment_keep_base"].to_dict("records")})


def genotype(work_dir, billing_project, gcs_dir):
    """Genotypes each person's minicram and calibration sample with ``spike_in_expansions.genotype_on_hail_batch``."""
    samples = []
    people_and_sexes = pd.read_csv(loci_tsv(work_dir), sep="\t").drop_duplicates("person")[["person", "sex"]]
    for person, sex in people_and_sexes.itertuples(index=False):
        for name, reads in (("%s_real" % calibration_label(person), spike.minicram_path(work_dir, person)),
                            ("%s_simulated" % calibration_label(person), simulated_bam(work_dir, person))):
            samples.append({"name": name, "reads": reads, "sex": sex, "out_json": genotyped_json(work_dir, name)})
    spike.genotype_on_hail_batch(samples, catalog_json(work_dir), billing_project, gcs_dir,
                                 "simulated_expansions: calibrate the simulated reads")


def paired_rows(loci, real_rows_by_person, simulated_rows_by_person):
    """One row per (person, locus, allele rank) the two runs both genotyped with the allele's own quality metrics:
    ``true_repeats`` and ``true_bp`` (the truth allele of that rank, 0 the short one) and ``real_<field>`` and
    ``simulated_<field>`` for ``COMPARED_FIELDS``.

    Args:
        loci: ``loci.tsv``.
        real_rows_by_person, simulated_rows_by_person: ``{person: spike_in_expansions.rows_at_loci output}``.
    """
    out = []
    for locus in loci.to_dict("records"):
        real, simulated = real_rows_by_person[locus["person"]], simulated_rows_by_person[locus["person"]]
        motif_size = len(locus["reference_motif"])
        for rank, true_repeats in enumerate((locus["short_allele_repeats"], locus["long_allele_repeats"])):
            key = (locus["locus_id"], rank)
            if key not in real or key not in simulated or not (real[key]["has_own_quality_metrics"]
                                                                and simulated[key]["has_own_quality_metrics"]):
                continue
            out.append({"person": locus["person"], "locus_id": locus["locus_id"], "allele_rank": rank,
                        "true_repeats": true_repeats, "true_bp": true_repeats * motif_size,
                        **{"real_%s" % f: real[key].get(f) for f in COMPARED_FIELDS},
                        **{"simulated_%s" % f: simulated[key].get(f) for f in COMPARED_FIELDS}})
    return pd.DataFrame(out)


def summary(pairs):
    """Per true-size bin (``TRUE_BP_BINS``) and field: the alleles, the median in each run, the median paired
    difference (simulated minus real), and for the call (``eh``) the share called exactly right."""
    pairs = pairs.assign(true_bp_bin=pd.cut(pairs["true_bp"], TRUE_BP_BINS, right=False).astype(str))
    rows = []
    for size_bin, group in pairs.groupby("true_bp_bin", sort=False):
        for field in COMPARED_FIELDS:
            # ANALYSIS_OK[imputation]: a metric ExpansionHunter left out (None) becomes NaN, which the medians skip.
            real = pd.to_numeric(group["real_%s" % field], errors="coerce")
            simulated = pd.to_numeric(group["simulated_%s" % field], errors="coerce")
            rows.append({"true_bp": size_bin, "field": field, "alleles": len(group), "median_real": real.median(),
                         "median_simulated": simulated.median(), "median_difference": (simulated - real).median(),
                         "mean_real": real.mean(), "mean_simulated": simulated.mean()})
        rows.append({"true_bp": size_bin, "field": "called exactly", "alleles": len(group),
                     "mean_real": (group["real_eh"] == group["true_repeats"]).mean(),
                     "mean_simulated": (group["simulated_eh"] == group["true_repeats"]).mean()})
    return pd.DataFrame(rows)


def compare(work_dir):
    """Writes ``comparison.pairs.tsv`` (``paired_rows``) and ``comparison.summary.tsv`` (``summary``), and prints the
    summary and how well a classifier tells the long alleles' real rows from their simulated ones."""
    loci = pd.read_csv(loci_tsv(work_dir), sep="\t", keep_default_na=False)
    real_rows, simulated_rows = {}, {}
    for person, person_loci in loci.groupby("person"):
        locus_ids = set(person_loci["locus_id"])
        real_rows[person] = spike.rows_at_loci(genotyped_json(work_dir, "%s_real" % calibration_label(person)),
                                               locus_ids)
        simulated_rows[person] = spike.rows_at_loci(
            genotyped_json(work_dir, "%s_simulated" % calibration_label(person)), locus_ids)
    pairs = paired_rows(loci, real_rows, simulated_rows)
    pairs.to_csv(os.path.join(work_dir, "comparison.pairs.tsv"), sep="\t", index=False)
    table = summary(pairs)
    table.to_csv(os.path.join(work_dir, "comparison.summary.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False, float_format=lambda v: "%.3f" % v))
    long_alleles = pairs[pairs["true_bp"] >= MIN_LONG_ALLELE_BP]
    inputs = list(compare_with_real.COMPARED_INPUTS)
    real = long_alleles[["real_%s" % f for f in inputs]].set_axis(inputs, axis=1)
    simulated = long_alleles[["simulated_%s" % f for f in inputs]].set_axis(inputs, axis=1)
    # An allele's real row and its simulated twin stay in one fold.
    allele = np.arange(len(long_alleles))
    print("\nlong alleles: %d; ROC-AUC of telling their real rows from their simulated ones: %.3f"
          % (len(long_alleles), compare_with_real.classifier_auc(real, simulated, groups=np.r_[allele, allele])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["select-loci", "make-minicrams", "measure-read-errors", "simulate",
                                            "genotype", "compare"])
    parser.add_argument("--person", action="append", help="default: %s" % ", ".join(spike.PILOT_PEOPLE))
    parser.add_argument("--work-dir", default=CALIBRATION_DIR)
    parser.add_argument("--training-tables-dir", help="as for spike_in_expansions.py (select-loci)")
    parser.add_argument("--reference-fasta",
                        help="as for spike_in_expansions.py (make-minicrams, measure-read-errors, simulate)")
    parser.add_argument("--bwa", default="bwa", help="the bwa executable (measure-read-errors, simulate)")
    parser.add_argument("--bwa-index", default=spike.DEFAULT_BWA_INDEX_PREFIX,
                        help="as for spike_in_expansions.py (simulate)")
    parser.add_argument("--repeat-read-errors", default=sim.REPEAT_READ_ERRORS_TSV,
                        help="the in-repeat read errors table measure-read-errors writes and simulate uses")
    parser.add_argument("--billing-project", help="Hail Batch and requester-pays billing project (make-minicrams, "
                                                  "genotype)")
    parser.add_argument("--gcs-dir", help="gs:// folder for the reads genotyped on Hail Batch (genotype)")
    args = parser.parse_args()
    people = args.person or spike.PILOT_PEOPLE
    if args.command == "select-loci" and not args.training_tables_dir:
        parser.error("select-loci needs --training-tables-dir")
    if args.command in ("make-minicrams", "measure-read-errors", "simulate") and not args.reference_fasta:
        parser.error("%s needs --reference-fasta" % args.command)
    if args.command in ("make-minicrams", "genotype") and not args.billing_project:
        parser.error("%s needs --billing-project" % args.command)
    if args.command == "genotype" and not args.gcs_dir:
        parser.error("genotype needs --gcs-dir")
    if args.command == "select-loci":
        select_loci(people, args.training_tables_dir, args.work_dir)
    elif args.command == "make-minicrams":
        for person in people:
            spike.make_minicram(person, args.work_dir, args.reference_fasta, catalog_json(args.work_dir),
                                args.billing_project)
    elif args.command == "measure-read-errors":
        measure_read_errors(args.work_dir, args.reference_fasta, args.bwa, args.repeat_read_errors)
    elif args.command == "simulate":
        simulate(args.work_dir, args.reference_fasta, args.bwa, args.bwa_index, args.repeat_read_errors)
    elif args.command == "genotype":
        genotype(args.work_dir, args.billing_project, args.gcs_dir)
    else:
        compare(args.work_dir)


if __name__ == "__main__":
    main()
