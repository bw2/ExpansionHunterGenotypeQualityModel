"""Checks each person's truth genotypes against that person's own short reads at the loci where ExpansionHunter
disagrees with the truth, and marks the rows whose truth the reads clearly contradict, so training can leave them out
and model comparisons can score with and without them.

Why: an assembly can miss a heterozygous allele or get a repeat's length wrong. Where ExpansionHunter's call agrees with
the reads but not with such a truth, the row is labeled TOO_LONG or TOO_SHORT although the call is right, and the model
learns that well-supported calls are wrong.

Steps (each person's loci are those where a call in any of the person's per-source parquets disagrees with the truth):

    python3 check_truth_genotypes_against_short_reads.py --dry-run
    python3 check_truth_genotypes_against_short_reads.py --no-wait -i HG002     # one person first
    python3 check_truth_genotypes_against_short_reads.py --no-wait              # all 133 people
    python3 check_truth_genotypes_against_short_reads.py --mark

1. Without --mark: one Hail Batch job per person copies the person's CRAM and measures the repeat length in every read at
   the person's loci (``measure_repeat_lengths_in_short_reads.py``; no genotyper is involved, so the check does not
   depend on ExpansionHunter). The output's name carries a hash of the loci list and of the measuring script's code
   (not its comments or docstrings), so a changed truth set or EH run gets new measurements and an unchanged one is not
   measured twice.
2. --mark: downloads each person's measurements, judges every locus against the truth genotype in the parquets
   (``judge_truth_genotype_against_reads``) and writes the column ``truth_contradicted_by_short_reads`` into each of the
   person's per-source parquets. ``dataset.label_and_filter`` drops the marked rows when asked to: training does
   (``dataset.assemble_branch`` refuses parquets without the column), and ``compare_models.py`` scores the held-out
   samples both with and without them. A rebuilt parquet loses the column, so it has to be marked again.

The judging rule is deliberately strict (``judge_truth_genotype_against_reads``): a locus is marked only when the reads
show an allele the truth genotype lacks, either as a cluster of clean spanning reads or as robust reads longer than the
truth's long allele (at least 2 of them spanning unless a robust one shows an allele longer than a read can span, which
is the only evidence against such an allele), and the truth genotype cannot explain the reads. Loci whose flanks are
not unique or that have fewer than 10 measured reads of mapping quality 40 or more are never marked, and differences of
1 repeat (or of 2 bp in a homopolymer) are never enough. Each person's judgments are written next to the
measurements (``<data dir>/short_read_checks_of_truth_genotypes/``) for inspection.
"""

import argparse
import ast
import collections
import gzip
import hashlib
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import dataset
import features
import measure_repeat_lengths_in_short_reads as measure

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_CONTRADICTED_COLUMN = dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN
DEFAULT_OUTPUT_DIR = "gs://tandem-repeat-explorer/tool_genotype_quality/short_read_checks_of_truth_genotypes"
LOCAL_SUBDIR = "short_read_checks_of_truth_genotypes"
# reads_path and reads_index_path of each person's CRAM, keyed by sample_label (the truth sample id), so HG002 uses
# its full-depth CRAM for all three of its coverages.
SHORT_READ_PATHS_TSV = os.path.join(HERE, "short_read_paths.tsv")
# hg38, the reference the catalog's coordinates are on. The HG002 CRAM was compressed against UCSC's hg38, whose
# checksums differ, so samtools fetches its contigs by checksum from the EBI reference server.
REFERENCE_FASTA = "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"
# An image with samtools 1.20 and Python 3.11, all the measuring script needs.
IMAGE = "weisburd/str-analysis-with-expansion-hunter@sha256:9e3090a2a69506a513a42ec49d1b9f83f1f0302c828741c3dfde001c629435d8"
JOB_DIR = "/io"
SAM_FLAGS_TO_EXCLUDE = "0xF04"  # unmapped, secondary, QC fail, duplicate, supplementary
PARQUET_METADATA_KEY = b"truth_check_against_short_reads"
MEASUREMENTS_SUFFIX = ".measured_reads.tsv.gz"
# The parquet columns needed to find the loci where a call disagrees with the truth.
CALL_AND_TRUTH_COLUMNS = ("locus_id", "eh", "true", "motif_size")

# The judging rule (judge_truth_genotype_against_reads). Mapping quality 40 rather than 20: reads of mapping quality
# 20-39 are more often mapped from a similar locus elsewhere, and marks resting on them were more often wrong.
MIN_MAPPING_QUALITY = 40
MIN_BASE_QUALITY = 20
MIN_MEASURED_READS = 10
MIN_REPEATS_DIFFERENCE = 2
MIN_BP_DIFFERENCE = 3
MIN_READS_SHOWING_AN_ALLELE_THE_TRUTH_LACKS = 4
MIN_FRACTION_SHOWING_AN_ALLELE_THE_TRUTH_LACKS = 0.25
MIN_ROBUST_READS_LONGER_THAN_TRUTH = 3
MIN_SPANNING_READS_LONGER_THAN_TRUTH_WHEN_SPANNABLE = 2
MAX_SPANNABLE_ALLELE_BP = 90
MIN_SPANNING_READS_TO_JUDGE_TRUTH_ALLELE_SUPPORT = 8
MAX_SUPPORT_FRACTION_OF_AN_UNSUPPORTED_TRUTH_ALLELE = 0.10

FLANKS_NOT_UNIQUE = "flanks next to the locus are not unique"
TOO_FEW_READS = "fewer than %d measured reads with mapping quality >= %d" % (MIN_MEASURED_READS, MIN_MAPPING_QUALITY)
NOT_CONTRADICTED = "not contradicted"
TRUTH_ALLELES_BOTH_SUPPORTED = "not contradicted: reads support both truth alleles"
SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS = "contradicted: spanning reads show an allele the truth lacks"
READS_LONGER_THAN_THE_TRUTH = "contradicted: reads longer than the truth's long allele"
CONTRADICTING_JUDGMENTS = (SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS, READS_LONGER_THAN_THE_TRUTH)


def parse_measured_reads(measured_reads, motif_size):
    """Returns the measured reads with mapping quality >= ``MIN_MAPPING_QUALITY`` as (kind, repeats, mean base quality,
    whether the motif run is at the read's sequencing 3' end) tuples; kind is s (spanning) or l / r (one-sided, a lower
    bound).

    Lengths become repeat counts the way the truth set counts them: floor(length in bp / motif length).
    """
    reads = []
    for read in filter(None, measured_reads.split(",")):
        kind, length_bp, mapping_quality, base_quality, strand = read.split(":")
        if int(mapping_quality) >= MIN_MAPPING_QUALITY:
            run_at_3_prime_end = (kind == "l" and strand == "+") or (kind == "r" and strand == "-")
            reads.append((kind, int(length_bp) // motif_size, int(base_quality), run_at_3_prime_end))
    return reads


def judge_truth_genotype_against_reads(reads, truth_short, truth_long, motif_size):
    """Returns how the reads at one locus bear on its truth genotype: one of the judgment constants above.

    A locus is contradicted (``CONTRADICTING_JUDGMENTS``) only when, with >= 10 measured reads,

    - the reads show an allele the truth lacks, as either
      (A) >= 4 spanning reads with base quality >= 20, >= 25% of the spanning reads, within 1 repeat of a common length
          (so spread over at most 3 adjacent lengths) and >= 2 repeats and >= 3 bp from both truth alleles; or
      (B) >= 4 reads, >= 25% of the measured reads, >= 2 repeats and >= 3 bp longer than the truth's long allele, of
          which >= 3 are robust (base quality >= 20, and spanning or with the motif run at the read's sequencing 5'
          end, where low-quality tails and poly-G artifacts can't produce it), and >= 2 spanning unless a robust one
          shows an allele too long to be spanned (> 90 bp: shorter one-sided reads alone mean nothing, since spanning
          reads would show such an allele);
    - and the truth genotype can't explain the reads: it is homozygous (the reads show a second allele), or one of its
      alleles short enough to be spanned has <= 10% of >= 8 clean spanning reads, or, for (B), its long allele is too
      long to be spanned (the longer reads then contradict it directly). A heterozygous truth whose two alleles both
      have read support is never contradicted: a third allele can be mosaicism or a mapping artifact.

    Args:
        reads: ``parse_measured_reads`` output.
        truth_short, truth_long: the truth genotype in repeats.
        motif_size: the motif length in bp.
    """
    if len(reads) < MIN_MEASURED_READS:
        return TOO_FEW_READS

    def far_from(repeats, truth_allele):
        return (abs(repeats - truth_allele) >= MIN_REPEATS_DIFFERENCE
                and abs(repeats - truth_allele) * motif_size >= MIN_BP_DIFFERENCE)

    spanning = [r for r in reads if r[0] == "s"]
    # A spanning read with no bases between its flanks (the repeat deleted) has no base quality to judge (-1); both of
    # its flanks matched next to each other, so it counts as clean.
    clean_spanning = [repeats for kind, repeats, base_quality, _ in spanning
                      if base_quality >= MIN_BASE_QUALITY or base_quality == -1]

    def support(truth_allele):
        """Clean spanning reads within 1 repeat of the truth allele and nearer to it than to the other one."""
        other = truth_long if truth_allele == truth_short else truth_short
        return sum(abs(v - truth_allele) <= 1 and (truth_short == truth_long or abs(v - truth_allele) < abs(v - other))
                   for v in clean_spanning)

    spannable_truth_allele_without_support = len(clean_spanning) >= MIN_SPANNING_READS_TO_JUDGE_TRUTH_ALLELE_SUPPORT and any(
        allele * motif_size <= MAX_SPANNABLE_ALLELE_BP
        and support(allele) <= MAX_SUPPORT_FRACTION_OF_AN_UNSUPPORTED_TRUTH_ALLELE * len(clean_spanning)
        for allele in {truth_short, truth_long})
    homozygous = truth_short == truth_long

    far = [v for v in clean_spanning if far_from(v, truth_short) and far_from(v, truth_long)]
    # The common length need not be one a read shows: reads at 6 and 8 repeats are within 1 repeat of 7.
    centers = {v + d for v in far for d in (-1, 0, 1)}
    largest_cluster = max((sum(abs(w - center) <= 1 for w in far) for center in centers), default=0)
    if (largest_cluster >= MIN_READS_SHOWING_AN_ALLELE_THE_TRUTH_LACKS
            and largest_cluster >= MIN_FRACTION_SHOWING_AN_ALLELE_THE_TRUTH_LACKS * len(spanning)
            and (homozygous or spannable_truth_allele_without_support)):
        return SPANNING_READS_SHOW_AN_ALLELE_THE_TRUTH_LACKS

    longer = [r for r in reads if r[1] > truth_long and far_from(r[1], truth_long)]
    robust = [r for r in longer if r[2] >= MIN_BASE_QUALITY and (r[0] == "s" or not r[3])]
    robust_reads_show_an_allele_too_long_to_span = any(r[1] * motif_size > MAX_SPANNABLE_ALLELE_BP for r in robust)
    if (len(longer) >= MIN_READS_SHOWING_AN_ALLELE_THE_TRUTH_LACKS
            and len(longer) >= MIN_FRACTION_SHOWING_AN_ALLELE_THE_TRUTH_LACKS * len(reads)
            and len(robust) >= MIN_ROBUST_READS_LONGER_THAN_TRUTH
            and (robust_reads_show_an_allele_too_long_to_span
                 or sum(r[0] == "s" for r in longer) >= MIN_SPANNING_READS_LONGER_THAN_TRUTH_WHEN_SPANNABLE)
            and (homozygous or truth_long * motif_size > MAX_SPANNABLE_ALLELE_BP
                 or spannable_truth_allele_without_support)):
        return READS_LONGER_THAN_THE_TRUTH
    support_of_both_truth_alleles_was_checked = (
        len(clean_spanning) >= MIN_SPANNING_READS_TO_JUDGE_TRUTH_ALLELE_SUPPORT
        and truth_long * motif_size <= MAX_SPANNABLE_ALLELE_BP)
    if not homozygous and support_of_both_truth_alleles_was_checked and not spannable_truth_allele_without_support:
        return TRUTH_ALLELES_BOTH_SUPPORTED
    return NOT_CONTRADICTED


def code_without_comments_and_docstrings(source_text):
    """Returns Python source text without its comments and docstrings (ast drops comments; docstrings are removed
    here), so a hash of it changes only when the code does. ast.unparse's output can differ between Python versions,
    so such a hash can also change with a new Python version."""
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def judging_rule_fingerprint():
    """Returns a hash of the judging code (``code_without_comments_and_docstrings``) and its thresholds, recorded in
    each marked parquet so a changed rule is noticed (``--mark`` then marks again)."""
    thresholds = {name: value for name, value in globals().items() if name.startswith(("MIN_", "MAX_"))}
    thresholds["CONTRADICTING_JUDGMENTS"] = list(CONTRADICTING_JUDGMENTS)
    code = code_without_comments_and_docstrings("".join(inspect.getsource(function) for function in (
        parse_measured_reads, judge_truth_genotype_against_reads, judge_person)))
    return hashlib.sha256((code + json.dumps(thresholds, sort_keys=True)).encode()).hexdigest()[:12]


def disagreeing_loci_and_truth_of_source(parquet_path):
    """Reads one per-source parquet and returns ``(locus ids where a call disagrees with the truth, truth genotypes)``.

    A locus disagrees when any row with a call and a truth allele would be labeled other than OK
    (``features.direction_codes``), including the rows training leaves out (the second copy of a homozygous call, a
    0-repeat truth allele) but the all-loci early warning scores (``heldout._fast_path_loci_of_sample``). The truth
    genotypes are a DataFrame indexed by locus_id with ``truth_short`` and ``truth_long`` (the smallest and largest truth
    allele of the locus, in repeats) and ``motif_size``.
    """
    df = pd.read_parquet(parquet_path, columns=list(CALL_AND_TRUTH_COLUMNS))
    # ANALYSIS_OK[imputation]: rows without a truth allele carry no truth genotype, and rows without a call (no-calls)
    # can't disagree with it.
    with_truth = df[df["true"].notna()]
    truth = with_truth.groupby("locus_id", sort=False).agg(
        truth_short=("true", "min"), truth_long=("true", "max"), motif_size=("motif_size", "first"))
    truth[["truth_short", "truth_long", "motif_size"]] = truth[["truth_short", "truth_long", "motif_size"]].round().astype(int)
    called = with_truth[with_truth["eh"].notna()]
    true = called["true"].to_numpy(float)
    codes = features.direction_codes(called["eh"].to_numpy(float), true,
                                     features.tol_repeats(called["motif_size"].to_numpy(float) * np.round(true)))
    return set(called.loc[codes != features.OK, "locus_id"]), truth


def loci_table(locus_ids):
    """Returns the loci to measure, sorted by position: locus_id, chrom (with "chr"), start_0based, end, motif."""
    parts = pd.Series(sorted(locus_ids), dtype=object).str.split("-", expand=True)
    table = pd.DataFrame({"locus_id": sorted(locus_ids), "chrom": "chr" + parts[0],
                          "start_0based": parts[1].astype(int), "end": parts[2].astype(int), "motif": parts[3]})
    chrom_order = {"chr%s" % c: i for i, c in enumerate([str(i) for i in range(1, 23)] + ["X", "Y"])}
    return (table.assign(chrom_rank=table["chrom"].map(chrom_order))
            .sort_values(["chrom_rank", "start_0based", "end", "locus_id"]).drop(columns="chrom_rank")
            .reset_index(drop=True))


def loci_table_text(table):
    return table.to_csv(sep="\t", index=False, lineterminator="\n")


def measurement_key(loci_text):
    """Returns the hash naming one person's loci file and measurements: it covers the loci and the measuring script's
    code (``code_without_comments_and_docstrings``), so a changed list or code means new measurements, while editing
    the script's comments or docstrings keeps them."""
    with open(measure.__file__) as f:
        code = code_without_comments_and_docstrings(f.read())
    return hashlib.sha256(loci_text.encode() + b"\0" + code.encode()).hexdigest()[:12]


def people_and_their_sources(data_dir, heldout_data_dir):
    """Returns ``{truth sample id: [local per-source parquet paths]}`` for the 54 training sources and 81 held-out
    samples, keyed the way ``build_per_source_parquets_on_hail_batch.sources_to_build`` lists them (HG002's three
    coverages are one person)."""
    import build_per_source_parquets_on_hail_batch  # local: it imports heldout, which only this needs
    people = collections.OrderedDict()
    for source in build_per_source_parquets_on_hail_batch.sources_to_build(data_dir, heldout_data_dir):
        people.setdefault(source["sample"], []).append(source["local_parquet_path"])
    return people


def person_loci_and_truth(person, parquet_paths):
    """Returns ``(loci table, truth genotypes)`` for one person from all of the person's per-source parquets.

    Raises:
        RuntimeError: if a per-source parquet is missing, or two of them disagree on a locus's truth genotype (they
            are built from the same truth TSV).
    """
    missing = [p for p in parquet_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError("%s: per-source parquet(s) missing: %s" % (person, ", ".join(missing)))
    locus_ids, truths = set(), []
    for path in parquet_paths:
        disagreeing, truth = disagreeing_loci_and_truth_of_source(path)
        locus_ids |= disagreeing
        truths.append(truth)
    truth = truths[0]
    for other, path in zip(truths[1:], parquet_paths[1:]):
        shared = truth.index.intersection(other.index)
        if not truth.loc[shared].equals(other.loc[shared]):
            raise RuntimeError("%s: %s disagrees with %s on the truth genotype of some loci"
                               % (person, path, parquet_paths[0]))
        truth = pd.concat([truth, other.loc[other.index.difference(truth.index)]])
    return loci_table(locus_ids), truth


def judge_person(measurements, truth):
    """Judges every measured locus of one person against its truth genotype.

    Args:
        measurements: the person's ``measure_repeat_lengths_in_short_reads.py`` output.
        truth: ``person_loci_and_truth``'s truth genotypes (indexed by locus_id).

    Returns:
        ``measurements`` with truth_short, truth_long, motif_size and judgment columns added.
    """
    judged = measurements.join(truth, on="locus_id", how="left", validate="one_to_one")
    if judged["truth_short"].isna().any():
        raise RuntimeError("%d measured loci have no truth genotype" % judged["truth_short"].isna().sum())
    judgments = []
    for row in judged.itertuples():
        if str(row.has_unique_flanks_next_to_locus) != "True":
            judgments.append(FLANKS_NOT_UNIQUE)
            continue
        judgments.append(judge_truth_genotype_against_reads(
            parse_measured_reads(row.measured_reads, int(row.motif_size)), int(row.truth_short), int(row.truth_long),
            int(row.motif_size)))
    judged["judgment"] = judgments
    return judged


def parquet_check_record(path):
    """Returns the record ``--mark`` wrote into a parquet's metadata (a dict), or None if it was never marked."""
    import pyarrow.parquet as pq  # local: only the parquet-facing functions need it
    schema = pq.ParquetFile(path).schema_arrow
    if TRUTH_CONTRADICTED_COLUMN not in schema.names:
        return None
    value = (schema.metadata or {}).get(PARQUET_METADATA_KEY)
    return json.loads(value) if value else None


def write_truth_contradicted_column(parquet_path, contradicted_locus_ids, record):
    """Rewrites one per-source parquet with ``truth_contradicted_by_short_reads`` (True at ``contradicted_locus_ids``)
    and ``record`` in its metadata. Every other column is left as it was.

    Returns:
        The number of rows marked.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = os.path.realpath(parquet_path)  # a promoted sample's training path is a symlink to its held-out parquet
    table = pq.read_table(path)
    if TRUTH_CONTRADICTED_COLUMN in table.column_names:
        table = table.drop_columns([TRUTH_CONTRADICTED_COLUMN])
    marked = np.isin(table.column("locus_id").to_numpy(zero_copy_only=False), np.array(sorted(contradicted_locus_ids),
                                                                                         dtype=object))
    table = table.append_column(TRUTH_CONTRADICTED_COLUMN, pa.array(marked, type=pa.bool_()))
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           PARQUET_METADATA_KEY: json.dumps(record, sort_keys=True).encode()})
    pq.write_table(table, path + ".tmp")
    os.replace(path + ".tmp", path)
    return int(marked.sum())


def gsutil(*args):
    return subprocess.run(["gsutil"] + list(args), capture_output=True, text=True)


def mark_person(person, parquet_paths, local_dir, output_dir, rule):
    """Judges one person's measurements and marks the person's per-source parquets.

    Returns:
        A summary dict, with ``status`` "marked", "already marked" or "not measured" (no measurements on GCS for the
        current loci list: launch the jobs first).
    """
    table, truth = person_loci_and_truth(person, parquet_paths)
    key = measurement_key(loci_table_text(table))
    name = "%s.%s%s" % (person, key, MEASUREMENTS_SUFFIX)
    record = {"measurements": name, "judging_rule": rule}
    if all(parquet_check_record(p) == record for p in parquet_paths):
        return {"person": person, "status": "already marked", "n_loci": len(table)}
    local_path = os.path.join(local_dir, name)
    if not os.path.exists(local_path):
        result = gsutil("cp", "%s/%s" % (output_dir, name), local_path + ".tmp")
        if result.returncode != 0:
            if "No URLs matched" in result.stderr or "matched no objects" in result.stderr:
                return {"person": person, "status": "not measured", "n_loci": len(table)}
            raise RuntimeError("%s: could not download %s/%s: %s" % (person, output_dir, name,
                                                                     result.stderr.strip()[-1000:]))
        os.replace(local_path + ".tmp", local_path)
    measurements = pd.read_csv(local_path, sep="\t", keep_default_na=False, dtype={"measured_reads": str})
    if set(measurements["locus_id"]) != set(table["locus_id"]):
        raise RuntimeError("%s: %s does not cover exactly the person's %d loci" % (person, local_path, len(table)))
    judged = judge_person(measurements, truth)
    judged.to_csv(os.path.join(local_dir, "%s.%s.judgments.tsv.gz" % (person, key)), sep="\t", index=False)
    contradicted = set(judged.loc[judged["judgment"].isin(CONTRADICTING_JUDGMENTS), "locus_id"])
    marked_rows = {os.path.basename(p): write_truth_contradicted_column(p, contradicted, record) for p in parquet_paths}
    return {"person": person, "status": "marked", "n_loci": len(table),
            "judgments": judged["judgment"].value_counts().to_dict(), "n_contradicted": len(contradicted),
            "marked_rows": marked_rows}


def print_mark_summary(summaries):
    """Prints one line per person and the totals of ``mark_person``'s summaries."""
    totals = collections.Counter()
    for s in summaries:
        if s["status"] != "marked":
            print("  %-12s %-15s %d loci" % (s["person"], s["status"], s["n_loci"]))
            continue
        judged = sum(n for j, n in s["judgments"].items() if j not in (FLANKS_NOT_UNIQUE, TOO_FEW_READS))
        print("  %-12s marked: %d of %d loci judged, %d contradicted (%.1f%%); rows marked %s"
              % (s["person"], judged, s["n_loci"], s["n_contradicted"], 100 * s["n_contradicted"] / max(judged, 1),
                 ", ".join("%s %d" % kv for kv in s["marked_rows"].items())))
        totals.update(s["judgments"])
    if totals:
        print("\n  judgments over the people marked in this run:")
        for judgment, n in totals.most_common():
            print("    %-62s %d" % (judgment, n))


def launch(bp, people, args):
    """Submits one measuring job per person whose measurements for the current loci list don't exist yet (step_pipeline
    skips a job whose output exists)."""
    import hailtop.fs as hfs  # local: only the launcher needs these
    from step_pipeline import Localize
    reads = pd.read_csv(SHORT_READ_PATHS_TSV, sep="\t").set_index("sample_label")
    unknown = [p for p in people if p not in reads.index]
    if unknown:
        sys.exit("no reads path in %s for %s" % (SHORT_READ_PATHS_TSV, ", ".join(unknown)))
    bp.set_name("check_truth_genotypes_against_short_reads: %d people -> %s" % (len(people), args.output_dir))
    bp.precache_file_paths(os.path.join(args.output_dir, "*" + MEASUREMENTS_SUFFIX))
    with open(measure.__file__, "rb") as f:
        script = f.read()
    script_path = "%s/code/measure_repeat_lengths_in_short_reads.%s.py" % (
        args.output_dir, hashlib.sha256(script).hexdigest()[:12])
    if not args.dry_run and not hfs.exists(script_path):
        with hfs.open(script_path, "wb") as f:
            f.write(script)
    for person, parquet_paths in people.items():
        table, _ = person_loci_and_truth(person, parquet_paths)
        text = loci_table_text(table)
        key = measurement_key(text)
        loci_path = "%s/loci/%s.%s.loci.tsv.gz" % (args.output_dir, person, key)
        print("%s: %d loci to measure (%s)" % (person, len(table), key), flush=True)
        if not args.dry_run and not hfs.exists(loci_path):
            with hfs.open(loci_path, "wb") as f:
                f.write(gzip.compress(text.encode(), mtime=0))
        step = bp.new_step("Measure repeat lengths in reads: %s" % person, arg_suffix="measure-step", step_number=1,
                           image=IMAGE, cpu=args.cpu, memory=args.memory, storage=args.storage,
                           localize_by=Localize.COPY, output_dir=args.output_dir)
        local_script = step.input(script_path).local_path
        local_loci = step.input(loci_path).local_path
        local_reads = step.input(reads.at[person, "reads_path"]).local_path
        step.input(reads.at[person, "reads_index_path"])
        local_fasta = step.input(REFERENCE_FASTA).local_path
        step.input(REFERENCE_FASTA + ".fai")
        out_path = os.path.join(JOB_DIR, "%s.%s%s" % (person, key, MEASUREMENTS_SUFFIX))
        step.command("set -euxo pipefail")
        step.command("cd %s" % JOB_DIR)
        step.command("python3 %s write-regions-bed %s regions.bed" % (local_script, local_loci))
        step.command("samtools view -M -F %s -T %s -L regions.bed -@ %d %s | python3 %s measure %s %s %s"
                     % (SAM_FLAGS_TO_EXCLUDE, local_fasta, max(1, int(args.cpu)), local_reads, local_script,
                        local_loci, local_fasta, out_path))
        step.output(out_path)
    bp.run()


def add_arguments(parser):
    """Adds this script's options to ``parser``: a plain one for --mark, step_pipeline's for launching jobs (whose
    ``run`` parses the command line again and rejects options it doesn't know)."""
    parser.add_argument("--mark", action="store_true",
                        help="judge the measurements and mark the per-source parquets instead of launching jobs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="cloud folder for the loci and measurements")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"),
                        help="dataset.py's --data-dir (training combos' parquets; local measurements go under it)")
    parser.add_argument("--heldout-data-dir", default=os.path.join(HERE, "data_eval_43"),
                        help="heldout.py's --data-dir (promoted and held-out samples' parquets)")
    parser.add_argument("-i", "--person", action="append",
                        help="only this person (truth sample id, e.g. HG002 or HG00738); repeatable")
    parser.add_argument("--training-people-only", action="store_true",
                        help="only the people training uses (dataset.COMBOS and dataset.PROMOTED_HELDOUT_SAMPLES), not "
                             "the held-out samples")
    parser.add_argument("--cpu", type=float, default=2)
    parser.add_argument("--memory", default="standard")
    parser.add_argument("--storage", default="60Gi", help="job disk: the CRAM (up to ~25 GB) plus the 3 GB reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args, unknown = parser.parse_known_args()
    if args.mark and unknown:  # launching leaves the rest to step_pipeline's parser
        parser.error("unrecognized arguments: %s" % " ".join(unknown))
    bp = None
    if not args.mark:  # only launching needs step_pipeline and Hail
        from step_pipeline import pipeline, Backend
        bp = pipeline("check_truth_genotypes_against_short_reads", backend=Backend.HAIL_BATCH_SERVICE,
                      config_file_path="~/.step_pipeline")
        add_arguments(bp.get_config_arg_parser())
        args = bp.parse_known_args()

    people = people_and_their_sources(args.data_dir, args.heldout_data_dir)
    if args.person:
        unknown = sorted(set(args.person) - set(people))
        if unknown:
            parser.error("unknown person(s): %s" % ", ".join(unknown))
        people = collections.OrderedDict((p, people[p]) for p in args.person)
    if args.training_people_only:
        training_people = {sample for sample, _, _ in dataset.COMBOS} | set(dataset.PROMOTED_HELDOUT_SAMPLES)
        people = collections.OrderedDict((p, paths) for p, paths in people.items() if p in training_people)
    # A person is checked at the union of the loci of all of their sources, so one missing source skips the person
    # (dataset.assemble_branch then refuses to train on an unmarked source).
    unbuilt = [p for p, paths in people.items() if not all(os.path.exists(path) for path in paths)]
    if unbuilt:
        print("skipping %d people whose per-source parquets are not all built yet: %s" % (len(unbuilt), ", ".join(unbuilt)))
        people = collections.OrderedDict((p, paths) for p, paths in people.items() if p not in unbuilt)
    if not args.mark:
        launch(bp, people, args)
        return

    local_dir = os.path.join(args.data_dir, LOCAL_SUBDIR)
    os.makedirs(local_dir, exist_ok=True)
    rule = judging_rule_fingerprint()
    summaries = []
    for person, parquet_paths in people.items():
        summaries.append(mark_person(person, parquet_paths, local_dir, args.output_dir, rule))
        print_mark_summary(summaries[-1:])
    print("\n==== summary (judging rule %s) ====" % rule)
    print_mark_summary(summaries)
    not_measured = [s["person"] for s in summaries if s["status"] == "not measured"]
    if not_measured:
        sys.exit("%d people have no measurements for their current loci (%s); launch their jobs first: python3 %s "
                 "--no-wait %s" % (len(not_measured), ", ".join(not_measured), os.path.basename(__file__),
                                   " ".join("-i %s" % p for p in not_measured)))


if __name__ == "__main__":
    main()
