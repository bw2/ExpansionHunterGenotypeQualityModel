"""Builds the per-source parquets that ExpansionHunter-bw2 5656d6a would give, without re-genotyping, then assembles
the training tables from them.

5656d6a stops the fast path's in-repeat hand-off (85bd281) at homopolymers, where poly-G artifact reads made the full
genotyper call false 200-300 repeat alleles. For homopolymers that restores the routing of EH b1fbc23. So, per source,
every homopolymer locus that the b1fbc23 run genotyped on the fast path and the e90307c run did not gets the rows
today's extraction gives from the b1fbc23 JSON
(``build_per_source_parquets_on_hail_batch.py --eh-results-root .../expansion_hunter_v2.1``). Every other row is kept.

A locus's genotyping branch in a run is that of its rows in the run's per-source parquet. Every variant locus is in
both parquets. A hom-ref locus is in a parquet only if it was called and its keep rate let it in
(``dataset.HOM_REF_KEEP_RATE_BY_CALLED_SIZE_BIN``: 1 on the full genotyper, from 0.02 up on the fast path, drawn
with ``dataset._locus_hash_fraction``). So:

- a homopolymer locus on the fast path in b1fbc23 and on the full genotyper in e90307c gets its b1fbc23 rows in place
  of its e90307c rows;
- a homopolymer hom-ref locus on the fast path in b1fbc23 and missing from the e90307c parquet (its full genotyper
  left it uncalled, so the sampling left it out) gets its b1fbc23 rows, after all the others, in the b1fbc23 order;
- a homopolymer hom-ref locus on the full genotyper in e90307c and missing from the b1fbc23 parquet was a b1fbc23
  fast-path call that the sampling left out, and gets no rows. It could also be a locus b1fbc23 left uncalled, which
  would keep its e90307c rows, but that can be told apart only when the locus's hash fraction is below the lowest
  fast-path keep rate (a fast-path call is kept there at any size). The summary counts such loci; there were none in
  HG002_10x and HG03492, so the case is treated as absent and a warning is printed if it occurs.

Checks, per source (in its summary, and printed): the rows of homopolymer loci that both runs genotyped on the fast
path must be identical in every column, which is the premise that the b1fbc23 rows are what 5656d6a gives; the rows of
the replaced variant loci must equal those of the old b1fbc23 table (``data/real_quick.eh_b1fbc23``,
``data_eval_43/real_43.eh_b1fbc23``: the same JSON extracted by the code of that time, variant loci only) in the
columns both have; and every homopolymer variant locus must be in both parquets.

Outputs, in folders of their own (the current tables are only read): ``<out-heldout-data-dir>/real_43/<sample>.parquet``
for the 131 promoted and held-out samples; ``<out-data-dir>/real_quick/<combo>.parquet`` for the 4 training combos
plus symlinks to the 50 promoted samples (the layout dataset.py uses); a ``<parquet>.homopolymer_handoff_fix.json``
summary next to each parquet; then ``<out-data-dir>/parquet/`` and ``parquet_all_loci/`` assembled by
``dataset.assemble_branch``. A source whose summary records the same input files (path, size, mtime) and the same
code (sha256 of this script, dataset.py and eh_json.py) is not rebuilt.
Each output parquet also gets the record dataset.py and heldout.py keep next to a per-source parquet
(``<parquet>.sources.json``): the current parquet's, which names the e90307c sources and EH build, plus, under
``homopolymer_handoff_fix``, the record of the b1fbc23 extraction its replaced loci come from.

These folders are inputs for ``train.py --data-dir`` and the evaluation scripts. Don't run dataset.py or heldout.py
builds on them: those would rebuild their parquets as plain e90307c tables (today they stop first, because the
e90307c build no longer counts as current against ExpansionHunter-bw2 HEAD).

Usage (the assembly leaves out the truth labels each person's own short reads contradict, so the outputs are marked
before it, as for dataset.py):
    python3 build_per_source_parquets_with_homopolymer_handoff_fix.py --skip-assembly
    python3 check_truth_genotypes_against_short_reads.py --no-wait --data-dir <out-data-dir> \\
        --heldout-data-dir <out-heldout-data-dir>
    python3 check_truth_genotypes_against_short_reads.py --mark --data-dir <out-data-dir> \\
        --heldout-data-dir <out-heldout-data-dir>
    python3 build_per_source_parquets_with_homopolymer_handoff_fix.py

Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

import build_per_source_parquets_on_hail_batch as per_source_builder
import dataset
import eh_json

HERE = os.path.dirname(os.path.abspath(__file__))
LOWEST_FAST_PATH_KEEP_RATE = min(dataset.HOM_REF_KEEP_RATE_BY_CALLED_SIZE_BIN[eh_json.BRANCH_QUICK])
SUMMARY_SUFFIX = ".homopolymer_handoff_fix.json"


def branch_of_each_locus(df):
    """Returns a Series locus_id -> genotyping branch.

    Raises:
        ValueError: if a locus's rows disagree on the branch.
    """
    branches = df.groupby("locus_id", sort=False)["genotyping_branch"]
    mixed = branches.nunique()
    if (mixed > 1).any():
        raise ValueError("loci with rows on both branches: %s" % ", ".join(mixed[mixed > 1].index[:5]))
    return branches.first()


COUNT_UNCALLED_IN_B1FBC23 = ("homopolymer hom-ref loci dropped although their hash fraction is below the lowest "
                             "fast-path keep rate, so b1fbc23 left them uncalled (expected 0)")


def homopolymer_loci_to_replace(current_df, old_df, individual):
    """Returns ``(locus ids to replace, counts)``: the homopolymer loci whose rows come from the old parquet, or that
    get none (see the module docstring), and counts of every case for the summary.

    Raises:
        ValueError: if a homopolymer variant locus is missing from either parquet.
    """
    def per_locus(df):
        branch_of_each_locus(df)  # raises if a locus is on both branches
        return df.groupby("locus_id", sort=False).agg(branch=("genotyping_branch", "first"),
                                                     motif_size=("motif_size", "first"),
                                                     variant_locus=("variant_locus", "first"))
    # ANALYSIS_OK[join]: outer on purpose; a locus missing from one parquet gets NaN columns from it, which the cases
    # below test for.
    loci = per_locus(current_df).join(per_locus(old_df), how="outer", lsuffix="_current", rsuffix="_old")
    homopolymer = (loci["motif_size_current"].eq(1) | loci["motif_size_old"].eq(1)).to_numpy()
    variant = (loci["variant_locus_current"].eq(True) | loci["variant_locus_old"].eq(True)).to_numpy()
    current_quick = loci["branch_current"].eq(eh_json.BRANCH_QUICK).to_numpy()
    current_full = loci["branch_current"].eq(eh_json.BRANCH_FULL).to_numpy()
    current_missing = loci["branch_current"].isna().to_numpy()
    old_quick = loci["branch_old"].eq(eh_json.BRANCH_QUICK).to_numpy()
    old_full = loci["branch_old"].eq(eh_json.BRANCH_FULL).to_numpy()
    old_missing = loci["branch_old"].isna().to_numpy()
    missing_variant = homopolymer & variant & (current_missing | old_missing)
    if missing_variant.any():
        raise ValueError("homopolymer variant loci missing from a parquet: %s"
                         % ", ".join(loci.index[missing_variant][:5]))
    from_old = homopolymer & old_quick & ~current_quick
    drop = homopolymer & current_full & old_missing
    hash_below_lowest_rate = np.array([dataset._locus_hash_fraction(locus_id, individual) < LOWEST_FAST_PATH_KEEP_RATE
                                       for locus_id in loci.index[drop]], bool)
    counts = {
        "homopolymer variant loci replaced by their b1fbc23 fast-path rows": int((from_old & variant).sum()),
        "homopolymer hom-ref loci replaced by their b1fbc23 fast-path rows": int((from_old & ~variant & current_full).sum()),
        "homopolymer hom-ref loci added from b1fbc23 (left out of e90307c, whose full genotyper left them uncalled)":
            int((from_old & ~variant & current_missing).sum()),
        "homopolymer hom-ref loci dropped (b1fbc23 fast-path calls the sampling left out)": int(drop.sum()),
        COUNT_UNCALLED_IN_B1FBC23: int(hash_below_lowest_rate.sum()),
        "homopolymer loci on the fast path in e90307c and the full genotyper in b1fbc23 (kept as they are)":
            int((homopolymer & current_quick & old_full).sum()),
        "other loci whose branch differs between the runs (kept as they are)":
            int((~homopolymer & ~current_missing & ~old_missing
                 & (loci["branch_current"] != loci["branch_old"]).to_numpy()).sum()),
    }
    return set(loci.index[from_old | drop]), counts


def rows_with_replaced_loci(current_df, old_df, locus_ids_to_replace):
    """Returns the current rows with every locus in ``locus_ids_to_replace`` replaced by its old rows (none, for a
    locus missing from the old parquet): a replaced locus's rows take the place of the rows they replace, and the
    rows of a locus missing from the current parquet go after all the others, in the old order."""
    current_locus = current_df["locus_id"].to_numpy()
    replaced = current_df["locus_id"].isin(locus_ids_to_replace).to_numpy()
    from_old = old_df["locus_id"].isin(locus_ids_to_replace).to_numpy()
    first_row_of_locus = pd.Series(np.arange(len(current_df))).groupby(current_locus).min()
    combined = pd.concat([current_df[~replaced], old_df[from_old]], ignore_index=True)
    row_within_its_source = np.concatenate([np.flatnonzero(~replaced), np.flatnonzero(from_old)])
    locus_position = combined["locus_id"].map(first_row_of_locus).to_numpy(dtype=float)
    # ANALYSIS_OK[imputation]: a locus missing from the current parquet has no position there, so it goes after all.
    locus_position[np.isnan(locus_position)] = len(current_df)
    order = np.lexsort((row_within_its_source, locus_position))
    return combined.iloc[order].reset_index(drop=True)


def _rows_that_differ(left, right):
    """Returns ``(rows compared, rows that differ, {column: rows that differ in it})`` for the rows of two tables
    matched by (locus_id, allele_rank), over the columns both have; NaN equals NaN."""
    key = ["locus_id", "allele_rank"]
    both = left.merge(right, on=key, suffixes=("_left", "_right"), validate="one_to_one")
    differs_by_column = {}
    any_difference = np.zeros(len(both), bool)
    for column in left.columns:
        if column in key or column not in right.columns:
            continue
        a, b = both[column + "_left"], both[column + "_right"]
        differs = ~((a == b) | (a.isna() & b.isna())).to_numpy(bool)
        if differs.any():
            differs_by_column[column] = int(differs.sum())
        any_difference |= differs
    return len(both), int(any_difference.sum()), differs_by_column


def differences_at_homopolymer_loci_on_the_fast_path_in_both(current_df, old_df):
    """``_rows_that_differ`` for the rows of homopolymer loci that both runs genotyped on the fast path."""
    def fast_path_homopolymer_rows(df):
        return df[(df["motif_size"] == 1) & (df["genotyping_branch"] == eh_json.BRANCH_QUICK)]
    return _rows_that_differ(fast_path_homopolymer_rows(current_df), fast_path_homopolymer_rows(old_df))


def differences_from_the_b1fbc23_tables(out_df, locus_ids, b1fbc23_table_path):
    """``_rows_that_differ`` for the output rows of ``locus_ids`` (replaced variant loci) against the old b1fbc23 table
    of the same source (the b1fbc23 JSON extracted by the code of that time; variant loci only, fewer columns)."""
    old_table = pd.read_parquet(b1fbc23_table_path)
    return _rows_that_differ(out_df[out_df["locus_id"].isin(locus_ids)], old_table[old_table["locus_id"].isin(locus_ids)])


def _file_signature(path):
    return {"path": os.path.realpath(path), "size": os.path.getsize(path), "mtime": os.path.getmtime(path)}


def _code_signature():
    """sha256 of the code that decides the outputs: this script and the modules whose functions it calls."""
    signature = {}
    for name in (os.path.basename(__file__), "dataset.py", "eh_json.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            signature[name] = hashlib.sha256(f.read()).hexdigest()
    return signature


def build_one_source(current_path, old_path, out_path, individual, b1fbc23_table_path):
    """Writes ``out_path`` and its summary from one source's current and old per-source parquets, unless the summary
    already records the same inputs; ``b1fbc23_table_path`` is the source's old b1fbc23 table, which the replaced
    variant loci are checked against. Returns the summary."""
    inputs = {"current": _file_signature(current_path), "old (today's extraction of the b1fbc23 JSON)": _file_signature(old_path),
              "b1fbc23 table": _file_signature(b1fbc23_table_path), "code": _code_signature()}
    summary_path = out_path + SUMMARY_SUFFIX
    if os.path.exists(out_path) and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if summary["inputs"] == inputs:
            return summary
    current_df, old_df = pd.read_parquet(current_path), pd.read_parquet(old_path)
    # The current parquets may carry the short-read truth check (check_truth_genotypes_against_short_reads.py --mark).
    # Its loci depend on the calls, which the substitution changes, so the parquets built here are marked on their own.
    current_df = current_df.drop(columns=[dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN], errors="ignore")
    if list(current_df.columns) != list(old_df.columns) or not current_df.dtypes.equals(old_df.dtypes):
        raise ValueError("%s and %s have different columns or column types" % (current_path, old_path))
    locus_ids_to_replace, counts = homopolymer_loci_to_replace(current_df, old_df, individual)
    out_df = rows_with_replaced_loci(current_df, old_df, locus_ids_to_replace)
    compared, differing, differing_by_column = differences_at_homopolymer_loci_on_the_fast_path_in_both(current_df, old_df)
    replaced_variant_locus_ids = set(old_df.loc[old_df["locus_id"].isin(locus_ids_to_replace)
                                                & old_df["variant_locus"].to_numpy(bool), "locus_id"])
    compared_with_old_table, differing_from_old_table, differing_from_old_table_by_column = \
        differences_from_the_b1fbc23_tables(out_df, replaced_variant_locus_ids, b1fbc23_table_path)
    summary = {"inputs": inputs, "rows": {"current": len(current_df), "out": len(out_df)}, "loci": counts,
               "homopolymer fast-path rows of both runs compared": compared,
               "homopolymer fast-path rows of both runs that differ": differing,
               "homopolymer fast-path rows of both runs that differ, by column": differing_by_column,
               "replaced variant-locus rows compared with the b1fbc23 table": compared_with_old_table,
               "replaced variant-locus rows that differ from the b1fbc23 table": differing_from_old_table,
               "replaced variant-locus rows that differ from the b1fbc23 table, by column":
                   differing_from_old_table_by_column}
    with open(dataset._sources_record_path(current_path)) as f:
        record = json.load(f)
    with open(dataset._sources_record_path(old_path)) as f:
        record["homopolymer_handoff_fix"] = {
            "what": "the e90307c rows, except the replaced homopolymer loci, whose rows come from today's extraction "
                    "of the b1fbc23 JSON (build_per_source_parquets_with_homopolymer_handoff_fix.py)",
            "record of the b1fbc23 extraction": json.load(f)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_parquet(out_path + ".tmp", index=False)
    os.replace(out_path + ".tmp", out_path)
    with open(dataset._sources_record_path(out_path), "w") as f:
        json.dump(record, f, indent=1)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current-data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--current-heldout-data-dir", default=os.path.join(HERE, "data_eval_43"))
    parser.add_argument("--old-data-dir", default=os.path.join(HERE, "data.eh_b1fbc23_jsons_extracted_with_todays_code"))
    parser.add_argument("--old-heldout-data-dir",
                        default=os.path.join(HERE, "data_eval_43.eh_b1fbc23_jsons_extracted_with_todays_code"))
    parser.add_argument("--out-data-dir", default=os.path.join(HERE, "data.eh_e90307c_with_homopolymer_handoff_fix"))
    parser.add_argument("--out-heldout-data-dir",
                        default=os.path.join(HERE, "data_eval_43.eh_e90307c_with_homopolymer_handoff_fix"))
    parser.add_argument("--b1fbc23-training-combos-dir", default=os.path.join(HERE, "data", "real_quick.eh_b1fbc23"),
                        help="the old b1fbc23 tables of the 4 training combos, which replaced variant loci are checked "
                             "against (default: %(default)s)")
    parser.add_argument("--b1fbc23-samples-dir", default=os.path.join(HERE, "data_eval_43", "real_43.eh_b1fbc23"),
                        help="the old b1fbc23 tables of the 131 promoted and held-out samples (default: %(default)s)")
    parser.add_argument("--skip-assembly", action="store_true", help="build the per-source parquets only")
    parser.add_argument("--keep-truth-labels-contradicted-by-short-reads", action="store_true",
                        help="assemble with every truth label, as dataset.py's flag of the same name; by default the "
                             "output parquets must have been marked by check_truth_genotypes_against_short_reads.py "
                             "--mark")
    args = parser.parse_args()
    # Absolute, so the promoted samples' symlinks below point at their parquets from any folder.
    for name in ("current_data_dir", "current_heldout_data_dir", "old_data_dir", "old_heldout_data_dir",
                 "out_data_dir", "out_heldout_data_dir"):
        setattr(args, name, os.path.abspath(getattr(args, name)))

    current = per_source_builder.sources_to_build(args.current_data_dir, args.current_heldout_data_dir)
    old = per_source_builder.sources_to_build(args.old_data_dir, args.old_heldout_data_dir)
    out = per_source_builder.sources_to_build(args.out_data_dir, args.out_heldout_data_dir)
    training_combos = {"%s_%s" % (sample, cov_label) for sample, cov_label, _ in dataset.COMBOS}
    totals = {}
    for current_source, old_source, out_source in zip(current, old, out):
        b1fbc23_table_path = os.path.join(
            args.b1fbc23_training_combos_dir if current_source["name"] in training_combos else args.b1fbc23_samples_dir,
            current_source["name"] + ".parquet")
        summary = build_one_source(current_source["local_parquet_path"], old_source["local_parquet_path"],
                                   out_source["local_parquet_path"],
                                   dataset.individual_of_part(current_source["local_parquet_path"]), b1fbc23_table_path)
        for name, count in summary["loci"].items():
            totals[name] = totals.get(name, 0) + count
        # The loci counts in the order of the totals printed at the end.
        print("%-16s rows %d -> %d; loci %s; rows that differ: fast-path homopolymer rows of both runs %d of %d %s, "
              "replaced variant-locus rows vs the b1fbc23 table %d of %d %s" % (
                  current_source["name"], summary["rows"]["current"], summary["rows"]["out"],
                  "/".join(str(count) for count in summary["loci"].values()),
                  summary["homopolymer fast-path rows of both runs that differ"],
                  summary["homopolymer fast-path rows of both runs compared"],
                  summary["homopolymer fast-path rows of both runs that differ, by column"] or "",
                  summary["replaced variant-locus rows that differ from the b1fbc23 table"],
                  summary["replaced variant-locus rows compared with the b1fbc23 table"],
                  summary["replaced variant-locus rows that differ from the b1fbc23 table, by column"] or ""),
              flush=True)
        if (summary["homopolymer fast-path rows of both runs that differ"] or summary["loci"][COUNT_UNCALLED_IN_B1FBC23]
                or summary["replaced variant-locus rows that differ from the b1fbc23 table"]):
            print("WARNING: %s breaks a premise of this script (see the module docstring)" % current_source["name"])
    print("\ntotals over %d sources:" % len(current))
    for name, count in totals.items():
        print("  %-110s %d" % (name, count))

    # The training pool: dataset.py links the promoted samples into real_quick/ (dataset._link_promoted_heldout_samples).
    for out_source in out:
        if out_source["name"] in dataset.PROMOTED_HELDOUT_SAMPLES:
            link = os.path.join(args.out_data_dir, dataset.SOURCE_SUBDIR, out_source["name"] + ".parquet")
            if not os.path.lexists(link):
                os.symlink(out_source["local_parquet_path"], link)
    if args.skip_assembly:
        return
    # As in dataset.py, the assembly leaves out the rows whose truth the person's own short reads contradict, and needs
    # the parquets built here marked first: run with --skip-assembly, then check_truth_genotypes_against_short_reads.py
    # (its jobs, then --mark) with --data-dir and --heldout-data-dir set to the output folders, then run this again.
    for training_population in dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION:
        for branch in (eh_json.BRANCH_QUICK, eh_json.BRANCH_FULL):
            dataset.assemble_branch(args.out_data_dir, branch, dataset.SOURCE_SUBDIR, training_population,
                                    drop_truth_contradicted_by_short_reads=not args.keep_truth_labels_contradicted_by_short_reads)


if __name__ == "__main__":
    main()
