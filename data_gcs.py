"""Build FULL-BRANCH REAL per-allele training rows from GCS ``EHv5`` outputs.

The ``EHv5`` variant is the low-mem-streaming full branch: every record is
full-genotyped (no ``QuickGenotype`` fast path), so every extracted row is a
``full``-contract row. For each ``(sample, coverage)`` combo this module:

  1. Downloads the full-branch JSON shards and the truth ``.alleles.tsv.gz`` from
     ``gs://str-truth-set-v2/tool_results/...`` (idempotent -- existing local
     files are not re-downloaded).
  2. Extracts per-allele feature rows from the JSON via
     ``eh_json_features.extract_rows`` -- ``eh`` and EVERY feature come from the
     JSON, never the TSV (SPEC sec 3).
  3. Joins the truth TSV on ``(LocusId, allele_rank)`` using size-sorted
     rank-pairing (SPEC "Real join key") to attach only ``true`` / ``purity`` /
     the negative-control flag / audit columns.
  4. Writes one parquet per combo to ``data/real_full/{sample}_{cov}.parquet``.

The added ``coverage`` column is the NOMINAL run coverage parsed from the GCS
path (10 / 20 / 31 / 46), overwriting the extractor's per-locus ``Coverage``:
``splits.cross_coverage_splits`` compares ``coverage == 10/20/31`` exactly, so it
requires the nominal value, not EH's measured per-locus depth (that local depth
is still available as the ``depth`` feature).

Determinism: stable size sort for rank assignment; no randomness. ``q`` / ``t`` /
``direction`` / row filtering are intentionally NOT computed here -- that is
``build_dataset.py``'s job (SPEC "Data flow"). Coding rules: no type hints,
Google-style docstrings, ``print()`` not logging, ``python3`` / ``gcloud`` (macOS).
"""

import argparse
import os
import subprocess

import pandas as pd

import eh_json_features as F


GCS_ROOT = "gs://str-truth-set-v2/tool_results"
VARIANT = "EHv5"  # low-mem-streaming full branch

# (sample, data_type, coverage-dir label) for the full-branch real TRAINING combos -- illumina only.
# element + illumina_exome were evaluated (reviewer_deepdive/domain_cv.py, 5-fold) and EXCLUDED:
# blending them regresses held-out WGS in both dominant regimes (full_spanning +both -0.175+-0.136,
# bimodal/unstable; fast -0.018+-0.002), while the illumina-trained model already GENERALIZES to them
# (LODO exome dist_red +0.186 full_spanning / +0.075 fast) at zero WGS cost. The data_type plumbing is
# kept (path component + parquet column) so the combos can be re-added behind a fix (e.g. an observable
# coverage-profile feature); to re-add, append the triples below and rebuild. The per-combo parquet name
# is ``{sample}_{cov}`` -- keep every (sample, cov_label) pair distinct to avoid clobbering.
COMBOS = [
    ("HG002", "illumina", "10x"),
    ("HG002", "illumina", "20x"),
    ("HG002", "illumina", "31x"),
    ("CHM1_CHM13", "illumina", "46x"),
]


def combo_base(sample, data_type, cov_label):
    """Returns the GCS directory for one ``(sample, data_type, coverage)`` combo."""
    return "%s/%s/%s/%s/%s_coverage/" % (GCS_ROOT, sample, data_type, VARIANT, cov_label)


def truth_tsv_remote(sample, data_type, cov_label):
    """Returns the GCS path of the per-allele truth TSV for one combo."""
    return "%s%s.tandem_repeat_genotypes.for_comparison.with_%s_vs_Truth_columns.alleles.tsv.gz" % (
        combo_base(sample, data_type, cov_label), sample, VARIANT)


def parse_nominal_coverage(cov_label):
    """Returns the nominal coverage (as float) parsed from a coverage-dir label like "10x" or "24G".

    WGS/exome/element labels are genome depth ("10x" -> 10.0). RNA-seq labels are total bases
    sequenced in Gbp ("24G" -> 24.0), since genome-wide depth is meaningless for transcript data;
    the value is only used for coverage stratification, not as a model feature.
    """
    return float(int(cov_label.rstrip("xG")))


def list_json_inputs(sample, data_type, cov_label):
    """Lists the JSON shard paths to use for one combo.

    Prefers a single combined ``*.json`` (no ``.shard`` suffix) if one is
    present; otherwise returns the sorted list of ``.shardNNN_of_NNN.json``
    shards. Using one or the other (never both) avoids double-counting loci.

    Args:
        sample: Sample name (e.g. ``"HG002"``).
        data_type: Sequencing data type (e.g. ``"illumina"``, ``"element"``).
        cov_label: Coverage-dir label (e.g. ``"10x"``).

    Returns:
        A sorted list of ``gs://`` JSON paths.
    """
    listing = subprocess.run(
        ["gsutil", "ls", combo_base(sample, data_type, cov_label) + "json/"],
        capture_output=True, text=True, check=True).stdout.split()
    jsons = [p for p in listing if p.endswith(".json")]
    combined = sorted(p for p in jsons if ".shard" not in os.path.basename(p))
    return combined if combined else sorted(p for p in jsons if ".shard" in os.path.basename(p))


def download(remote_paths, dest_dir):
    """Downloads ``remote_paths`` into ``dest_dir`` (skips files already present).

    Uses a multi-file ``gcloud storage cp`` (fast -- it parallelizes streams)
    and retries the whole batch on a non-zero exit. Each retry only re-fetches
    the files still missing, so a transient per-task ``FileNotFoundError`` /
    ``HashMismatchError`` (gcloud leaves no final file on a bad hash) recovers
    cleanly. NOTE: never run two builds of the same combo at once -- concurrent
    writers race on gcloud's ``.gstmp`` temp files and corrupt the download.

    Args:
        remote_paths: List of ``gs://`` source paths.
        dest_dir: Local destination directory (created if needed).

    Returns:
        The list of local paths (one per ``remote_paths`` entry).
    """
    os.makedirs(dest_dir, exist_ok=True)
    locals_ = [os.path.join(dest_dir, os.path.basename(p)) for p in remote_paths]
    for attempt in (1, 2, 3, 4):
        missing = [r for r, l in zip(remote_paths, locals_) if not os.path.exists(l)]
        if not missing:
            break
        print("    attempt %d: downloading %d/%d file(s) -> %s"
              % (attempt, len(missing), len(locals_), dest_dir))
        subprocess.run(["gcloud", "storage", "cp"] + missing + [dest_dir + os.sep])
    still_missing = [r for r, l in zip(remote_paths, locals_) if not os.path.exists(l)]
    if still_missing:
        raise RuntimeError("failed to download after 4 attempts: %s" % still_missing)
    return locals_


def load_json_rows(json_paths, sample_id):
    """Extracts per-allele rows from the combo's JSON shards into a DataFrame.

    Args:
        json_paths: Local paths to the combo's JSON shard(s).
        sample_id: ``sample_id`` to stamp on every extracted row.

    Returns:
        A DataFrame of raw extractor rows (one per allele).
    """
    rows = []
    for path in json_paths:
        before = len(rows)
        rows.extend(F.extract_rows(path, sample_id=sample_id))
        print("      %s -> %d rows" % (os.path.basename(path), len(rows) - before))
    return pd.DataFrame(rows)


def load_truth_tsv(tsv_path, variant=VARIANT):
    """Loads the per-allele truth TSV and synthesizes ``allele_rank``.

    The TSV is one row per ``(LocusId, allele)`` with no rank/sample/coverage
    columns. Within each ``LocusId`` the allele rows are sorted by
    ``(EHv5 size, Truth size)`` and assigned ``allele_rank`` 0, 1, ... -- the
    same size ordering ``eh_json_features`` uses on the JSON side, so equal keys
    pair correctly (a homozygous EH call's two rows tie harmlessly). Only the
    truth/purity/negative-flag/audit columns are returned; ``eh`` and all
    features come from the JSON.

    Args:
        tsv_path: Local path to the gzipped ``.alleles.tsv.gz``.
        variant: Tool-call variant name templated into the ``NumRepeats: Allele:
            <variant>`` / ``Allele: Concordance: <variant> vs Truth`` column
            headers (``"EHv5"`` for the full branch, ``"EHv5-bw2-optimized"`` for
            the fast branch). Truth/purity are tool-independent.

    Returns:
        A DataFrame keyed by ``(LocusId, allele_rank)`` with columns
        ``true``, ``purity``, ``is_negative_locus``, ``tsv_eh``, ``concordance``.

    Raises:
        ValueError: If ``TruthSetOrNegativeLocus`` holds any token other than the
            expected ``"TruthSet"`` / ``"NegativeLocus"`` (a stray blank / case /
            whitespace variant would otherwise be silently treated as a negative
            control and the locus dropped from the training set).
    """
    cols = {
        "LocusId": "LocusId",
        "NumRepeats: Allele: Truth": "true",
        "RepeatPurity: Allele: Truth": "purity",
        "NumRepeats: Allele: %s" % variant: "tsv_eh",
        "TruthSetOrNegativeLocus": "neg_flag",
        "Allele: Concordance: %s vs Truth" % variant: "concordance",
    }
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip",
                     usecols=list(cols), dtype={"LocusId": str}).rename(columns=cols)
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["purity"] = pd.to_numeric(df["purity"], errors="coerce")
    df["tsv_eh"] = pd.to_numeric(df["tsv_eh"], errors="coerce")
    flag = df["neg_flag"].astype(str).str.strip()
    unexpected = sorted(set(flag.unique()) - {"TruthSet", "NegativeLocus"})
    if unexpected:
        raise ValueError(
            "TruthSetOrNegativeLocus in %s has unexpected token(s) %s; expected only "
            "['NegativeLocus', 'TruthSet']" % (tsv_path, unexpected))
    # Compare against the NEGATIVE token (the drop side): an unvalidated/unknown value then
    # defaults to NOT-negative (kept) rather than silently dropping a real truth-set locus.
    df["is_negative_locus"] = flag == "NegativeLocus"
    # Rank-pairing: stable sort by (EHv5 size, Truth size) within each LocusId.
    df = df.sort_values(["LocusId", "tsv_eh", "true"], kind="mergesort", na_position="last")
    df["allele_rank"] = df.groupby("LocusId", sort=False).cumcount()
    return df[["LocusId", "allele_rank", "true", "purity",
               "is_negative_locus", "tsv_eh", "concordance"]]


def join_truth(json_df, tsv_df):
    """Left-joins truth onto the JSON rows by ``(LocusId, allele_rank)``.

    Asserts the join key is unique on both sides first (SPEC "Real join key").
    Unmatched JSON rows are kept with NaN truth columns.

    Args:
        json_df: Extractor rows (``locus_id`` + ``allele_rank``).
        tsv_df: Truth rows from ``load_truth_tsv`` (``LocusId`` + ``allele_rank``).

    Returns:
        A ``(merged_df, n_matched)`` tuple. ``n_matched`` is the number of JSON
        rows that found a truth row.
    """
    assert not json_df.duplicated(["locus_id", "allele_rank"]).any(), \
        "JSON (LocusId, allele_rank) key is not unique"
    assert not tsv_df.duplicated(["LocusId", "allele_rank"]).any(), \
        "TSV (LocusId, allele_rank) key is not unique"
    merged = json_df.merge(tsv_df, how="left", indicator=True,
                           left_on=["locus_id", "allele_rank"],
                           right_on=["LocusId", "allele_rank"])
    n_matched = int((merged["_merge"] == "both").sum())
    return merged.drop(columns=["_merge", "LocusId"]), n_matched


def build_combo(sample, data_type, cov_label, data_dir, force):
    """Downloads + joins one combo and writes its parquet.

    Args:
        sample: Sample name (e.g. ``"HG002"``).
        data_type: Sequencing data type (e.g. ``"illumina"``, ``"element"``).
        cov_label: Coverage-dir label (e.g. ``"10x"``).
        data_dir: Base ``genotype_quality/data`` directory.
        force: Rebuild even if the output parquet already exists.

    Returns:
        A summary dict for this combo.
    """
    coverage = parse_nominal_coverage(cov_label)
    out_path = os.path.join(data_dir, "real_full", "%s_%s.parquet" % (sample, cov_label))
    print("=== %s %s %s -> %s ===" % (sample, data_type, cov_label, out_path))

    if os.path.exists(out_path) and not force:
        flag = pd.read_parquet(out_path, columns=["is_negative_locus"])["is_negative_locus"]
        n_matched = int(flag.notna().sum())
        print("    parquet exists; skipping build (use --force to rebuild)")
        return {"sample": sample, "cov": cov_label, "n_rows": len(flag),
                "n_matched": n_matched, "match_rate": n_matched / max(len(flag), 1),
                "n_negative": int((flag == True).sum()), "path": out_path, "built": False}

    download_dir = os.path.join(data_dir, "real_full", "_downloads", "%s_%s" % (sample, cov_label))
    json_remote = list_json_inputs(sample, data_type, cov_label)
    print("    JSON inputs: %d %s file(s)"
          % (len(json_remote), "combined" if ".shard" not in os.path.basename(json_remote[0]) else "shard"))
    json_local = download(json_remote, download_dir)
    tsv_local = download([truth_tsv_remote(sample, data_type, cov_label)], download_dir)[0]

    json_df = load_json_rows(json_local, sample_id="%s_%s" % (sample, cov_label))
    tsv_df = load_truth_tsv(tsv_local)
    print("    JSON rows: %d | TSV rows: %d (%d loci)"
          % (len(json_df), len(tsv_df), tsv_df["LocusId"].nunique()))

    merged, n_matched = join_truth(json_df, tsv_df)
    merged["sample"] = sample
    merged["data_type"] = data_type  # path/identity + CV domain-split only; NOT a model feature
    merged["coverage"] = coverage  # nominal run coverage (overwrites per-locus)
    merged["source"] = "real"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    n_unmatched = len(merged) - n_matched
    print("    matched %d/%d JSON rows (%.4f); %d unmatched (true=NaN); %d negative-control rows"
          % (n_matched, len(merged), n_matched / max(len(merged), 1),
             n_unmatched, int((merged["is_negative_locus"] == True).sum())))
    return {"sample": sample, "cov": cov_label, "n_rows": len(merged),
            "n_matched": n_matched, "match_rate": n_matched / max(len(merged), 1),
            "n_negative": int((merged["is_negative_locus"] == True).sum()),
            "path": out_path, "built": True}


def main():
    """Builds the full-branch real parquet for every requested combo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                        help="genotype_quality/data directory")
    parser.add_argument("--combos", nargs="+", default=None,
                        help="subset of combos to build, e.g. HG002_10x CHM1_CHM13_46x")
    parser.add_argument("--force", action="store_true", help="rebuild even if the parquet exists")
    args = parser.parse_args()

    wanted = set(args.combos) if args.combos else None
    summaries = []
    for sample, data_type, cov_label in COMBOS:
        if wanted is not None and ("%s_%s" % (sample, cov_label)) not in wanted:
            continue
        summaries.append(build_combo(sample, data_type, cov_label, args.data_dir, args.force))

    print("\n==================== SUMMARY ====================")
    total = 0
    for s in summaries:
        total += s["n_rows"]
        print("  %-16s %-4s rows=%-9d matched=%-9d match_rate=%.4f negative=%-6d %s"
              % (s["sample"], s["cov"], s["n_rows"], s["n_matched"], s["match_rate"],
                 s["n_negative"], "(built)" if s["built"] else "(cached)"))
    print("  ---------------------------------------------")
    print("  TOTAL full-branch real rows: %d across %d combo(s)" % (total, len(summaries)))


if __name__ == "__main__":
    main()
