"""Download the training data from GCS and assemble the per-branch parquets.

For each ``(sample, coverage)`` combo on each genotyping branch this module:

  1. Idempotently downloads the per-shard EH JSON (``*.json.gz``) and the truth
     ``*.alleles.tsv.gz`` from ``gs://str-truth-set-v2/tool_results/...`` (existing
     local files are not re-fetched).
  2. Extracts per-allele feature rows from the JSON via ``eh_json.extract_rows``
     (``eh`` and every feature come from the JSON, never the TSV).
  3. Joins the truth TSV on ``(locus_id, allele_rank)`` using size-sorted
     rank-pairing, attaching only ``true`` / ``purity`` / the negative-control flag.
  4. Writes one parquet per combo, then assembles + labels + filters into
     ``data/parquet/{quick,full}.parquet``.

Single EH source -- the optimized-streaming run (``EHv5-bw2-optimized``), the variant
deployed in production. Both branches are carved from the SAME run by ``genotyping_branch``
so each expert trains on exactly the subpopulation it is served at deploy time:
  - ``quick`` -- the ``QuickGenotype`` fast-path rows (``genotyping_branch == "quick"``),
    routed to the ``quick`` genotyping_regime.
  - ``full``  -- the full-genotyper FALLBACK rows (``genotyping_branch == "full"``: the loci
    the fast path punted on), routed to ``full_spanning`` / ``full_nonspanning``.

This replaces the earlier design that sourced ``full`` from a separate low-mem-streaming
``EHv5`` run over ALL loci -- an easier, different locus population than the fast-path fallback
the full experts actually see at deploy, which left the full experts train/serve-skewed. The old
leakage concern (optimized fallback rows colliding with ``EHv5`` full rows on the same loci) no
longer applies: there is no ``EHv5`` pool to collide with, within one optimized run a locus is
either fast-path-called OR fallback (never both), and ``quick`` / ``full`` are separate experts;
cross-coverage repeats of a locus are held out together by the chromosome-clean CV.

The committed model is real-data-only (HG002 10x/20x/31x + CHM1_CHM13 46x); no
simulated rows. Determinism: stable size sort for rank pairing; no randomness.
Coding rules: no type hints, Google docstrings, ``print()``, ``gcloud`` (macOS).
"""

import argparse
import glob
import os
import subprocess

import pandas as pd

import eh_json
import features

GCS_ROOT = "gs://str-truth-set-v2/tool_results"
HERE = os.path.dirname(os.path.abspath(__file__))

# Single source variant: the optimized-streaming run deployed in production. Its QuickGenotype
# rows feed the `quick` branch and its full-genotyper fallback rows feed the `full` branch (the
# split happens in assemble_branch by genotyping_branch). The variant token is also templated into
# the truth-TSV column headers. SOURCE_SUBDIR keeps its historical name ("real_quick") so existing
# local downloads/parquets are reused; it now feeds BOTH branches.
SOURCE_VARIANT = "EHv5-bw2-optimized"
SOURCE_SUBDIR = "real_quick"

# (sample, coverage-dir label) -- illumina WGS only.
COMBOS = [
    ("HG002", "10x"),
    ("HG002", "20x"),
    ("HG002", "31x"),
    ("CHM1_CHM13", "46x"),
]

VALID_CHROMS = set(str(i) for i in range(1, 23)) | {"X", "Y"}


def _combo_dir(sample, variant, cov_label):
    return "%s/%s/illumina/%s/%s_coverage/" % (GCS_ROOT, sample, variant, cov_label)


def _truth_tsv_remote(sample, variant, cov_label):
    return ("%s%s.tandem_repeat_genotypes.for_comparison.with_%s_vs_Truth_columns.alleles.tsv.gz"
            % (_combo_dir(sample, variant, cov_label), sample, variant))


def _list_json_inputs(sample, variant, cov_label):
    """Lists the combo's JSON shard paths (prefers a single combined file if present)."""
    listing = subprocess.run(["gsutil", "ls", _combo_dir(sample, variant, cov_label) + "json/"],
                             capture_output=True, text=True, check=True).stdout.split()
    jsons = [p for p in listing if p.endswith(".json") or p.endswith(".json.gz")]
    combined = sorted(p for p in jsons if ".shard" not in os.path.basename(p))
    return combined if combined else sorted(p for p in jsons if ".shard" in os.path.basename(p))


def _download(remote_paths, dest_dir):
    """Downloads ``remote_paths`` into ``dest_dir`` (skips files already present, retries).

    Uses a single multi-file ``gcloud storage cp`` (parallelized) and retries the
    whole batch on failure; each retry only re-fetches the still-missing files, so a
    transient bad-hash leaves no partial file and recovers cleanly.
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
    still = [r for r, l in zip(remote_paths, locals_) if not os.path.exists(l)]
    if still:
        raise RuntimeError("failed to download after 4 attempts: %s" % still)
    return locals_


def _load_truth_tsv(tsv_path, variant):
    """Loads the per-allele truth TSV and synthesizes ``allele_rank``.

    Within each ``LocusId`` the allele rows are sorted by ``(tool size, truth size)``
    and assigned ``allele_rank`` 0, 1, ... -- the same size ordering ``eh_json`` uses
    on the JSON side, so equal keys pair correctly. Returns only the
    truth/purity/negative-flag columns; ``eh`` and all features come from the JSON.
    """
    cols = {
        "LocusId": "LocusId",
        "NumRepeats: Allele: Truth": "true",
        "RepeatPurity: Allele: Truth": "purity",
        "NumRepeats: Allele: %s" % variant: "tool_size",
        "TruthSetOrNegativeLocus": "neg_flag",
    }
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip",
                     usecols=list(cols), dtype={"LocusId": str}).rename(columns=cols)
    for c in ("true", "purity", "tool_size"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    flag = df["neg_flag"].astype(str).str.strip()
    unexpected = sorted(set(flag.unique()) - {"TruthSet", "NegativeLocus"})
    if unexpected:
        raise ValueError("TruthSetOrNegativeLocus in %s has unexpected token(s) %s; expected only "
                         "['NegativeLocus', 'TruthSet']" % (tsv_path, unexpected))
    df["is_negative_locus"] = flag == "NegativeLocus"
    df = df.sort_values(["LocusId", "tool_size", "true"], kind="mergesort", na_position="last")
    df["allele_rank"] = df.groupby("LocusId", sort=False).cumcount()
    return df[["LocusId", "allele_rank", "true", "purity", "is_negative_locus"]]


def _join_truth(json_df, tsv_df):
    """Left-joins truth onto the JSON rows by ``(locus_id, allele_rank)`` (keys asserted unique).

    A leading ``chr`` is stripped from both locus ids before the join: some catalogs emit
    ``chr1-..`` LocusIds in the JSON while the truth TSV uses ``1-..`` (or vice versa), and the
    two must compare equal.
    """
    json_df["locus_id"] = json_df["locus_id"].astype(str).str.replace(r"^chr", "", regex=True)
    tsv_df["LocusId"] = tsv_df["LocusId"].astype(str).str.replace(r"^chr", "", regex=True)
    assert not json_df.duplicated(["locus_id", "allele_rank"]).any(), \
        "JSON (locus_id, allele_rank) key is not unique"
    assert not tsv_df.duplicated(["LocusId", "allele_rank"]).any(), \
        "TSV (LocusId, allele_rank) key is not unique"
    merged = json_df.merge(tsv_df, how="left", left_on=["locus_id", "allele_rank"],
                           right_on=["LocusId", "allele_rank"]).drop(columns=["LocusId"])
    return merged


def build_combo(variant, subdir, sample, cov_label, data_dir, force):
    """Downloads + joins one combo and writes its per-combo parquet (all rows, both branches)."""
    out_path = os.path.join(data_dir, subdir, "%s_%s.parquet" % (sample, cov_label))
    print("=== %s %s -> %s ===" % (sample, cov_label, out_path))
    if os.path.exists(out_path) and not force:
        n = len(pd.read_parquet(out_path, columns=["eh"]))
        print("    parquet exists; skipping (use --force to rebuild)  [%d rows]" % n)
        return n

    dl_dir = os.path.join(data_dir, subdir, "_downloads", "%s_%s" % (sample, cov_label))
    json_remote = _list_json_inputs(sample, variant, cov_label)
    print("    %d JSON file(s) + 1 truth TSV" % len(json_remote))
    json_local = _download(json_remote, dl_dir)
    tsv_local = _download([_truth_tsv_remote(sample, variant, cov_label)], dl_dir)[0]

    rows = []
    for path in json_local:
        rows.extend(eh_json.extract_rows(path, sample_id="%s_%s" % (sample, cov_label)))
    json_df = pd.DataFrame(rows)
    tsv_df = _load_truth_tsv(tsv_local, variant)
    merged = _join_truth(json_df, tsv_df)
    merged = merged.drop(columns=["sample_id"])
    for c in merged.select_dtypes("float64").columns:
        merged[c] = merged[c].astype("float32")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    matched = int(merged["true"].notna().sum())
    print("    %d rows, %d matched truth (%.4f), %d quick / %d full"
          % (len(merged), matched, matched / max(len(merged), 1),
             int((merged["genotyping_branch"] == "quick").sum()),
             int((merged["genotyping_branch"] == "full").sum())))
    return len(merged)


def _chrom_from_locus(locus_id):
    """Returns the normalized chromosome ({1..22,X,Y}) from a ``locus_id`` like ``1-5917-..``."""
    chrom = locus_id.astype(str).str.split("-").str[0].str.replace(r"^chr", "", regex=True)
    return chrom.where(chrom.isin(VALID_CHROMS))


def label_and_filter(df):
    """Adds ``chrom`` + the q/direction labels and drops unusable rows.

    Drops chrM/unknown contigs, negative-control loci, missing/non-positive ``eh``/``true``, and
    missing motif, then derives the labels + genotyping-regime routing via ``features.add_labels``.
    Shared by the training-pool assembly and the held-out-sample benchmark so both apply identical
    filtering. Does NOT filter on truth repeat purity -- impure loci are kept in both training and
    eval (purity is available to the report's opt-in stratification pill via ``accuracy_by_size.py``
    instead of being used as a blanket exclusion).

    Returns:
        A ``(kept_df, drop_counts)`` tuple. ``kept_df`` has ``chrom`` + the label columns added and
        the heavy ``locus_id`` / ``is_negative_locus`` columns dropped.
    """
    df = df.copy()
    df["chrom"] = _chrom_from_locus(df["locus_id"])
    eh = pd.to_numeric(df["eh"], errors="coerce")
    true = pd.to_numeric(df["true"], errors="coerce")
    motif = pd.to_numeric(df["motif_size"], errors="coerce")
    negative = df["is_negative_locus"].fillna(False).astype(bool)

    keep = pd.Series(True, index=df.index)
    drops = {}
    for name, bad in (("chrM_or_unknown_contig", df["chrom"].isna()),
                      ("negative_control_locus", negative),
                      ("missing_eh_or_true", eh.isna() | true.isna()),
                      ("nonpositive_eh_or_true", (eh <= 0) | (true <= 0)),
                      ("missing_motif_size", motif.isna() | (motif <= 0))):
        bad = bad & keep
        drops[name] = int(bad.sum())
        keep &= ~bad

    df = df[keep].reset_index(drop=True)
    features.add_labels(df)
    df = df.drop(columns=[c for c in ("locus_id", "is_negative_locus") if c in df.columns])
    return df, drops


def assemble_branch(data_dir, branch, subdir):
    """Assembles one branch's per-combo parquets, labels + filters, writes data/parquet/<branch>."""
    parts = sorted(glob.glob(os.path.join(data_dir, subdir, "*.parquet")))
    if not parts:
        print("\n%s branch: no per-combo parquets in %s/ -- skipping" % (branch, subdir))
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # Both branches are carved from the same optimized-streaming run by genotyping_branch:
    # quick = fast-path QuickGenotype rows; full = full-genotyper fallback rows (the deploy-matched
    # subpopulation the full experts are served).
    df = df[df["genotyping_branch"] == ("quick" if branch == "quick" else "full")].reset_index(drop=True)

    n0 = len(df)
    df, drops = label_and_filter(df)

    out_path = os.path.join(data_dir, "parquet", "%s.parquet" % branch)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    print("\n==================== %s branch ====================" % branch.upper())
    print("  input %d -> kept %d" % (n0, len(df)))
    for k, v in drops.items():
        print("    dropped %-26s %d" % (k, v))
    print("  genotyping_regimes:", df["genotyping_regime"].value_counts().to_dict())
    print("  directions:", df["direction"].value_counts().to_dict())
    print("  wrote %d rows -> %s" % (len(df), out_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--force", action="store_true", help="rebuild even if parquets exist")
    args = parser.parse_args()

    for sample, cov_label in COMBOS:
        build_combo(SOURCE_VARIANT, SOURCE_SUBDIR, sample, cov_label, args.data_dir, args.force)
    for branch in ("quick", "full"):
        assemble_branch(args.data_dir, branch, SOURCE_SUBDIR)


if __name__ == "__main__":
    main()
