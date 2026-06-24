"""Build per-sample held-out rows for the 43 HPRC short-read population samples.

These 43 samples (HG00438..NA20129) were genotyped against the full 1.6M-locus combined catalog with
the latest patched EHv5-bw2-optimized (optimized-streaming) image, so each per-sample JSON carries the
QuickGenotype fast-path metrics + AlleleQualityMetrics + a real CI -- i.e. both fast-branch
(QuickGenotype) and full-branch (full-genotyper fallback) rows. They are absent from the model
training pool (HG002 + CHM1_CHM13 + sim), so they serve as an external cross-population held-out
benchmark for the trained q + direction models (see BENCHMARK_43_SAMPLES_PLAN.md).

This module, for each sample:
  1. Auto-discovers the {N}x_coverage dir + locates the single combined .json.gz and the per-allele
     truth comparison table under the combined_catalog_.../ subdir.
  2. STREAMS the (large, ~1-2GB-parsed) .json.gz via ijson per LOCUS so the whole file is never
     resident -- essential on this RAM-constrained machine. Each locus's allele rows are extracted by
     the shared eh_json_features.extract_variant_rows (so features match every other source).
  3. Joins truth per batch (reuses data_gcs.load_truth_tsv / join_truth on (LocusId, allele_rank)),
     keeping only truth-matched rows.
  4. Writes one parquet per sample to {data_dir}/real_fast/{sample}_{cov}.parquet (same schema as
     data_fast_real -- both branches mixed, split downstream by genotyping_branch).

eh + every feature come from the JSON; the TSV supplies only true / purity / the negative flag.
q / t / direction / filtering are deferred to build_dataset.add_labels_and_filter (run by
benchmark_43.py). Coding rules: no type hints, Google docstrings, print(), gcloud/gsutil.
"""

import argparse
import gc
import gzip
import os
import subprocess

import ijson
import pandas as pd

import data_gcs as G
import eh_json_features as F

TRUTH_ROOT = "gs://str-truth-set-v2/tool_results"
VARIANT = "EHv5-bw2-optimized"  # optimized-streaming; truth columns templated on this
SUBDIR = "combined_catalog_43_samples_1.6M_loci_maxdepth150"
DATA_TYPE = "illumina"

# Flush accumulated extractor rows to a join + matched-row filter once a batch reaches this many rows
# (at a locus boundary, so a locus is never split across batches and the join key stays unique).
BATCH_ROWS = 300_000

# The 43 HPRC short-read population samples genotyped into the maxdepth150 combined-catalog run.
SAMPLES = [
    "HG00438", "HG00514", "HG00621", "HG00673", "HG00733", "HG00735", "HG00741", "HG01071",
    "HG01106", "HG01109", "HG01175", "HG01243", "HG01258", "HG01358", "HG01361", "HG01891",
    "HG01928", "HG01952", "HG01978", "HG02055", "HG02080", "HG02145", "HG02148", "HG02257",
    "HG02572", "HG02622", "HG02630", "HG02717", "HG02723", "HG02818", "HG02886", "HG03098",
    "HG03125", "HG03453", "HG03486", "HG03492", "HG03516", "HG03540", "HG03579", "NA12878",
    "NA18906", "NA19240", "NA20129",
]


def discover_cov_label(sample):
    """Returns the ``{N}x_coverage`` dir's ``{N}x`` label for one sample, or None if absent."""
    base = "%s/%s/%s/%s/" % (TRUTH_ROOT, sample, DATA_TYPE, VARIANT)
    try:
        listing = subprocess.run(["gsutil", "ls", base], capture_output=True, text=True,
                                 check=True).stdout.split()
    except subprocess.CalledProcessError:
        return None
    for p in listing:
        name = p.rstrip("/").split("/")[-1]
        if name.endswith("_coverage"):
            return name[:-len("_coverage")]
    return None


def json_gz_remote(sample, cov_label):
    """Returns the gs:// path(s) of the combined optimized-streaming .json.gz for one sample."""
    d = "%s/%s/%s/%s/%s_coverage/%s/json/" % (TRUTH_ROOT, sample, DATA_TYPE, VARIANT, cov_label, SUBDIR)
    try:
        listing = subprocess.run(["gsutil", "ls", d], capture_output=True, text=True,
                                 check=True).stdout.split()
    except subprocess.CalledProcessError:
        return []
    return sorted(p for p in listing if p.endswith(".json.gz") or p.endswith(".json"))


def truth_tsv_remote(sample, cov_label):
    """Returns the gs:// path of the per-allele truth comparison table for one sample."""
    return ("%s/%s/%s/%s/%s_coverage/%s/"
            "%s.tandem_repeat_genotypes.for_comparison.with_%s_vs_Truth_columns.alleles.tsv.gz"
            % (TRUTH_ROOT, sample, DATA_TYPE, VARIANT, cov_label, SUBDIR, sample, VARIANT))


def stream_extract_rows(json_local, sample_id):
    """Yields per-allele extractor rows from a (gzipped or plain) EH JSON, one LOCUS at a time.

    Uses ijson.kvitems so only one locus_result dict is resident at a time -- the whole multi-GB
    JSON is never loaded. Each locus_result is routed through the shared extractor so features are
    identical to every other data source.

    Args:
        json_local: Local path to the EH ``.json`` or ``.json.gz``.
        sample_id: sample_id stamped on every row (encodes sample + coverage).

    Yields:
        Per-allele row dicts (from ``eh_json_features.extract_variant_rows``).
    """
    opener = gzip.open if json_local.endswith(".gz") else open
    with opener(json_local, "rb") as fh:
        for locus_id, locus_result in ijson.kvitems(fh, "LocusResults"):
            if not isinstance(locus_result, dict):
                continue
            locus_result.setdefault("LocusId", locus_id)
            for variant in (locus_result.get("Variants") or {}).values():
                yield from F.extract_variant_rows(variant, locus_result, sample_id)


def build_sample(sample, data_dir, force):
    """Streams + joins one sample's optimized-streaming JSON and writes its parquet.

    Returns a summary dict, or None if the JSON / truth table is not present yet.
    """
    cov_label = discover_cov_label(sample)
    if cov_label is None:
        print("=== %s : NO %s coverage dir yet -- skipping" % (sample, VARIANT))
        return None
    sample_id = "%s_%s" % (sample, cov_label)
    out_path = os.path.join(data_dir, "real_fast", "%s.parquet" % sample_id)
    print("=== %s %s -> %s ===" % (sample, cov_label, out_path))
    if os.path.exists(out_path) and not force:
        df = pd.read_parquet(out_path, columns=["genotyping_branch"])
        print("    parquet exists; skipping (use --force)")
        return {"sample": sample, "cov": cov_label, "n_rows": len(df),
                "n_fast": int((df["genotyping_branch"] == "fast").sum()), "built": False}

    json_remote = json_gz_remote(sample, cov_label)
    if not json_remote:
        print("    NO JSON yet (batch still running?) -- skipping")
        return None
    tsv_remote = truth_tsv_remote(sample, cov_label)
    try:
        tsv_ok = subprocess.run(["gsutil", "-q", "stat", tsv_remote]).returncode == 0
    except Exception:
        tsv_ok = False
    if not tsv_ok:
        print("    NO truth comparison table yet -- skipping")
        return None

    dl = os.path.join(data_dir, "real_fast", "_downloads", sample_id)
    json_local = G.download(json_remote, dl)
    tsv_local = G.download([tsv_remote], dl)[0]
    tsv_df = G.load_truth_tsv(tsv_local, variant=VARIANT)

    parts, n_matched, batch = [], 0, []

    def flush():
        nonlocal n_matched
        if not batch:
            return
        jdf = pd.DataFrame(batch)
        # The combined.43_catalogs.EHv5 catalog uses chr-prefixed LocusIds (e.g. "chr1-28588-..."),
        # but the str-truth-set-v2 comparison tables key on the un-prefixed form ("1-598934-..."),
        # so strip the leading "chr" before the (LocusId, allele_rank) join (else 0 rows match).
        jdf["locus_id"] = jdf["locus_id"].str.replace(r"^chr", "", regex=True)
        m, nm = G.join_truth(jdf, tsv_df)
        m = m[m["is_negative_locus"].notna()].copy()  # keep only truth-matched rows
        parts.append(m)
        n_matched += nm
        del jdf, m
        gc.collect()

    n_seen = 0
    last_locus = None
    for row in _iter_all(json_local, sample_id):
        # flush only at a locus boundary so a locus is never split across batches
        if len(batch) >= BATCH_ROWS and row["locus_id"] != last_locus:
            flush()
            batch = []
        batch.append(row)
        last_locus = row["locus_id"]
        n_seen += 1
    flush()

    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    del parts, tsv_df
    gc.collect()
    if merged.empty:
        print("    WARNING: 0 truth-matched rows")
        return None
    merged["sample"] = sample
    merged["data_type"] = DATA_TYPE
    merged["coverage"] = G.parse_nominal_coverage(cov_label)
    merged["source"] = "real"
    merged = merged.drop(columns=[c for c in ("variant_id", "sample_id", "repeat_unit",
                                              "ref_chrom", "tsv_eh", "concordance")
                                  if c in merged.columns])
    for c in merged.select_dtypes("float64").columns:
        merged[c] = merged[c].astype("float32")
    gc.collect()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    n_fast = int((merged["genotyping_branch"] == "fast").sum())
    print("    seen=%d  matched_rows=%d  fast=%d  full_fallback=%d  (json_match=%d)"
          % (n_seen, len(merged), n_fast, len(merged) - n_fast, n_matched))
    return {"sample": sample, "cov": cov_label, "n_rows": len(merged),
            "n_fast": n_fast, "n_matched": n_matched, "built": True}


def _iter_all(json_locals, sample_id):
    """Chains stream_extract_rows over one or more local JSON files for a sample."""
    for path in json_locals:
        yield from stream_extract_rows(path, sample_id)


def main():
    """Builds the per-sample held-out parquet for every of the 43 samples whose JSON+truth is ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data_eval_43"))
    parser.add_argument("--samples", nargs="+", default=None,
                        help="subset of the 43 sample names (default: all ready)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    wanted = set(args.samples) if args.samples else None
    summaries = []
    for sample in SAMPLES:
        if wanted is not None and sample not in wanted:
            continue
        s = build_sample(sample, args.data_dir, args.force)
        if s is not None:
            summaries.append(s)

    print("\n==================== SUMMARY ====================")
    total = total_fast = 0
    for s in summaries:
        total += s["n_rows"]
        total_fast += s["n_fast"]
        print("  %-9s %-4s rows=%-9d fast=%-9d %s"
              % (s["sample"], s["cov"], s["n_rows"], s["n_fast"],
                 "(built)" if s["built"] else "(cached)"))
    print("  TOTAL held-out rows: %d (fast=%d, full=%d) across %d/43 sample(s)"
          % (total, total_fast, total - total_fast, len(summaries)))


if __name__ == "__main__":
    main()
