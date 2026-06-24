"""Build FAST-BRANCH REAL per-allele rows from the regenerated optimized-streaming JSON.

The fast-branch real data comes from the production EHv5-bw2-optimized (optimized-streaming)
per-shard JSON under ``tool_results`` -- now genotyped with the latest EH image, which carries the
QuickGenotype fast-path metrics, so the separate ``fast_real_pipeline.py`` scratch run is obsolete.
Each variant in that JSON carries ``QuickGenotype: true`` on the fast path (and is absent on the
full-genotyper fallback). This module, for each combo:

  1. Downloads the per-shard JSON from the production EHv5-bw2-optimized ``json/`` dir (idempotent).
  2. Extracts per-allele rows via ``eh_json_features.extract_rows`` -- the rows split into
     ``fast`` (QuickGenotype) and ``full`` (fallback) by ``genotyping_branch`` automatically.
  3. Joins truth from the EXISTING ``with_EHv5-bw2-optimized_vs_Truth_columns.alleles.tsv.gz``
     (same Truth/RepeatPurity columns as EHv5; only the tool-call columns differ) on
     ``(LocusId, allele_rank)`` via size-sorted rank-pairing (reuses ``data_gcs`` helpers).
  4. Writes one parquet per combo to ``data/real_fast/{sample}_{cov}.parquet``.

``eh`` and every feature come from the regenerated JSON (SPEC sec 3); the TSV supplies only
``true`` / ``purity`` / the negative flag. ``q`` / ``t`` / direction / filtering are deferred to
``build_dataset.py``. Coding rules: no type hints, Google docstrings, ``print()``, ``gcloud``.
"""

import argparse
import gc
import os
import subprocess

import pandas as pd

import data_gcs as G
import eh_json_features as F


TRUTH_ROOT = "gs://str-truth-set-v2/tool_results"
VARIANT = "EHv5-bw2-optimized"  # optimized-streaming; truth columns are templated on this

# (sample, data_type, coverage-dir label) -- illumina only. element + illumina_exome were evaluated
# and EXCLUDED (they regress held-out WGS on the fast branch -0.018+-0.002; see data_gcs.COMBOS note
# + reviewer_deepdive/domain_cv.py). data_type plumbing kept so they can be re-added behind a fix.
# Mirrors data_gcs.COMBOS.
COMBOS = [
    ("HG002", "illumina", "10x"),
    ("HG002", "illumina", "20x"),
    ("HG002", "illumina", "31x"),
    ("CHM1_CHM13", "illumina", "46x"),
]

# Held-out, EVALUATION-ONLY short-read combos (non-illumina-WGS data types). These are deliberately
# NOT in COMBOS so the core training parquet (data/real_fast -> data/parquet/fast.parquet) stays
# illumina-WGS-only; blending element/exome regresses held-out WGS (see data_gcs.COMBOS note), and
# RNA-seq is highly out-of-distribution (spliced, expressed-loci-only coverage). Generated separately
# (--include-eval-combos, typically into a distinct --data-dir) to measure how the illumina-trained
# corrected-depth model GENERALIZES to these domains. rnaseq uses a "{N}G" (Gbp) coverage label.
EVAL_COMBOS = [
    ("HG002", "element", "30x"),
    ("HG002", "illumina_exome", "3x"),
    ("CHM1_CHM13", "illumina_exome", "3x"),
    ("HG002", "illumina_rnaseq", "24G"),
]


def scratch_json_dir(sample, data_type, cov_label):
    """Returns the production EHv5-bw2-optimized ``json/`` dir of per-shard optimized-streaming JSON.

    Repointed from the old scratch bucket: production tool_results are now genotyped with the same
    latest image (carrying the QuickGenotype fast-path metrics), so the scratch run is obsolete.
    """
    return "%s/%s/%s/%s/%s_coverage/json/" % (TRUTH_ROOT, sample, data_type, VARIANT, cov_label)


def truth_tsv_remote(sample, data_type, cov_label):
    """Returns the GCS path of the EHv5-bw2-optimized per-allele truth TSV for one combo."""
    return ("%s/%s/%s/%s/%s_coverage/"
            "%s.tandem_repeat_genotypes.for_comparison.with_%s_vs_Truth_columns.alleles.tsv.gz"
            % (TRUTH_ROOT, sample, data_type, VARIANT, cov_label, sample, VARIANT))


def list_scratch_json(sample, data_type, cov_label):
    """Lists the regenerated per-shard JSON paths for one combo (sorted)."""
    listing = subprocess.run(["gsutil", "ls", scratch_json_dir(sample, data_type, cov_label)],
                             capture_output=True, text=True, check=True).stdout.split()
    return sorted(p for p in listing if p.endswith(".json"))


def build_combo(sample, data_type, cov_label, data_dir, force):
    """Downloads + joins one fast-branch real combo and writes its parquet.

    Args:
        sample: Sample name.
        data_type: Sequencing data type (e.g. ``"illumina"``, ``"element"``).
        cov_label: Coverage-dir label.
        data_dir: Base ``genotype_quality/data`` directory.
        force: Rebuild even if the output parquet exists.

    Returns:
        A summary dict, or None if the scratch JSON is not present yet (batch still running).
    """
    coverage = G.parse_nominal_coverage(cov_label)
    out_path = os.path.join(data_dir, "real_fast", "%s_%s.parquet" % (sample, cov_label))
    print("=== %s %s %s -> %s ===" % (sample, data_type, cov_label, out_path))
    if os.path.exists(out_path) and not force:
        print("    parquet exists; skipping (use --force)")
        df = pd.read_parquet(out_path, columns=["genotyping_branch"])
        return {"sample": sample, "cov": cov_label, "n_rows": len(df),
                "n_fast": int((df["genotyping_branch"] == "fast").sum()), "built": False}

    try:
        json_remote = list_scratch_json(sample, data_type, cov_label)
    except subprocess.CalledProcessError:
        json_remote = []
    if not json_remote:
        print("    NO scratch JSON yet (Hail Batch still running?) -- skipping")
        return None

    download_dir = os.path.join(data_dir, "real_fast", "_downloads", "%s_%s" % (sample, cov_label))
    json_local = G.download(json_remote, download_dir)
    tsv_local = G.download([truth_tsv_remote(sample, data_type, cov_label)], download_dir)[0]

    # Merge each shard against the (small) truth table separately so the two ~1.1M-row frames
    # never coexist -- keeps peak RAM bounded on this machine (EH shards by locus range, so a
    # locus is wholly within one shard and the per-shard join key stays unique).
    tsv_df = G.load_truth_tsv(tsv_local, variant=VARIANT)
    parts, n_matched = [], 0
    for path in json_local:
        jdf = pd.DataFrame(list(F.extract_rows(path, sample_id="%s_%s" % (sample, cov_label))))
        m, nm = G.join_truth(jdf, tsv_df)
        parts.append(m)
        n_matched += nm
        print("      %s -> %d rows (%d matched)" % (os.path.basename(path), len(m), nm))
        del jdf, m
        gc.collect()
    merged = pd.concat(parts, ignore_index=True)
    del parts, tsv_df
    gc.collect()
    merged["sample"] = sample
    merged["data_type"] = data_type  # path/identity + CV domain-split only; NOT a model feature
    merged["coverage"] = coverage
    merged["source"] = "real"

    # Trim heavy unused columns + downcast floats before writing to bound RAM and parquet size
    # (build_dataset/run_cv only need features + labels + keys; audit string cols are dropped).
    merged = merged.drop(columns=[c for c in ("variant_id", "sample_id", "repeat_unit",
                                              "ref_chrom", "tsv_eh", "concordance")
                                  if c in merged.columns])
    for c in merged.select_dtypes("float64").columns:
        merged[c] = merged[c].astype("float32")
    gc.collect()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    n_fast = int((merged["genotyping_branch"] == "fast").sum())
    print("    rows=%d fast=%d full_fallback=%d matched=%d (%.4f)"
          % (len(merged), n_fast, len(merged) - n_fast, n_matched,
             n_matched / max(len(merged), 1)))
    return {"sample": sample, "cov": cov_label, "n_rows": len(merged),
            "n_fast": n_fast, "n_matched": n_matched, "built": True}


def main():
    """Builds the fast-branch real parquet for every combo whose JSON is ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--combos", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-eval-combos", action="store_true",
                        help="also build the held-out EVAL_COMBOS (element/exome/rnaseq); use a distinct "
                             "--data-dir so these eval-only rows do not enter the core training parquet")
    parser.add_argument("--only-eval-combos", action="store_true",
                        help="build ONLY the held-out EVAL_COMBOS (skip the illumina training combos)")
    args = parser.parse_args()

    combos = [] if args.only_eval_combos else list(COMBOS)
    if args.include_eval_combos or args.only_eval_combos:
        combos += EVAL_COMBOS

    wanted = set(args.combos) if args.combos else None
    summaries = []
    for sample, data_type, cov_label in combos:
        if wanted is not None and ("%s_%s" % (sample, cov_label)) not in wanted:
            continue
        s = build_combo(sample, data_type, cov_label, args.data_dir, args.force)
        if s is not None:
            summaries.append(s)

    print("\n==================== SUMMARY ====================")
    total = total_fast = 0
    for s in summaries:
        total += s["n_rows"]
        total_fast += s["n_fast"]
        print("  %-16s %-4s rows=%-9d fast=%-9d %s"
              % (s["sample"], s["cov"], s["n_rows"], s["n_fast"],
                 "(built)" if s["built"] else "(cached)"))
    print("  TOTAL fast-branch real rows: %d (fast=%d) across %d combo(s)"
          % (total, total_fast, len(summaries)))


if __name__ == "__main__":
    main()
