"""Generate simulated per-allele training rows for the genotype-quality models.

This is the SIMULATED data source of the genotype_quality pipeline (SPEC.md sec
5.3, "Sim data"). For every simulated locus it builds representative diploid
genotypes by merging two single-allele simulated bams, runs ExpansionHunter in
both relevant analysis modes, and turns each run's JSON into tidy per-allele rows
via the shared extractor ``eh_json_features.extract_rows``.

Branch routing / de-dup (SPEC.md sec 5.3)
-----------------------------------------
Both analysis modes are run on every merged bam and EVERY produced row is
emitted, tagged with an ``analysis_mode`` column:

- ``low-mem-streaming``    -> the full genotyper; every row is ``genotyping_branch
  == "full"`` (no ``QuickGenotype``). This is the sim FULL-branch source.
- ``optimized-streaming``  -> a mix: fast-path rows (``QuickGenotype == true`` ->
  ``genotyping_branch == "fast"``) plus full-genotyper-fallback rows
  (``genotyping_branch == "full"``). This is the sim FAST-branch source.

Downstream ``build_dataset.py`` selects FULL-branch rows from ``low-mem-streaming``
and FAST-branch rows from ``optimized-streaming``; emitting all rows here (rather
than pre-filtering) keeps that choice in one place and avoids losing information.

Truth
-----
Each single-allele bam ``sim_<N>x__...bam`` carries reads from one allele of size
``N`` repeat units (``<N>`` is the allele size, NOT coverage). The diploid truth
is ``sorted((n1, n2))``; per allele we use rank-pairing -- the extractor yields
``allele_rank`` 0 (short) / 1 (long) sorted by the EH call, so
``true = sorted((n1, n2))[allele_rank]`` pairs the smaller call with the smaller
truth (standard for sim).

Determinism / idempotency
-------------------------
Loci and pairs are processed in sorted order. Merged bams, single-locus catalogs
and EH JSON outputs are cached on disk and reused if already present, so re-running
is cheap and produces identical output.

Run with:  python3 data_sim.py
"""

import os
import subprocess
import sys

import pandas as pd

# Reuse the simulation harness machinery (paths + bam/catalog builders) from the
# repo-root integration test module rather than duplicating it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mode_consistency_and_accuracy_tests as M

import eh_json_features as F


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_OUT_DIR = os.path.join(REPO_DIR, "data", "sim")
MERGED_BAM_DIR = os.path.join(SIM_OUT_DIR, "merged_bams")
CATALOG_DIR = os.path.join(SIM_OUT_DIR, "catalogs")
EH_JSON_DIR = os.path.join(SIM_OUT_DIR, "eh_json")
OUTPUT_PARQUET = os.path.join(SIM_OUT_DIR, "sim_rows.parquet")

# low-mem-streaming = full-branch source; optimized-streaming = fast-branch source.
ANALYSIS_MODES = ["low-mem-streaming", "optimized-streaming"]

# Columns appended to every extractor row (extractor already sets sample_id and coverage).
ADDED_COLUMNS = ["true", "source", "purity", "is_negative_locus", "chrom_raw", "analysis_mode"]


def run_eh_json(reads_bam, reference, catalog, output_prefix, mode):
    """Runs ExpansionHunter and returns the path to its ``<prefix>.json`` output.

    The run is skipped (and the cached JSON returned) when ``<prefix>.json``
    already exists, so the whole pipeline is idempotent. Raises
    ``CalledProcessError`` on a non-zero exit.
    """
    json_path = output_prefix + ".json"
    if os.path.exists(json_path):
        return json_path
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    subprocess.run(
        [M.EH_BINARY, "--reads", reads_bam, "--reference", reference,
         "--variant-catalog", catalog, "--output-prefix", output_prefix,
         "--analysis-mode", mode, "--sort-catalog-by", "position"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json_path


def rows_for_run(json_path, sample_id, truth_sorted, chrom_raw, analysis_mode):
    """Extracts per-allele rows from one EH JSON and annotates them with sim labels.

    ``truth_sorted`` is the sorted ``(n1, n2)`` diploid truth; each row's ``true``
    is ``truth_sorted[allele_rank]`` (rank-pairing). Returns a list of row dicts.
    """
    rows = []
    for row in F.extract_rows(json_path, sample_id=sample_id):
        rank = row["allele_rank"]
        row["true"] = float(truth_sorted[rank] if rank < len(truth_sorted) else truth_sorted[-1])
        row["source"] = "sim"
        row["purity"] = 1.0
        row["is_negative_locus"] = False
        row["chrom_raw"] = chrom_raw
        row["analysis_mode"] = analysis_mode
        rows.append(row)
    return rows


def generate_rows():
    """Runs the full sim pipeline over all loci and returns (rows, stats).

    ``rows`` is a list of per-allele row dicts; ``stats`` is a dict with counts
    used for the run summary (loci processed, diploid samples, per-mode/branch row
    counts, fast-path trigger numbers, and any errored (locus, pair, mode) units).
    """
    reference = M.find_reference()
    if reference is None:
        raise SystemExit("No hg38 reference FASTA (+.fai) found among candidates.")
    if not (os.path.exists(M.EH_BINARY) and os.access(M.EH_BINARY, os.X_OK)):
        raise SystemExit(f"ExpansionHunter binary not found/executable: {M.EH_BINARY}")
    if not os.path.isdir(M.SIM_DATA_DIR):
        raise SystemExit(f"Simulated data dir not found: {M.SIM_DATA_DIR}")

    for d in (MERGED_BAM_DIR, CATALOG_DIR, EH_JSON_DIR, SIM_OUT_DIR):
        os.makedirs(d, exist_ok=True)

    loci = sorted(filter(None, (
        M.parse_locus_dir(d) for d in os.listdir(M.SIM_DATA_DIR)
        if os.path.isdir(os.path.join(M.SIM_DATA_DIR, d)))))

    rows = []
    stats = {"loci": 0, "samples": 0, "errors": [],
             "opt_variants": 0, "opt_quick_variants": 0}
    for locus_id, chrom, start, end, motif in loci:
        locus_dir = os.path.join(M.SIM_DATA_DIR, locus_id)
        pairs = M.select_pairs(M.allele_sizes_in_dir(locus_dir))
        if not pairs:
            continue
        stats["loci"] += 1
        catalog = os.path.join(CATALOG_DIR, f"{locus_id}.catalog.json")
        if not os.path.exists(catalog):
            M.write_catalog(locus_id, chrom, start, end, motif, catalog)

        for n1, n2 in pairs:
            merged_bam = os.path.join(MERGED_BAM_DIR, locus_id, f"a1_{n1}__a2_{n2}.bam")
            M.build_merged_bam(
                os.path.join(locus_dir, f"sim_{n1}x__10_150_450_50.bam"),
                os.path.join(locus_dir, f"sim_{n2}x__10_150_450_50.bam"),
                merged_bam)
            stats["samples"] += 1
            sample_id = f"sim_{locus_id}_{n1}_{n2}"
            truth_sorted = tuple(sorted((n1, n2)))
            for mode in ANALYSIS_MODES:
                prefix = os.path.join(EH_JSON_DIR, locus_id, f"{n1}_{n2}.{mode}")
                try:
                    json_path = run_eh_json(merged_bam, reference, catalog, prefix, mode)
                except subprocess.CalledProcessError as e:
                    stats["errors"].append((locus_id, f"{n1}/{n2}", mode, str(e)))
                    continue
                run_rows = rows_for_run(json_path, sample_id, truth_sorted, chrom, mode)
                rows.extend(run_rows)
                if mode == "optimized-streaming":
                    seen = set()
                    for r in run_rows:
                        key = (r["sample_id"], r["locus_id"], r["variant_id"])
                        if key in seen:
                            continue
                        seen.add(key)
                        stats["opt_variants"] += 1
                        if r["quick_genotype"]:
                            stats["opt_quick_variants"] += 1
    return rows, stats


def main():
    """Generates sim rows, writes the parquet, and prints a run summary."""
    rows, stats = generate_rows()
    if not rows:
        raise SystemExit("No rows were produced -- check sim data / EH binary / reference.")

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\nWrote {len(df)} rows to {OUTPUT_PARQUET}")
    print(f"Loci processed:   {stats['loci']}")
    print(f"Diploid samples:  {stats['samples']}")
    print("\nRows per (analysis_mode, genotyping_branch):")
    for (mode, branch), n in sorted(
            df.groupby(["analysis_mode", "genotyping_branch"]).size().items()):
        print(f"  {mode:<20} {branch:<5} {n:>7}")
    if stats["opt_variants"]:
        rate = stats["opt_quick_variants"] / stats["opt_variants"]
        print(f"\nFast-path trigger rate (optimized-streaming QuickGenotype=true): "
              f"{stats['opt_quick_variants']}/{stats['opt_variants']} = {rate:.3f}")
    if stats["errors"]:
        print(f"\n{len(stats['errors'])} errored (locus, pair, mode) units:")
        for locus_id, pair, mode, err in stats["errors"]:
            print(f"  {locus_id} {pair} {mode}: {err}")
    else:
        print("\nNo errors.")


if __name__ == "__main__":
    main()
