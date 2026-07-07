# Plan: migrate training/held-out truth ingestion off the stale `for_comparison` TSV

## Background

`dataset.py`'s `_join_truth()` attaches ground-truth allele sizes to EH-called alleles by
joining our own fresh JSON extraction (`json_df`, from `eh_json.extract_rows`) against each
sample's `*.tandem_repeat_genotypes.for_comparison.with_EHv5-bw2-optimized_vs_Truth_columns.alleles.tsv.gz`
(the "for_comparison TSV"), downloaded from `gs://str-truth-set-v2/tool_results/...`.

That TSV is stale relative to the current JSON: it was built from an older snapshot of the
`json/` shards than what's in the bucket today. Confirmed via `dataset._assert_catalog_agreement`
(added earlier this session, currently `fatal=False`): every existing training combo shows
0 loci "only in JSON" and a growing "only in TSV" gap at large allele sizes (73-85% missing at
>=200 truth repeats, universally across HG002 10x/20x/31x and CHM1_CHM13). Root-caused by
finding a locus present in the sample's own `*.alleles.tsv.gz` JSON-flattening (same directory,
implying it reflects an older JSON run) but absent from the current `json/` shards entirely
(confirmed absent via `grep` on the raw JSON text, not just an extraction bug).

Two fix paths were identified:
1. **Upstream fix** — regenerate the `for_comparison` TSVs in `str-truth-set-v2` against
   current `json/` shards. Not pursued: unfamiliar codebase, found only the legacy v1 pipeline
   (`run_step_D__combine_tool_results.sh`), not the current v2 generator.
2. **Local re-derivation** (this plan) — stop depending on the `for_comparison` TSV's truth
   values; join against a tool-independent truth source instead.

## The replacement source

`gs://str-truth-set-v2/filter_vcf_v2/<sample>/<sample>.tandem_repeat_genotypes.tsv.gz` — built
directly from the dipcall/long-read-assembly pipeline, never tied to any specific EH run, so it
cannot go stale relative to one. One file per **sample** (not per sample+coverage — truth doesn't
depend on short-read coverage).

**Validated** (this session, read-only, no production changes):
- Schema: `LocusId`, `Zygosity` (HET/HOM/HEMI), `NumRepeatsShortAllele`, `NumRepeatsLongAllele`,
  `RepeatPurityShortAllele`, `RepeatPurityLongAllele`, plus sequence/motif columns we don't need.
- HOM and HEMI (male chrX) rows have `NumRepeatsShortAllele == NumRepeatsLongAllele` (no NaNs
  anywhere in either column) — safe to always treat as two allele rows, no special-casing needed.
- Cross-checked against HG002 31x's existing `for_comparison` TSV: **578,296 common loci, 0 value
  mismatches**, and the new source has **4,875 more loci** than the old TSV (consistent with being
  the fuller, current catalog).
- Contains **only variant (non-reference) loci** — no negative-control loci at all. Per explicit
  instruction, negative-locus handling is dropped: every row derived from this source gets
  `is_negative_locus = False`.

## Scope boundary (what this does NOT fix)

`accuracy_by_size.py`'s report-generation join (`categorize_tsv` + `add_corrected_categories`,
used for the "Accuracy by true allele size" plots on `hg002_genome`/`hg002_exome`) needs the
**tool's own** comparison columns (`DiffRepeats`, `category`, etc.), not just truth values. The
tool-independent source has no tool-comparison columns at all, so it can't replace that join.
Those report plots will keep showing the same residual All-vs-p<0.5+p>=0.5 gap after this change.
This plan only fixes **training-pool assembly** (`dataset.build_combo`) and **held-out-sample
building** (`heldout.build_sample`) — i.e., what the model actually trains and validates on.

## Implementation steps

### 1. `dataset.py`: new source + loader

- Add `TRUTH_CATALOG_ROOT = "gs://str-truth-set-v2/filter_vcf_v2"`.
- Add `_truth_genotypes_tsv_remote(sample)` → `"%s/%s/%s.tandem_repeat_genotypes.tsv.gz" % (TRUTH_CATALOG_ROOT, sample, sample)`.
- Add `_load_truth_from_genotypes_tsv(tsv_path)`:
  - Read `LocusId`, `NumRepeatsShortAllele`, `NumRepeatsLongAllele`, `RepeatPurityShortAllele`,
    `RepeatPurityLongAllele`.
  - Reshape wide → long: `allele_rank=0` row uses the Short columns, `allele_rank=1` row uses the
    Long columns (ascending-by-truth-value pairing; simpler and more self-contained than the old
    tool_size-based sort, and doesn't depend on any EH-run-specific ordering).
  - Set `is_negative_locus = False` for every row.
  - Return the same `{LocusId, allele_rank, true, purity, is_negative_locus}` contract
    `_join_truth` already expects — a drop-in replacement, no changes needed to `_join_truth`
    itself or to `_assert_catalog_agreement`.

### 2. `heldout.py`: full swap

`build_sample()`'s downloaded `for_comparison` TSV is used *only* for `_join_truth` — nothing
else in the held-out path touches it (the `heldout43` report dataset is parquet-only, no TSV
involved). Clean full replacement:
- Replace `tsv_remote = ".../for_comparison..."` + `dataset._download(...)` +
  `dataset._load_truth_tsv(tsv_local, VARIANT)` with
  `dataset._download([dataset._truth_genotypes_tsv_remote(sample)], dl_dir)[0]` +
  `dataset._load_truth_from_genotypes_tsv(...)`.
- `dl_dir` here is already keyed by sample only (`_downloads/<sample>/`), so no download-sharing
  changes needed.

### 3. `dataset.py`: partial swap for `build_combo`

Unlike held-out, `gen_datasets.py`'s report path for `hg002_genome`/`hg002_exome` still reads the
`for_comparison` TSV that `build_combo` downloads (for the tool-comparison columns — see Scope
boundary above). So:
- **Keep** the existing `_truth_tsv_remote` download (still needed as a side effect for report
  generation).
- **Add** a second download of `_truth_genotypes_tsv_remote(sample)` into a sample-level (not
  sample+coverage-level) dest dir, e.g. `os.path.join(data_dir, subdir, "_downloads", sample)`,
  so HG002's three coverages (10x/20x/31x) share one download instead of fetching it three times.
- Feed `_load_truth_from_genotypes_tsv(...)`'s output into `_join_truth` instead of
  `_load_truth_tsv(...)`'s.

### 4. Remove dead code

Once both call sites stop calling it, `_load_truth_tsv` has no remaining callers — remove it.
`_truth_tsv_remote` stays (still used by `build_combo` for the report-side download).

### 5. Re-validate before trusting

Before retraining on the new truth values:
- Re-run `dataset._assert_catalog_agreement` (still wired into `_join_truth`) on a fresh
  `build_combo`/`build_sample` run and confirm the global/per-bin mismatch collapses (expect
  near-0%, not the current 73-85% at the largest bin) — this is the direct before/after proof the
  fix works.
- Spot-check that alleles previously missing from training (e.g. the `1-875829-876434-...`
  example) now flow through `label_and_filter` instead of silently vanishing.

### 6. Rebuild + retrain + re-report

Same shape as earlier today's promoted-sample work:
- `--force` rebuild the 4 existing training combos (HG002 10x/20x/31x, CHM1_CHM13 46x) so they
  pick up the new truth source (existing cached parquets won't rebuild without `--force`).
- Rebuild the 43 held-out samples' parquets similarly (`--force`), including the 13 already
  promoted into training — they need the corrected truth too.
- Re-run `train.py` (new model artifact, don't overwrite today's `_plus13` model).
- Re-run `report.py` (full 5-fold CV) + `gen_datasets.py` for all 3 datasets.
- Compare held-out accuracy at large allele sizes before/after — expect a real improvement at
  the `full_nonspanning` large-size tail, since training now sees the previously-invisible large
  truth loci.

## Risks / things to watch

- **Rank-pairing assumption**: ascending-by-truth-value (short=rank0, long=rank1) matches
  ascending-by-EH-call-value in the normal case, but could occasionally mis-pair for a badly
  miscalled heterozygous locus where EH's own reported order is inverted relative to truth. Rare;
  arguably more principled than the old tool-order-dependent pairing anyway (doesn't chase a
  possibly-wrong tool ordering), but worth a spot-check on a few heterozygous loci post-rebuild.
- **Negative-control loci**: now entirely unfiltered from this source (`is_negative_locus` always
  False) per explicit instruction to ignore them. If any of the 4 training combos' negative-control
  loci were meaningfully affecting training composition, that filtering signal is gone. Not
  expected to matter much (negative controls are a small, deliberately-inert QC subset) but
  flagging since it's a real behavior change, not just a data-freshness fix.
- **Compute cost**: this is a `--force` rebuild of the full training pool (~17 combos including
  the 13 promoted samples) + a full retrain + full report regen — comparable in wall-clock cost to
  today's earlier retrain (order of an hour+ for the 5-fold CV alone).
