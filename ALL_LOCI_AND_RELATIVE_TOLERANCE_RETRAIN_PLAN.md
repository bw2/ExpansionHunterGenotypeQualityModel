# Plan: retrain the genotype-quality model on all loci, with a relative-tolerance pOk

*Written 2026-10-20. Replaces `ALL_LOCI_TRAINING_PLAN.md` and `pOk_relative_tolerance_label_plan.md`; those two
files stay for their background and measurements, but progress and decisions are recorded here. Status
(2026-10-26): steps 0-4 done; step 5's held-out views done for 9 models, including the three D3 (b) retrains.
A with D3 (b) still fails step 6's gated-error rule by 0.4-3.7% in 4 of 6 strata. The 10 cohort samples and the
gnomAD comparison are not started. D3 (b) code, compare_retrain_variants.py and the compare_models.py refactor
are reviewed but uncommitted.*

## Two changes, one rebuild

| | Change A: train on all loci | Change B: relative-tolerance pOk |
|---|---|---|
| Problem | Male chrX/chrY calls get low pOk; pOk ranks fast-path misses worse than spanning depth alone (ROC-AUC 0.725 vs 0.855) | On 2,719 manually reviewed gnomAD genotypes, pOk collapses for alleles of 270 bp and above (AUC 0.63 vs Q's 0.90) |
| Cause | Training keeps only loci where the person's truth differs from the reference (about 15% of loci) | The OK tolerance (exact below 50 bp, up to +-4 repeats at 270 bp and above) is almost never met by long calls: 83% of training alleles of 270 bp or more are labeled TOO_SHORT |
| Evidence so far | Pilot on 10 cohort samples: ROC-AUC 0.920-0.931 and 31-34% of misses in the lowest 1%, vs 0.725 and 7.8% | By allele length, pOk beats Q at 50-270 bp (AUC 0.83-0.92 vs 0.69-0.87) and loses outside that range |
| ExpansionHunter change | None beyond embedding the new model | Model reader, the pOk/pTooShort/pTooLong output, `docs/07_AlleleQualityMetrics.md` (the three would no longer sum to 1) |
| Details | `ALL_LOCI_TRAINING_PLAN.md` | `pOk_relative_tolerance_label_plan.md` |

Both changes need new training tables built from re-genotyped JSONs, so they share the re-genotyping, the table
rebuild and the evaluation. They ship separately (step 6), so a failure of one doesn't hold back the other.

## Who does what

- **One session owns all code in this repository** (`dataset.py`, `eh_json.py`, `features.py`, `model.py`,
  `train.py`, `heldout.py`, `report.py` and their tests). Two sessions editing the same uncommitted files at once
  is the thing to avoid.
- **The gnomAD session owns only the gnomAD manual-review comparison**, in
  `~/p1_iter3/2023_11_24__updated_gnomad_callset/manual_review_vs_EH_pOk/`. It reads model files from this
  repository's `model/` folder and does not edit files here.

## Decided (2026-10-20)

- **A:** training and calibration rows come from all comparable loci. Loci where the truth equals the reference
  for both alleles are sampled while streaming each JSON (`--keep-hom-ref-loci`), and each row records its keep
  probability so later steps can draw rows with equal probability regardless of variant status.
- **A:** one-size calls (homozygous or one-allele) are labeled against both truth alleles (`dir_code_genotype`):
  OK only if within tolerance of both, otherwise the direction of the allele missed by more.
- **B:** one direction classifier with 4 classes: OK, PAST_LIMIT, TOO_SHORT, TOO_LONG. Reported as
  pOk = P(OK) + P(PAST_LIMIT), pTooShort = P(TOO_SHORT) + P(PAST_LIMIT), pTooLong = P(TOO_LONG).
- **B:** the limit is the locus's own FragmentLength (FL) from the ExpansionHunter JSON. The model also gets
  `eh_bp / FL` as an input.
- **B:** label, per allele, with `true_bp = motif_size * round(true)`, `eh_bp = motif_size * round(eh)`,
  `dr = round(eh) - round(true)`:
  - if `true_bp < FL`: `tol = max(1, x * round(true))` repeats; OK if `|dr| <= tol`, TOO_LONG if `dr > tol`,
    TOO_SHORT if `dr < -tol`;
  - if `true_bp >= FL` (one-sided): TOO_LONG if `eh > (1 + x) * true`, TOO_SHORT if `eh_bp < (1 - x) * FL`,
    otherwise PAST_LIMIT.
- **B:** x = 10% and x = 15% are both trained and compared, unless D7 picks one first.
- **A + B (decided 2026-10-25):** homozygous calls keep A's genotype-level rule with B's labels. The call takes
  the label of the truth allele it misses, or of the bigger miss if it misses both. If it misses neither but is
  PAST_LIMIT against the other allele, the label is PAST_LIMIT. Labeling the two copies as separate rows was
  considered and rejected: their inputs are identical, so a missed expansion on one haplotype could not get a pOk
  below about 0.5. With x = 10%, the genotype-level rule changes 0.16% of fast-path and 0.45% of
  full-genotyper representative rows; the PAST_LIMIT case alone is 283 rows.
- **Both:** the length head (q-median, which gives PredictedLengthCorrectionFactor, LCF) keeps its target.
- **Both:** `strand_bias_phred` returns to the full-genotyper features with the rebuild. The 10-17 change that
  removed it (`features.py`, `features_tests.py`, `report.py`, `report_tests.py`, committed as `97772dc`) is
  reverted at step 3, and the rebuilt full-genotyper rows must show the fixed binary's values (median about 2, not
  about 30).
- **Order:** implement and review the extraction code before re-genotyping (step 1 before step 2).
- **Extra stored columns** for later experiments (step 1 item 4), not model inputs.
- **The LCF-by-depth check** (step 0) is in scope.

## Decisions still needed

- **D1. Re-genotype all 135 samples** (about $155, 10-12 h on Hail Batch), after step 1's code is reviewed.
- **D2. Which regimes change A covers.** (a) All three (same code path; full-genotyper calls on male chrX/chrY
  show a smaller version of the same pOk drop). (b) Fast path only first.
- **D3. Rows for the LCF head**, which is applied only where pOk < 0.5. (a) Rows where the out-of-fold gate fires,
  from the representative rows only (faithful to deployment, but about 0.8% of calls). (b) Rows where the gate
  fires from all kept rows, including every variant-locus row. Decide by the variant-view gated error. Change B
  moves this too: it changes which rows have pOk < 0.5 (see Risks). Until decided, `train.py --training-population
  all_loci` does a third thing: rows where the gate fires among the all-loci table's training rows (representative
  rows plus the size top-up), as today's recipe does on its own table. (b) needs a `train.py` change that takes the
  LCF head's rows from today's variant-locus table (`data/parquet/`), but no new extraction: the all-loci table
  leaves out the variant-locus rows outside its sample, and the per-source parquets keep them all.
  - **2026-10-26: (b) implemented after step 5 showed A's gated error worse in all 6 strata (user's choice).**
    - `train.py --length-head-training-population variant_loci`: the direction head trains on the all-loci table.
      The q-median head trains on the variant-locus table's rows where the gate fires, judged out of fold by
      direction heads fit on the other half of the people. Both tables hold out the same calibration people.
    - `/code-review-until-done`: 2 rounds, 2 fixes (a test that could not catch an ignored `people` argument;
      an unused variable), then clean. 3% check run: the fast-path q-median head gets 5.9% of the variant-locus
      pool, the baseline's rate.
    - A, A + B 10% and A + B 15% retrained with it as
      `model/..._all_loci_dir_code_genotype[_relative_tolerance_{10,15}pct]_length_head_variant_loci.20261026.json.gz`.
    - Result (step 5): better than the same model before in all 6 strata (81 of 81 samples in 4 of them), closing
      most of A's gap. A is still worse than 10-17 in 4 strata and tied in 2. See step 5.
- **D4. Sampling rate for loci where truth equals the reference.** Implemented with first rates (fast path 2% to
  100% by called-size bin, full genotyper 100%). On HG002 10x (old JSON, test only, after the review fixes): 1.90M
  rows, 22% more than today (fast path +245k rows, full genotyper +99k), peak 5.0 GB for the whole test script;
  the variant-locus rows are identical to today's parquet. In the representative sample, fast-path
  reference-length rows are 97.0% OK and male chrX/chrY exact-reference one-allele rows 99.5% (26.3% in today's
  pool). Confirm or adjust after review.
- **D5. A separate "hidden variant" head** for the early warning. Deferred: most of its edge is on misses with a
  zero-repeat truth allele (likely truth artifacts), and it needs a new ExpansionHunter output.
- **D6. Where the table rebuild runs. Decided 2026-10-23: (b) Hail Batch.** (a) Locally, one source at a time (the 2026-10-02 way: about a day, about
  95 GB downloaded and deleted, about 6 GB peak per source on this RAM-limited Mac, under the memory guard).
  (b) Hail Batch, one job per sample next to the data (no download, about an hour, a few dollars; needs new job
  code that ships this repository's extraction code into the job).
- **D7. An early, research-only prototype of change B. Decided 2026-10-23: dropped; step 4 trains both x = 10%
  and x = 15% (6 trainings).** It would have run while step 2's re-genotyping ran: train the 4-class
  head on today's tables with each sample's median FL (read from the start of each old JSON, about 20 MB each)
  instead of the per-locus FL, and score it on the gnomAD reviews. It would pick x and show whether the 270 bp+
  collapse goes away before the rebuild, cutting step 4 from 6 trainings to 4. Limits: old build, variant-only
  tables (so the male chrX/chrY flaw remains), and per-sample FL (on HG002 31x, 90% of loci are within about 5%
  of the sample median, less than the tolerance). Never embedded.

## Step 0: does LCF hurt at low depth? (now, no rebuild)

The June REViewer deep-dive found 4 of 20 failures at HG002 10x: with 3-8 reads, the length correction overshot
or went the wrong way. Coverage has been a model input since 2026-10-01 (per-locus flank depth), but nothing makes
the correction smaller when reads are few, and nobody has rechecked those cases with the current model.

- **Data:** the deployed 10-17 model on the 81 held-out samples' tables in `data_eval_43/real_43/` (excluding the
  50 promoted to training). These come from the same build as the model's training data, so they are consistent
  with it. If `train.py`'s split shows the LCF head never trained on HG002 chr7, HG002 10x chr7 is a second,
  low-coverage test set.
- **Measure:** for each regime and homopolymer vs not, bin the rows where LCF is applied (pOk < 0.5) by per-allele
  `depth` and by `spanning_total`. Per bin: rows, share moved closer and share moved further from the truth, mean
  absolute error in repeats before and after, and distance reduction.
- **Files:** `lcf_effect_by_depth.py` with `lcf_effect_by_depth_tests.py`; report
  `report/lcf_effect_by_depth.2026-10-17_plus50_model.html`.
- **Outcome:** if LCF is net harmful below some depth, add the simplest fix to the retrain: a minimum depth for
  applying LCF, stated in `docs/07_AlleleQualityMetrics.md` (or ExpansionHunter omits LCF below it, a small C++
  change). Shrinking LCF toward 1 at low depth, or adding spanning-mode inputs, are larger changes and stay out
  of scope unless this result calls for them. If LCF is not harmful at low depth, no change.
- **Result (2026-10-23): not harmful at low depth, so no change.** 10.7M corrected calls of the 10-17 model
  (`lcf_effect_by_depth.py`, report `report/lcf_effect_by_depth.2026-10-17_plus50_model.html`). In the 81 held-out samples
  the correction helps in every depth and spanning-read bin of every regime and motif stratum, the lowest included
  (depth below 2: +5% quick non-homopolymer, +15% quick homopolymer, +33% to +95% full genotyper; distance
  reduction, intervals clear of 0). HG002 chr7 at 10x/20x/31x agrees: at 10x, depth 2-8, quick calls gain 45-51%.
  Weakest spot: quick homopolymer calls with a single spanning read (held-out +3%, 34% of calls closer vs 27%
  further; about even in HG002). One bin is net harmful, HG002 31x quick homopolymer with depth below 2 (394
  calls, -21%), but there 68% of the gated calls were already exact, so the gate fired on right calls rather than
  the LCF overshooting; the same bin is positive at 10x and 20x and in all 81 held-out samples.

## Step 1: code changes in this repository

1. **A: keep sampled loci where truth equals the reference.** Done, under review: `--keep-hom-ref-loci` in
   `dataset.py` and `heldout.py` (also reaching the 50 promoted training samples), columns `variant_locus`,
   `all_loci_keep_probability`, `in_all_loci_pool`, `all_loci_representative`. Readers that use today's recipe
   keep only variant-locus rows. Changed in review rounds 1-2: the option now reaches the 50 promoted samples;
   loci are sampled per individual (hash of individual and LocusId, so each person contributes different loci and
   one person's coverages share them); loci with no called allele are left out of the all-loci sample.
2. **A: genotype-level label for one-size calls** (`dir_code_genotype`, with `true_other_allele`). Done, under
   review. When both truth alleles are missed, the one missed by more repeats wins (the longer one on a tie).
3. **A: assembly, training options and the all-loci evaluation view** (`ALL_LOCI_TRAINING_PLAN.md` step 1
   items 3-5). Done, under review:
   - `dataset.py --training-population all_loci` (repeatable with `variant_loci`, today's and the default)
     writes `data/parquet_all_loci/{quick,full}.parquet`. Per regime, the representative rows are the
     `all_loci_representative` rows (uniformly capped as today), and the training-only size top-up comes from
     the `in_all_loci_pool` rows (the whole pool when it fits within the cap). It refuses per-source parquets
     built without `--keep-hom-ref-loci`, and `dataset.py` refuses `--training-population all_loci` without
     `--keep-hom-ref-loci` before building anything. Rebuild hints in error messages no longer suggest
     `--force` for the branch tables, and add `--keep-hom-ref-loci` wherever a rebuild without it would drop the
     hom-ref loci.
   - `train.py --training-population` and `--direction-label dir_code_genotype`. A non-default choice goes into
     the default model file name, so it never overwrites today's default model.
   - Evaluation: `heldout.all_loci_early_warning`, printed by `compare_models.py` after its verdict. It covers
     the representative fast-path loci of held-out parquets built with `--keep-hom-ref-loci`. It reports
     ROC-AUC of 1 - pOk for misses among reference-length calls, the share below pOk 0.5, and recall of misses
     at 1/2/5/10% flag rates (all, without chrX/chrY, by sex).
   - Smoke test on HG002 10x (old JSON, 2026-10-20): fast-path reference-length training rows labeled OK
     84.4% per allele / 76.0% genotype-level in the variant-locus table, against 96.3% / 94.6% in the all-loci
     table. The deployed 10-17 model on this sample (a training source, so a smoke test only): 98,853
     representative reference-length calls, 848 misses, ROC-AUC 0.62, 14% of misses in the lowest 1%.
4. **Extra columns stored at extraction**, so later experiments need no second rebuild. Done, under review
   (`eh_json.py`; `truth_reference_repeats` from the truth TSV in `dataset.py`); on HG002 10x every called row
   has `fragment_length` (median 509) and every row `truth_reference_repeats`:
   - `fragment_length`: the locus's FragmentLength from the JSON (change B needs it).
   - `locus_id`, kept in the tables (for per-locus joins and change B's analysis).
   - `eh_minus_spanning_mode_naive`, `spanning_mode_naive_support` and `het_gap`. Stored only, not model inputs.
     Same definitions as `original/eh_json_features.py` (`spanning_mode_features`, `_sister_eh`), now computed
     from the fixed spanning-read counts:
     - the naive spanning mode is the size with the most spanning reads, over all spanning reads at the locus,
       after dropping sizes seen in only 1 read when any size has 2 or more (ties go to the first listed size);
       `eh_minus_spanning_mode_naive` is the allele's call minus that size, and `spanning_mode_naive_support` is
       its read count (0 and empty when there are no spanning reads);
     - `het_gap` is the allele's call minus the other allele's call (signed), empty for one-allele calls; with
       more than 2 alleles, the other allele nearest in size.
     Testing them as inputs later happens in Python; shipping them would need ExpansionHunter C++ changes
     (`GenotypeQualityFeatures`).
5. **B: label function** in `features.py`, next to the current one, with x as a parameter.
6. **B: 4-class direction head** with per-class calibration in `model.py`, the reported sums, and a new model
   file format version.

   Items 5-6 done 2026-10-25 (uncommitted, waiting for the user's review):
   - `features.relative_tolerance_direction_codes`: a missing or 0 FL means no limit.
   - `features.genotype_level_direction_codes`: now shared with `dir_code_genotype`.
   - The `eh_bp_over_fragment_length` input.
   - `model.reported_probabilities`, `predict_reported` and `predict_reported_json`.
   - `train.py --relative-tolerance X`: writes `format_version` 3 with a `relative_tolerance` key.
   - The serialized direction head names its classes, and lists PAST_LIMIT only if its training rows had some.
     Only full_nonspanning does; the fast path and full_spanning have none.
   - Every reader of a saved model uses the reported pOk.
   - `/code-review-until-done`: 1 round, no findings from the three reviewers. The two pandas-lint items were left
     open as Controversial (missing waiver comments at `features.py:142-143`). 292 tests pass.
7. **Tests** for each item: loci where truth equals the reference survive extraction with the right truth; the
   genotype-level label on hand-made one-size calls; sampling that doesn't depend on variant status; every branch
   of the B label (including the 1-repeat floor and the one-sided rule past FL); the 3 stored columns on
   hand-made histograms (ties, single-read sizes, no spanning reads, one-allele calls).
8. **Review:** `/code-review-until-done`, then the user reviews before step 2.

## Step 2: re-genotype (D1)

- Run `run_expansion_hunter_on_selected_samples.py` (tandem-repeat-explorer) on all 135 samples with the current
  `weisburd/str-analysis-with-expansion-hunter` image (ExpansionHunter-bw2 e90307c), to a new directory, for
  example `gs://tandem-repeat-explorer/tool_genotype_quality/expansion_hunter_v2.1_eh_e90307c/`. The 2026-10-02
  run cost $152.84 and took about 10 h, plus a rerun of 11 samples that ran out of memory at 13 GB.
- Rerun the comparison step for the 10 cohort samples (about $1).
- Point `dataset.EH_RESULTS_ROOT` at the new directory.
- If D7 is approved, the B prototype runs during this step.
- **Result (2026-10-23): done.** All 135 samples re-genotyped with EH e90307c (Hail Batch 8450427 and 8450428)
  into `gs://tandem-repeat-explorer/tool_genotype_quality/expansion_hunter_v2.1_eh_e90307c/`.
  `dataset.EH_RESULTS_ROOT` was switched there at step 3. The 10 cohort samples' comparison rerun is needed only
  by step 5, so it moved there.

## Step 3: rebuild the tables (D6)

- All 54 training sources and 81 held-out samples, with step 1's extraction.
- Undo the 10-17 strand-bias change so full models use `FULL_FEATURES` again. It was committed on 2026-10-21
  as `97772dc` (to match the deployed 10-17 model), so this is `git revert 97772dc`, not a stash.
- Checks before training: median full-genotyper `strand_bias_phred` about 2; `fragment_length` present at every
  locus; the 3 stored columns filled where expected; variant-locus rows match what today's recipe would have kept.
- Before step 4: make `report.report_html_name` keep what sits between
  `CHM1_CHM13` and the date. Today it keeps only the date, so step 4's same-day models (`train.py` names them
  `..._all_loci.<date>.json.gz` and so on) would all write `report/model_report.<date>_model.html` and overwrite
  each other's report. Until then, pass each one its own `--out-dir`. (Found in review on 2026-10-21, when
  `report.py` still held the uncommitted 10-17 change. That change is now `97772dc` and touches only
  `_feature_glossary`, so this fix can go in any time without conflicting with the revert above.)
- **Result (2026-10-24): done; the code changes are uncommitted.**
  - Code: the inverse of `97772dc` (applied by hand; `97772dc`'s own diff applies cleanly on top again);
    `report_html_name` now gives `model_report.2026-10-24_all_loci_model.html` and `..._plus50_model.html`;
    `label_and_filter` keeps `locus_id`; `train_model.sh` passes `--keep-hom-ref-loci`; the new
    `build_per_source_parquets_on_hail_batch.py` builds the 135 per-source parquets, one Hail Batch job each,
    with the same `dataset.write_per_source_parquet` the local builders use. `/code-review-until-done` ran twice
    (on the existing step 3 code, then on these changes), each ending with a clean round.
  - Batches: 8450641 (HG00738, $0.02) and https://batch.hail.is/batches/8450644 (the other 134, $3.38, 19 min),
    into `gs://tandem-repeat-explorer/tool_genotype_quality/genotype_quality_model_parquets_from_expansion_hunter_v2.1_eh_e90307c/`.
    `--download` put them at their local paths, and `dataset.py` and `heldout.py --build-only` reused all 135
    without extracting again.
  - Checks: all 135 are EH 053dcd8 builds. `fragment_length`, `locus_id`, the mode difference, `het_gap` and
    `truth_reference_repeats` are never missing where they should be present. The median full-genotyper
    `strand_bias_phred` is 2.36 per source (2.17-2.94; HG002 10x 0.0 and 20x 1.6, where depth is low), against
    33.1 in the old tables. HG00738's variant-locus rows equal a variant-only extraction of the same JSON
    (1,571,662 rows, all 40 columns).
  - Tables: `data/parquet/` quick 7,024,500 and full 8,803,581 rows (the b1fbc23 tables: 7,033,619 and
    8,627,113); `data/parquet_all_loci/` quick 8,129,066 and full 10,675,371 rows. Assembly peaked at 9.0 GB.
  - The b1fbc23 tables are kept as `data/real_quick.eh_b1fbc23`, `data/parquet.eh_b1fbc23` and
    `data_eval_43/real_43.eh_b1fbc23`.

## Step 4: train

On the same tables, one at a time under the memory guard (40-60 min each):

1. Baseline: today's recipe (variant loci, 3-class label), with strand bias restored.
2. A.
3. B (x from D7; otherwise both 10% and 15%).
4. A + B (same x).

That is 4 trainings, or 6 without D7.

Progress:
- **Baseline, done 2026-10-24:** `model/genotype_quality_model_from_HG002_and_CHM1_CHM13.20261024.json.gz`.
  35 min, 6.0 GB peak.
- **A, done 2026-10-25:** `model/genotype_quality_model_from_HG002_and_CHM1_CHM13_all_loci_dir_code_genotype.20261024.json.gz`.
  Trained from the committed code (4dbbbfb). 58 min, 7.3 GB peak.
  - A first run was stopped by the memory guard when macOS memory pressure stayed at warning level for 30 s,
    with 43% of memory free. The rerun ignored warning-level pressure, with the user's OK for that run.
- **B and A + B, done 2026-10-25:** trained one at a time from a snapshot of the uncommitted B code
  (`scratchpad/step4/code_with_change_b`), each in 38-75 min, peak 9.0 GB. All are `format_version` 3, and every
  regime passed the round-trip check. Only the full_nonspanning heads have 4 classes.
  - `model/genotype_quality_model_from_HG002_and_CHM1_CHM13_relative_tolerance_{10,15}pct.20261025.json.gz`
  - `model/genotype_quality_model_from_HG002_and_CHM1_CHM13_all_loci_dir_code_genotype_relative_tolerance_{10,15}pct.20261025.json.gz`
- **Rows the fast-path LCF head trains on** (rows where the out-of-fold gate fires; D3):

  | Model | Rows |
  |---|---|
  | baseline | 377,577 |
  | A | 234,059 |
  | B 10% | 140,419 |
  | B 15% | 109,818 |
  | A + B 10% | 90,450 |
  | A + B 15% | 72,402 |
  | A, LCF on variant loci (D3 b) | 462,745 |
  | A + B 10%, LCF on variant loci | 166,462 |
  | A + B 15%, LCF on variant loci | 143,266 |

  The full-genotyper heads still get their 1M-row cap. The three D3 (b) models trained on 2026-10-26 in 29-33 min
  each, peak 11.3 GB, and every regime passed the round-trip check. Step 5's variant-locus view must show whether the fast-path
  length correction suffers.

## Step 5: evaluate

Every variant, plus the deployed 10-17 model:

- **Variant-locus view:** today's metrics, including gated mean absolute error in the 6 regime x motif strata,
  each model using its own pOk < 0.5 gate.
- **All-loci view, 81 held-out samples:** ROC-AUC for misses, share of misses in the lowest 1/2/5/10% of pOk,
  share of calls below pOk 0.5; by sex and with chrX/chrY left out.
- **10 cohort samples:** `rescore_fast_path_calls_with_models.py` and `compare_models_on_fast_path_misses.py`
  (tandem-repeat-explorer), on the new JSONs. First rerun the comparison step for these 10 samples (about $1,
  moved here from step 2). `compare_models_on_fast_path_misses.py` reads one model's pOk from an earlier run's
  comparison files and stops with "differs between the two runs" when that run's calls differ, which they now do
  (5210d1f changed the fast-path read counts). So rescore that model with
  `rescore_fast_path_calls_with_models.py` too.
- **gnomAD manual reviews (gnomAD session):** Spearman rho, AUC and median rho within a locus, by allele length,
  against Q. These are seeking-mode calls, so they test the full-genotyper models.
- Show each change's effect separately: A vs baseline, B vs baseline, A + B vs A.
- **Result, held-out views (2026-10-25):** `compare_retrain_variants.py` scored all 7 models on the 81 held-out
  samples (report `report/retrain_variants/retrain_variants.2026-10-25_model.html`).
  - **Gated MAE, variant-locus view:**

    | Model | Compared with the 10-17 model |
    |---|---|
    | baseline | within about 1% (better in 1 stratum, worse in 3, tied in 2) |
    | A | worse in all 6 strata (+3% to +15%; full_nonspanning non-homopolymer 9.72 to 11.17) |
    | B 10% / 15% | worse in 4 strata, better in full_spanning non-homopolymer (6.95 to 6.82 / 6.78) |
    | A + B | worse in all 6 strata, like A |

  - **Early warning, all-loci view:**

    | Model | ROC-AUC | Recall at 1% (all / male) |
    |---|---|---|
    | 10-17 | 0.740 | 0.088 / 0.076 |
    | baseline | 0.700 | 0.077 / 0.056 |
    | A | 0.889 | 0.328 / 0.326 |
    | B 10% | 0.740 | 0.176 |
    | A + B 10% | 0.912 | 0.406 / 0.407 |
    | A + B 15% | 0.912 | 0.397 |

  - So A gives the early warning the pilot promised, with males and females now alike, but fails step 6's rule that
    gated error must not be worse than 10-17 in any stratum. The likely cause is D3: A's LCF head trains on the
    all-loci table's gated rows (fast path 234k vs 378k, mostly reference-length calls), not on the variant-locus
    rows the gated MAE is measured on.
  - B's direction accuracy and gate accuracy in run_eval are scored against today's 3-class label, so they are not
    meaningful for B. B's own test is the gnomAD comparison at 270 bp and above.
- **Result, D3 (b) models (2026-10-26):** report `report/retrain_variants/retrain_variants.2026-10-26_model.html`.
  The direction heads are unchanged (same early warning and gate accuracy as A and A + B), so only the LCF differs.
  - **Gated MAE, change from 10-17** (* = the paired per-sample interval excludes 0, so the stratum is lost):

    | Model | quick non-homo | quick homo | spanning non-homo | spanning homo | non-spanning non-homo | non-spanning homo |
    |---|---|---|---|---|---|---|
    | baseline | -1.0% (won) | +0.5%* | +0.6%* | -0.5% | +1.2%* | -1.0% |
    | A before | +2.8%* | +6.2%* | +10.0%* | +8.4%* | +15.0%* | +9.6%* |
    | A, LCF on variant loci | +0.4%* | +3.7%* | +1.0%* | +0.1% | +2.0%* | -1.8% |
    | A + B 10%, LCF on variant loci | +1.1%* | +7.4%* | +0.5%* | +5.4%* | +4.3%* | +0.3% |
    | A + B 15%, LCF on variant loci | +1.5%* | +8.4%* | +0.1% | +8.1%* | +5.8%* | +1.0% |

  - What is left of A's gap comes from its direction head, which decides where the gate fires: its gate decision
    accuracy on variant loci is lower than the baseline's in every stratum (fast path non-homopolymer 0.891 vs
    0.912), and the net number of alleles the gated correction helps falls from 19,945 to 15,675 there.
  - Share of the correction's error reduction (raw MAE minus gated MAE) that A keeps, relative to 10-17: 92%
    (fast path non-homopolymer), 79% (fast path homopolymer), 99% and 100% (spanning), 97% (non-spanning
    non-homopolymer); A does better than 10-17 in non-spanning homopolymer.
  - A still fails step 6's rule, by 0.4% to 3.7% in 4 strata. Its early warning is unchanged (ROC-AUC 0.889, 33%
    of misses in the lowest 1% of pOk). Against the baseline (same tables and build), A is worse by 0.4% to 3.2%
    in 4 strata and tied in 2.
  - The baseline itself, today's recipe retrained on the new tables, also loses 3 strata to 10-17 by 0.5% to 1.2%.
    So no retrain meets the rule as written; the rule needs a tolerance or the baseline as its reference.
  - **By allele size (2026-10-26, per-call comparison on the same rows, other motifs; scripts in the session
    scratchpad `step5/`):** A hurts the direction head on expansions, and the baseline shows the cause is A, not
    the new data. pTooShort ROC-AUC for true alleles 150-299 bp longer than the reference: 10-17 0.851, baseline
    0.834, A 0.770; 300 bp or more: 0.868, 0.867, 0.793. pTooLong ROC-AUC over all sizes: 0.927, 0.928, 0.868.
    A's pOk flags 45% of correct calls at +150-299 bp (baseline 31%). A's LCF applied to every call is as good
    as the baseline's on expansions and better on small alleles. A gains only at reference-size alleles
    (pTooLong ROC-AUC 0.858 to 0.951).
  - **Under- or overfitting (2026-10-26, `step5/direction_head_fit_on_expansions.py`):** A is not overfitting
    expansions (log loss on its training people nearly as high as on held-out people: non-spanning 0.758 vs
    0.792); it underfits them. A at 3,000 trees is worse than the baseline at 100 (0.792 vs 0.728), so more trees
    won't close the gap. A's training sample holds far fewer expansion rows: in full_nonspanning both tables hold
    549,860 expansion rows, but the 1M-row cap keeps 42% of them for the baseline and 20% for A (15% vs 40% of
    under-called ones), and in the fast path the all-loci table itself has half of them (38,036 vs 76,462). That
    was the suspected cause, but the "baseline rows plus 500k hom-ref rows" model below rules it out.
  - **Reweighted A (2026-10-26):** `train.py --variant-locus-weight-share 0.8` (new, reviewed, uncommitted)
    trained `model/..._length_head_variant_loci_variant_locus_weight_share_80pct.20261026.json.gz`. It recovers
    about a quarter to a third of A's loss on expansions (pTooShort ROC-AUC at +300 bp: baseline 0.867, A 0.793,
    reweighted 0.815; pTooLong over all sizes 0.928, 0.868, 0.900) and keeps the early warning (ROC-AUC 0.904,
    32% of misses in the lowest 1%). Gated MAE still loses 4 strata to the baseline (by 0.2% to 1.3%).
  - **Baseline rows plus 500k hom-ref rows (2026-10-26):** `train.py --hom-ref-locus-direction-training-rows
    500000` (new, reviewed, uncommitted) trains the direction heads on exactly the baseline's 1M variant-locus rows
    plus 500,000 quota-sampled rows at loci where the truth equals the reference
    (`model/..._length_head_variant_loci_variant_locus_direction_rows_plus_500000_hom_ref_locus_rows.20261026.json.gz`).
    Best early warning so far by ROC-AUC (0.924; misses in the lowest 1/5/10% of pOk: 28/70/81%, A 33/60/68%,
    baseline 8/39/48%, 10-17 9/39/58%), but on expansions it equals the reweighted A (pTooShort ROC-AUC at +300 bp 0.819, pTooLong over
    all sizes 0.906), and gated MAE loses 4 strata to the baseline (fast path +1.7% and +4.1%, spanning
    homopolymer +1.1%, non-spanning other motifs +0.9%) and wins non-spanning homopolymer (-1.3%). So the number of
    expansion rows was not the cause. What still differs from the baseline: the genotype-level label (changes
    0.2-3% of the full genotyper's variant-locus training rows, 2-9% of the fast path's), the hom-ref rows, and
    calibration on all-loci rows. `step5/direction_training_rows_by_label_and_size.py` shows the full genotyper's
    hom-ref rows include many large over-calls (non-spanning: 92,277 rows called 50 bp or more over the reference,
    81% TOO_LONG, beside 201,043 variant-locus rows there, 40% TOO_LONG). These are real in deployment: hom-ref loci are
    42% (spanning) and 47% (non-spanning) of the representative calibration rows. Every metric above except the
    early warning scores variant loci only, so it cannot credit a model for those calls.
  - **Same with the per-allele label (2026-10-26, user's choice):** `--direction-label dir_code` instead of
    `dir_code_genotype`, everything else the same
    (`model/..._all_loci_length_head_variant_loci_variant_locus_direction_rows_plus_500000_hom_ref_locus_rows.20261026.json.gz`).
    The early warning is as good or better (ROC-AUC 0.928; misses in the lowest 1/5/10% of pOk 33/69/81%; males and
    females alike), though half as many fast-path calls fall below pOk 0.5 (0.4% vs 0.8%). It beats the
    genotype-label version in 3 strata (both fast-path ones by 1.4% and 3.3%) and ties 3. Against the baseline it
    is worse by 0.3% to 0.7% in 4 strata and tied in 2; against 10-17 it wins 2, ties 1, loses 3 (0.6% to 1.9%),
    like the baseline. pTooShort ROC-AUC at +300 bp 0.846 and +150-299 bp 0.819 (baseline 0.867 / 0.834,
    genotype label 0.819 / 0.799), so the label caused about half of that loss. pTooLong over all sizes is
    unchanged by the label (0.905; baseline 0.928). The rest comes from the hom-ref rows or the all-loci
    calibration. The genotype-level label buys nothing measurable here, so this per-allele version is the lead
    candidate for change A.
  - **Calibration's share (2026-10-26, `step5/direction_head_fit_on_expansions.py`, held-out expansions of 25 bp or
    more, other motifs):** none. Calibration moves ROC-AUC by at most 0.002 in every regime for both the baseline
    and the per-allele version, so the gap is in the trees, from the hom-ref rows. It sits mostly in the fast path:
    uncalibrated pTooShort ROC-AUC baseline 0.739 vs per-allele version 0.654 (spanning 0.955 vs 0.946,
    non-spanning 0.915 vs 0.908), while fast-path pTooLong improves (0.747 vs 0.837; spanning 0.916 vs 0.906,
    non-spanning 0.907 vs 0.896). Likely reason: fast-path hom-ref rows are calls at or just under the reference
    size that are all correct (343k of the 500k are such calls of other motifs), so a reference-size call that misses an expansion looks safer than it does to the
    baseline, which trains as if every locus were a variant locus. That is the price of the early warning, and
    hom-ref loci are 81% of the representative fast-path rows, more than the 33% in this training mix.
  - **Per-allele version vs deployed 10-17 with 95% intervals (2026-10-26, `step5/bootstrap_by_size_differences.py`,
    people resampled; loci shared across people are not accounted for, so intervals may be a little narrow):** on
    variant loci it is worse almost everywhere except reference-size alleles. Other motifs, expansions of 25 bp or
    more: pTooShort ROC-AUC lower by 0.02-0.04 in every bin, pOk < 0.5 catches 5-7 points fewer wrong calls and
    flags 5-9 points more right calls (all intervals exclude 0); pTooLong not distinguishable (few too-long calls).
    All sizes: pTooShort -0.006, pTooLong -0.022, wrong calls caught -6.4 points (homopolymers -0.036, -0.023,
    -3.4 points). Better: reference-size alleles (pTooShort +0.028, pTooLong +0.054; homopolymers +0.098, +0.041)
    and fewer right calls flagged near the reference size (-1.6 to -2.1 points). The gate fires less overall: pOk
    is pushed up because most training loci are right.
  - **Old-table experiment (2026-10-26, user's request): today's baseline recipe trained on the EH b1fbc23 tables
    that 10-17 was trained on** (same truth files; `scratchpad/step4/run_baseline_recipe_on_old_tables_training.sh`).
    Without `strand_bias_phred` among the full genotyper's features, as in 10-17
    (`model/..._old_eh_b1fbc23_tables_without_full_strand_bias_phred.20261026.json.gz`), the held-out predictions
    equal 10-17's exactly (pOk, pTooShort, pTooLong and LCF identical for all 2.4M held-out alleles). So nothing in
    the code since 10-17 changes the baseline recipe, and every difference between 10-17 and the baseline comes from
    the new EH e90307c tables or from `strand_bias_phred`. With `strand_bias_phred`
    (`model/..._old_eh_b1fbc23_tables.20261026.json.gz`), all 4 full-genotyper strata get worse by 2.0-2.9%. That is
    the old, buggy feature (median 33.1) scored on the fixed one (median 2.4), so it says nothing about whether the
    fixed feature helps.
  - **Baseline vs 10-17 with 95% intervals (2026-10-27, `step5/bootstrap_deployed_10_17_vs_baseline.txt`):** the
    fast path has no `strand_bias_phred`, so it shows the new tables alone: other motifs 1.0% better, homopolymers
    0.5% worse, early warning ROC-AUC 0.700 vs 0.740. Full genotyper: other motifs worse by 0.6% (spanning) and 1.2%
    (non-spanning), homopolymers tied. Other-motif expansions of 25-299 bp: pTooShort ROC-AUC lower by 0.015-0.023
    and 0.5-1.7 points fewer wrong calls caught; at 300 bp or more the same pTooShort, 1.7 points fewer caught. So
    40-80% of the per-allele version's pTooShort ROC-AUC loss against 10-17 at 25-299 bp (0.028-0.038) is already in
    the baseline, but only 15-25% of its loss in wrong calls caught (5-7 points), and none of its loss at 300 bp or
    more. Still open: whether the new tables or `strand_bias_phred` cause the baseline's full-genotyper and expansion
    losses. Answering it needs the baseline recipe on the new tables without `strand_bias_phred`.
  - **Baseline recipe on the new tables without `strand_bias_phred` (2026-10-27, user's choice,
    `model/..._without_full_strand_bias_phred.20261027.json.gz`):** no clear difference from the baseline in any of
    the 6 strata, the early warning, or any size bin (bootstrap: every expansion difference within 0.01 and none
    clear). So the fixed feature is neutral, and the baseline's differences from 10-17 all come from the new tables
    (85bd281 routing, 5210d1f fast-path counts, f37a139 full-path pairs). Against 10-17 it shows the same pattern as
    the baseline: fast path other motifs 1.0% better, homopolymers 0.5% worse, early warning 0.700 vs 0.740; full
    genotyper other motifs 0.5% (spanning) and 1.7% (non-spanning) worse, spanning homopolymers 0.7% better;
    other-motif expansions of 25-299 bp pTooShort ROC-AUC lower by 0.016-0.023. Caveat: the 95% intervals cover
    the held-out people, not training randomness. Dropping this neutral feature alone moved full-genotyper gated
    error by up to 1.0% (non-significant), comparable to the 0.5-1.7% gaps to 10-17, while it moved expansion
    pTooShort ROC-AUC by at most 0.002, a tenth of the expansion gap.
  - **Training noise (2026-10-27, user's choice, baseline recipe with `SEED` 20261027 instead of 20260616 in
    train.py and model.py, `model/..._seed_20261027.20261027.json.gz`):** against the baseline, gated error moved
    by at most 0.3% in any stratum, early-warning ROC-AUC by 0.005 (0.700 to 0.705), and other-motif pTooShort
    ROC-AUC by at most 0.004 in size bins with 2,000 or more calls; wrong calls caught moved by up to 0.7 points;
    the lowest-percentile recall rows moved by up to 6 points (too noisy to decide on). The per-person intervals
    and the bootstrap mark some of these seed-only differences as clear (fast path gated error, several large size
    bins), so a clear interval alone is not enough. Margins for step 6, from this one pair of seeds: 0.5% gated
    error, 0.005 ROC-AUC per size bin with 2,000 or more calls, 1 point of wrong calls caught, 0.01 early-warning
    ROC-AUC. Against those margins the baseline's differences from 10-17 are real: fast path other motifs 1.0%
    better; full genotyper other motifs 0.5% (spanning, at the margin) and 1.2-1.4% (non-spanning) worse;
    other-motif expansions of 25-299 bp pTooShort lower by 0.015-0.025; early warning 0.035-0.040 lower. Homopolymer
    differences are within the margin. A lead from the training logs, not tested: in the full genotyper's
    non-spanning rows the quota sample is 40% homopolymer on the new tables against 27% on the old (pool 2.06M vs
    1.91M rows), so its other-motif training rows fell from about 730k to 600k. That fits the non-spanning
    other-motif loss but not the spanning one (40% homopolymer on both tables).
  - **Where the expansion gap is (2026-10-27, user's choice, scratchpad step5/expansion_gap_by_genotyping_path.py):**
    replacing one path's pTooShort with 10-17's in the baseline's predictions shows that the fast path holds 72-88%
    of the other-motif 25-299 bp gap (67-85% with the other seed) and the full genotyper without spanning reads
    6-26% (9-35%); with spanning reads, nothing. Within the full genotyper's paths, the pTooShort ranking of
    expansions is the same as 10-17's. The fast path's expansion calls are nearly all too short, and the baseline
    flags 3-11 points fewer of them with pOk < 0.5 (10-17: 38-57%; the seed moves this by at most 3 points). The
    early-warning loss is also the fast path's. So both point at the fast path's inputs on the new tables (5210d1f
    read counts, 85bd281 routing); the non-spanning homopolymer-share lead covers at most a quarter of the gap.
  - **Cause of the fast-path expansion gap (2026-10-27, scratchpad step5/fast_path_expansion_inputs_old_vs_new_tables.py,
    all 81 held-out samples without the per-sample cap):** routing is not it: all 454,481 wrong fast-path calls at
    other-motif expansions of 25 bp or more were on the fast path in both table builds. Of their inputs only the
    read counts changed (5210d1f): `flanking_total` in 79% of calls (median 11 to 17), `flanking_frac` with it,
    `spanning_total` in 15%. On these calls 10-17 flags 39.4% with pOk < 0.5 on the new inputs, 33.4% on its own old
    inputs, and the baseline 34.6%; putting `flanking_total` alone back to its old value takes 10-17 from 39.4% to
    33.4%. So the baseline did not lose skill: 10-17 reads the higher, corrected flanking counts as more evidence of
    an expansion. On the whole fast path (other motifs) 10-17 over-predicts pTooShort (0.39 predicted vs 0.30
    observed in its 0.3-0.5 bin; the baseline 0.40 vs 0.39) and flags more right calls (2.73% vs 2.38%); its pTooShort
    ROC-AUC is lower (0.900 vs 0.903). At 10-17's false-flag rate (baseline pOk threshold 0.536) the baseline catches
    37.3% of these wrong expansion calls against 10-17's 39.5% (1-6 points by size, near the 3-point seed noise).
    Not checked: whether the early-warning gap has the same cause (the old tables have no hom-ref loci), and the full
    genotyper's other-motif losses (0.5% spanning, 1.2-1.4% non-spanning).
  - **Cause of the full genotyper's other-motif loss, and an ExpansionHunter regression (2026-10-27, user's choice:
    /code-review-until-done over this repository and EH 5210d1f, 85bd281, f37a139, 3759cac; plus scratchpad
    step5/full_genotyper_inputs_old_vs_new_tables.py on a seeded 3,000 calls per person and regime):**
    - Other-motif calls in the same full-genotyper regime in both builds have identical inputs for 10-17 (all 29 of
      its features); 10-17's error is the same on old and new inputs, and the baseline is worse on those same
      inputs by 0.48% (spanning) and 2.59% (non-spanning). So the loss is in the baseline's training, not in input
      shift.
    - EH 85bd281 hands a fast-path locus to the full genotyper when any 2 of its reads (unmapped and far mates
      included, no MAPQ or anchoring check) score 0.90 purity for the motif. At C/G homopolymers poly-G artifact
      reads trip it in about a quarter of loci in many 1000 Genomes runs (0.3% in HG002_31x), and the full genotyper
      then calls false alleles about 200-300 repeats long where the fast path was right (median error 1 before).
      In the 81 held-out people 690,865 homopolymer calls moved from the fast path to the full genotyper (117,757 to
      full_nonspanning). All three validators agreed. This is an EH accuracy regression on its own.
    - In training, those false calls grow the full_nonspanning homopolymer cells called at 150-300 bp (about 94k to
      167k rows) and 300 bp or more (13k to 59k). `dataset.quota_sample` gives every (homopolymer or not) x
      called-size cell the same quota, so the quota level falls 23% and the large other-motif cells lose about
      a fifth of their direction-head and q-median rows. Reviewers computed this; no retrain has tested it.
    - Also agreed by all three validators: the fast path's strand bias counts both mates, the full genotyper only
      read 1, so `strand_bias_phred` means different things in the two branches.
    - Unexplained: the 0.48% full_spanning loss (just above the 0.3% seed noise; its quota barely changed).
  - **Even motif sampling (2026-10-27, user's request; `train.py --quota-by-motif-size-category`, model
    `..._quota_by_motif_size_category.20261027`, report `report/retrain_variants_motif_size_categories/`):** quota
    cells are motif-size category (1, 2, 3, 4, 5, 6, 7-15, 16-24, 25+ bp) x called size, so homopolymers fall from
    about 40% of training rows to 11% (26% in full_nonspanning). Gated MAE, mean over people, vs the baseline (the
    other seed moves no stratum more than 0.3%): other motifs better by 3.9% (full_spanning, 80 of 81 people) and
    1.5% (full_nonspanning), fast path unchanged; homopolymers worse by 1.1% (fast path), 5.9% (full_spanning) and
    5.0% (full_nonspanning). Vs 10-17: other motifs 0.9% and 3.4% better and a tie without spanning reads, so both
    other-motif losses are gone; homopolymers 1.6%, 5.4% and 4.0% worse. By category the full_spanning gains are
    largest at 3 bp (-16.5%), 6 bp (-10%), 5 bp (-7.7%) and 16-24 bp (-5.3%). Early warning 0.688 (baseline 0.700,
    10-17 0.740). Other-motif expansion pTooShort ROC-AUC up 0.002-0.008 vs the baseline at 25-149 bp; the gap to
    10-17 (0.012-0.019 at 25-299 bp) is unchanged and still 80-93% in the fast path. Homopolymer pTooShort ROC-AUC
    down 0.011-0.014. The 16-24 and 25+ bp cells are still under-filled in the fast path (table top-up used the
    old cells).
  - **Even motif sampling with 3M rows per head (2026-10-27, `--train-cap 3000000`, model
    `..._quota_by_motif_size_category_train_cap_3000000.20261027`; homopolymers 16-21% of rows; calibration rows
    about 520k):** better than the 1M run in all 6 strata, so the homopolymer loss was row count, not share. Vs the
    baseline: other motifs -0.09% (fast path), -4.6% (full_spanning), -2.3% (full_nonspanning), homopolymers tie
    in all three. Vs 10-17: other motifs -1.1%, -4.1%, -1.1%; homopolymers +0.6% on the fast path (baseline +0.5%),
    tie elsewhere. Other-motif expansion pTooShort ROC-AUC vs 10-17 at 25-299 bp: -0.003 to +0.004 (gap closed by
    the full genotyper's paths; the fast path still flags 3-9 points fewer wrong expansion calls). Early warning
    0.681, and 9.4% of fast-path alleles at all loci get pOk < 0.5 (baseline 5.7%).
  - **The pOk < 0.5 rule at all fast-path loci (scratchpad step5/all_loci_fast_path_gated_error.py, 81 held-out
    people, representative loci):** following the rule makes the fast path WORSE than raw EH for every model,
    10-17 included: mean |error| 0.1256 raw vs 0.148 (10-17), 0.139 (baseline), 0.134 (other seed), 0.140 (3M).
    At hom-ref loci (79% of fast-path alleles) EH is nearly always exact, and the correction turns 1.8-3.4% of
    exact calls wrong while fixing 0.02-0.05%; at variant loci it helps (other motifs 0.962 to 0.910). The two
    seeds of the baseline differ by 13-17% at hom-ref loci, so variant-only models are unconstrained there. The
    variant-locus view never sees this. The 3M model's higher flag rate does no extra harm (2.3% of exact
    other-motif calls changed, baseline 2.2%, 10-17 3.3%).
  - **Models trained with hom-ref rows fix it (same check):** A (all loci, genotype-level label) 0.1207 and the
    per-allele label + 500k hom-ref rows model 0.1197, both below raw EH (0.1256), so the rule helps overall; at
    hom-ref loci they flag 0.3-0.4% of alleles and change 0.05-0.14% of exact calls. The per-allele + 500k model is
    -0.5% vs 10-17 at variant loci (other motifs) but fixes fewer wrong calls there than the 3M model (1.75% vs
    2.25%); early warning 0.928 (10-17 0.740). In the variant-locus view it still loses the full genotyper's other
    motifs to 10-17 (+0.6% spanning, +1.9% non-spanning) and fast-path homopolymers (+1.1%), the losses even
    sampling fixed. Next candidate: 3M even sampling + per-allele label + hom-ref rows.
  - **3M even sampling + per-allele label + hom-ref rows (2026-10-28, models
    `..._plus_{1500000,500000}_hom_ref_locus_rows_quota_by_motif_size_category_train_cap_3000000.20261027`, in
    report/retrain_variants_motif_size_categories):** 1.5M and 500k hom-ref rows tie in 5 of 6 strata; 500k is
    better for expansions. 500k vs 10-17: other motifs -0.8% (fast path), -3.9% (full_spanning), tie
    (full_nonspanning, -0.9%); homopolymers +2.2% on the fast path, tie elsewhere. Vs the 3M model without hom-ref
    rows: +0.3% (fast path, other motifs) and +1.5% (fast path, homopolymers), tie in the full genotyper. All
    fast-path loci: gated error 0.1202 (1.5M 0.1198), -18.9% vs 10-17 and below raw EH (0.1256), the same fix as
    the per-allele + 500k model; early warning 0.929 (1.5M 0.936). Cost: missed expansions on the fast path
    (variant loci, other motifs, true allele 25+ bp over the reference, call too short; scratchpad
    step5/fast_path_expansion_ranking.py): pOk < 0.5 on 19% (10-17 40%, 3M model 35%), ROC-AUC 0.820 (10-17 0.845,
    3M 0.840, 1.5M 0.800, per-allele + 500k 0.786), share in the lowest 5% of pOk 23.7% (10-17 28.0%, 3M 25.1%).
    Hom-ref rows raise fast-path pOk broadly (0.4% of fast-path calls below 0.5 at all loci, vs 9.6% for the 3M
    model), which fixes hom-ref loci and costs expansion flags; more hom-ref rows cost more expansion ranking for no
    all-loci gain.
  - **150k hom-ref rows (2026-10-28, `..._plus_150000_..._train_cap_3000000.20261028`):** going below 500k does not
    bring the expansion flags back. Missed expansions with pOk < 0.5: 15.8% (500k 18.9%), ROC-AUC 0.825 (500k
    0.820), share in the lowest 5% of pOk 23.3% (500k 23.7%); pTooShort ROC-AUC at +25-99 bp is 0.016-0.018 below
    500k. Vs 500k it is worse on the fast path (other motifs +0.25%, homopolymers +1.4%), ties or is slightly better
    in the full genotyper, and has a lower early warning (0.905 vs 0.929). All fast-path loci: 0.1209 (500k 0.1202).
    So 500k stays the best hom-ref row count; the expansion cost is not set by the hom-ref row count.
  - **500k + fast-path expansion rows weighted x5 / x10 (2026-10-28, train.py `--quick-regime-expansion-row-weight`,
    `..._train_cap_3000000_quick_regime_expansion_rows_weighted_{5,10}x.20261028`):** only the quick regime's direction
    heads weight rows whose true allele is 25+ bp over the reference (1.9% of the rows; 8.9% / ~16% of the weight); the
    full-genotyper regimes are identical to the 500k model. Fast-path missed expansions (other motifs): ROC-AUC 0.856 /
    0.865 (10-17 0.845, 500k 0.820), share in the lowest 5% of pOk 31.9% / 35.9% (10-17 28.0%), lowest 1% 8.7% / 10.3%
    (10-17 6.8%); pOk < 0.5 still only 24.1% / 25.5% (10-17 39.6%, 500k 18.9%), since the hom-ref rows keep fast-path
    pOk high. pTooShort ROC-AUC at 25+ bp (other motifs, all paths) now at or above 10-17 (x5 +0.002 to +0.009, x10
    +0.004 to +0.022; 500k was -0.012 to -0.025), at a cost at +1-24 bp (-0.019 / -0.032). Gated error vs 10-17: fast
    path other motifs -0.4% / tie, fast-path homopolymers +3.8% / +5.6% (500k +2.2%); full genotyper as 500k. Early
    warning 0.931 / 0.930; all fast-path loci -18.6% / -18.2% vs 10-17 (500k -18.9%).
  - A + B is further behind in the homopolymer strata. The likely reason (not checked row by row): B's label calls
    any allele within 1 repeat OK, so its gate fires less on off-by-one calls, the bulk of homopolymer errors, and
    the correction misses them. That follows from B's definition rather than from training.

## Step 6: ship decisions (separate)

- **A ships** if the early warning is clearly better (pilot: 7.8% to about 30% of misses in the lowest 1%), males
  and females score alike, and gated error in the variant-locus view is not worse than 10-17 in any of the 6
  strata. No C++ change.
- **B ships** if it beats Q on the gnomAD reviews overall and at 270 bp and above, keeps A's early-warning gain
  when combined with it, and the variant-locus view is not worse. Then: read the new model format in
  `ehunter/genotype_quality/GenotypeQualityModel.cpp`, update the pOk/pTooShort/pTooLong output and
  `docs/07_AlleleQualityMetrics.md`, add tests, build a new Docker image, and rerun the gnomAD comparison end to
  end (about $1 on Hail Batch).
- If only A passes, ship A with the 3-class head.
- **Decision (2026-10-28, user):** ship the 3M even + per-allele label + 500k hom-ref rows model (unweighted,
  `..._plus_500000_hom_ref_locus_rows_quota_by_motif_size_category_train_cap_3000000.20261027`). It passes the early
  warning (31% of misses in the lowest 1% of pOk vs 8.8% for 10-17) and males vs females (31% vs 32%), and the rule
  "not worse in any stratum" is waived for its one loss, fast-path homopolymers at variant loci (+2.2%), since it is
  52% better at all fast-path homopolymer loci. Known cost: fast-path missed expansions rank worse than with 10-17
  (ROC-AUC 0.820 vs 0.845). The user preferred it over the x5 / x10 expansion-weighted versions. Before step 7: a
  comparison with 10-17 on the known-pathogenic-expansions test and the gnomAD manual reviews.
- **That comparison (2026-10-28, report/new_model_vs_deployed_10_17/new_model_vs_deployed_10_17.2026-10-27_model.html):**
  the model runs in today's ExpansionHunter via `--genotype-quality-model` with C++ values identical to Python (21,860
  minicram alleles), so no C++ change. Known pathogenic expansions (56 samples): the same calls get pOk < 0.5 (20 of 22
  certainly-too-short calls), but mean pTooLong on them rises 0.49 to 0.60, the LCF correction's median error vs
  validated sizes rises 20 to 28 repeats (raw call 9.5), and 3 pathogenic-range CNBP calls are corrected below
  PathogenicMin (10-17: 1). gnomAD reviews (2,719 genotypes, seeking mode, Python rescoring that reproduces the C++
  values exactly): pOk AUC 0.722 to 0.712 overall (CI crosses 0), worse below 270 bp (< 100 bp 0.852 to 0.821),
  tie at 270+ (0.65 vs Q 0.895); 1 - pTooLong improves 0.716 to 0.770. Shipping is on hold pending the user's call.
  Graded by the long allele only (user, 2026-10-28; Q too): pOk AUC 0.729 to 0.718, < 100 bp 0.857 to 0.827.
- **Cause (2026-10-28, 13 models on both outside tests, report section 4):** the hom-ref rows. With 500k or more,
  long alleles < 100 bp lose 0.030-0.054 gnomAD AUC (150k: -0.009, n.s.), and every hom-ref model raises pTooLong on
  under-called pathogenic expansions (0.52-0.60 vs 0.45-0.47 without them, 0.49 for 10-17). 3M even sampling without
  hom-ref rows ties 10-17 on both (gnomAD 0.736 vs 0.729). x5 / x10 score like the new model (same full heads); the
  variant-locus share 80% model is worst (0.663). Candidate: hom-ref rows only in the quick regime (the 3M even and
  500k models share a feature contract, so the quick heads of one and the full heads of the other can be spliced).
  Unmeasured: full-genotyper calls at hom-ref loci (seeking mode, optimized-streaming hand-offs).
- **Zygosity (2026-10-28, report section 5):** the hom-ref rows' fast-path loss is in homozygous and one-allele calls
  (missed-expansion ROC-AUC 0.824 -> 0.767 and 0.890 -> 0.860; heterozygous unchanged), the same group the hom-ref rows
  come from, so heads split by zygosity would not separate them. gnomAD: pOk near random on the 96 homozygous genotypes
  for every model (Q 0.78).
- **Spliced model (2026-10-28, report section 6):** `..._quota_by_motif_size_category_train_cap_3000000_quick_regime_
  plus_500000_hom_ref_locus_rows.20261028` = the 500k model's quick regime + the 3M even model's full regimes. Vs 10-17:
  fast path other -0.8%, homopolymers +2.2%, full_spanning other -4.1%, full_nonspanning other -1.1%, two ties; all
  fast-path loci -18.9%, early warning 0.929; all full-genotyper loci (optimized-streaming hand-offs, hom-ref loci
  included; first measurement, all_loci_full_genotyper_gated_error.py) -2.8% (500k -3.3%); gnomAD 0.736 vs 0.729; known
  expansions mean pTooLong 0.45 vs 0.49, 2-3 CNBP calls corrected below PathogenicMin vs 1. Best candidate so far.
- **Shipped (step 7, 2026-10-28, uncommitted):** the spliced model is embedded in ExpansionHunter-bw2 as
  `ehunter/data/genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.20261028.json.gz`; 450/450 unit tests;
  known-expansions minicram runs give identical calls. Group-by-group comparison with 10-17 (motif size, purity,
  chromosome, genotyping path; variant loci and all loci):
  report/genotype_quality_model_20261028_vs_20261017/genotype_quality_model_20261028_vs_20261017.2026-10-28_model.html.
- **Evaluation code review (2026-10-28, 4 rounds of /code-review-until-done on the evaluation scripts):** the
  known-expansion numbers in the bullets above were measured before these fixes and are superseded. Fixed: validated
  sizes parsed as the long allele (lower bounds such as ">75" left out of size errors), findings whose validated
  allele is below PathogenicMin left out, corrected sizes rounded to whole repeats, and one row per sample and gene
  (a gene's alternative catalog definitions, e.g. RFC1_WITH_IUPAC, had counted the same expansion twice). Corrected,
  seeking mode, 10-17 vs shipped: 11 certainly-too-short calls, all flagged by both; pTooShort > pTooLong for 2 of 11
  with either; mean pTooLong 0.687 vs 0.637; median |corrected - validated| over 9 validated sizes 8 vs 11 (raw call
  8; the shipped model is closer for 1, further for 6, tied for 2); one CNBP call (66, PathogenicMin 55) corrected
  below PathogenicMin by both in seeking mode, by every new-tables model but not 10-17 in optimized-streaming. Hom-ref
  rows still raise pTooLong on these calls (0.695-0.796 vs 0.637-0.685 without, 10-17 0.687). Held-out
  fast-path missed expansions recounted with the truth's reference count (floor, not round): 10,286, ROC-AUC 0.847
  (10-17) vs 0.822, pOk < 0.5 40.1% vs 19.0%. Seeking mode at all loci (1,500 loci of HG00438 and HG00609): a tie
  (gated error +0.001 [-0.040, +0.031] repeats, weighted to a callset).
- **Not shipped (2026-10-29):** the 20261028 model fails the user's 2026-10-20 ship gate in seeking mode (pTooLong >=
  0.5 on 18 of 30 confirmed expansions, 10-17: 10; no retrained model passes). The user kept 10-17; the embedding is
  stashed in ExpansionHunter-bw2 (stash@{0}).

## Step 8: tables with the homopolymer hand-off fix, without re-genotyping (2026-10-29)

ExpansionHunter-bw2 5656d6a stops the fast path's in-repeat hand-off (85bd281) at homopolymers, where poly-G artifact
reads made the full genotyper call false 200-300 repeat alleles. For homopolymers this restores the routing of EH
b1fbc23, the build of the old tables. Checked on HG03492: 6,153 of 457,521 homopolymer variant loci went fast path ->
full genotyper between the b1fbc23 and e90307c tables, none the other way, and homopolymer fast-path rows of both
builds match in all 36 shared columns (784,285 rows). The user chose to substitute rather than re-genotype:

1. `build_per_source_parquets_on_hail_batch.py` gets an option to read the JSONs of another run (the b1fbc23 one,
   `gs://tandem-repeat-explorer/tool_genotype_quality/expansion_hunter_v2.1/`) and write to new folders, so today's
   extraction gives the fast path's values the old tables don't store (`fragment_length`, which the fast path and the
   full genotyper measure differently; the spanning-mode columns; the routing of hom-ref loci). Review loop.
2. Hail Batch on the 135 old JSONs (approved 2026-10-29; ~$3.40, ~20 min as in 8450644).
3. A substitution script: per source, the rows at homopolymer loci that b1fbc23 genotyped on the fast path and
   e90307c on the full genotyper are replaced by the b1fbc23-JSON rows; everything else is kept. New folders only;
   checks that all other rows are unchanged and the replaced rows match the old tables in the shared columns.
   Review loop.
4. Reassemble `data/parquet` and `data/parquet_all_loci` from the new per-source parquets into new folders.
5. Retrain and rerun the reviewed evaluation suite against 10-17.

**Result (2026-10-29):** done, in `data.eh_e90307c_with_homopolymer_handoff_fix/` and
`data_eval_43.eh_e90307c_with_homopolymer_handoff_fix/`. Over all 135 people (54 training + 81 held-out): 717,500
homopolymer variant-locus calls and 27,611 hom-ref calls replaced by their b1fbc23 fast-path rows (training people:
279,078 and 10,715), 1,162,351 hom-ref calls dropped (left out by the fast path's sampling; training: 450,575), 9 added;
every check found 0 differing rows. The 20261028 splice retrained on them with the same code snapshot
(`model/..._quick_regime_plus_500000_hom_ref_locus_rows_homopolymer_handoff_fix.20261029.json.gz`). Report:
`report/genotype_quality_model_20261029_vs_20261017/genotype_quality_model_20261029_vs_20261017.2026-10-29_model.html`.
- Held-out (corrected tables): a tie with 20261028; the same gains and losses against 10-17 (all-loci fast path gated
  error -19.7%, full genotyper -1.6% to -3.1%; variant-locus fast path ROC-AUCs 0.009-0.013 lower, chrY +11.9%). The
  full-genotyper heads lose 20261028's edge at hom-ref homopolymer calls (-3.0% -> +0.1% vs 10-17) and its loss at
  variant homopolymers (+2.0% -> +0.4%).
- Still fails the ship gate: pTooLong >= 0.5 on 16 of 30 confirmed expansions in seeking mode (10-17: 10, 20261028:
  18), 14 in optimized-streaming (10-17: 12, 20261028: 13).
- Worse than 10-17 in seeking mode at all loci (+0.052 [+0.006, +0.119] repeats, weighted to a callset), where 20261028
  tied; cause not investigated. Corrected long alleles further from the validated sizes (median 15 repeats vs 8), and
  NOTCH2NLC RGP_1637_3 (called 82, validated ~145) corrected to 58, below PathogenicMin 60.
- Not shipped; 10-17 stays embedded.

## Step 9: balance too-long and too-short calls by called size (2026-10-29)

The 20261029 model's extra pTooLong >= 0.5 flags on confirmed expansions are all calls longer than a read (seeking:
10 of the 18 calls of 150 bp or more vs 4 for 10-17; shorter calls 6 vs 6). These are scored by the full_nonspanning
heads, which learn from healthy genomes that the longer the call, the more often it is too long (held-out: 0.9
too-long rows per too-short row at 150-200 bp, 1.6 at 250-300 bp, 4.9 at 450 bp and above). The user rejected a
disease-locus rule (ExpansionHunter must not know which loci cause disease) and a fourth regime, and chose weighting:

1. `train.py --full-nonspanning-balance-too-long-and-too-short-by-called-size`: in full_nonspanning, the direction
   heads' too-long and too-short rows get equal weight in each called-size bin (150, 200, ..., 450 bp, last bin
   open-ended), and so do the TOO_LONG and TOO_SHORT calibrators' rows (user's choice: calibration weighted too, so the
   reported pTooLong itself moves; the pOk calibrator stays unweighted, though normalization still moves the reported
   pOk). No ExpansionHunter change. Review loop: 2 rounds, 2 documentation fixes.
2. Train it at two seeds, plus the unweighted 3M even model at the second seed (seed noise of the gate counts is about
   3 calls), each spliced with the 20261029 fast-path heads.
3. Evaluate: ship-gate counts, gnomAD reviews, held-out against 10-17 and 20261029 (full-genotyper calls of every size,
   since the shared calibrators also move calls below 150 bp), seeking mode. (20261028 added on the user's request.)

Result (2026-10-30): the weighting is nearly free on healthy genomes but fails the ship gate; not shipped.

- Ship gate (pTooLong >= 0.5 on the 30 confirmed expansions; 10-17 / 20261028 / 20261029 / unweighted seed 2 /
  weighted / weighted seed 2): seeking all 10 / 18 / 16 / 15 / 13 / 14, calls >= 150 bp 4 / 11 / 10 / 10 / 8 / 8;
  optimized-streaming all 12 / 13 / 14 / 12 / 11 / 12. The weighting removes 2 flags at >= 150 bp at both seeds
  (CNBP RGP_1281_3, ATXN2 RGP_1647_3, DMPK 46N_RL_1 at one seed); FXN STRetch_sample10 (0.78-0.85 vs 0.48), DMPK
  STRetch_sample09, DMPK BEG_1230_1_01 and NOTCH2NLC RGP_1637_3 (0.50 / 0.49) stay flagged.
- gnomAD pOk AUC: 0.729 / 0.736 / 0.735 / 0.715 / 0.737 / 0.704. The second seed alone moves it by 0.020, so every
  difference here is within seed noise.
- Held-out, full_nonspanning, all loci: gated error 6.578 / 6.474 / 6.426 / - / 6.454 / 6.431 (weighted vs 20261029
  +0.2%, CI includes 0); AUCs unchanged; full_spanning and fast-path heads identical to 20261029. Calls >= 450 bp
  (variant loci): under-calls with pTooLong >= 0.5 10.7% -> 2.7% / 1.3%, over-calls with pTooLong >= 0.5 96.0% ->
  93.1% / 93.9%. At >= 150 bp every model, 10-17 included, gives pTooLong >= 0.5 to only ~4% of held-out under-calls,
  but to 8-11 of 18 confirmed expansions: the expansions (alleles many reads long, e.g. FXN ~2550 bp called 345 bp)
  look like the healthy genomes' over-calls, and the training data has almost no alleles that long. Rebalancing
  classes cannot supply them.
- Seeking-mode check: gated error minus 10-17 +0.053 [+0.007, +0.122], the same as 20261029 (fast-path heads unchanged).

Cleanup (2026-10-30, user's request): the retrained models that are no longer contenders were moved to the Trash (21
model files, their report files and scratch test outputs), and the training options only they used were removed from
the code: `--direction-label`, `--relative-tolerance`, `--variant-locus-weight-share` and
`--quick-regime-expansion-row-weight`. Kept: 10-17 and earlier, 20261028, 20261029, both balanced models, the seed-2
control, the four source models of their splices, and the old b1fbc23 tables (inputs of the homopolymer-fix tables).

## Step 10: why the old tables give a safer full_nonspanning head (2026-10-30)

A splice of the new fast-path and full_spanning heads with 10-17's full_nonspanning heads ties 10-17 on the ship gate
(seeking 10, calls >= 150 bp 4) and keeps most of the new model's held-out gains; the best model trained only on the
new tables (step 9's) flags 13 / 8. Today's code on the old b1fbc23 tables reproduces 10-17 exactly (2026-10-26), so
the safer full_nonspanning head comes from the old tables, not the recipe. The user chose to find out why and fix the
new tables or labels so that a single training run passes the gate. Leads:

1. 85bd281's routing: fast-path loci with >= 2 in-repeat reads go to the full genotyper, which may add many "long
   call, many in-repeat reads" alleles, labeled too long, to the full_nonspanning pool.
2. Truth errors: in the expansion-like region (2-6 bp motif, short pure reference, call >= 120 bp over it, many pure
   in-repeat reads), anchored reads contradicted the truth in 9 of 20 sampled held-out "too long" labels.

Plan: match each training allele across the two builds' per-source tables (both extracted with today's code) and
compare the full_nonspanning pools by origin and label (scratchpad step12/compare_old_new_nonspanning_pools.py); then
test the candidate fixes by training on new tables filtered accordingly and scoring the ship gate.

Results (2026-10-30):
- The two builds agree exactly on the 1,908,367 full_nonspanning variant-locus training alleles they share (call,
  label and inputs). The new pool adds 15,606: 6,972 the old build called on the fast path (85bd281's in-repeat
  hand-off) and 8,634 the old build did not have; ~80% labeled too long, and they raise the expansion-like region
  (2-6 bp motif, reference <= 60 bp and >= 0.9 pure, call >= 120 bp over it) from 17,434 to 22,151 rows.
- full_nonspanning-only retrains on the new tables, scored inside the weighted model (seeking all / >= 150 bp,
  optimized-streaming all / >= 150 bp; 10-17 10 / 4, 12 / 7; weighted 13 / 8, 11 / 7): 10-17's recipe 16 / 9, 13 / 7;
  the same without the 15,606 added alleles 12 / 6, 11 / 6; today's recipe without strand_bias_phred 17 / 11, 14 / 9.
  So neither the recipe nor strand_bias_phred is the cause; the added alleles are most of it.
- The 6,972 moved alleles: the old fast path called 52% OK (median error 3 bp); the full genotyper calls 80% too long
  (median error 115 bp, larger in all of them); 3,538 went from OK to too long (1,404 of 2,563 in the expansion-like
  region, median error 0 -> 146 bp). So 85bd281's hand-off makes ExpansionHunter over-call alleles the fast path got
  right, and those over-calls, which look like expansions, teach the head to flag expansions.
- Truth errors (reads with MAPQ >= 20 overlapping the repeat, with a motif run longer than the truth allele) affect
  ~25% of too-long long-allele labels in that region, equally in both builds (added 8/30, kept 7/30; held-out 5/20).
  An earlier, looser check (any read within 1 kb) overstated this as ~45%.
- Clean fix: narrow the hand-off in ExpansionHunter, rebuild the tables, and train once (no splicing).

## Step 7: ship

Embed the winning model in ExpansionHunter-bw2 (`ehunter/CMakeLists.txt` `GQ_MODEL_FILE`), rebuild, run the unit
tests and the known-pathogenic-expansions test, write `report/model_report.<date>_model.html`, and update the
fast-path pOk report.

## Risks

- **B changes which calls the documented correction rule touches.** ExpansionHunter only reports LCF; it never
  changes a call. `docs/07_AlleleQualityMetrics.md` tells users to divide the call by LCF only where pOk < 0.5,
  and this repository's training (the LCF head trains only on rows where that rule fires) and evaluations (gated
  error) follow the same rule. Under B, a call whose true allele is past FL gets a high pOk. So a user following
  the rule stops correcting it, and the LCF head stops training on such rows (D3), which also makes their LCF less
  reliable for anyone who corrects them anyway. That may stop the correction from shrinking known expansions below
  PathogenicMin (4 of 50 today: C9ORF72 three times, CNBP once), or it may drop real corrections. The
  variant-locus view and the known-expansions test must check this.
- A changes every regime's calibration; the variant-locus view guards the length correction.
- 44% of fast-path misses have a zero-repeat truth allele and are likely truth artifacts; A brings more of them
  into training. The pilot's results held with and without them.
- Training has few pathogenic-scale expansions (52 individuals at polymorphic loci); the known-pathogenic test
  samples are the real test for long alleles.
- Per-locus FL is somewhat noisy (HG002 31x: 5th-95th percentile 481-534 bp around a median of 509), which blurs
  B's label near the limit by a few percent.
- Memory on this Mac: one source or one training at a time, under the memory guard.
