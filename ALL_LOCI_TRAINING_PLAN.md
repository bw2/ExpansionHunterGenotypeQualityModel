# Plan: train the genotype-quality model on all loci

> **Replaced on 2026-10-20 by `ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md`**, which combines this plan with
> `pOk_relative_tolerance_label_plan.md`. Record progress and decisions there; this file is kept for background.

*Written 2026-10-20. Status: step 1 items 1-2 implemented and tested on one source (below); not yet
reviewed by the user, nothing run on Hail Batch.*

### Progress (2026-10-20)

- `dataset.extract_rows_and_join_truth(..., keep_hom_ref_loci=True)` (`--keep-hom-ref-loci` in `dataset.py` and
  `heldout.py`; off by default) and `dir_code_genotype` in `label_and_filter` are implemented, with tests (229 pass).
  Readers of per-source parquets (`assemble_branch`, `heldout.run_eval`, `accuracy_by_size`) keep only
  variant-locus rows (`dataset.variant_locus_rows`), so today's recipe is unchanged with either kind of parquet.
- On HG002 10x (old b1fbc23 JSON, test only): the variant-locus rows equal the existing parquet exactly (1,557,590
  rows); 2.11M rows in all (+35%; fast path +245k hom-ref rows, full genotyper +308k); peak 5.6 GB. In the
  representative sample, fast-path reference-length rows are 97.1% OK and male chrX/chrY exact-reference
  one-allele rows 99.4% (26.3% in today's variant-only pool). `dir_code_genotype` relabels 71,756 rows from OK to
  TOO_SHORT and 1,751 from TOO_LONG to TOO_SHORT.
- Not done: items 3-5 (all-loci assembly, training options, all-loci evaluation views).

## Why

The training pool keeps only loci where the person's truth differs from the reference
(`dataset._load_truth_from_genotypes_tsv`, filter (b)), about 15% of the loci ExpansionHunter genotypes. Two
problems follow, both measured in tandem-repeat-explorer's
`fast_path_pok_early_warning_regression.2026-10-17_model.html`:

1. **Male chrX/chrY calls get low pOk.** A one-allele locus enters training only when its single allele
   differs from the reference, so exact-reference one-allele calls are labeled OK 36% of the time in training
   but are right 99.1% of the time in practice. Since `n_alleles` became a feature (2026-09-06), the model
   acts on this: 83-94% of male chrX/chrY reference-length fast-path calls get pOk < 0.5, and they fill the
   lowest-pOk 1%. That is the 10-08 review's "early warning" regression (1% flag rate catches 7.8% of misses
   with the 10-17 model, against 18.0% with 07-08).
2. **pOk ranks misses worse than spanning depth alone** (ROC-AUC 0.725 against 0.855).

A pilot (`pilot_fast_path_heads_trained_on_cohort.py`; fast-path heads trained on 4 cohort people and scored
out of fold on others) showed the fix:

| fast-path direction head | ROC-AUC, misses | recall at 1% flag rate |
|---|---|---|
| shipped 10-17 pOk | 0.725 | 7.8% |
| spanning depth alone | 0.855 | 19.9% |
| shipped recipe on 4 people (variant loci only) | 0.655 | 14.3% |
| same recipe, all loci | 0.920 | 31.4% |
| + one-size calls judged against both truth alleles | 0.931 | 33.8% |

The gains hold on chromosomes the heads never saw (0.922 / 0.936), male and female samples score the same,
and extra depth features add nothing.

## Goal

Retrain all three regimes with (1) training and calibration rows drawn from all comparable loci and (2)
genotype-level labels for one-size calls (homozygous or one-allele), with no C++ change, and ship it if it
beats the 10-17 model on the early warning without making the length correction worse.

## Step 0: re-genotype the samples (required either way)

The 135 training and held-out JSONs were made by ExpansionHunter-bw2 b1fbc23. `dataset._eh_build_is_current`
rejects them at HEAD (e90307c): since then 5210d1f changed the fast path's `CountsOfSpanningReads` /
`CountsOfFlankingReads` (model features), 85bd281 changed which loci take the fast path, and 3759cac / f37a139
changed streaming read handling. Training on the old JSONs would fit features the current binary no longer
writes.

- Run `run_expansion_hunter_on_selected_samples.py` (tandem-repeat-explorer) on all 135 samples with the
  current `weisburd/str-analysis-with-expansion-hunter` image, to a new output directory (for example
  `gs://tandem-repeat-explorer/tool_genotype_quality/expansion_hunter_v2.1_eh_e90307c/`). The 2026-10-02 run
  cost $152.84 and took about 10 h, plus a rerun of the 11 samples that ran out of memory at 13 GB.
- Rerun the comparison step for the 10 cohort samples (about $1), so the early-warning evaluation uses the
  same calls.
- Point `dataset.EH_RESULTS_ROOT` at the new directory.

## Step 1: code changes in this repo

1. **Keep hom-ref loci at extraction** (`dataset.extract_rows_and_join_truth`, `_load_truth_from_genotypes_tsv`,
   also used by `heldout.build_sample`). A source JSON has about 11M allele rows, so stream and keep:
   - every row at a variant truth locus (as now), flagged `variant_locus = True`;
   - a seeded sample of rows at hom-ref truth loci (truth = reference for both alleles), kept with a
     per-quota-cell probability. Every row records in a new `inclusion_probability` column the probability a
     hom-ref row of its cell had of being kept, so a later step can draw rows with equal probability whether
     or not the locus is a variant.
   Hom-ref loci still need the high-confidence-region and read-length filters that define comparable loci.
2. **Genotype-level label for one-size calls** (`label_and_filter`, before the rank-1 rows of one-size calls
   are dropped): the row is OK only if it is within tolerance of both truth alleles; otherwise it takes the
   direction of the truth allele it misses by more. Keep the current per-allele label as `dir_code_allele` for
   the reports that use it. `t` (the length head's target) stays per-allele.
3. **Assembly** (`_cap_rows_per_genotyping_regime`, `assemble_branch`): the representative rows become a
   seeded uniform sample over all loci (each row kept with the same probability, using the recorded
   inclusion probabilities), and the training-only quota top-up draws from all loci in a cell with equal
   probability, never from variant loci alone. Drawing top-up rows from variant loci only would bring back
   the selection the pilot removed. The variant-locus rows not chosen this way are left out of the all-loci
   table (`data/parquet_all_loci/`); they stay in the per-source parquets and in today's variant-locus table
   (`data/parquet/`, assembled from the same per-source parquets), for the length head and the variant-only
   reports.
4. **Training** (`train.fit_heads`): direction heads train on the representative rows plus the top-up, and
   calibrate on the 5 held-out people's representative rows (unchanged logic, new population). For the length
   head, see decision D3.
5. **Evaluation** (`heldout.py`, `compare_models.py`, `report.py`): report two views side by side. The
   **variant-locus view** keeps today's metrics (gated mean absolute error and so on) on the same rows as before,
   so the new model can be compared with 10-17 directly. The **all-loci view** adds the early-warning metrics:
   ROC-AUC for misses, recall at 1/2/5/10% flag rates, share of calls below pOk 0.5, by sex and with chrX/chrY
   left out.
6. **Tests** for each change, especially: hom-ref rows survive extraction with the right truth, the
   genotype-level label on hand-made one-size calls (hidden expansion, hidden contraction, both alleles
   right), and representative sampling that does not depend on variant status.

## Step 2: rebuild the parquets

All 54 training sources and 81 held-out samples: about 95 GB of JSON downloaded one sample at a time and
deleted after use, under the memory guard. The 2026-10-02 rebuild took about a day. Peak memory should stay near
today's (kept rows plus two locus-id sets), since hom-ref rows are sampled while streaming.

**Restore `strand_bias_phred` in the full-genotyper features here** (user's decision, 2026-10-20), once the
rebuilt parquets exist and not before: until then all training data carries the full genotyper's buggy strand
counts. The 2026-10-17 change that removed it (`FULL_MODEL_FEATURES` in `features.py`, plus `features_tests.py`,
`report.py`, `report_tests.py`) is uncommitted; move it to a stash so it stays recoverable
(`git stash push -m "10-17 model: full features without strand_bias_phred" -- features.py features_tests.py
report.py report_tests.py`), which returns those files to HEAD, where full models use `FULL_FEATURES`. Before
training, check that the rebuilt full-branch rows have the fixed binary's values (median `strand_bias_phred`
about 2, not about 30).

## Step 3: train, evaluate, decide

1. Train (about 40-60 min). Model size and runtime should not change (same iterations).
2. Compare with the 10-17 model on the 81 held-out samples, both views. **Ship criteria:** the all-loci early
   warning clearly better (pilot: 1% recall 8% to about 34%), no male/female gap, and gated mean absolute
   error in the variant-locus view not worse than 10-17 in any of the 6 regime x motif strata.
3. Rescore the 10 cohort samples with `rescore_fast_path_calls_with_models.py` and rerun
   `compare_models_on_fast_path_misses.py` (new JSONs, new model and 10-17).
4. If it ships: embed in ExpansionHunter-bw2 (`ehunter/CMakeLists.txt` `GQ_MODEL_FILE`), rebuild, run the unit
   tests and the known-expansions test, write `report/model_report.<date>_model.html`, and update the fast-path
   pOk report.

## Interaction with `pOk_relative_tolerance_label_plan.md`

Another plan started the same day redefines the direction label: a relative tolerance (10% or 15%, floor of 1
repeat), a PAST_LIMIT class for alleles longer than the fragment length, and 4 classes. It touches the same
head and the same parquets:

- Both need step 0 (re-genotyping) and step 2 (parquet rebuild); doing them once for both saves a day.
- The two changes compose. The training population (all loci) is independent of how OK is defined, and the
  genotype-level rule for one-size calls applies to any per-allele label: a one-size call is OK only if it is
  OK against both truth alleles under whichever tolerance is used.
- Evaluating them together needs both test sets: the cohort early warning (this plan) and the 2,719
  manually reviewed gnomAD genotypes (that plan), with each change's effect shown separately.

Points to settle between the two plans:

- That plan trains on the current parquets and reads only the start of each JSON (for fragment length) to avoid
  the 90 GB download. Those JSONs come from ExpansionHunter-bw2 b1fbc23 and are stale at HEAD (see step 0), so
  a model trained on them is fine as a research comparison on the gnomAD manual review, but should not be
  embedded.
- Its held-out evaluation uses the variant-only held-out parquets, which cannot show the early-warning problem.
  A model trained on variant loci keeps the male chrX/chrY flaw whatever the tolerance.
- Its step 6 (C++ changes for 4 classes, new model format, Docker image) should wait for a retrain on the
  refreshed all-loci parquets.
- The uncommitted 2026-10-17 change that drops `strand_bias_phred` from the full-genotyper features also needs a
  retrain; all three changes can share one.

Suggested order:

1. Now: that plan's label experiment on the current parquets (cheap, no downloads) to choose the tolerance.
   In parallel, step 0's re-genotyping on Hail Batch, which every option needs.
2. Then: this plan's code changes, one parquet rebuild, and four trainings on the same parquets: 10-17's recipe,
   all loci, the new tolerance, and both together. Compare them on the cohort early warning and the gnomAD
   manual review.
3. Embed the winner; that plan's C++ step goes in only if its label wins.

## Decisions needed

- **D1. Re-genotype all 135 samples now** (about $155 and 10-12 h on Hail Batch)? Required for any retrain on
  the current build.
- **D2. Which regimes.** The pilot tested only the fast path. Option (a): all three regimes (same code path;
  full-genotyper calls on male chrX/chrY show a smaller version of the same pOk drop). Option (b): fast path
  only first, full regimes unchanged.
- **D3. Rows for the length (q-median) head**, which is applied only where pOk < 0.5. Option (a): rows where
  the out-of-fold gate fires, from the representative rows only (faithful to deployment, but the gate will fire
  on far fewer rows, about 0.8% of calls). Option (b): rows where the gate fires from all kept rows, including
  every variant-locus row (more rows, tilted toward real variants). Decide by the variant-view gated error.
- **D4. Hom-ref sampling rate per source**, which sets parquet size and memory. Start with enough to give each
  quota cell its 1M-row share across the 54 sources; check with one source first.
- **D5. A separate "hidden variant" head** for the early warning (pilot ROC-AUC 0.963, recall 48% at 1%). Most
  of its edge over the genotype-level labels (step 1.2) is on misses with a zero-repeat truth allele, which are
  likely truth artifacts, and it needs a new model output in C++. Suggest deferring it.

## Risks

- Changing the training population changes every regime's calibration; the variant-view comparison guards the
  length correction, which is the model's other job.
- 44% of fast-path misses have a zero-repeat truth allele and are likely truth artifacts; more of them now
  enter training as hom-ref-locus negatives or misses. The pilot's results held with and without them.
- Memory on this Mac: run one source at a time under the memory guard, as in 2026-10-02.
