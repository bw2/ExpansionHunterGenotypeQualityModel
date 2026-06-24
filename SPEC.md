# genotype_quality — implementation SPEC (stable contract)

Realizes `GENOTYPE_QUALITY_METRICS_PLAN.md`. All modules code against THIS file.
Determinism: global `SEED = 20260616`; pass `random_state=SEED` to every estimator;
persist fold/split definitions to disk.

## Binaries / paths
- EH binary (has `QuickGenotype` + GCS/libcurl htslib): `ehunter/build/ExpansionHunter` (repo-relative).
- Reference: `/Users/weisburd/code/str-truth-set/ref/hg38.fa` (+ `.fai`); fallback `/Users/weisburd/hg38.fa`.
- Positive-loci catalog (downloaded): `genotype_quality/data/catalog/positive_loci.EHv5.*_of_293.json`.
- GCS tool results: `gs://str-truth-set-v2/tool_results/{sample}/illumina/{variant}/{cov}_coverage/`
  - `variant ∈ {EHv5 (=low-mem-streaming, full branch), EHv5-bw2-optimized (=optimized-streaming)}`
  - `{sample,cov}`: `HG002 10x/20x/31x`, `CHM1_CHM13 46x`.
  - truth TSV: `…/{sample}.tandem_repeat_genotypes.for_comparison.with_{variant}_vs_Truth_columns.alleles.tsv.gz`
  - full-branch JSON shards: `…/EHv5/{cov}_coverage/json/*.shard*_of_020.json`
- Real illumina bams (stream via gs://, `GCS_OAUTH_TOKEN=$(gcloud auth print-access-token)`):
  - `gs://str-truth-set-v2/raw_data/HG002/illumina/HG002.pcr_free.downsampled_to_{10x,20x}.bam` (+ 31x: confirm name)
  - CHM 46x: `gs://broad-public-datasets/CHM1_CHM13_WGS2/CHM1_CHM13_WGS2.cram` (public).
- Sim data: `~/code/PolymorphicTandemRepeatFinder/simulated_data` (48 loci; `sim_<N>x__…bam`, `<N>`=allele size).

## Data flow
`eh_json_features.extract_rows(json)` → per-allele raw rows (auto-routed to `full`/`fast`
by `QuickGenotype`). `build_dataset.py` joins truth + adds labels → one parquet per branch:
`genotype_quality/data/parquet/{full,fast}.parquet`.

## Per-allele raw row schema (from `eh_json_features.py`)
Columns: see `COMMON_FIELDS` + `FULL_ONLY_FIELDS` in `eh_json_features.py` (authoritative).
`genotyping_branch ∈ {full, fast}` per row. `eh` = JSON `Genotype` allele (the call; label source).

## build_dataset.py — adds these columns
- `true` (float): truth allele size. Real = matched truth TSV `NumRepeats: Allele: Truth`. Sim = bam `sim_<N>x`.
- `q = eh / true`; `t = log(eh) - log(true)`. Drop rows `eh<=0 or true<=0` (count them).
- `direction` ∈ {OK, OVER, UNDER}: `dr = round(eh)-round(true)`; OK if `|dr|<=1`, OVER if `dr>1`, UNDER if `dr<-1`.
- `dir_code` int: OK=0, OVER=1, UNDER=2.
- `chrom` (str): normalized, **strip `chr`** → `{1..22,X,Y}`. Drop chrM. Real from `LocusId` (`"1-..."`→`1`),
  sim from locus dir (`"chr1-..."`→`1`). Used for folds.
- `sample` (str), `coverage` (float; also a feature), `source ∈ {real, sim}`.
- `purity` (float): real = truth `RepeatPurity: Allele: Truth`; sim = 1.0.
- audit-only (NOT features): `concordance` (real, if present), `tsv_eh` (real TSV call, for audit).

## Row filtering (build_dataset.py; documented + counted)
- Real only: keep `purity >= 0.9`; drop negative-control loci (`TruthSetOrNegativeLocus`) + no-truth rows.
- Sim: `purity:=1.0`, all positive; skip the two truth-derived filters.
- Both: drop EH no-calls (no `eh`), drop `eh<=0 or true<=0`, drop chrM.
- Keep haploid/HEMI (flagged `is_hemi`), keep multiallelic per-allele rows.

## Real join key (data_gcs.py / fast real)
`(sample, coverage, LocusId, allele_rank)`. sample/coverage from GCS path. The
`…vs_Truth_columns.alleles.tsv.gz` is one row per (LocusId, allele); melt `Allele 1/2`→rows
(drop the 1/2). NO `VariantId`/`allele_idx`/`sample`/`coverage` columns in TSV. Synthesize
`allele_rank` by sorting each LocusId's allele rows by `(eh_size, allele_string)` → 0,1.
Match JSON alleles to TSV rows by the SAME size sort (rank-pairing). Assert key unique per table.
Column names templated on `<V>` (`EHv5` or `EHv5-bw2-optimized`) — do NOT hardcode `EHv5`.
Truth + purity + `true` come from TSV; **`eh` and all features come from the JSON**, not TSV.

## features.py — per-branch model feature lists
Bools → int (0/1). Excluded as features (leakage): `true, q, t, direction, dir_code, purity,
sample_id, sample, source, locus_id, variant_id, ref_chrom, chrom, genotyping_branch,
repeat_unit, concordance, tsv_eh, ci_start, ci_end, ref_start, ref_end`.

`FAST_FEATURES` (fast branch):
```
motif_size, coverage, read_length, fragment_length, num_repeats_in_reference, ref_size_bp,
eh, eh_minus_ref, allele_rank, is_long, is_hemi, is_multi,
ci_width, ci_over_size, ci_asymmetry, ci_over_eh,
frac_spanning, frac_flanking, frac_inrepeat,
spanning_total, flanking_total, inrepeat_total, hq_unamb_total,
spanning_at_called, spanning_above_called, flanking_above_called, support_frac,
depth, hq_unambiguous_reads, strand_bias_phred, mean_inserted_bases, mean_deleted_bases
```
`FULL_FEATURES` = `FAST_FEATURES` + `[qd, eh_q, left_flank_norm_depth, right_flank_norm_depth]`.
Engineered (compute in features.py from raw row):
- `ci_asymmetry = ((ci_end - eh) - (eh - ci_start)) / (ci_width + 1)` (0 if ci missing)
- `ci_over_eh = ci_width / (eh + 1)`
`features.build_matrix(df, branch) -> (X: float ndarray/DataFrame, feature_names: list)`. Keep NaN
(HistGBM handles natively). Provide `feature_families` dict (Tier2-AQM, CI, read-fractions, eh_q)
for grouped ablation.

## Labels API
`build_dataset` writes columns above. Models read:
- q: target `t`; recover `true_pred = eh / exp(t_hat)`.
- direction: target `dir_code` (0/1/2).

## Module interfaces (Phase 3)
- `splits.py`:
  - `SEED=20260616`. `make_cv_folds(chroms, n_folds=10, n_test=5, seed=SEED) -> [{"test":[...], "calib":[...], "train":[...]}]`
    (each fold: 5 random test chroms from 24; from remaining 19, 2 seeded calib, 17 train). Persist to JSON.
  - `cross_sample_split(df)`, `cross_coverage_splits(df)` (HG002 10x/20x/31x leave-one-out),
    `cross_domain_splits(df)` (train real/test sim and vice-versa). Each returns train/test row masks.
  - Splitting is BY `chrom` (group), never by row.
- `model_q.py`:
  - `QUANTILES=[0.05,0.1,0.5,0.9,0.95]`. `train_q(X,t,Xcalib,tcalib,...) ->` fitted per-quantile
    `HistGradientBoostingRegressor(loss='quantile', quantile=a, random_state=SEED, learning_rate~0.05,
    min_samples_leaf 200-1000, l2_regularization>0)`, early stop via manual `warm_start`+growing `max_iter`
    monitoring the calib set (NOT `early_stopping=True`). `predict_q(models,X) -> {a: yhat}` with per-row
    sorting to fix quantile crossing. Point q = `exp(yhat_0.5)`.
- `model_direction.py`:
  - `train_direction(X,y,Xcalib,ycalib) -> (clf multinomial HistGBM, calibrators)`; per-class one-vs-rest
    isotonic fit on calib, then renormalize to simplex. `predict_proba(...) -> [P_OK,P_OVER,P_UNDER]` summing to 1.
    Calibration assessed AFTER renormalization.
- `evaluate.py`:
  - q: MAE/RMSE on `t` and recovered `true`; exact-match rate `round(true_pred)==round(true)` vs raw EH
    `round(eh)==round(true)`; interval coverage of 80%/90% at fitted quantiles. Stratify by source/motif/coverage/sample.
  - direction: multiclass log-loss, per-class reliability+ECE (post-renorm), OVER/UNDER one-vs-rest ROC/PR-AUC,
    confusion matrix. Baselines: full branch uses `eh_q` + `ci_width`; fast branch uses `ci_width` only
    (NO `eh_q` baseline — absent on fast). Guard against missing columns.
  - Report real-only, sim-only, pooled.
- `diagnostics.py`: loss-vs-iteration from the warm_start monitor loop (train + calib loss per step);
  optional 3-size train-size curve on one fold.
- `ablation.py`: importance-ranked add-one curve + grouped family ablation, scored on the 2 CALIB chroms
  of ONE fixed fold (never test); → minimal feature set, then scored once on test. Greedy backward off by default.
- `explain.py`: permutation importance (held-out chroms) bars + PDP/ICE for top features. No native HistGBM importance.
- `run_cv.py`: orchestrate 10 folds × {full,fast} branch × {q,direction}; cross splits; write
  `results/{branch}_fold{k}.json` + plots under `plots/`.

Outputs: results JSON per (branch, fold/ split); plots SVG/PNG under `plots/`; final markdown report
`genotype_quality/REPORT.md` (Phase 5).
