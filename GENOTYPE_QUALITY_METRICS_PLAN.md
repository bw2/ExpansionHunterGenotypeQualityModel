# Plan: per-allele genotype-quality metrics for ExpansionHunter (q, P_OVER, P_UNDER)

**Status:** draft for review — the optimized-streaming fast-path metric edits (§5.4) are now
implemented; the modeling work is unstarted.
**Date:** 2026-06-16.
**Scope this phase:** train + evaluate two models (one per genotyping branch — §6). The
**optimized-streaming fast-path C++ edits that emit the richer per-allele metrics** (real flanking
counts, a real genotype CI, and `AlleleQualityMetrics` — strand bias, indel bases, DP,
high-quality-unambiguous reads, CI/allele-size) are now **implemented** in
`HtsLowMemStreamingHelpers.cpp` (working tree), **including** the `"QuickGenotype": true` branch flag
that partitions fast- vs full-branch records (§5.4) — so the fast-path EH work is **complete**. The
eventual EH wiring that *consumes* the trained models is still deferred.

---

## 1. Goal

Produce three **per-allele** quality metrics for an ExpansionHunter (EH) STR call, trained
de novo against truth:

- **`q`** — a single continuous "how far off" score with the multiplicative meaning
  `eh_called / q ≈ true_allele_size`. So `q > 1` ⇒ EH over-called, `q < 1` ⇒ EH
  under-called, `q = 1` ⇒ exact. Example: EH called 20, true 15 ⇒ `q = 20/15 = 1.333`.
- **`P_OVER`** — probability the call is an over-call (EH > true, beyond a tolerance band).
- **`P_UNDER`** — probability the call is an under-call (EH < true, beyond a tolerance band).

`P_OK = 1 − P_OVER − P_UNDER` is implied (the within-band / correct class).

The eventual goal is for EH itself to emit these, but that wiring is **deferred**. This
phase delivers trained models + an evaluation report only.

This is a **new project**. Prior work in `ehunter/calibrator/` (NGBoost delta calibrator
`model_v2`, directional Tier-0 classifier) is used **for reference only** and will be
archived (§9).

---

## 2. Training data

Two data sources are pooled: **(1)** real reads with truth-set genotypes (GCS), and
**(2)** simulated reads with exact known truth (§2.2). The chromosome-based CV (§7) pools
both for the primary training run, while preserving a `source ∈ {real, simulated}` column for
source-stratified reporting and optional source-aware training weights (§6–§8).

### 2.1 Source 1 — real reads (GCS, str-truth-set-v2)

Source: the str-truth-set-v2 tool-comparison outputs on GCS (discovered via
`~/code/str-truth-set-v2`). Per-allele tables that already join EH calls to truth:

```
gs://str-truth-set-v2/tool_results/{sample}/illumina/{variant}/{cov}_coverage/
    {sample}.tandem_repeat_genotypes.for_comparison.with_{variant}_vs_Truth_columns.alleles.tsv.gz
```

Eight tables = **2 EH variants × 4 sample×coverage combos** (~265 MB total, downloadable
locally):

| EH variant            | HG002 10x | HG002 20x | HG002 31x | CHM1_CHM13 46x |
|-----------------------|-----------|-----------|-----------|----------------|
| `EHv5` (low-mem-streaming) | ✓ | ✓ | ✓ | ✓ |
| `EHv5-bw2-optimized` (optimized-streaming) | ✓ | ✓ | ✓ | ✓ |

- HG002 truth = str-truth-set-v2 (SynDip-derived). CHM1_CHM13 truth = SynDip (CHM1+CHM13).
  Both already populated in the `*_vs_Truth_columns` tables (verified).
- Raw EH JSON shards are **expected** under `…/json/` and are the **full-branch** feature source
  (low-mem-streaming shards → §5.1; the optimized-streaming shards are *not* used for fast-branch
  features — those are regenerated locally, §5.3). This GCS layout is **unverified** — Phase-1 (§10)
  must confirm the shards exist before committing to JSON-derived features. **Fallback if absent:**
  re-run local EH on the real bams to regenerate JSON, or fall back to Tier-1-from-TSV for full-branch
  real data — but then the real↔sim feature-equivalence guarantee (§5) is lost, so each TSV column
  must be unit-tested against its JSON definition and Tier-2 is dropped for real.

**Partition axis = genotyping branch, not analysis-mode variant (decision).** Each per-allele
row carries the branch that produced its genotype, and two independent models are trained (§6):
- **full-branch model** — the full genotyper: seeking / streaming / low-mem-streaming (the repo
  integration test asserts these three are call-identical), **and** optimized-streaming's
  full-genotyper fallback for loci `processLocusFast` declines (`HtsLowMemStreamingSampleAnalysis.cpp:583`).
  Gets the complete JSON → full feature set (§5.1).
- **fast-branch model** — optimized-streaming's heuristic fast path (`processLocusFast`,
  `HtsLowMemStreamingSampleAnalysis.cpp:536`), which now emits a near-full JSON → fast feature set (§5.2).
Both models still output the **same three scores** `q`, `P_OVER`, `P_UNDER`. The `EHv5` (low-mem-streaming) vs
`EHv5-bw2-optimized` (optimized-streaming) GCS tables are now **data sources feeding these two
branch models**, not two separately-modeled variants: low-mem-streaming rows are all full-branch;
optimized-streaming rows split across both branches by their tag (§5.3).

### 2.2 Source 2 — simulated reads (`~/code/PolymorphicTandemRepeatFinder/simulated_data`)

48 STR loci (real hg38 coordinates `chrom-start-end-motif`), each with single-allele bams at
controlled repeat sizes — the `sim_<N>x__…bam` suffix `<N>x` is the **allele size in repeat
units** (not coverage). Truth is therefore **exact and known**, including large/controlled
expansions that the HG002/CHM truth sets under-sample. This is the main value: it pins down
the q model at allele sizes where real truth is sparse.

No EH outputs exist for these yet, so they must be **generated**:
- Build diploid samples by merging pairs of single-allele bams (hom + representative het
  pairs), reusing the bam-merge / pair-selection machinery (`build_merged_bam`, `select_pairs`,
  `write_catalog`, `parse_locus_dir`, `allele_sizes_in_dir`) from the **repo-root**
  `mode_consistency_and_accuracy_tests.py` (per-allele read-name prefixes so coverage adds;
  single-locus catalog per dir).
- Run the **local** (patched, §5.4) EH binary to populate **both branches**:
  `--analysis-mode low-mem-streaming` → full-branch sim rows; `--analysis-mode optimized-streaming`
  → a `QuickGenotype`-flagged mix of fast-branch rows (the `processLocusFast` loci) and
  full-branch fallback rows. Emit EH's `.json` and parse with the shared `eh_json_features.py`
  extractor (§5). Note the repo-root helper's `run_expansion_hunter()` parses the `.vcf`, so the
  JSON-output + parse path is **new code**, not reuse of that helper.
- Truth = the two controlled allele sizes parsed from the bam filenames (`sim_<N>x__…bam` via
  the existing `BAM_NAME_RE`); there is **no** `sim_stats.tsv` (only a `sim_stats.twb` Tableau
  workbook, not used).

Caveats (flagged in the report): only 48 loci (small contribution by row count); the local
binary version may differ slightly from the pinned GCS Docker images; simulated reads are
idealized and may not reproduce all real-data error modes. Report metrics separately for
real-only, sim-only, and pooled slices (§7–§8), and use the **real↔sim cross-domain**
generalization tests (§7) to detect source-specific failure modes.

### Relevant columns (per-allele table)

Target / label inputs:
- `NumRepeats: Allele: Truth` → `true`
- `NumRepeats: Allele: EHv5` → `eh_called`
- `DiffRepeats: Allele: EHv5 - Truth` → `delta` (cross-check; we recompute `eh − true`)
- `RepeatPurity: Allele: Truth` → purity filter (truth-only, **not a feature**)
- `TruthSetOrNegativeLocus` → row filtering. `Allele: Concordance: EHv5 vs Truth` is a
  prediction-vs-truth **outcome** and is **not** used to filter rows (filtering on it would
  select on correctness and bias every metric); it is kept only for post-hoc audit counts
  computed **after** the metrics.

Candidate features (EH-side, available at deployment):
- `Coverage: EHv5`, `MotifSize`, `NumRepeatsInReference`, `ReferenceLocusSize (bp)`
- `CI start/end/size: Allele: EHv5`, `DiffFromRefRepeats: Allele: EHv5`
- `NumReadsTotal/NumSpanningReads/NumFlankingReads/NumInrepeatReads: EHv5`
- `FractionOfReadsThatSupportsGenotype: Allele`, `NumReadsTotalThatSupportGenotype: Allele`,
  `NumSpanningReadsThatSupportGenotype: Allele`
- `Q: Allele: EHv5` (EH's own existing quality scalar — useful baseline + feature)
- `NumAllelesSupportedTotal`, `HET_or_HOM_or_HEMI_or_MULTI`, `IsMultiallelic`,
  `IsFoundInReference`, allele index (short vs long)
- `coverage` (numeric depth, spans 10–46 across the pool) — a real feature.

**Excluded as features** (leakage / not available at deployment): all `…: Truth` columns,
`RepeatPurity`, `Concordance`, `delta`, `sample_id`, EH-variant id.

---

## 3. Metric definitions

Let `true = NumRepeats: Allele: Truth` (from the TSV truth column) and `eh` = the called repeats
**from the same EH JSON the features are extracted from** — `NumRepeats: Allele: <V>` is shown only
to name the call; `<V>` is the table's EH-variant suffix (`EHv5` for low-mem-streaming,
`EHv5-bw2-optimized` for optimized-streaming — **verified** against the local str-truth-set-v2 tables:
columns suffixed `: EHv5`, e.g. `NumRepeats: Allele: EHv5`, `Q: Allele: EHv5`, `Coverage: EHv5`).
**Do not hardcode `EHv5`**: every `…: EHv5` column name below is templated on `<V>` per source table.

**Label source (decision):** `eh` for `q`/direction labels comes from the **JSON call**, never
the TSV `<V>` column, whenever the two can disagree. For full-branch real rows the GCS json shards
*are* the Docker run that produced the TSV, so `<V>` == JSON and either works. But §5.3 regenerates
fast-branch real rows with the **patched local optimized-streaming binary**, whose calls can differ
from the old Docker-run `<V>` column; using the TSV `eh` there would mislabel every call that changed
between builds (feature from new JSON, label from old call). So: **TSV supplies truth + row keys
only; `eh` and all features come from the JSON being trained/evaluated.** (Alternatively, regenerate
the comparison table from the new run — but JSON-sourced `eh` avoids the extra step.)

### q (continuous)
- **Exact ratio** (decision): **`q = eh / true`**, modeling target **`t = log q = log(eh) − log(true)`**.
- Recovery: `true_pred = eh / q_pred`.
- Rows with `eh ≤ 0` or `true ≤ 0` are dropped (~0.1%; q undefined / log-unsafe). Counted in
  the report.
- `t` is signed, sharply peaked, heavy-tailed → handled by quantile modeling (§6).

### P_OVER / P_UNDER (3-class, ±1 RU band)
- `delta_round = round(eh) − round(true)`.
- Label: **OK** if `|delta_round| ≤ 1`; **OVER** if `delta_round > 1`; **UNDER** if
  `delta_round < −1`. (Direction is consistent with q: `OVER ⇔ eh>true ⇔ q>1`.)
- The ±1 band absorbs the ubiquitous off-by-one (esp. homopolymers).

---

## 4. Row filtering & edge cases

Applied before training/eval (documented + counted in the report):
- **Simulated-row policy (decision):** the GCS truth columns `RepeatPurity: Allele: Truth` and
  `TruthSetOrNegativeLocus` **do not exist for simulated rows** (§2.2 truth is exact, from the bam
  filename). Applying the truth filters globally would drop every sim row as missing. So sim rows
  are assigned `RepeatPurity := 1.0` (exact by construction) and treated as positive (non-negative-
  control) loci, i.e. they **pass** the purity + defined-truth filters by definition. The two
  truth-derived filters below are scoped to `source == real`; sim rows skip them.
- **Purity filter (real only):** keep `RepeatPurity: Allele: Truth ≥ 0.9`. Impure loci have
  unreliable truth counts (prior finding: median |delta| jumps 0.1→5–19 RU); training on them
  collapsed multi-bp strata. Truth-derived ⇒ filter only, never a feature.
- **Defined truth (real only):** drop negative-control loci (`TruthSetOrNegativeLocus`) and rows
  with no truth allele.
- **EH no-calls (both sources):** drop rows where EH emitted no genotype (no `eh`); q is undefined there.
- **Haploid / HEMI** (chrX/Y; HG002 is male): single allele per locus — kept, flagged.
- **Multiallelic / MULTI:** the table is already per-allele; keep as-is.
- **Chromosome normalization (both sources):** parse the chrom from the locus id and normalize to a
  single canonical form before fold assignment — **strip any `chr` prefix** so real `LocusId`
  (`"1-590658-…"` → `1`) and simulated locus dirs (`"chr1-…"`/`"chrX-…"` → `1`/`X`, via the repo's
  `LOCUS_DIR_RE`) map to the same `{1…22, X, Y}` space the folds (§7) are drawn over. Without this,
  sim rows miss fold assignment or split inconsistently with real rows. (chrM is dropped — STR truth
  sets exclude it.)

---

## 5. Features — two branch-specific feature sets

The two models (§6) use **mostly overlapping feature sets**. Until recently the optimized-streaming
**fast branch** emitted a strict subset of the JSON, but the fast path now populates the previously-
empty fields directly (`HtsLowMemStreamingHelpers.cpp`): `CountsOfFlankingReads` is filled from
one-flank-anchored read votes; the genotype CI is a **real** interval derived from the spread of
high-quality spanning reads (plus soft-clip upper-bound extension), not a zero-width point; and
`AlleleQualityMetrics` is set with DP, high-quality-unambiguous reads, strand-bias phred, mean
inserted/deleted bases within the repeat, and CI/allele-size (`CountsOfHighQualityUnambiguousReads`
is also populated). **Still degenerate on the fast branch:** `CountsOfInrepeatReads` (left empty),
`qd`, and the flank-normalized depths (left at 0); and the fast-path metrics are **approximations**
of the full genotyper's graph-realignment-derived values, so they will not match it exactly — the
two-branch split is retained for that distributional difference, not a missing-field gap (§5.2).

All features in both sets are computed by the one shared `eh_json_features.py` extractor over raw EH
JSON (two contracts: `full`, `fast`) so real and sim rows are identical **within a branch**. Real
features are recomputed from the `…/json/` shards, **not** read from the precomputed TSV columns (a
TSV-vs-JSON definitional mismatch — e.g. `frac_spanning` rounding — would inject a real↔sim covariate
shift). Permutation importance (§8) reports whether each tier earns its keep, per branch.

### 5.1 Full-branch feature set (full model)
- Tier-1 engineered: `ci_width = CI end − CI start`, `ci_over_size = ci_width / (eh + 1)`, CI
  asymmetry; `frac_spanning = NumSpanningReads / NumReadsTotal`, `frac_flanking`, `frac_inrepeat`;
  `support_frac = FractionOfReadsThatSupportsGenotype`; `eh_minus_ref = DiffFromRefRepeats`,
  `motif_size`, `is_long`, `is_hemi`, `is_multi`; `coverage` (depth), `Q` (EH's own quality).
- Tier-2 (JSON-mined): candidate grid (`cand_min/max`, `called_is_cand_*`), per-size read profile
  (`flanking_above_called`, `spanning_above_called`), and `AlleleQualityMetrics` (strand bias,
  indel bases). Prior work showed `flanking_above_called` lifted the directional classifier.

### 5.2 Fast-branch feature set (fast model)
- **Already in fast JSON:** `coverage`, `motif_size`, `NumRepeatsInReference`, `ReferenceLocusSize`,
  `eh` (called repeats), `eh_minus_ref`, `is_long`, `is_hemi`, `is_multi`, `GenotypeType`, the
  spanning-read count + spanning-vote distribution (`CountsOfSpanningReads`), and — **now emitted
  natively** (§5.4 done) — `CountsOfFlankingReads` → `frac_flanking`, `flanking_above_called`; the
  **real** genotype CI → `ci_width`, `ci_over_size`; and `AlleleQualityMetrics` → strand-bias phred,
  mean inserted/deleted bases, DP, high-quality-unambiguous reads, CI/allele-size. The fast set now
  largely **matches** the full set (§5.1); it differs mainly in that the fast metrics are
  approximations and the items below are absent.
- **Still absent on the fast branch:** `CountsOfInrepeatReads` (empty), `qd`, the flank-normalized
  depths (left at 0), and EH's per-allele `Q` scalar.

### 5.3 Branch tag & training-data sourcing
- EH emits `"QuickGenotype": true` on fast-path records only (absent on full-genotyper records — §5.4),
  so rows partition exactly: **fast-branch iff** the JSON record has `QuickGenotype == true`, **else
  full-branch**. No fragile heuristics.
- **Full-branch data:** low-mem-streaming GCS tables (all full-branch, no tag needed) + full-branch
  sim rows (run a full-genotyper mode). Optimized-streaming's full-branch fallback rows may be
  pooled in too; their calls differ slightly from low-mem-streaming because optimized-streaming
  enables the improved mixing weight (`ParameterLoading.cpp:462`) — same JSON fields, so the full
  feature set still applies; flag the mixing-weight regime as a `source`-style column.
- **Fast-branch data:** fast-branch rows from optimized-streaming only. The **pre-existing GCS
  optimized-streaming JSON shards lack the new fast fields and the tag**, so fast-branch real data
  must be **regenerated by re-running the rebuilt local optimized-streaming binary on the real bams**
  (sim already runs the local binary). Budget this in Phase-2 (§10).

### 5.4 EH (C++) changes — status
**Most of this is now implemented** in `processLocusFast` (`HtsLowMemStreamingHelpers.cpp`, working
tree). Unlike the originally-planned "separate additive JSON block," the implementation populates the
**existing** output fields directly, so the fast path now emits the same per-allele metrics the full
genotyper does (as approximations). This is a deliberate **product** change to fast-mode output, not a
training-only side block, so the fast-mode JSON/VCF derivatives (`ADFL`, `REPCI`, the
`AlleleQualityMetrics` block) now differ from the old fast-path output — the
`mode_consistency_and_accuracy_tests.py` fast-vs-full comparisons must be updated accordingly.

**Done:**
- `CountsOfFlankingReads` — filled from one-flank-anchored read votes (was empty).
- Real genotype CI — `setShortAlleleSizeInUnitsCi` / `setLongAlleleSizeInUnitsCi` from the spread of
  high-quality spanning reads, plus soft-clip-length upper-bound extension (was a zero-width point).
- `AlleleQualityMetrics` per allele — DP, high-quality-unambiguous reads, strand-bias binomial phred,
  mean inserted/deleted bases within the repeat, CI/allele-size — plus
  `CountsOfHighQualityUnambiguousReads`. (Per-read inserted/deleted-base tracking added to
  `FastReadAnalysisResult`.)
- `QuickGenotype` branch flag — `setQuickGenotype(true)` on the fast-path `RepeatFindings`, carried
  through `LocusFindings` and emitted by `JsonWriter` as `"QuickGenotype": true`, **only** on fast-path
  records (absent on full-genotyper records — `JsonWriter.cpp:174`). On the concurrency-safe
  main-thread output path the plan required, not a worker-thread `JsonWriter` write.

**No remaining required EH work** — the fast-path metric emission and the branch flag are all
implemented. (Parity unit-testing of each new/changed field against `eh_json_features.py`'s `fast`
contract on a shared fixture is **planned** as part of building that extractor — §9 — and is not yet
written; the existing C++ tests do not cover the new fast-path JSON metric fields.)

---

## 6. Models

**Engine:** `sklearn.ensemble.HistGradientBoosting{Regressor,Classifier}` (already installed;
no new dependency). Fast on the ~6–8M-row pool, handles mixed-scale heteroscedastic
features, supports `loss='quantile'` and multiclass, and is **interpretable** via sklearn's
built-in `permutation_importance` + `PartialDependenceDisplay` (no SHAP dependency needed).
This is the most interpretable option that is still statistically appropriate here; simpler
linear/logistic or GAM models underfit the strong homopolymer / coverage nonlinearities,
and NGBoost is too slow at this row count for 10-fold × 2-branch CV.

### Source-aware weighting
- Primary fit starts as a pooled real+sim model with `source` **excluded** as a feature.
- Track row counts, effective training weight, and fold metrics by source. If simulated rows
  have disproportionate influence — e.g. dominate the pooled objective, improve sim-only metrics
  while degrading real-only metrics, or drive feature/stratum importance in a way not reflected
  in real held-out data — add source-aware `sample_weight` to both q and direction training.
- Candidate weighting schemes: cap total simulated weight to a fixed fraction of the real total,
  equalize total weight across `source`, or tune a small grid of simulated-source downweights
  using the inner calibration chromosomes. Selection is based on calibration-split performance
  with real-only metrics treated as the primary guardrail; final test reporting remains
  unweighted and source-stratified.

### q model — quantile regression on `t = log q`
- Fit `HistGradientBoostingRegressor(loss='quantile', quantile=α)` for
  `α ∈ {0.05, 0.1, 0.5, 0.9, 0.95}` (one independent fit per α). These five knots cover the
  deployed q deliverables — median point estimate + 80% (`.1`/`.9`) and 90% (`.05`/`.95`)
  predictive intervals — and the primary calibration check is **interval-coverage reliability at
  the fitted quantiles** (§8). `.25`/`.75` are omitted: they add only finer CDF resolution, losing
  no point/interval accuracy while cutting q-model fits ~28%.
  **Caveat (PIT):** five independent quantiles do **not** define a full predictive CDF, so a proper
  PIT histogram and arbitrary-threshold direction probabilities are not available from them. Either
  (a) restrict q calibration to interval coverage at the fitted quantiles (default), or (b) if a PIT
  histogram is wanted, state an explicit interpolation contract — piecewise-linear CDF between the
  five knots with assumed tail behaviour — and label the result a **coarse, approximate PIT**. Add a
  denser quantile grid only if a fine PIT becomes a hard requirement.
- **Point q** = `exp(t̂_0.5)` (median). Quantile spread = calibrated predictive distribution
  (statistically appropriate for a heteroscedastic, asymmetric residual; pinball loss is a
  proper scoring rule). Quantile crossing fixed by per-row sorting.
- The distribution can also yield direction probabilities as a cross-check against the dedicated
  classifier — but this is **conditional on the explicit CDF interpolation contract above (option b)**:
  computing `P(round(true) < round(eh) − 1)` evaluates the predictive CDF at a threshold, which the
  five independent quantiles do **not** provide on their own. Without the interpolated (or denser-grid)
  CDF the cross-check is undefined/arbitrary, so **omit it** in the default 5-knot configuration. When
  it is computed, it MUST use the **same ±1 RU band** as §3 — i.e.
  `P_OVER = P(round(true) < round(eh) − 1)`, `P_UNDER = P(round(true) > round(eh) + 1)`
  (`round(true) < round(eh) − 1 ⇔ delta_round > 1 ⇔ OVER`, consistent with §3)
  — **not** the zero-tolerance `P(true<eh)`/`P(true>eh)`, otherwise off-by-one calls (which the
  band absorbs) are directional for q but OK for the classifier, conflating a definitional
  band-mismatch with genuine model disagreement.

### direction model — 3-class classifier (OK/OVER/UNDER)
- `HistGradientBoostingClassifier` (multinomial log-loss), then recalibration on a held-out
  calibration split (§7) so `P_OVER`/`P_UNDER` are calibrated. **Calibration contract:** per-class
  one-vs-rest isotonic maps do **not** preserve the simplex, so the calibrated scores are
  renormalized to sum to 1 and **calibration is then assessed on that exact post-normalization
  output** (reliability / ECE computed after renormalization), not on the raw per-class isotonic
  scores. (If simplex-coherence matters more than per-class monotonicity, use multinomial /
  Dirichlet / temperature calibration instead — open choice.)
- Outputs `P_OK, P_OVER, P_UNDER` (sum to 1 by construction).

Trained **de novo per metric** (separate q + direction models), **per genotyping branch**
(full-branch model, fast-branch model — §2.1, §5), each on its own branch-specific feature set.
Both branches emit the same three scores `q`, `P_OVER`, `P_UNDER`; at deployment a row is routed to
the **fast-branch model** if its JSON record carries `QuickGenotype: true`, otherwise the
**full-branch model**.

**Alternative noted for review:** NGBoost-Laplace over `t` (parametric, matches the prior
`model_v2`) instead of quantile HistGBM — cleaner parametric distribution but much slower
(open question Q2).

---

## 7. Validation: 10-fold Monte-Carlo CV by chromosome

Per user instruction:
- **10 folds.** Each fold holds out **5 random chromosomes** as the test set (drawn from all
  24: `1–22, X, Y` — not just the smallest). Train = the remaining 19 chromosomes.
- Folds are independent random draws (Monte-Carlo / repeated-subsampling CV), so test sets
  overlap across folds — that is the intended "different random set each time."
- **Seeded** (`numpy.random.default_rng(SEED)` drawing 10 sets of 5) for full reproducibility
  / determinism (per CLAUDE.md). Fold definitions written to disk.
- **Leakage safety:** splitting by chromosome (across the *pooled* combos) keeps every
  instance of a locus — across 10x/20x/31x/46x and across both samples — entirely in train
  *or* test. A pure row-split would leak the same locus between coverages.
- **Calibration inside a fold:** from the 19 train chromosomes, hold out 2 (seeded) as a
  calibration set, fit on the other 17, evaluate on the 5 test chromosomes. The 2 calib
  chromosomes serve double duty: isotonic recalibration (direction) + quantile-coverage
  checks (q), **and** the early-stopping monitor set for HistGBM (§12). Because sklearn's
  `HistGradientBoosting.fit` accepts no explicit eval set, chromosome-clean early stopping
  requires a **manual `warm_start` / staged-`max_iter` monitor loop** scoring the 2 calib chroms
  — *not* `early_stopping=True` (whose internal `validation_fraction` would take a random
  within-train row split and leak loci across coverages). See §12 item 3.
- Report **mean across the 10 folds** for every metric, per branch, per source slice
  (**real-only**, **sim-only**, **pooled**) and per motif stratum {1,2,3,4,5,6+}. The
  across-fold **std is a descriptive spread only, not a confidence interval**:
  the MC test sets overlap (5 of 24 chroms each, large chroms recurring), so folds are positively
  correlated and the naive std under-estimates true generalization variance (Nadeau-Bengio). For an
  honest uncertainty estimate use a chromosome block-bootstrap or non-overlapping `GroupKFold`-by-chrom.

### Secondary generalization checks (additive, outside the 10-fold CV)
Distribution-shift tests the chrom CV does not cover:
- **Cross-sample (confounded with coverage):** train on HG002 (10x/20x/31x), test on
  CHM1_CHM13 46x. 46x exists **only** for CHM and 10/20/31x **only** for HG002, so this is
  simultaneously a 46x coverage shift — report it as a joint sample+coverage shift, not a pure
  sample shift.
- **Cross-coverage (clean only within HG002):** leave-one-coverage-out restricted to HG002's
  10x/20x/31x (e.g. train {10x,31x}, test 20x). Do **not** pull CHM's 46x into a coverage split —
  that would confound coverage with sample.
- **Cross-domain:** train on real (GCS), test on simulated (§2.2), and vice versa.
  Report these alongside the pooled-model **real-only**, **sim-only**, and **pooled** CV results
  so source-specific gains/losses are visible.

These reuse the same harness with a different split (decision: include now).

---

## 8. Evaluation metrics & report

**q:**
- MAE / RMSE on `t` (log q) and on recovered `true`; "exact-match" rate of `round(true_pred)`
  vs **`round(true)`**, compared to raw EH (`round(eh)` vs `round(true)`). (Truth alleles can be
  fractional — `load_truth.py` keeps them as floats — so compare against `round(true)`, as
  `build_features.py` does; a fractional truth like 10.5 could never exact-match an integer call.)
- **Primary:** predictive-interval coverage — empirical coverage of the 80%/90% intervals at the
  fitted quantiles vs nominal (does the quantile spread match reality). A proper full PIT histogram
  is **not** computed from the five knots (§6 caveat); only a coarse interpolated PIT if explicitly
  contracted there.
- Stratified by source (real-only, sim-only, pooled), motif size, coverage, and sample.

**direction:**
- Multiclass log-loss, per-class reliability diagrams + ECE, OVER-vs-rest and UNDER-vs-rest
  ROC-AUC / PR-AUC. **Baselines are branch-specific** (the fast branch now emits `AlleleQualityMetrics`
  and a real CI, but still **not** EH's per-allele `Q` scalar — §5.2): the **full branch** compares
  against the `Q: Allele: <V>` baseline and the real CI-width baseline; the **fast branch** skips the
  `Q` baseline (column absent / left at 0 on the fast path) but now uses its **real** CI-width (no
  longer a proxy). Do not apply the `Q` baseline unconditionally — it would error on missing columns
  or mis-score the fast branch.
- Confusion matrix; failure breakdown.
- Report each metric for real-only, sim-only, and pooled test rows. Pooled metrics are useful for
  headline comparison, but real-only and sim-only are required to diagnose whether simulated data
  is helping real calls or mainly optimizing an idealized source.

**Comparison:** full-branch vs fast-branch model side by side (how much accuracy the fast branch's
reduced feature set costs). Compare q-distribution-derived direction probabilities vs the dedicated
classifier **only if** the explicit interpolated/denser CDF contract from §6 is enabled; otherwise
omit this coherence check instead of inventing threshold probabilities from five independent
quantiles.

### 8.1 Training diagnostics (objective + overfitting curves)

Standard learning curves, one set per model (q-quantile regressor at the median, and the
direction classifier):
- **Loss vs boosting iteration** — sourced from the manual `warm_start` monitor loop's own
  per-step scores (§7/§12 item 3), **not** sklearn's `train_score_`/`validation_score_` attributes:
  those are empty unless `early_stopping=True`, which §12 item 3 deliberately disables to stay
  chromosome-clean. At each staged-`max_iter` step the loop already scores the 2 calib
  chromosomes; record the train-set objective at the same steps and plot both vs number of
  trees. The train/val gap and the stopping point visualize overfitting directly. q:
  pinball/quantile loss; direction: multinomial log-loss.
- **Loss vs training-set size** — **not on the required path.** sklearn's `learning_curve`
  refits at ~5 sizes × 5 internal `GroupKFold` splits ≈ 25 near-full-scale fits per call (×2
  models ×2 branches ≈ 100 fits — comparable to the entire core CV) for a near-foregone result
  given the low-variance N:capacity regime (§12 item 7), and it is redundant with the
  per-iteration train/val gap above. If a size curve is still wanted, run **3 sizes on one
  representative fold** (~9 fits), not the full `GroupKFold` sweep.
- Reported per branch; representative fold (or averaged) to keep the report compact.

### 8.2 Feature ablation → minimal effective feature set

Goal: the smallest feature set that retains (near-)peak performance. **All ranking and minimal-set
selection here uses the 2 calibration chromosomes, never the 5 test chromosomes** — a feature-set
choice is a model choice, so selecting it on test-chrom score would violate the no-test-selection
rule (§12 item 10) and optimistically bias the reported minimal-set performance. All §8.2 sweeps
run on **one representative fixed train+calib split** (a single fold), not the full 10-fold harness —
a minimal-set diagnostic needs a stable split, not 10-fold averaging, and routing it through the
harness would multiply every re-score ×10.
- **Importance-ranked add-one curve** (primary): rank features by permutation importance on the
  calib chromosomes, then plot calib score vs number-of-top-k features (k = 1…all). The knee
  identifies the minimal set; report the smallest k within a small tolerance (e.g. ≤1% of
  peak) of full performance.
- **Grouped ablation**: drop whole feature *families* (Tier-2 JSON, CI-derived, read-class
  fractions, EH's own `Q`) to quantify each family's marginal contribution — directly answers
  "does Tier-2 earn its keep" (§5). Cheap (4 family-drop fits) and orthogonal to the add-one curve.
- **Greedy backward elimination** is **not run by default** — it re-scores ~F times only to
  confirm the add-one knee (O(F) extra fits for no new question). Reinstate it on the single
  chosen (metric, branch) split only if the add-one knee is genuinely ambiguous.
- Run per metric and per branch; report the chosen minimal set, then score it **once on the
  test chromosomes** alongside the full set.

### 8.3 Explainability

- **Permutation importance** (held-out chroms) — bar plot, most→least important, with the
  importance-std whiskers across repeats; the model-agnostic, leakage-aware ranking.
- (No native HistGBM split/gain importance: sklearn's `HistGradientBoosting{Regressor,Classifier}`
  expose **no** `feature_importances_`. Permutation importance is the only model-derived ranking;
  a fast cross-check, if wanted, is a single-pass drop-column score delta — not a native attribute.)
- **Partial-dependence / ICE** plots for the top features (coverage, motif size, CI size,
  support fraction, frac_spanning) — shows the *direction* and shape of each effect, not just
  magnitude.
- Least-important features are called out explicitly (candidates for removal / the §8.2 set).

Outputs: a results JSON per (branch, fold), all plots (learning curves, ablation curves,
importance bars, PDPs) as SVG/PNG, + a single markdown report under the project dir.

---

## 9. Project structure & archiving

**Archive prior work** (untracked; move is reversible):
`ehunter/calibrator/` → `ehunter/archive/calibrator_2026_06_16/`.

**New project:** `genotype_quality/` (repo root — decision), with:
- `data_gcs.py` — gsutil download the 8 GCS tables (truth + row keys) **and** the `…/json/`
  shards; features **and** the called `eh` (for labels — §3) come from the shared
  `eh_json_features.py` extractor over the JSON; the TSV columns supply only **truth + row keys**.
  low-mem-streaming shards → full-branch rows. The
  optimized-streaming shards predate the fast fields + the `QuickGenotype` flag, so fast-branch real rows are
  sourced from a **fresh local re-run of the patched optimized-streaming binary** on the real bams
  (§5.3), not these shards. **Verified schema of the `…with_*_vs_Truth_columns.alleles.tsv.gz`
  tables** (checked against local str-truth-set-v2 examples): one row per (locus, allele), keyed by
  `LocusId`; columns are `…: Allele: <V>` (the wide table's `Allele 1`/`Allele 2` pairs are melted
  into rows, dropping the `1`/`2`). There is **no `VariantId` column and no `allele_idx`/allele-number
  column**, and **no `sample`/`coverage` column** — these loci are single-variant, and sample +
  coverage are encoded in the **GCS path** (`…/{sample}/illumina/{variant}/{cov}_coverage/…`). So the
  earlier `(sample, coverage, LocusId, VariantId, allele_idx)` key is not constructible as written.
  **Real join key (decision):** `(sample, coverage, LocusId, allele_rank)` — `sample`/`coverage`
  parsed from the path; `allele_rank` synthesized by sorting each LocusId's allele rows by
  `(eh_size, allele_string)` and indexing 0,1. Match JSON-extracted alleles to TSV allele rows by the
  **same** size sort (rank-pairing); a homozygous call's two identical rows tie harmlessly (same
  features). Assert the key is unique per real table before any feature join. (If a future v2 catalog
  introduces genuinely compound/multi-variant loci, a `VariantId` would have to be recovered from the
  companion `…json_files.alleles.tsv.gz` — not needed for the current single-variant tables.)
- `data_sim.py` — build merged diploid sim samples, run local EH, parse EH JSON + known truth →
  per-allele rows tagged by the `QuickGenotype` flag. Reuses the repo-root
  `mode_consistency_and_accuracy_tests.py` bam-merge/pair-selection machinery (the JSON parse is new,
  shared with `eh_json_features.py`).
- `eh_json_features.py` — the shared EH-JSON feature extractor exposing **two contracts**
  (`full`, `fast` — §5.1/§5.2); used by both real and sim so features are identical within a branch;
  unit-tested for column-by-column JSON↔extractor parity per contract on shared fixtures.
- `build_dataset.py` — unify sources → **one parquet per genotyping branch** (`full`, `fast`);
  compute `q`, `t`, direction label, `chrom`, `sample`, `coverage`, `source∈{real,sim}`,
  `genotyping_branch` (full|fast, from the `QuickGenotype` flag); purity flag; drop no-call /
  negative / zero-allele rows.
- `features.py` — the two branch feature specs + engineering (§5.1/§5.2).
- `splits.py` — seeded 10×5-chrom MC-CV + cross-sample / cross-coverage / cross-domain splits.
- `model_q.py` — quantile HistGBM for `t`; point + distribution; recovery to `true`.
- `model_direction.py` — multiclass HistGBM + isotonic calibration.
- `evaluate.py` — held-out metrics (§8): q + direction scores, reliability, stratified.
- `diagnostics.py` — training curves (loss vs iteration, loss vs train-size) (§8.1).
- `ablation.py` — importance-ranked add-one + grouped ablation on one fold, calib-chrom scored →
  minimal set (§8.2); greedy backward elimination off by default.
- `explain.py` — permutation importance bars (no native HistGBM importance — unsupported), PDP/ICE (§8.3).
- `run_cv.py` — orchestrate folds × branches; write results.
- `*_tests.py` — unittest per module (`_tests.py` suffix, no type hints, Google docstrings).
- `README.md`.

Determinism: fixed global seed; `random_state` on every estimator; fold defs persisted.

---

## 10. Subagent decomposition (the build "using subagents")

- **Phase 0 (C++ — fast-branch outputs, §5.4):** the flanking counts, real CI, `AlleleQualityMetrics`,
  **and** the `QuickGenotype` branch flag are **all implemented** in the fast path (working tree).
  Remaining: rebuild the local EH binary and update the `mode_consistency_and_accuracy_tests.py`
  fast-vs-full comparisons for the changed fast-mode output. Done criteria: JSON↔extractor parity
  tests pass. Blocks fast-branch data generation.
- **Phase 1 (main thread — correctness-critical, not parallelized):** archive old project,
  scaffold new dir, define the per-branch parquet schemas + the two EH-JSON feature contracts
  (`full`, `fast`). Fixing the schema here lets agents work against a stable spec. The schema
  explicitly records `eh` as the JSON call used for labels, keeps TSV calls as audit-only fields
  where retained, and encodes branch-specific baseline availability.
- **Phase 2 (parallel subagents — data generation, independent):**
  - Agent(s): download the low-mem-streaming GCS tables + JSON shards → full-branch real rows
    (parallelize across datasets — JSON parse is the slow part).
  - Agent: re-run the patched optimized-streaming binary on the real bams → fast-branch real rows
    (+ its full-branch fallback rows), distinguished by the `QuickGenotype` flag; labels use the regenerated JSON
    calls, while the old GCS comparison TSV supplies truth + row keys only.
  - Agent: generate simulated-data EH outputs (merge diploid sims, run local EH) → branch-tagged sim rows.
  Main thread merges → one verified parquet **per branch** (schema, counts, JSON-call-sourced
  q/label sanity, tag split).
- **Phase 3 (parallel subagents — implement modules against the spec):**
  - Agent A: `splits.py` + `features.py` (+ tests).
  - Agent B: `model_q.py` (+ tests).
  - Agent C: `model_direction.py` (+ tests).
  - Agent D: `evaluate.py` (+ tests), including branch-specific direction baselines and conditional
    q-derived direction-probability reporting only when the §6 CDF contract is enabled.
  - Agent E: `diagnostics.py` + `ablation.py` + `explain.py` (+ tests).
  Main thread integrates + runs unit tests.
- **Phase 4 (parallel subagents — execute, compute-bound, independent):** run full 10-fold CV
  for the **full-branch model** (agent) and the **fast-branch model** (agent); plus the
  cross-sample / cross-coverage / cross-domain splits, training-curve diagnostics, ablation, and
  explainability — each on its branch's feature set.
- **Phase 5 (subagent):** synthesize the report from the results JSONs + all plots, comparing the
  two branch models (e.g. how much accuracy the fast branch's reduced features cost vs the full branch).
Each subagent's output is verified by the main thread before the next phase.

---

## 11. Resolved decisions

1. **Project dir** — `genotype_quality/` (repo root); prior work archived to
   `ehunter/archive/calibrator_2026_06_16/`.
2. **q model** — quantile HistGBM (sklearn), not NGBoost.
3. **q target** — exact ratio `q = eh/true` (`t = log eh − log true`); drop `eh≤0`/`true≤0` rows.
4. **Calibration** — 2 held-out train chromosomes per fold, also the early-stopping monitor set
   (via a manual `warm_start` loop, since sklearn HistGBM takes no explicit eval set — §12 item 3).
5. **Generalization** — include cross-sample + cross-coverage + cross-domain (real↔sim) now.
6. **Features** — two branch-specific sets (§5.1 full, §5.2 fast), now **largely overlapping** after
   the fast path gained the real CI / flanking counts / `AlleleQualityMetrics` (§5.2); JSON-mined via
   one shared extractor with `full`/`fast` contracts; full-branch contingent on the `…/json/` shards
   existing (verify in Phase-1; Tier-1-only fallback — §2.1).
7. **Row filter** — exclude EH no-calls and negative-control loci from train + eval.
8. **Partition by genotyping branch (not analysis-mode variant)** — two models: full-branch
   (seeking/streaming/low-mem-streaming + optimized-streaming's full-genotyper fallback) and
   fast-branch (optimized-streaming `processLocusFast`). Each emits `q`/`P_OVER`/`P_UNDER`; rows
   routed by the `QuickGenotype` flag (present on fast-path records, absent on full).
8b. **Fast-branch EH C++ edit (this phase)** — the flanking counts, real CI, `AlleleQualityMetrics`,
   and the `QuickGenotype` branch flag are **all done** in the fast path
   (`HtsLowMemStreamingHelpers.cpp` + `JsonWriter.cpp`) (§5.4). Regenerate optimized-streaming JSON to
   source fast-branch real data from the rebuilt binary.
9. **Direction band** — ±1 RU (OK/OVER/UNDER).
10. **Extra data** — simulated reads (§2.2) pooled into the primary fit, with mandatory
    real-only, sim-only, and pooled reporting; consider source-aware training weights if
    simulated rows have disproportionate influence (§6–§8).
11. **Required deliverables** — training-curve diagnostics (§8.1), feature ablation → minimal
    set (§8.2), and explainability importance/PDP plots (§8.3), in addition to held-out metrics.

---

## 12. Overfitting controls

Concern raised in review. Defenses, layered:

1. **Chromosome-held-out evaluation is the headline.** Every reported number is on
   chromosomes absent from training. 10 Monte-Carlo folds → **mean** (the across-fold std is a
   descriptive spread, **not** a CI — the folds' test sets overlap and are correlated, §7). The
   train-vs-test gap is reported per fold; a persistently large gap is the overfitting alarm.
2. **No locus leakage.** Chromosome splitting (not row splitting) keeps each locus wholly in
   train or test across all coverages, both samples, and real/sim — so the model cannot
   "recognize" a locus it saw at another coverage. This is the biggest leakage risk in this
   pooled, multi-coverage setup, and it is structurally eliminated.
3. **Early stopping** on the chromosome-clean calib set. sklearn's `HistGradientBoosting.fit`
   takes **no explicit validation set** and has no `partial_fit`, so `early_stopping=True` would
   fall back to an internal **random** `validation_fraction` row split (leaking the same locus
   across coverages/samples). Instead use a manual **`warm_start` + increasing-`max_iter` monitor
   loop**: refit with growing tree counts, score the 2 calib chromosomes each step, and stop once
   their held-out loss stops improving (`n_iter_no_change`-style patience). This keeps early
   stopping genuinely chromosome-clean.
4. **Capacity limits / regularization** (HistGBM): modest `learning_rate` (~0.05),
   capped `max_leaf_nodes`/`max_depth`, **large `min_samples_leaf`** (e.g. 200–1000 — trivial
   given millions of rows; stops leaves memorizing noise), and `l2_regularization > 0`.
   Hyperparameters fixed by principled defaults, not tuned on the test chroms.
5. **Monotonic constraints — scoped to where they are valid.** They do **not** apply to the
   3-class direction model (sklearn `HistGradientBoostingClassifier` raises `ValueError` for
   `monotonic_cst` with >2 classes) nor to the **signed** q target `t = log(eh/true)` (higher
   coverage/support shrinks `|t|` toward 0 from *both* sides — monotone in uncertainty, not in
   signed `t`). They may be used only on outer-quantile *spread* outputs or a binary recast where
   a feature has a genuinely unidirectional effect; otherwise omit. (Net: a weaker regularizer
   than first assumed — lean on capacity limits (4) and early stopping (3).)
6. **Honest calibration.** Isotonic recalibration (direction) and quantile-coverage checks
   (q) are fit on the calib chroms and assessed on the test chroms — never on the fit data —
   so reported calibration is not optimistic. Predictive intervals that are too narrow on
   test (coverage < nominal) flag over-confidence.
7. **Favourable N:capacity ratio.** ~6–8M rows vs a few hundred shallow trees → low-variance
   regime; the dominant risk is leakage (handled in 2), not raw capacity. The full-branch pool
   is the bulk; the fast-branch pool is a (still-large) subset — the simple STR loci
   `processLocusFast` handles — so its N:capacity remains favourable, but report its row count
   explicitly so a thin-data branch can't hide.
8. **Feature hygiene.** All truth-derived columns, `RepeatPurity`, and `delta` are excluded
   as features; permutation importance is checked for any single feature that "explains" the
   target suspiciously well (leak detector).
9. **Stratified reporting.** Metrics per motif stratum {1,2,3,4,5,6+}, coverage, and source
   surface overfitting concentrated in rare strata (5, 6+) that pooled averages would hide.
10. **No test-set selection.** Any hyperparameter/model choice uses inner calib performance
    only; the 10×5-chrom test sets are touched once, for final reporting.
```
