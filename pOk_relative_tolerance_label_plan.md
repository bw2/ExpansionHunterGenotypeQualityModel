# Plan: redefine pOk with a relative tolerance and a fragment-length limit

> **Replaced on 2026-10-20 by `ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md`**, which combines this plan with
> `ALL_LOCI_TRAINING_PLAN.md`. Record progress and decisions there; this file is kept for background.
>
> **2026-10-30:** the relative-tolerance models lost, and `train.py --relative-tolerance` and its code were removed.

Started 2026-10-20. Motivation: on 2,719 manually reviewed gnomAD genotypes, pOk beats gnomAD's Q for alleles of
50-270 bp (AUC 0.83-0.92 vs 0.69-0.87) but collapses at 270 bp and above (AUC 0.63 vs 0.90), because the current
label's tolerance (exact below 50 bp, +-1/+-2/+-4/+-8 repeats by size tier) is almost never met by long calls. See
`~/p1_iter3/2023_11_24__updated_gnomad_callset/manual_review_vs_EH_pOk/pOk_vs_manual_review.html`.

## Decisions (made with the user on 2026-10-20)

- One direction classifier with 4 classes instead of 3: OK, PAST_LIMIT, TOO_SHORT, TOO_LONG.
- ExpansionHunter keeps reporting three numbers, which no longer sum to 1:
  `pOk = P(OK) + P(PAST_LIMIT)`, `pTooShort = P(TOO_SHORT) + P(PAST_LIMIT)`, `pTooLong = P(TOO_LONG)`.
  So for a true allele longer than ExpansionHunter can measure, both pOk and pTooShort are high.
- The limit is the locus's own FragmentLength (FL) from ExpansionHunter's output, the same value ExpansionHunter has
  when it scores the allele. The model also gets `eh_bp / FL` as an input, so it can carry over between libraries:
  the training libraries' FL is about 500 bp (HG002 31x: median 509, 99% of loci within 10% of it) and gnomAD's is
  about 380 bp (1st-99th percentile 314-456).
- Tolerance x: train with both 10% and 15% and compare.
- Short alleles: the tolerance has a floor of 1 repeat.
- Past the limit, the rule is one-sided (a call above FL is fine as long as it doesn't exceed the true size by more
  than x).
- The size-correction head (q-median, which gives PredictedLengthCorrectionFactor) is unchanged.

## Label, per allele

Inputs: `eh` and `true` (repeats), `motif_size`, and FL (bp). `true_bp = motif_size * round(true)`,
`eh_bp = motif_size * round(eh)`, `dr = round(eh) - round(true)`.

- If `true_bp < FL`: `tol = max(1, x * round(true))` repeats.
  OK if `|dr| <= tol`; TOO_LONG if `dr > tol`; TOO_SHORT if `dr < -tol`.
- If `true_bp >= FL`:
  TOO_LONG if `eh > (1 + x) * true`; TOO_SHORT if `eh_bp < (1 - x) * FL`; otherwise PAST_LIMIT.

## Steps

1. **Per-locus fragment length in the training table.** ExpansionHunter reports FragmentLength per locus (the mean
   over read pairs in that locus's flanks). The parquets store neither it nor locus IDs, so they have to be rebuilt
   from the training JSONs in `gs://tandem-repeat-explorer/tool_genotype_quality/expansion_hunter_v2.1/`, with
   `eh_json.py` also extracting FragmentLength. Each JSON is about 680 MB compressed (about 90 GB for all samples),
   and this Mac has 79 GB free, so the JSONs can't all be downloaded at once.
2. **Label.** New label function in `features.py` next to the current one, with x as a parameter, plus tests in
   `features_tests.py` for each branch of the rule (including the 1-repeat floor and the one-sided limit rule).
3. **Model.** 4-class direction head with per-class isotonic calibration in `model.py`; the reported pOk/pTooShort/
   pTooLong sums; a new serialization format version.
4. **Train** with x = 10% and x = 15% (serially, under memguard), and compare with the current model on held-out
   people and held-out chromosomes (`report.py`), including the class balance by allele size.
5. **Evaluate on the gnomAD reviews in Python**, before touching ExpansionHunter: compute the model inputs from the
   2,123 gnomAD JSONs with `eh_json.py`, apply the old and new models, and compare with Q by allele length, using
   the same statistics as the report (Spearman rho, AUC, median rho within a locus). Success means beating Q
   overall and at 270 bp and above.
6. **Only if step 5 succeeds:** read the new format in `ehunter/genotype_quality/GenotypeQualityModel.cpp`, update
   the pOk/pTooShort/pTooLong output and `docs/07_AlleleQualityMetrics.md` (which says the three sum to 1), add
   tests, build a new Docker image, and rerun the gnomAD comparison end to end (about $1 on Hail Batch).

## Notes and risks

- `features.py`, `features_tests.py`, `report.py` and `report_tests.py` have uncommitted changes from 2026-10-17
  that drop `strand_bias_phred` from the full-genotyper inputs (after ExpansionHunter-bw2's strand-counting fix). They
  also require a retrain, so the two changes could share one; they are not mine to commit.
- The per-locus FL is a little noisy (HG002 31x: 5th-95th percentile 481-534 bp around a median of 509), which blurs
  the label near the limit by a few percent, less than the 10-15% tolerance.
- The training data (52 individuals at polymorphic loci) has few pathogenic-scale expansions; the known-pathogenic
  test samples in ExpansionHunter-bw2 are the real test for those.
