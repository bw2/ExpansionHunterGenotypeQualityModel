# Regime-split + size-dependent-tolerance redesign — plan

Supersedes the weighting-only objective: redefines "correct" by allele size, splits the full
genotyper into spanning / non-spanning experts, and (optionally) gates q by the direction head.
Tails matter more than the ±2 bulk, so bulk degradation from weighting is acceptable.

## Locked decisions
- **Tolerance tiers** (repeats), keyed on **true allele bp = `motif_size × round(true)`**, SAME for
  spanning and non-spanning:
  | true bp | tol (repeats) |
  |---|---|
  | < 50 | 0 (exact) |
  | 50–120 | ±1 |
  | 120–270 | ±2 |
  | 270–600 | ±4 |
  | ≥ 600 | ±8 |
  Boundary rule (to fix exactly): `<50→0`, `[50,120]→1`, `(120,270)→2`, `[270,600)→4`, `≥600→8`.
- **Regimes (3 model-sets, each = q head + direction head):**
  - `fast` — `processLocusFast` rows (unchanged, one model).
  - `full_spanning` — full-branch rows with `spanning_at_called ≥ 1`.
  - `full_nonspanning` — full-branch rows with `spanning_at_called == 0` (flanking/IRR only).
- **Weighting** — SHELVED. Training is **unweighted** (`--bin-weight-cap < 0`, the new default).
  The regime split + size-tolerance are the structural fix; the equal-per-bin weighting plumbing
  stays dormant (re-enable later if wanted). Per-bin REPORTING (macro + breakdown) stays on — it's
  visibility into the tail, not a training weight.
- **Gating** — directional gate (`t_final = (P_OVER+P_UNDER)·t_hat`, veto sign-disagreement) is a
  LATER cleanup; the regime split already protects the bulk (spanning model learns t≈0 on small
  clean alleles). Add only if residual over-correction remains.

## Module changes

### build_dataset.py  (labels — the foundational change)
- Add `allele_bp = motif_size * round(true)` and `tol_repeats = tier(allele_bp)` columns.
- Redefine `direction` / `dir_code` with the size-dependent band:
  `dr = round(eh) - round(true)`; OK if `|dr| <= tol`, OVER if `dr > tol`, UNDER if `dr < -tol`.
  (Replaces the fixed ±1 band. tol=0 small alleles ⇒ OK only if exact.)
- Add `regime` column: `fast` for fast-branch rows; for full-branch rows
  `full_spanning` if `spanning_at_called >= 1` else `full_nonspanning`.
- Keep `q`/`t` unchanged (continuous target). Document the new columns in SPEC.

### size_tolerance.py  (new, pure + tested)
- `tier_repeats(allele_bp)` vectorized → tol per allele.
- `within_tol(eh, true, tol)` → bool (the within-tolerance "correct" predicate).
- Reuse `size_bins.py` unchanged for the Δ weighting bins.

### evaluate.py
- Replace/augment `exact_match_rate` with **`within_tol_rate = mean(within_tol(round(true_pred), true, tol))`**
  and keep `eh_within_tol_rate` as the raw-EH baseline. Keep round-equal exact-match for continuity.
- `evaluate_q` takes a `tol` array; `_eval_frame` carries `tol_repeats`.
- Direction metrics already read `dir_code`, which is now size-aware — no change beyond labels.

### run_cv.py
- Generalize the `--branch full|fast` axis to a **regime** axis (`fast`, `full_spanning`,
  `full_nonspanning`). Load the branch parquet, split full into the two regimes in-memory by
  `spanning_at_called`, and train/eval one model-set per regime. Weights from `size_bins.bin_weights`
  on that regime's rows. Results JSON keyed by regime.
- Per-regime macro-over-bins + per-Δ-bin tables unchanged in shape.

### make_report.py
- Report 3 regimes instead of 2 branches: headline, per-source/motif/Δ-bin tables, within-tol metric.
- Add a regime-size + tolerance-tier summary (how many alleles per regime, how the OK band shifts).

## Optional / later
- Tolerance-aware loss (don't penalize inside the band) — start with labels+metrics only.
- Directional gating composition (P_OVER/P_UNDER) if the spanning model still over-corrects bulk.
- Monotone-increasing-with-|Δ| weighting (tails > bulk made explicit beyond equal-per-bin).

## Sequencing (recommended)
1. `size_tolerance.py` + tests.
2. `build_dataset.py` relabel + regime column → rebuild `data/parquet/{full,fast}.parquet` (needs the
   build inputs available; confirm) ; re-derive labels.
3. `evaluate.py` within-tol metric (+ tests).
4. `run_cv.py` regime axis; smoke on a subset.
5. `make_report.py` 3-regime report.
6. Subset-sweep convergence per regime (non-spanning is the data-limited one), then full run.

## Open items to confirm before step 2
- Does the dataset need a full `build_dataset.py` re-run from source JSON/TSV, or can labels be
  re-derived in-place from the existing parquet (it already has `eh`, `true`, `motif_size`,
  `spanning_at_called`)? In-place relabel is far cheaper and avoids re-touching GCS/Hail.
