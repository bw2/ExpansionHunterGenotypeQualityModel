#!/usr/bin/env bash
# Trains the genotype quality model that ExpansionHunter ships, from training tables built beforehand, then evaluates
# it. The defaults are the inputs and recipe of ExpansionHunter-bw2's
# ehunter/data/genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.20261104.json.gz.
#
#   1. assemble -- simulated_expansions/add_to_training_tables.py links BASE_DATA_DIR's per-source parquets and the
#                  spiked samples' parquets of every SPIKED_TABLES_DIRS folder into DATA_DIR, then assembles its
#                  variant-locus and all-loci training tables.
#   2. train    -- train.py twice, once per training recipe, each drawing up to TRAIN_CAP training rows with one quota
#                  per motif-size category. The first model trains on the variant-locus tables, and only its
#                  full-genotyper regimes are kept. The second trains its direction head on the variant-locus training
#                  rows plus 500,000 rows at loci where the truth equals the reference, and its length correction on
#                  the variant-locus rows; only its quick regime is kept.
#   3. combine  -- combine_genotyping_regimes_of_two_models.py writes MODEL_OUT from those two models' regimes.
#   4. report   -- report.py: 5-fold chromosome-clean cross-validation and the HTML report. The cross-validation fits
#                  train.py's default recipe, not the two above; the report's held-out HPRC section (HELDOUT_DATA_DIR's
#                  samples) and HG002 genome section (a training sample) score MODEL_OUT itself.
#   5. compare  -- compare_models.py: the new model vs the one ExpansionHunter ships now, on HELDOUT_DATA_DIR's samples.
#
# The inputs come from one-time steps, mostly on Hail Batch (README.md, "Building the training tables"). Every step
# runs again on every run; the two models of step 2 are kept in a models_whose_regimes_are_combined/ folder next to
# MODEL_OUT.
#
# Env overrides: PYTHON, BASE_DATA_DIR, SPIKED_TABLES_DIRS (space-separated), DATA_DIR, HELDOUT_DATA_DIR, MODEL_OUT,
# TRAIN_CAP, CV_TRAIN_CAP. DATA_DIR's default is the shipped model's training folder, so setting BASE_DATA_DIR or
# SPIKED_TABLES_DIRS requires setting DATA_DIR too.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -z "${DATA_DIR:-}" ] && { [ -n "${BASE_DATA_DIR:-}" ] || [ -n "${SPIKED_TABLES_DIRS:-}" ]; }; then
  echo "ERROR: set DATA_DIR too when setting BASE_DATA_DIR or SPIKED_TABLES_DIRS, so that the shipped model's" \
    "training folder is not rebuilt from other inputs" >&2
  exit 1
fi
PYTHON="${PYTHON:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-$HERE/data.eh_e90307c_with_homopolymer_handoff_fix_and_new_truth}"
SPIKED_TABLES_DIRS="${SPIKED_TABLES_DIRS:-$HERE/simulated_expansions/work_v3/tables $HERE/simulated_expansions/work_v4/tables}"
DATA_DIR="${DATA_DIR:-$HERE/data.eh_e90307c_with_homopolymer_handoff_fix_and_new_truth_plus_simulated_expansions_v4}"
HELDOUT_DATA_DIR="${HELDOUT_DATA_DIR:-$HERE/data_eval_43.eh_e90307c_with_homopolymer_handoff_fix_and_new_truth}"
DATE="$(date +%Y%m%d)"
MODEL_OUT="${MODEL_OUT:-$HERE/model/genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.${DATE}.json.gz}"
TRAIN_CAP="${TRAIN_CAP:-3000000}"
CV_TRAIN_CAP="${CV_TRAIN_CAP:-400000}"
# In a subfolder, so that compare_models.py, which falls back to the newest model/*.json.gz when no ExpansionHunter
# checkout holds a model, never compares MODEL_OUT with one of these.
MODELS_WHOSE_REGIMES_ARE_COMBINED_DIR="$(dirname "$MODEL_OUT")/models_whose_regimes_are_combined"
MODEL_OUT_NAME="$(basename "${MODEL_OUT%.json.gz}")"
FULL_GENOTYPER_REGIMES_MODEL="$MODELS_WHOSE_REGIMES_ARE_COMBINED_DIR/$MODEL_OUT_NAME.trained_for_its_full_genotyper_regimes.json.gz"
QUICK_REGIME_MODEL="$MODELS_WHOSE_REGIMES_ARE_COMBINED_DIR/$MODEL_OUT_NAME.trained_for_its_quick_regime.json.gz"

stop_because_input_is_missing() {
  echo "ERROR: $1; build it first (README.md, \"Building the training tables\")" >&2
  exit 1
}
[ -d "$BASE_DATA_DIR" ] || stop_because_input_is_missing "BASE_DATA_DIR $BASE_DATA_DIR not found"
spiked_tables_dir_args=()
for spiked_tables_dir in $SPIKED_TABLES_DIRS; do
  [ -d "$spiked_tables_dir" ] || stop_because_input_is_missing "spiked tables folder $spiked_tables_dir not found"
  spiked_tables_dir_args+=(--spiked-tables-dir "$spiked_tables_dir")
done
[ ${#spiked_tables_dir_args[@]} -gt 0 ] || stop_because_input_is_missing "SPIKED_TABLES_DIRS names no folder"

echo "==== [1/5] assemble the training tables with the spiked samples ======="
"$PYTHON" simulated_expansions/add_to_training_tables.py --base-data-dir "$BASE_DATA_DIR" \
  "${spiked_tables_dir_args[@]}" --out-data-dir "$DATA_DIR" \
  --training-population variant_loci --training-population all_loci

echo "==== [2/5] train one model for each training recipe ==================="
"$PYTHON" train.py --data-dir "$DATA_DIR" --out "$FULL_GENOTYPER_REGIMES_MODEL" --train-cap "$TRAIN_CAP" \
  --quota-by-motif-size-category
"$PYTHON" train.py --data-dir "$DATA_DIR" --out "$QUICK_REGIME_MODEL" --train-cap "$TRAIN_CAP" \
  --quota-by-motif-size-category --training-population all_loci --length-head-training-population variant_loci \
  --hom-ref-locus-direction-training-rows 500000

echo "==== [3/5] combine their genotyping regimes ==========================="
"$PYTHON" combine_genotyping_regimes_of_two_models.py --model-to-take-quick-regime-from "$QUICK_REGIME_MODEL" \
  --model-to-take-full-genotyper-regimes-from "$FULL_GENOTYPER_REGIMES_MODEL" --out "$MODEL_OUT"

echo "==== [4/5] evaluate (5-fold CV) + HTML report ========================="
# Homopolymer-only CV first (writes results_homopolymer.json + dir_oof_homopolymer.npz), so the
# render below includes the homopolymer panels + the Excluded/Only-Homopolymers toggles.
"$PYTHON" report.py --data-dir "$DATA_DIR" --cv-train-cap "$CV_TRAIN_CAP" --homopolymer-cv
# The main render also regenerates the held-out HPRC artifacts from HELDOUT_DATA_DIR's parquets (skipped if absent).
"$PYTHON" report.py --data-dir "$DATA_DIR" --model "$MODEL_OUT" --cv-train-cap "$CV_TRAIN_CAP" \
  --heldout-eval-data-dir "$HELDOUT_DATA_DIR"

echo "==== [5/5] compare new model vs previous (held-out) ==================="
# Post-hoc convenience: apply the new model AND the previous/deployed model to the held-out HPRC set
# and print per-regime metric deltas. Self-skips (prints a note) if HELDOUT_DATA_DIR has no held-out
# parquets or no previous model exists. Non-fatal: never fails the pipeline.
"$PYTHON" compare_models.py --new-model "$MODEL_OUT" --data-dir "$HELDOUT_DATA_DIR" \
  || echo "(model comparison step failed: non-fatal)"

echo "======================================================================="
echo "DONE"
echo "  model:  $MODEL_OUT"
echo "  report: $HERE/report/$("$PYTHON" -c 'import sys, report; print(report.report_html_name(sys.argv[1]))' "$MODEL_OUT")"
