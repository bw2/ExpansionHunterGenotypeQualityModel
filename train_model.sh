#!/usr/bin/env bash
# End-to-end genotype-quality model pipeline (idempotent).
#
#   1. download + build  -- fetch the EH JSON + truth TSVs from GCS and assemble
#                           data/parquet/{quick,full}.parquet (skips work already done).
#   2. train + export    -- fit the per-genotyping_regime experts and write the dated model .json.gz.
#   3. evaluate + report  -- 5-fold chromosome-clean CV + a standalone HTML report.
#
# Env overrides: PYTHON, DATA_DIR, MODEL_OUT, TRAIN_CAP, CV_TRAIN_CAP.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-$HERE/data}"
DATE="$(date +%Y%m%d)"
MODEL_OUT="${MODEL_OUT:-$HERE/model/genotype_quality_model_from_HG002_and_CHM1_CHM13.${DATE}.json.gz}"
TRAIN_CAP="${TRAIN_CAP:-1000000}"
CV_TRAIN_CAP="${CV_TRAIN_CAP:-400000}"

echo "==== [1/3] download + build parquet ===================================="
"$PYTHON" dataset.py --data-dir "$DATA_DIR"

echo "==== [2/3] train + export model ======================================="
"$PYTHON" train.py --data-dir "$DATA_DIR" --out "$MODEL_OUT" --train-cap "$TRAIN_CAP"

# Optional external validation: build the 30 held-out HPRC per-sample parquets (absent from training;
# of the original 43-sample panel, 13 were promoted into training) so the report's held-out HPRC
# section is populated. Off by default (a ~7-8 GB download). Enable with RUN_HELDOUT_SAMPLES=1. Once
# the parquets exist, report.py regenerates the held-out eval/stacked artifacts from them by default
# (no extra gen_datasets.py step) -- pass --skip-heldout-samples to opt out.
if [ "${RUN_HELDOUT_SAMPLES:-0}" = "1" ]; then
  echo "==== build held-out HPRC per-sample parquets (30 samples) ============="
  "$PYTHON" heldout.py --model "$MODEL_OUT" --build-only
fi

echo "==== [3/3] evaluate (5-fold CV) + HTML report ========================="
# Homopolymer-only CV first (writes results_homopolymer.json + dir_oof_homopolymer.npz), so the
# render below includes the homopolymer panels + the Excluded/Only-Homopolymers toggles.
"$PYTHON" report.py --data-dir "$DATA_DIR" --cv-train-cap "$CV_TRAIN_CAP" --homopolymer-cv
# The main render also regenerates the held-out HPRC artifacts from the parquets above (if built).
"$PYTHON" report.py --data-dir "$DATA_DIR" --model "$MODEL_OUT" --cv-train-cap "$CV_TRAIN_CAP"

echo "==== [4/4] compare new model vs previous (held-out) ==================="
# Post-hoc convenience: apply the new model AND the previous/deployed model to the held-out HPRC set
# and print per-regime metric deltas. Self-skips (prints a note) if the held-out parquets weren't
# built (no RUN_HELDOUT_SAMPLES=1) or no previous model exists. Non-fatal -- never fails the pipeline.
"$PYTHON" compare_models.py --new-model "$MODEL_OUT" || echo "(model comparison step failed -- non-fatal)"

echo "======================================================================="
echo "DONE"
echo "  model:  $MODEL_OUT"
echo "  report: $HERE/report/model_report.html"
