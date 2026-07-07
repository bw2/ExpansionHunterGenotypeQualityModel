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

echo "==== [3/3] evaluate (5-fold CV) + HTML report ========================="
# Homopolymer-only CV first (writes results_homopolymer.json + dir_oof_homopolymer.npz), so the
# render below includes the homopolymer panels + the Excluded/Only-Homopolymers toggles.
"$PYTHON" report.py --data-dir "$DATA_DIR" --cv-train-cap "$CV_TRAIN_CAP" --homopolymer-cv
"$PYTHON" report.py --data-dir "$DATA_DIR" --model "$MODEL_OUT" --cv-train-cap "$CV_TRAIN_CAP"

# Optional external validation: apply the exported model to the 30 held-out HPRC samples (of the
# original 43-sample panel; 13 were promoted into training) and fold the results into the report.
# Off by default (a ~7-8 GB download). Enable with RUN_HOLDOUT43=1.
if [ "${RUN_HOLDOUT43:-0}" = "1" ]; then
  echo "==== [4] external held-out benchmark (30 HPRC samples) ================"
  # Download + build the 30 held-out per-sample parquets (into data_eval_43/real_43/).
  "$PYTHON" heldout.py --model "$MODEL_OUT" --build-only
  # Generate the artifacts report.py actually reads for the held-out-43 section
  # (report/eval_heldout43.json + report/stacked_heldout43.json). heldout.py's own
  # report/heldout.json is a standalone dump the report does NOT consume.
  "$PYTHON" gen_datasets.py --dataset heldout43 --model "$MODEL_OUT"
  "$PYTHON" report.py --data-dir "$DATA_DIR" --model "$MODEL_OUT" --render-only
fi

echo "======================================================================="
echo "DONE"
echo "  model:  $MODEL_OUT"
echo "  report: $HERE/report/model_report.html"
