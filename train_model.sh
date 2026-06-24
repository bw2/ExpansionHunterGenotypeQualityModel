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
"$PYTHON" report.py --data-dir "$DATA_DIR" --model "$MODEL_OUT" --cv-train-cap "$CV_TRAIN_CAP"

echo "======================================================================="
echo "DONE"
echo "  model:  $MODEL_OUT"
echo "  report: $HERE/report/model_report.html"
