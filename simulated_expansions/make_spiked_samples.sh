#!/usr/bin/env bash
# Makes the spiked samples the shipped genotype quality model was trained on: each of the training people in
# spike_in_expansions.SPIKED_PEOPLE, with simulated reads of expanded alleles added to their own reads at the known
# disease loci, genotyped the way the training runs were and turned into per-source parquets. The two tables/ folders it
# writes are train_model.sh's SPIKED_TABLES_DIRS. Runs spike_in_expansions.py's steps in this order:
#
#   make-minicrams           each person's reads at the disease loci, from their CRAM (short_read_paths.tsv)
#   check-parity             on Hail Batch: the un-spiked minicrams must give exactly the training runs' rows there
#                            (the shipped samples' check covered 14 of the 45 people, all without a difference)
#   spike                    simulates, aligns (bwa) and merges the expanded alleles, in SPIKED_SAMPLES_WORK_DIR
#   genotype                 on Hail Batch: ExpansionHunter on the spiked samples
#   tables                   their per-source parquets, in SPIKED_SAMPLES_WORK_DIR/tables
#   downsample               a copy of every spiked sample at 15-30x and a matched control of the person's own reads
#                            downsampled alike, in LOWER_COVERAGE_WORK_DIR
#   genotype-lower-coverage  on Hail Batch: ExpansionHunter on those
#   tables-lower-coverage    their per-source parquets, in LOWER_COVERAGE_WORK_DIR/tables
#
# Usage: simulated_expansions/make_spiked_samples.sh [STEP ...]   (default: every step, in the order above)
#
# The steps read the committed disease_loci.catalog.json, disease_loci.tsv, pathogenic_allele_structures.tsv and
# repeat_read_errors.tsv (README.md says how those were made). Required env, for the steps that use it: REFERENCE_FASTA
# (a local, indexed copy of the UCSC hg38 the training runs genotyped against), BILLING_PROJECT (Hail Batch and
# requester-pays) and GCS_DIR (a gs:// folder for the reads genotyped on Hail Batch). spike also needs the bwa index of
# the hg38 the people's reads were aligned to, at BWA_INDEX_PREFIX (default
# simulated_expansions/work/bwa_index/Homo_sapiens_assembly38.fasta.64): the six files
# gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.64.{alt,amb,ann,bwt,pac,sa}. The script
# stops before running any step if spike is to run and the index is missing. It also stops if spike or downsample is
# to run in a work folder that already holds spiked samples, such as the shipped ones: set SPIKED_SAMPLES_WORK_DIR and
# LOWER_COVERAGE_WORK_DIR to new folders to make them again next to those (and compare), or REPLACE_EXISTING_SPIKED_SAMPLES=1
# to replace them (the folder's spiked/ is then moved aside to spiked.replaced_<time>/, not deleted). Other env overrides: PYTHON, BWA, TRAINING_TABLES_DIR, and ONLY_MISSING_OUTPUTS=1 to finish a
# genotyping step that failed part way (genotypes only the samples without an output).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

PYTHON="${PYTHON:-python3}"
BWA="${BWA:-bwa}"
TRAINING_TABLES_DIR="${TRAINING_TABLES_DIR:-$REPO/data.eh_e90307c_with_homopolymer_handoff_fix_and_new_truth/real_quick}"
SPIKED_SAMPLES_WORK_DIR="${SPIKED_SAMPLES_WORK_DIR:-$HERE/work_spiked_samples}"
LOWER_COVERAGE_WORK_DIR="${LOWER_COVERAGE_WORK_DIR:-$HERE/work_lower_coverage_copies_of_spiked_samples}"
BWA_INDEX_PREFIX="${BWA_INDEX_PREFIX:-$HERE/work/bwa_index/Homo_sapiens_assembly38.fasta.64}"
ALL_STEPS=(make-minicrams check-parity spike genotype tables downsample genotype-lower-coverage tables-lower-coverage)

only_missing_outputs_args=()
[ "${ONLY_MISSING_OUTPUTS:-0}" != "1" ] || only_missing_outputs_args=(--only-missing-outputs)

spike_in_expansions() {
  "$PYTHON" -u "$HERE/spike_in_expansions.py" "$@"
}

# spike and downsample replace a work folder's spiked samples, which by default are the ones the shipped model was
# trained on (train_model.sh's SPIKED_TABLES_DIRS), so they run there only when asked to.
stop_if_folder_holds_spiked_samples() {
  if [ -e "$1/spiked/manifest.tsv" ] && [ "${REPLACE_EXISTING_SPIKED_SAMPLES:-0}" != "1" ]; then
    echo "ERROR: $1 already holds spiked samples. To make them again next to these, set SPIKED_SAMPLES_WORK_DIR and" \
      "LOWER_COVERAGE_WORK_DIR to new folders; to replace them, set REPLACE_EXISTING_SPIKED_SAMPLES=1" >&2
    exit 1
  fi
}

# Moves a work folder's earlier spiked samples (reads, truth and ExpansionHunter output) aside before spike or downsample
# makes new ones there, so no earlier output is genotyped or tabled as the new reads' (the genotype step's
# ONLY_MISSING_OUTPUTS=1 would otherwise keep an earlier output JSON).
move_earlier_spiked_samples_aside() {
  if [ -e "$1/spiked" ]; then
    local moved_to="$1/spiked.replaced_$(date +%Y%m%d_%H%M%S)"
    mv "$1/spiked" "$moved_to"
    echo "moved the earlier spiked samples of $1 to $moved_to"
  fi
}

run_step() {
  echo "==== $1 ($(date '+%F %T')) ===="
  case "$1" in
    make-minicrams)
      spike_in_expansions make-minicrams --work-dir "$SPIKED_SAMPLES_WORK_DIR" \
        --reference-fasta "${REFERENCE_FASTA:?set REFERENCE_FASTA}" --billing-project "${BILLING_PROJECT:?set BILLING_PROJECT}" ;;
    check-parity)
      spike_in_expansions check-parity --work-dir "$SPIKED_SAMPLES_WORK_DIR" --training-tables-dir "$TRAINING_TABLES_DIR" \
        --billing-project "${BILLING_PROJECT:?set BILLING_PROJECT}" --gcs-dir "${GCS_DIR:?set GCS_DIR}/check_parity" ;;
    spike)
      move_earlier_spiked_samples_aside "$SPIKED_SAMPLES_WORK_DIR"
      spike_in_expansions spike --work-dir "$SPIKED_SAMPLES_WORK_DIR" --training-tables-dir "$TRAINING_TABLES_DIR" \
        --reference-fasta "${REFERENCE_FASTA:?set REFERENCE_FASTA}" --bwa "$BWA" --bwa-index "$BWA_INDEX_PREFIX" ;;
    genotype)
      spike_in_expansions genotype --work-dir "$SPIKED_SAMPLES_WORK_DIR" \
        --billing-project "${BILLING_PROJECT:?set BILLING_PROJECT}" --gcs-dir "${GCS_DIR:?set GCS_DIR}/spiked_samples" \
        ${only_missing_outputs_args[@]+"${only_missing_outputs_args[@]}"} ;;
    tables)
      spike_in_expansions tables --work-dir "$SPIKED_SAMPLES_WORK_DIR" ;;
    downsample)
      move_earlier_spiked_samples_aside "$LOWER_COVERAGE_WORK_DIR"
      spike_in_expansions downsample --source-work-dir "$SPIKED_SAMPLES_WORK_DIR" --work-dir "$LOWER_COVERAGE_WORK_DIR" \
        --reference-fasta "${REFERENCE_FASTA:?set REFERENCE_FASTA}" ;;
    genotype-lower-coverage)
      spike_in_expansions genotype --work-dir "$LOWER_COVERAGE_WORK_DIR" \
        --billing-project "${BILLING_PROJECT:?set BILLING_PROJECT}" \
        --gcs-dir "${GCS_DIR:?set GCS_DIR}/lower_coverage_copies_of_spiked_samples" \
        ${only_missing_outputs_args[@]+"${only_missing_outputs_args[@]}"} ;;
    tables-lower-coverage)
      spike_in_expansions tables --work-dir "$LOWER_COVERAGE_WORK_DIR" ;;
  esac
}

if [ $# -eq 0 ]; then
  set -- "${ALL_STEPS[@]}"
fi
for step in "$@"; do
  case " ${ALL_STEPS[*]} " in
    *" $step "*) ;;
    *) echo "ERROR: unknown step '$step'; the steps are: ${ALL_STEPS[*]}" >&2; exit 1 ;;
  esac
  if [ "$step" = spike ] && [ ! -e "$BWA_INDEX_PREFIX.bwt" ]; then
    echo "ERROR: spike needs the bwa index $BWA_INDEX_PREFIX.* (download the six files" \
      "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.64.{alt,amb,ann,bwt,pac,sa} there," \
      "or set BWA_INDEX_PREFIX)" >&2
    exit 1
  fi
  case "$step" in
    spike) stop_if_folder_holds_spiked_samples "$SPIKED_SAMPLES_WORK_DIR" ;;
    downsample) stop_if_folder_holds_spiked_samples "$LOWER_COVERAGE_WORK_DIR" ;;
  esac
done
for step in "$@"; do
  run_step "$step"
done
echo "DONE: train_model.sh's SPIKED_TABLES_DIRS are $SPIKED_SAMPLES_WORK_DIR/tables and $LOWER_COVERAGE_WORK_DIR/tables"
