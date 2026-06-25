# ExpansionHunterGenotypeQualityModel

Training pipeline for the per-allele **genotype-quality model** used by the bw2 fork of
[ExpansionHunter](https://github.com/Illumina/ExpansionHunter). Given an EH allele call it predicts:

- **`PredictedLengthCorrectionFactor`** = `eh/true` (the q-median head; `<1` call too short, `>1` too long;
  recovered truth ≈ `AlleleSize / LCF`),
- **`pOk` / `pTooShort` / `pTooLong`** — the direction head's calibrated class probabilities.

The trained model serializes to a gzipped JSON (`format_version: 2`) that the EH binary compiles in
(`ehunter/genotype_quality/GenotypeQualityModel.cpp` parses it). The downstream rule is "apply the LCF
only when `pOk < 0.5`" — EH emits the raw fields and does no thresholding itself.

## One command

```bash
pip install -r requirements.txt
./train_model.sh
```

`train_model.sh` runs three idempotent stages and writes the dated model + an HTML report:

1. **`dataset.py`** — downloads the EH JSON shards + truth TSVs from the `str-truth-set-v2` GCS
   buckets (skipping anything already local) and assembles `data/parquet/{quick,full}.parquet`.
2. **`train.py`** — fits the three genotyping_regime experts and exports
   `model/genotype_quality_model_from_HG002_and_CHM1_CHM13.<date>.json.gz`, round-trip-verifying that
   the serialized trees/softmax/isotonic reproduce sklearn's predictions.
3. **`report.py`** — 5-fold chromosome-clean cross-validation, then `report/model_report.html` with the
   raw-EH-vs-gated-LCF MAE chart, per-genotyping_regime held-out accuracy, and relative feature importance.

Requires `gcloud`/`gsutil` authenticated for `gs://str-truth-set-v2` (read access).

## Model structure

Three **genotyping_regime experts**, routed per allele (`features.genotyping_regime_of`):

| genotyping_regime | rows it serves |
|---|---|
| `quick` | QuickGenotype fast-path (`processLocusFast`) alleles |
| `full_spanning` | full-genotyper alleles with ≥1 spanning read at the called size |
| `full_nonspanning` | full-genotyper alleles with 0 spanning reads (flanking/IRR) |

Each expert = two scikit-learn `HistGradientBoosting` heads (`model.py`):
- **q-median** — quantile regressor on `t = log(eh) − log(true)`; `LCF = exp(t)`.
- **direction** — 3-class classifier + per-class isotonic calibration.

Both use a manual `warm_start` early-stopping loop scored on a held-out **chromosome-clean**
calibration set (never sklearn's internal random-split early stopping, which would leak loci).

## Training data

**HG002** (10× / 20× / 31×) + **CHM1_CHM13** (46×), illumina WGS. The model uses **no
locus-id or raw-coverage feature** — the realistic "train on some samples, apply to new samples"
setting. `data/`, `model/`, `report/`, `results/` are gitignored and rebuilt from sources.

## Feature-order contract

`features.py` (`QUICK_FEATURES` / `FULL_FEATURES` + the engineered CI columns) defines the feature
vector order. The C++ side (`GenotypeQualityFeatures.cpp`) must assemble features in the **exact same
order**; the JSON carries `feature_names` for this contract. Bump `format_version` on any feature
change.

## Files

| file | role |
|---|---|
| `train_model.sh` | end-to-end orchestrator |
| `eh_json.py` | EH-JSON → per-allele feature rows (gzip-aware) |
| `features.py` | feature contracts, size-tolerance band, genotyping_regime routing, labels |
| `dataset.py` | GCS download + parquet assembly |
| `model.py` | the two heads + JSON serialization + round-trip verification + vectorized JSON inference |
| `metrics.py` | held-out accuracy / direction / gated-MAE metrics |
| `report.py` | 5-fold CV + HTML report (MAE chart, feature importance, ablation, optional 43-sample section) |
| `holdout43.py` | external validation: apply the exported `.json.gz` to 43 held-out HPRC samples |
| `*_tests.py` | unit tests (`python3 -m pytest -q`) |
| `original/` | the prior multi-module implementation, kept for reference |

## External validation (optional)

`holdout43.py` loads the exported model from its `.json.gz` (the exact format ExpansionHunter
consumes) and **applies it unchanged — no re-fitting** — to 43 HPRC short-read samples absent from the
training pool, scoring against their truth. It writes `report/holdout43.json`, which `report.py` folds
into the report as an "external held-out validation" section. Run via `RUN_HOLDOUT43=1 ./train_model.sh`
or directly: `python3 holdout43.py --model <model.json.gz>` (a ~7-8 GB download).

## Tests

```bash
python3 -m pytest -q
```
