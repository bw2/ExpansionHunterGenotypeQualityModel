# ExpansionHunterGenotypeQualityModel

Training harness for the per-allele **genotype-quality model** used by the bw2 fork of
[ExpansionHunter](https://github.com/Illumina/ExpansionHunter). Given an EH allele call it
predicts:

- **`LENGTH_CORRECTION_FACTOR`** = `eh/true` (the q-head median; `<1` call too short, `>1` too
  long; recovered truth ≈ `AlleleSize / LCF`),
- **`P_OK` / `P_TOO_SHORT` / `P_TOO_LONG`** — the direction head's calibrated class probabilities.

The trained model is serialized to a gzipped JSON that is compiled into the EH binary
(`ehunter/genotype_quality/` reads it; see `GENOTYPE_QUALITY_CPP_INTEGRATION_PLAN.md`). The
downstream consumer rule is "apply the LCF only when `P_OK < 0.5`" — EH emits the raw fields, it
does no thresholding itself.

## Model structure

Three **regime experts**, routed per allele (mirrors `size_tolerance.regime_of`):

| regime | rows it serves |
|---|---|
| `fast` | QuickGenotype (fast-path) alleles |
| `full_spanning` | full-genotyper alleles with ≥1 spanning read at the called size |
| `full_nonspanning` | full-genotyper alleles with 0 spanning reads (flanking/IRR) |

Each expert = two `HistGradientBoosting` heads (scikit-learn):
- **q-median** (`model_q.py`) — quantile regressor on `t = log(eh) − log(true)`; `LCF = exp(t)`.
- **direction** (`model_direction.py`) — 3-class classifier + per-class isotonic calibration.

## Pipeline

```
EH JSON + truth set ──> data builders ──> per-allele parquet ──> build_dataset (labels: t, dir_code, regime)
  data_gcs / data_sim / data_fast_real / data_43           │
                                                           ├─> run_cv.py        chrom-clean CV training + eval
                                                           └─> export_gq_model.py  fit + serialize -> model .json.gz
```

- `run_cv.py` — chromosome-clean cross-validated training/eval (regimes, size-bin weighting,
  ablation/diagnostics/explain reporting). The research entry point.
- `export_gq_model.py` — fits the deployable per-regime experts and **exports the `.json.gz`** in
  the C++ schema, round-trip-verifying that the serialized trees/softmax/isotonic reproduce
  sklearn's predictions. The deployment entry point.

```bash
pip install -r requirements.txt
python3 export_gq_model.py --out model/genotype_quality_model.<date>.json.gz
# then drop that file into ExpansionHunter at ehunter/data/genotype_quality_model.json.gz and rebuild
```

## Training data

The committed model is trained on **HG002** (downsampled 10× / 20× / 31×) + **CHM1_CHM13** (46×),
real data only (simulated rows excluded). Models use **no locus-id or raw-coverage feature** — the
realistic "train on some samples, apply to new samples" regime.

## ⚠ Feature-order contract

`features.py` (`FAST_FEATURES` / `FULL_FEATURES`, plus the engineered features) defines the feature
vector order. The C++ side (`GenotypeQualityFeatures.cpp`) must assemble features in the **exact
same order** — the JSON carries `feature_names` for this contract. There is no automatic cross-repo
check yet, so: bump `format_version` on any feature change, and keep an EH-side golden test that the
C++ assembler order equals the exported `feature_names`.

## Data dependencies (not in this repo)

`data/`, `data_eval*/`, `results/` (multi-GB parquets / npz / joblib) are gitignored and rebuilt
from sources. The data builders (`data_gcs.py`) additionally require Hail Batch + GCS access and the
str-truth-set-v2 comparison-table outputs.

## Tests

```bash
python3 -m pytest -q   # *_tests.py
```
