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
   raw-EH-vs-gated-LCF MAE chart, per-genotyping_regime held-out accuracy, and — separately for each of
   the two heads — a relative-feature-importance panel and an add-one-feature ablation curve (see
   [Feature importance and ablation](#feature-importance-and-ablation)).

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

## Feature importance and ablation

Both heads get the same two diagnostics, computed independently of each other, because a feature that
pins down *how far off* a call is need not be the one that says *whether* it is off:

| | q-median head (LCF) | direction head (`pOk` / `pTooShort` / `pTooLong`) |
|---|---|---|
| permutation importance | rise in pinball loss (sklearn `permutation_importance`) | rise in multinomial log-loss of the **calibrated** probabilities (`model.predict_proba`, i.e. classifier + isotonic — what EH actually emits) |
| add-one-feature ablation | held-out MAE `mean\|true − eh/LCF\|` | held-out multinomial log-loss |
| `k = 0` anchor | raw EH, no correction | the class prior: that fold's training-set class frequencies emitted for every allele |

Both ablations re-fit their head on the top-1, then top-2, ... features of **their own** importance
order, on one fold and the same rows (`report._ablation_split`), so the two curves describe the same
alleles. Each ablation costs one head fit per prefix length, so the direction curve roughly doubles the
ablation stage's runtime; `report.py --ablation-only` recomputes just the curves from a cached
`results.json`. Alongside the plotted log-loss, every `dir_ablation` point also stores the one-vs-rest
AUCs, average precisions and calibration error, so the curve can be re-plotted against a different
metric without re-fitting.

## Training data

One row per allele **ExpansionHunter actually scores**. EH emits one `AlleleQualityMetrics` entry per
allele and runs the model once per entry, but only **one** entry for a homozygous or hemizygous call —
so the second genotype copy of a hom call is an allele inference never produces. `eh_json.py` still
emits it (the accuracy-by-size report's truth join needs two rows per locus) but marks it
`has_own_quality_metrics = False`, and `dataset.label_and_filter` drops those before training and
isotonic calibration. Without that filter ~15.8% of training rows were duplicate feature vectors
joined to a *different* truth allele than their twin.

**HG002** (10× / 20× / 31×) + **CHM1_CHM13** (46×), illumina WGS, plus 44 single-coverage 1kGP/HPRC
samples (`dataset.PROMOTED_HELDOUT_SAMPLES`; 48 training sources in all) chosen for ancestry/sex
diversity at large allele sizes: 15 of the 18 populations among the 138 samples that have both
a DipCall-based truth set and a Broad short-read CRAM are represented, 23 female / 22 male. Eight of
the 138 are excluded from both training and held-out because their DipCall truth lost almost all of
chrX/chrY (HGSVC2 males listed in str-truth-set-v2's
`filter_vcfs_v2/samples_excluded_from_downstream_analyses.tsv`); the 3 missing populations (IBS, ITU,
MXL) had only excluded samples. The remaining 87 of the 138 are the held-out panel
(`heldout.SAMPLES`); the comment above
`PROMOTED_HELDOUT_SAMPLES` records the selection rule. The model uses **no locus-id feature** — the realistic "train on some samples,
apply to new samples" setting. It *does* use the per-locus `coverage` EH reports
(`LocusResults.Coverage`), so the pool's coverage range (10x-46x) is the range the model is
calibrated over; a sample far outside it is extrapolation. `data/`, `model/`, `report/`, `results/`
are gitignored and rebuilt from sources.

## Feature-order contract

`features.py` (`QUICK_FEATURES` / `FULL_FEATURES` + the engineered CI columns) defines the feature
vector order, and `GenotypeQualityFeatures.cpp` carries the same list on the C++ side. The model JSON
declares its own `feature_names`, and `GenotypeQualityAnnotator.cpp` materializes the model's vector
**by name** from the assembler's output, rejecting a model that asks for a feature the binary does
not produce. So adding a feature does **not** need a `format_version` bump: `format_version` tracks
the serialization schema (trees / softmax / isotonic), and an older binary already rejects a newer
model with a precise "requires feature 'x'" error. What a feature change does require is rebuilding
every parquet (`dataset.py --force`, and `heldout.build_sample(..., force=True)` for the held-out
panel), since the new columns come out of `eh_json.py` at extract time.

`features_tests.py` checks the two lists against each other three ways: `FULL_FEATURES` is
`QUICK_FEATURES` plus the two flank depths, the list matches a spelled-out literal in the test (so a
one-sided edit is deliberate), and — whenever a local `~/code/ExpansionHunter-bw2` checkout is present
— the list parsed straight out of `GenotypeQualityFeatures.cpp` matches too.

## Retraining after an ExpansionHunter change

A feature's *value* can change without its *name* changing, and nothing in the feature-name contract
catches that. The training parquets are parsed from EH's own output JSON, so any EH change that alters
a field the model reads makes every existing parquet describe a binary that no longer exists: the model
would be fit on one distribution and deployed against another, silently.

`dataset._check_freshness` is the guard. Every EH output JSON stamps the build that produced it in
`RunInfo.Version`, and the download step refuses to proceed when that sha differs from the local
`ExpansionHunter-bw2` HEAD, since re-downloading the same object cannot fix it — the JSONs have to be
regenerated by re-running EH. **Note that EH's CMake reads the sha at *configure* time**, so a rebuild
in an existing build directory stamps whatever sha that directory was configured with; re-run `cmake`
before generating JSONs meant for training, or the stamp will name the wrong commit.

## Files

| file | role |
|---|---|
| `train_model.sh` | end-to-end orchestrator |
| `eh_json.py` | EH-JSON → per-allele feature rows (gzip-aware) |
| `features.py` | feature contracts, size-tolerance band, genotyping_regime routing, labels |
| `dataset.py` | GCS download + parquet assembly |
| `model.py` | the two heads + JSON serialization + round-trip verification + vectorized JSON inference |
| `metrics.py` | held-out accuracy / direction / gated-MAE metrics |
| `report.py` | 5-fold CV + HTML report (MAE chart, feature importance, ablation, optional 87-sample held-out section) |
| `heldout.py` | external validation: apply the exported `.json.gz` to the 87 held-out samples |
| `*_tests.py` | unit tests (`python3 -m unittest discover -p "*_tests.py"`) |
| `original/` | the prior multi-module implementation, kept for reference |

## External validation (optional)

`heldout.py` loads the exported model from its `.json.gz` (the exact format ExpansionHunter
consumes) and **applies it unchanged — no re-fitting** — to the 87 held-out short-read samples absent
from the training pool (see Training data above), scoring against their truth. `heldout.py` itself writes a standalone
`report/heldout.json` benchmark dump; the report's held-out HPRC section is fed by the
`report/eval_heldout_hprc.json` / `report/stacked_heldout_hprc.json` artifacts, which `report.py`
**regenerates by default** from the locally-built held-out parquets (via `gen_datasets.generate`, no
download) — pass `--skip-heldout-samples` to opt out. Building those parquets (a ~7-8 GB download for
the original 30 samples; roughly 3x that for all 87) is off by default; enable it with
`RUN_HELDOUT_SAMPLES=1 ./train_model.sh`, which builds them before the report step so the render picks
them up.

## Tests

```bash
python3 -m unittest discover -p "*_tests.py"
```
