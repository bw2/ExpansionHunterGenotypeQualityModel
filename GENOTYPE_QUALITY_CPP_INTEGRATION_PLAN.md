# Genotype-quality model → ExpansionHunter C++ integration plan

Wire the `genotype_quality/` model into the EH binary so it emits per-allele quality
fields. Decisions locked over sessions 2026-06-22 .. 2026-06-24.

**STATUS 2026-06-24: Steps 1–5 IMPLEMENTED + verified (301 unit tests pass; end-to-end
fast+full paths emit fields; default/no-model output byte-identical minus the 4 keys).
Only remaining item is Step 6 — the deferred python export producing the real
`ehunter/data/genotype_quality_model.json.gz` (currently an empty/inert placeholder).**

## Output contract (per allele, inside `AlleleQualityMetrics.Alleles[]`)

- `LENGTH_CORRECTION_FACTOR` = `eh/true` (q median head, `= exp(t)`). `<1` call too short,
  `>1` too long. Recover truth ≈ `AlleleSize / LCF`.
- `P_OK`, `P_TOO_SHORT`, `P_TOO_LONG` (direction head; three sum to ~1 after rounding).
  P_OK is emitted explicitly (it is the consumer gate variable: apply LCF when P_OK<0.5).
- UPPER_SNAKE names (match the python), emitted only when a model is loaded AND the
  allele has an AlleleQualityMetrics record. Rounded to 3 decimals (round3, like depth).

These are emitted ALWAYS (when a model is loaded); the "apply LCF only when P_OK<0.5"
gate is a downstream consumer rule, not an EH-side decision. No EH-side thresholding.

## Model delivery

- The model is COMPILED INTO the binary: a gzipped JSON file checked into the repo at
  `ehunter/data/genotype_quality_model.json.gz` is embedded at build time as a generated
  byte-array source (`cmake/EmbedModel.cmake` + an add_custom_command in CMakeLists.txt →
  `GenotypeQualityModelData.cpp` defining `kEmbeddedModelGz[]`/`kEmbeddedModelGzLen`).
- `--genotype-quality-model PATH` (Advanced options) overrides the embedded model at
  runtime, loading a `.json` or `.json.gz`.
- The checked-in file is currently EMPTY (0 bytes) → the generator emits `len = 0` →
  `loadEmbeddedGenotypeQualityModel()` returns null → no quality fields by default
  (output unchanged). Dropping in the real exported model + rebuilding compiles it in.
- In-memory gunzip (zlib inflate, gzip window 15+16) decompresses the embedded blob;
  the flag path uses zlib gzread (transparent for plain + gzip files).

## Model JSON schema (the export contract this C++ reader defines)

```jsonc
{
  "format_version": 1,
  "feature_names": { "fast": [...20...], "full": [...22...] },   // order == features.py
  "regimes": {
    "fast":            { "q_median": <QHEAD>, "direction": <DIRHEAD> },
    "full_spanning":   { "q_median": <QHEAD>, "direction": <DIRHEAD> },
    "full_nonspanning":{ "q_median": <QHEAD>, "direction": <DIRHEAD> }
  }
}
// QHEAD   = { "baseline": <float>, "trees": [ <TREE>, ... ] }            // predicts t; LCF=exp(t)
// DIRHEAD = { "baseline": [<f>,<f>,<f>],                                  // [OK,TOO_LONG,TOO_SHORT]
//             "trees": [ [<TREE_ok>,<TREE_long>,<TREE_short>], ... ],     // one triple per boosting iter
//             "calibrators": [ <ISO_ok>, <ISO_long>, <ISO_short> ] }
// TREE    = { "nodes": [ <NODE>, ... ] }   // flat array, root=index 0
// NODE leaf:     { "leaf": true, "value": <float> }
// NODE internal: { "feature": <int>, "threshold": <float>, "missing_left": <bool>,
//                  "left": <int>, "right": <int> }
// ISO     = { "x": [<float>...], "y": [<float>...], "increasing": true } // clip out-of-bounds; passthrough = empty x/y
```

## Inference math (must match python bit-for-close)

- tree.eval(x): n=0; while !nodes[n].leaf: n = (isnan(x[f]) ? (missing_left?L:R) : (x[f]<=thr?L:R)); return value.
- q:   t = baseline + Σ_trees eval(x);  LCF = exp(t).
- dir: raw[c] = baseline[c] + Σ_iters trees[iter][c].eval(x);  p = softmax(raw);
       cal[c] = clip(iso[c](p[c]), 0, 1);  s = Σ cal;  out = s>0 ? cal/s : {1/3,1/3,1/3};
       P_OK=out[0], P_TOO_LONG=out[1], P_TOO_SHORT=out[2].
- iso(v): empty(passthrough) ⇒ v; else clipped linear interpolation over (x,y) knots,
          clamp v to [x0,x_last] then piecewise-linear (y clamped to [y0,y_last]).

## Regime routing (C++, per allele, mirrors size_tolerance.regime_of)

- `fast` if the variant is QuickGenotype; else `spanning_at_called>=1 ? full_spanning : full_nonspanning`.

## Feature assembly (C++ at the writer, mirrors eh_json_features.extract_variant_rows)

All from typed accessors on `repeatFindings` + `variantSpec_` (NOT by re-parsing JSON strings):
motif_size, num_repeats_in_reference, ref_size_bp, eh, eh_minus_ref, allele_rank,
ci_width, ci_asymmetry, ci_over_eh, spanning_total, hq_unamb_total, spanning_at_called,
spanning_above_called, flanking_above_called, support_frac, depth, hq_unambiguous_reads,
strand_bias_phred, mean_inserted_bases, mean_deleted_bases (+ full: left/right_flank_norm_depth).
Engineered: ci_asymmetry=((ci_end-eh)-(eh-ci_start))/(ci_width+1); ci_over_eh=ci_width/(eh+1)
(0 when inputs missing). NaN passed through (model handles natively).

## Integration point

- Single chokepoint: `VariantJsonWriter::visit(RepeatFindings*)` (JsonWriter.cpp), used by
  BOTH batch `JsonWriter` and streaming `IterativeJsonWriter` (both construct a
  VariantJsonWriter per variant). AQM is *computed* in two sites (LocusAnalyzer.cpp:269 +
  HtsLowMemStreamingHelpers.cpp:725) — the writer is the only shared serialization point.
- Thread a `const GenotypeQualityModel*` (nullptr ⇒ skip) into VariantJsonWriter via both
  writer ctors, sourced from ProgramParameters (model loaded once at startup).

## Files

NEW:
- `ehunter/genotype_quality/GenotypeQualityModel.hh/.cpp` — schema structs + JSON(.gz) loader.
- `ehunter/genotype_quality/GenotypeQualityEvaluator.hh/.cpp` — tree/softmax/isotonic eval + feature assembly + regime routing.
- `ehunter/genotype_quality/GenotypeQualityModel_tests.cpp` — fixture-model unit tests.
- `ehunter/data/genotype_quality_model.json.gz` — placeholder model (real one from export step).

EDIT:
- `ehunter/io/ParameterLoading.cpp` (+`.hh`, UserParameters) — `--genotype-quality-model` flag.
- `ehunter/core/Parameters.hh` — carry the loaded model (shared_ptr) on ProgramParameters.
- `ehunter/io/JsonWriter.hh/.cpp` + `ehunter/io/IterativeJsonWriter.hh/.cpp` — thread model into VariantJsonWriter; emit the 3 fields.
- `ehunter/app/*` (main) — load the model once.
- `ehunter/CMakeLists.txt` + tests CMake — new sources, configured default-path header.

## Build order (each step independently testable, default behavior unchanged until last)

1. Evaluator + schema structs + loader, with a hand-written fixture model + unit tests
   (pure, no EH wiring). ← START HERE.
2. Feature assembly + regime routing from typed findings, unit-tested on a synthetic variant.
3. CLI flag + Parameters carrier + startup load (no output change yet).
4. Writer wiring: emit the 3 fields when model present. Golden-output test.
5. Placeholder model file + CMake default-path plumbing.
6. (Deferred, separate task) python export producing the real `.json.gz`.
```
