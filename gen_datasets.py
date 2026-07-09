"""Generate the per-dataset evaluation artifacts that feed the report's dataset pill.

For each dataset (HG002 genome 31x / HG002 exome 3x / the held-out HPRC samples) this writes:
  - ``report/eval_<key>.json`` + ``report/eval_<key>_violin.npz`` -- the apply-based eval (MAE,
    helped/hurt, by-pOk / pdiff / LCF violins), via ``heldout.run_eval``.
  - ``report/stacked_<key>.json`` -- the stacked "accuracy by true allele size" counts (raw +
    LCF-corrected, each split non-homopolymer / homopolymer), via ``accuracy_by_size``.

Every dataset is scored straight from its per-allele parquet(s) via
``accuracy_by_size.categorize_parquet`` -- the EH calls come from the JSON (flattened into ``eh`` by
``eh_json``) and the truth from the truth-genotypes TSV (joined into ``true`` by ``dataset``), so no
precomputed "for_comparison" TSV is downloaded or read. Parquets built by the current ``eh_json``
carry no-call rows, so the "No Call" category is reconstructed too (except for the legacy hand-built
HG002 exome parquet, which predates that and has none -- see ``datasets``).

The deployed model is applied unchanged (no fitting). Run one dataset at a time (foreground) so the
heavy model-apply survives the environment's background-job limits. Coding rules: no type hints,
Google docstrings, ``print()``.
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

import accuracy_by_size as A
import features
import heldout

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = "EHv5-bw2-optimized"
# Per-parquet allele caps (shared by this CLI and report.py's default held-out regeneration).
EVAL_CAP_DEFAULT = 30_000        # apply-based eval (violins / MAE)
CORRECTED_CAP_DEFAULT = 400_000  # stacked LCF-corrected apply


def _heldout_parquets():
    """Returns the per-sample parquets for the held-out HPRC samples still in ``heldout.SAMPLES``.

    Filters the ``data_eval_43/real_43/`` glob down to ``heldout.SAMPLES`` so samples promoted into
    training (``dataset.PROMOTED_HELDOUT_SAMPLES``) -- whose parquets are left on disk for reuse as
    training-pool symlinks -- are never scored as held-out (that would leak train rows into external
    validation).
    """
    all_parquets = glob.glob(os.path.join(HERE, "data_eval_43", "real_43", "*.parquet"))
    return sorted(p for p in all_parquets
                 if os.path.splitext(os.path.basename(p))[0] in heldout.SAMPLES)


# key -> {label, coverage_label, list of per-allele parquets, optional no_call_note}.
def datasets():
    return {
        "hg002_genome": {
            "label": "HG002 genome (31x)", "coverage_label": "31x Illumina Genome data",
            "parquets": [os.path.join(HERE, "data/real_quick/HG002_31x.parquet")]},
        # Disabled: the legacy hand-built input parquet (data_eval_misc/HG002_exome_3x.parquet) is
        # unavailable and has no downloader, so its eval can't be refreshed against the current model.
        # Re-enable (and uncomment the DATASETS entry in report.py) once the parquet is rebuilt.
        # "hg002_exome": {
        #     "label": "HG002 exome (3x)", "coverage_label": "3x Illumina exome data",
        #     "no_call_note": " (No-Call alleles not shown)",
        #     "parquets": [os.path.join(HERE, "data_eval_misc/HG002_exome_3x.parquet")]},
        "heldout_hprc": {
            "label": "%d held-out HPRC samples" % len(heldout.SAMPLES),
            "coverage_label": "%d held-out HPRC samples (short-read WGS)" % len(heldout.SAMPLES),
            "parquets": _heldout_parquets()},
    }


def gen_stacked(spec, model_path, out_json, corrected_cap):
    """Builds the stacked accuracy-by-size counts (raw + each LCF-correction variant, non-homo + homo).

    Each parquet is categorized via ``accuracy_by_size.categorize_parquet`` (model applied row-aligned,
    no join). The per-sample category counts are ACCUMULATED (additive) rather than concatenated, so a
    many-sample pool stays bounded in memory. ``corrected_cap`` caps the model apply per parquet;
    ``None`` = all alleles.

    The JSON nests ``out[homo|nonhomo][correction_variant][purity_variant][pok_variant]``: each
    ``A.CORRECTION_VARIANTS`` key (``raw`` reads the ``category`` column, every gated variant its
    ``category__<key>`` column) crossed with each ``A.PURITY_VARIANTS`` key (``off`` = all alleles, the
    filtered key keeps only alleles above its truth-purity threshold) crossed with each
    ``A.POK_VARIANTS`` key (``all`` = every allele, the other two split on the model's own predicted
    ``pok`` regardless of which correction variant is selected).
    """
    nb = len(A.X_LABELS)
    variants = [(key, "category" if gate is None else "category__" + key)
                for key, _, _, gate in A.CORRECTION_VARIANTS]
    purities = [(pk, pmin) for pk, _, pmin in A.PURITY_VARIANTS]
    poks = [(kk, mode) for kk, _, mode in A.POK_VARIANTS]
    acc = {(h, vk, pk, kk): {"counts": {cat: np.zeros(nb, int) for cat in A.CATEGORIES},
                             "alleles_per_bin": np.zeros(nb, int), "same": 0, "total": 0, "loci": set()}
           for h in (False, True) for vk, _ in variants for pk, _ in purities for kk, _ in poks}

    def fold(m, src):
        for h in (False, True):
            for vk, col in variants:
                for pk, pmin in purities:
                    for kk, mode in poks:
                        bc = A.bin_counts(m, col, h, purity_min=pmin, pok_stratum=mode)
                        a = acc[(h, vk, pk, kk)]
                        for cat in A.CATEGORIES:
                            a["counts"][cat] += np.asarray(bc["counts"][cat], int)
                        a["alleles_per_bin"] += np.asarray(bc["alleles_per_bin"], int)
                        a["same"] += bc["same"]
                        a["total"] += bc["total"]
                        a["loci"].update(bc["loci"])  # union distinct loci across samples (catalog loci repeat across held-out samples)
        print("  stacked: %s (%d alleles)" % (src, len(m)), flush=True)

    for p in spec["parquets"]:
        fold(A.categorize_parquet(p, model_path, corrected_cap=corrected_cap), os.path.basename(p))

    out = {"label": spec["label"], "coverage_label": spec["coverage_label"],
           "tool_label": A.TITLE_TOOL_LABELS[TOOL]}
    if spec.get("no_call_note"):
        out["no_call_note"] = spec["no_call_note"]
    for homo, name in ((False, "nonhomo"), (True, "homo")):
        out[name] = {}
        for vk, _ in variants:
            out[name][vk] = {}
            for pk, _ in purities:
                out[name][vk][pk] = {}
                for kk, _ in poks:
                    a = acc[(homo, vk, pk, kk)]
                    out[name][vk][pk][kk] = {
                        "counts": {cat: a["counts"][cat].tolist() for cat in A.CATEGORIES},
                        "alleles_per_bin": a["alleles_per_bin"].tolist(),
                        "same": int(a["same"]), "total": int(a["total"]),
                        "total_loci": int(len(a["loci"]))}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f)
    print("wrote %s" % out_json, flush=True)


def _check_feature_columns(parquets):
    """Raises a clear error if a parquet predates the current model feature contract.

    Catches a cached external-validation parquet built before a ``features.py`` feature-list
    change (e.g. it lacks a newly added column) with an actionable message, instead of the
    unrelated-looking ``AssertionError`` that ``features.build_matrix`` would raise deep inside
    ``heldout.run_eval``.
    """
    required = set(features.FULL_FEATURES) - {"ci_asymmetry", "ci_over_eh"}  # engineered by add_engineered
    for p in parquets:
        missing = required - set(pq.ParquetFile(p).schema.names)
        if missing:
            raise RuntimeError(
                "%s is missing feature column(s) %s -- it was built with an older feature "
                "contract. Rebuild it (re-run its download/extract step, e.g. "
                "heldout.build_sample(..., force=True) for a held-out-43 sample) before "
                "re-running gen_datasets.py." % (p, sorted(missing)))


def generate(dataset_key, model_path, out_dir, eval_cap=EVAL_CAP_DEFAULT,
             corrected_cap=CORRECTED_CAP_DEFAULT, skip_eval=False):
    """Writes one dataset's eval + stacked artifacts into ``out_dir`` (no download; no fitting).

    Applies the exported ``model_path`` to that dataset's local per-allele parquets. Returns the number
    of parquets processed -- 0 (a no-op that writes nothing) when none are present locally, so callers
    like report.py can regenerate the held-out section only when its parquets have been built. Shared by
    this module's CLI and report.py's default held-out regeneration.
    """
    spec = datasets()[dataset_key]
    parquets = spec["parquets"]
    print("==== dataset %s: %d parquet(s) ====" % (dataset_key, len(parquets)), flush=True)
    if not parquets:
        return 0
    _check_feature_columns(parquets)
    if not skip_eval:
        heldout.run_eval(parquets, model_path,
                         os.path.join(out_dir, "eval_%s.json" % dataset_key), eval_cap)
    print("\n==== stacked accuracy-by-size: %s ====" % dataset_key, flush=True)
    gen_stacked(spec, model_path, os.path.join(out_dir, "stacked_%s.json" % dataset_key),
                corrected_cap or None)
    return len(parquets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(datasets().keys()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "report"))
    parser.add_argument("--eval-cap", type=int, default=EVAL_CAP_DEFAULT,
                        help="per-parquet allele cap for the apply-based eval (violins/MAE)")
    parser.add_argument("--corrected-cap", type=int, default=CORRECTED_CAP_DEFAULT,
                        help="per-parquet allele cap for the stacked LCF-corrected apply (0 = all)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="only (re)generate the stacked-bar JSON")
    args = parser.parse_args()

    if generate(args.dataset, args.model, args.out_dir, args.eval_cap,
                args.corrected_cap, args.skip_eval) == 0:
        print("no parquet(s) found for dataset %s -- nothing to do" % args.dataset, flush=True)


if __name__ == "__main__":
    main()
