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

import accuracy_by_size as A
import heldout
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = "EHv5-bw2-optimized"
# Per-parquet allele caps (shared by this CLI and report.py's default held-out regeneration).
EVAL_CAP_DEFAULT = 30_000        # apply-based eval (violins / MAE)
CORRECTED_CAP_DEFAULT = 400_000  # stacked accuracy-by-size: whole-locus cohort for raw AND corrected panels
DEFAULT_DATA_DIR = os.path.join(HERE, "data")  # training data dir (dataset.py / report.py --data-dir)


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
def datasets(data_dir=DEFAULT_DATA_DIR):
    """Returns the dataset specs; ``data_dir`` is the training data dir holding ``real_quick/``."""
    heldout_parquets = _heldout_parquets()
    return {
        "hg002_genome": {
            "label": "HG002 genome (31x)",
            "coverage_label": "HG002 31x Illumina genome (a training sample)",
            "parquets": [os.path.join(data_dir, "real_quick", "HG002_31x.parquet")]},
        # Disabled: the legacy hand-built input parquet (data_eval_misc/HG002_exome_3x.parquet) is
        # unavailable and has no downloader, so its eval can't be refreshed against the current model.
        # Re-enable (and uncomment the DATASETS entry in report.py) once the parquet is rebuilt.
        # "hg002_exome": {
        #     "label": "HG002 exome (3x)", "coverage_label": "3x Illumina exome data",
        #     "no_call_note": " (No-Call alleles not shown)",
        #     "parquets": [os.path.join(HERE, "data_eval_misc/HG002_exome_3x.parquet")]},
        # Labels count the parquets actually on disk, not heldout.SAMPLES: a partially-built panel
        # would otherwise produce artifacts captioned with the full cohort size.
        "heldout_hprc": {
            "label": "%d held-out HPRC samples" % len(heldout_parquets),
            "coverage_label": "%d held-out HPRC samples (short-read WGS)" % len(heldout_parquets),
            "parquets": heldout_parquets},
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
    ``A.POK_VARIANTS`` key (``all`` = every allele, every other key keeps only the alleles below, or at
    or above, its threshold on the model's own predicted ``pok``, regardless of which correction variant
    is selected; the "below" strata are nested, not disjoint).
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

    out = {"model": M.fingerprint(model_path), "label": spec["label"],
           "coverage_label": spec["coverage_label"], "tool_label": A.TITLE_TOOL_LABELS[TOOL]}
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


def generate(dataset_key, model_path, out_dir, eval_cap=EVAL_CAP_DEFAULT,
             corrected_cap=CORRECTED_CAP_DEFAULT, skip_eval=False, data_dir=DEFAULT_DATA_DIR):
    """Writes one dataset's eval + stacked artifacts into ``out_dir`` (no download; no fitting).

    Applies the exported ``model_path`` to that dataset's local per-allele parquets. Returns the number
    of parquets processed -- 0 (a no-op that writes nothing) when none are present locally, so callers
    like report.py can regenerate the held-out section only when its parquets have been built. Shared by
    this module's CLI and report.py's default held-out regeneration.
    """
    spec = datasets(data_dir)[dataset_key]
    # Configured paths are not necessarily present: hg002_genome names one path unconditionally, so
    # without this filter an unbuilt dataset raised FileNotFoundError instead of the documented no-op.
    parquets = [p for p in spec["parquets"] if os.path.exists(p)]
    missing = [p for p in spec["parquets"] if p not in parquets]
    if missing:
        print("  %d configured parquet(s) not built locally, skipping them: %s"
              % (len(missing), ", ".join(os.path.basename(p) for p in missing)), flush=True)
    print("==== dataset %s: %d parquet(s) ====" % (dataset_key, len(parquets)), flush=True)
    if not parquets:
        return 0
    spec = dict(spec, parquets=parquets)
    # run_eval checks this too, but --skip-eval bypasses run_eval entirely, so the stacked-accuracy
    # pass below would otherwise read a stale parquet unguarded.
    heldout.assert_parquets_carry_contract(parquets)
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
                        help="per-parquet cap on the called alleles ExpansionHunter scores (one per "
                             "homozygous call) for the stacked accuracy-by-size plots; it selects a seeded "
                             "whole-locus sample used by the raw and the corrected panels alike, which also "
                             "keeps both copies of each homozygous call and the no-calls (0 = all)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="only (re)generate the stacked-bar JSON")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="training data dir holding real_quick/ (for the HG002 genome dataset)")
    args = parser.parse_args()

    if generate(args.dataset, args.model, args.out_dir, args.eval_cap,
                args.corrected_cap, args.skip_eval, args.data_dir) == 0:
        print("no parquet(s) found for dataset %s -- nothing to do" % args.dataset, flush=True)


if __name__ == "__main__":
    main()
