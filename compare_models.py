"""Compares a newly-trained genotype-quality model against the previous/deployed model on the
held-out HPRC benchmark, printing per-regime metric deltas. Run at the end of ``train_model.sh``.

Both models are applied UNCHANGED (no fitting) to the same held-out parquets -- the samples in
``heldout.SAMPLES``, absent from either model's training pool -- so the comparison is a fair
generalization test. Metrics (raw/gated MAE, distance reduction, pOk accuracy, and the pOk<0.5
net-helped count) reuse ``heldout.run_eval`` with the same per-sample allele cap the report uses
(``gen_datasets.EVAL_CAP_DEFAULT``), so the new model's numbers match the report's held-out section.

The "previous model" defaults to whichever model is deployed in the local ExpansionHunter-bw2 checkout
(``$EXPANSIONHUNTER_BW2_REPO/ehunter/data/genotype_quality_model_*.json.gz`` -- i.e. the "should we
ship the new one?" comparison), falling back to the most recent other ``model/*.json.gz`` in this repo.
Override with ``--prev-model``.

Self-skips (prints a note, exit 0) when the held-out parquets haven't been built (no
``RUN_HELDOUT_SAMPLES=1`` run) or no previous model exists, so it is safe to always invoke.
"""

import argparse
import glob
import os
import re
import tempfile

import dataset
import features
import gen_datasets
import heldout
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))


def default_prev_model(new_model):
    """Returns the previous model to compare against, or None.

    Prefers the model deployed in the ExpansionHunter-bw2 checkout (the one currently compiled into
    the binary); falls back to the most recent other dated ``model/*.json.gz`` in this repo.
    """
    deployed = sorted(glob.glob(os.path.join(
        dataset.EXPANSIONHUNTER_BW2_REPO, "ehunter", "data", "genotype_quality_model_*.json.gz")),
        key=_model_sort_key)
    if deployed:
        return deployed[-1]
    others = sorted((p for p in glob.glob(os.path.join(HERE, "model", "*.json.gz"))
                     if os.path.abspath(p) != os.path.abspath(new_model)),
                    key=_model_sort_key)
    return others[-1] if others else None


def _model_sort_key(path):
    """Sorts dated model files by their trailing ``YYYYMMDD`` token, then by name.

    A plain filename sort is wrong here: the prefix dominates it, so
    ``..._plus13.20260707.json.gz`` sorts after ``....20260708.json.gz`` and the OLDER model wins.
    A file with no parseable date sorts first, so a dated model is always preferred.
    """
    match = re.search(r"\.(\d{8})\.json(?:\.gz)?$", os.path.basename(path))
    return (match.group(1) if match else "", os.path.basename(path))


def _winner(prev, new, higher_is_better):
    """Returns 'new'/'prev'/'tie' for a metric where new/prev are the two values."""
    if new == prev:
        return "tie"
    return "new" if (new > prev) == higher_is_better else "prev"


def compare(new_model, prev_model, data_dir, max_alleles):
    """Applies both models to the held-out parquets and prints the per-regime comparison."""
    paths = sorted(p for p in (os.path.join(data_dir, "real_43", "%s.parquet" % s)
                               for s in heldout.SAMPLES) if os.path.exists(p))
    if not paths:
        print("\n[compare_models] skipped: no held-out parquets under %s/real_43/ -- run with "
              "RUN_HELDOUT_SAMPLES=1 to build them." % data_dir)
        return
    if prev_model is None:
        print("\n[compare_models] skipped: no previous model found to compare against.")
        return
    if os.path.abspath(prev_model) == os.path.abspath(new_model):
        print("\n[compare_models] skipped: previous model resolves to the new model itself.")
        return
    print("\n==================== MODEL COMPARISON (held-out HPRC, %d samples, cap %s alleles/sample) "
          "====================" % (len(paths), max_alleles or "none"))
    print("  prev: %s" % os.path.basename(prev_model))
    print("  new:  %s" % os.path.basename(new_model))
    # A feature-contract change is the MAIN reason to run this comparison (does the model with the new
    # features actually beat the deployed one?), so differing contracts are reported, not a reason to
    # skip: run_eval builds each model's matrix from that model's own declared feature_names, and both
    # models still score the same rows of the same parquets.
    prev_names, new_names = (M.feature_names_of(M.load(p), p) for p in (prev_model, new_model))
    if prev_names != new_names:
        added = [f for f in new_names[features.BRANCH_FULL] if f not in prev_names[features.BRANCH_FULL]]
        dropped = [f for f in prev_names[features.BRANCH_FULL] if f not in new_names[features.BRANCH_FULL]]
        print("  note: the two models declare different feature contracts (prev %d/%d quick/full, new "
              "%d/%d); added: %s; dropped: %s"
              % (len(prev_names[features.BRANCH_QUICK]), len(prev_names[features.BRANCH_FULL]),
                 len(new_names[features.BRANCH_QUICK]), len(new_names[features.BRANCH_FULL]),
                 ", ".join(added) or "(none)", ", ".join(dropped) or "(none)"))
    # run_eval writes an out_json + violin npz; both are throwaway here, so use a temp dir.
    with tempfile.TemporaryDirectory() as td:
        prev = heldout.run_eval(paths, prev_model, os.path.join(td, "prev.json"), max_alleles)
        new = heldout.run_eval(paths, new_model, os.path.join(td, "new.json"), max_alleles)

    wins = {"new": 0, "prev": 0, "tie": 0}
    for rk in features.GENOTYPING_REGIMES:
        p = prev["genotyping_regimes"][rk]
        n = new["genotyping_regimes"][rk]
        if not p.get("n"):
            continue
        pnet = p["helped_lt"] - p["hurt_lt"]
        nnet = n["helped_lt"] - n["hurt_lt"]
        print("\n  [%s] n=%d  (raw MAE %.4f, identical input for both)" % (rk, p["n"], p["mae_raw"]))
        for label, pv, nv, fmt, higher in (
                ("gated MAE",            p["mae_gated"],      n["mae_gated"],      "%.4f", False),
                ("pOk accuracy",         p["p_ok_accuracy"],  n["p_ok_accuracy"],  "%.4f", True),
                ("median |err| (gated)", p["median_gated"],   n["median_gated"],   "%.3f", False),
                ("net helped (pOk<0.5)", pnet,                nnet,                "%+d",  True)):
            win = _winner(pv, nv, higher)
            wins[win] += 1
            print("     %-22s prev %-11s new %-11s -> %s" % (label, fmt % pv, fmt % nv, win))
        # Printed for context but deliberately NOT tallied: dist_reduction = 1 - mae_gated/mae_raw and
        # mae_raw is identical for both models, so it can only ever agree with the gated-MAE row --
        # counting it would weight that one signal twice and could flip the verdict on its own.
        print("     %-22s prev %-11s new %-11s    (derived from gated MAE; not counted)"
              % ("distance reduction", "%+.4f" % p["dist_reduction"], "%+.4f" % n["dist_reduction"]))
    verdict = ("NEW is better overall" if wins["new"] > wins["prev"]
               else "PREVIOUS is better overall" if wins["prev"] > wins["new"]
               else "MIXED / roughly tied")
    print("\n  VERDICT: %s  (metric wins -- new: %d, prev: %d, tie: %d)"
          % (verdict, wins["new"], wins["prev"], wins["tie"]))
    print("=" * 108)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-model", required=True, help="the newly-trained model .json[.gz]")
    parser.add_argument("--prev-model", default=None,
                        help="model to compare against (default: the deployed ExpansionHunter-bw2 "
                             "model, else the most recent other model/*.json.gz)")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data_eval_43"),
                        help="dir holding the held-out real_43/<sample>.parquet files")
    parser.add_argument("--max-alleles-per-sample", type=int, default=gen_datasets.EVAL_CAP_DEFAULT,
                        help="seeded per-sample allele cap (0 = all alleles); both models use the same "
                             "cap+seed so they score identical rows. Defaults to the same cap the report's "
                             "held-out section uses (gen_datasets.EVAL_CAP_DEFAULT) so the new model's "
                             "numbers match it.")
    args = parser.parse_args()
    compare(args.new_model, args.prev_model or default_prev_model(args.new_model),
            args.data_dir, args.max_alleles_per_sample)


if __name__ == "__main__":
    main()
