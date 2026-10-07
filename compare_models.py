"""Compares a newly-trained genotype-quality model against the previous/deployed model on the
held-out HPRC benchmark, printing per-regime metric deltas. Run at the end of ``train_model.sh``.

Both models are applied UNCHANGED (no fitting) to the same held-out parquets -- the samples in
``heldout.SAMPLES``, absent from either model's training pool -- and to the same rows, via
``heldout.run_eval`` with the per-sample allele cap the report uses (``gen_datasets.EVAL_CAP_DEFAULT``).

Results are reported per genotyping regime AND per motif stratum (homopolymer vs non-homopolymer;
``run_eval`` keeps them apart, and homopolymers are over half of the alleles EH scores). The verdict
counts only gated MAE, as a per-sample paired difference with a bootstrap interval over samples; gate
decision accuracy, 3-class direction accuracy, the median and the net-helped count are printed as
context. The result is also given without ``heldout.FASTPATH_DIAGNOSIS_SAMPLES``, which were used during
model development. The evaluated population is variant loci only (truth differs from the reference)
and one row per emitted quality-metric entry; the printout says so.

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

import numpy as np

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


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260616
MIN_SAMPLES_FOR_VERDICT = 10  # fewer samples in a stratum: reported, but it cannot be won or lost
# The two motif strata run_eval reports per genotyping regime: (label, key of its metrics within the
# regime's summary or None for the top level, per-sample allele-count key, per-sample error-sum key).
MOTIF_STRATA = (("non-homopolymer", None, "n", "sum_gated_abs_error"),
                ("homopolymer", "homopolymer", "homopolymer_n", "homopolymer_sum_gated_abs_error"))


def _per_sample_gated_mae_deltas(prev_per_sample, new_per_sample, n_key, sum_key, samples):
    """Returns new-minus-previous gated MAE for each of ``samples`` that has alleles in this stratum."""
    return np.array([new_per_sample[s][sum_key] / new_per_sample[s][n_key]
                     - prev_per_sample[s][sum_key] / prev_per_sample[s][n_key]
                     for s in samples if s in prev_per_sample and prev_per_sample[s][n_key] > 0])


def _bootstrap_mean_interval(values):
    """Returns the 95% percentile-bootstrap interval of the mean of ``values`` (resampling samples)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _pooled_gated_mae(per_sample, n_key, sum_key, samples):
    """Returns the gated MAE pooled over ``samples`` (all alleles weighted equally)."""
    n = sum(per_sample[s][n_key] for s in samples if s in per_sample)
    return sum(per_sample[s][sum_key] for s in samples if s in per_sample) / n if n else float("nan")


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

    print("  scope: alleles at loci whose truth genotype differs from the reference (truth hom-ref loci are "
          "not built into the held-out parquets), and one row per emitted quality-metric entry (the second "
          "copy of a homozygous call is not scored, so a missed heterozygous allele is not counted).")
    print("  verdict: per genotyping regime and motif stratum, the per-sample paired difference in gated MAE "
          "(new - prev) with a %d-resample bootstrap 95%% interval over samples; a stratum is won only when "
          "the interval excludes 0. The other rows are context and are not counted." % BOOTSTRAP_RESAMPLES)

    samples = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    without_diagnosis = [s for s in samples if s not in heldout.FASTPATH_DIAGNOSIS_SAMPLES]
    wins = {"new": 0, "prev": 0, "tie": 0, "too few samples": 0}
    for rk in features.GENOTYPING_REGIMES:
        for stratum, key, n_key, sum_key in MOTIF_STRATA:
            p_regime, n_regime = prev["genotyping_regimes"][rk], new["genotyping_regimes"][rk]
            p, n = (p_regime, n_regime) if key is None else (p_regime[key], n_regime[key])
            if not p.get("n"):
                continue
            deltas = _per_sample_gated_mae_deltas(p_regime["per_sample"], n_regime["per_sample"],
                                                  n_key, sum_key, samples)
            low, high = _bootstrap_mean_interval(deltas)
            # A bootstrap over a handful of samples gives an interval far too narrow to mean anything
            # (one sample gives [d, d]), so such a stratum is reported but never decides the verdict.
            win = ("too few samples" if len(deltas) < MIN_SAMPLES_FOR_VERDICT
                   else "new" if high < 0 else "prev" if low > 0 else "tie")
            wins[win] += 1
            print("\n  [%s, %s] n=%d alleles in %d samples  (raw MAE %.4f, identical input for both)"
                  % (rk, stratum, p["n"], len(deltas), p["mae_raw"]))
            print("     %-34s prev %-9.4f new %-9.4f -> %s" % ("gated MAE", p["mae_gated"], n["mae_gated"], win))
            print("     %-34s mean %+.4f, 95%% interval [%+.4f, %+.4f]; new better in %d of %d samples"
                  % ("paired per-sample delta", deltas.mean(), low, high, int((deltas < 0).sum()), len(deltas)))
            print("     %-34s prev %-9.4f new %-9.4f"
                  % ("gated MAE without the %d diagnosis samples" % (len(samples) - len(without_diagnosis)),
                     _pooled_gated_mae(p_regime["per_sample"], n_key, sum_key, without_diagnosis),
                     _pooled_gated_mae(n_regime["per_sample"], n_key, sum_key, without_diagnosis)))
            for label, pv, nv, fmt in (
                    ("gate decision accuracy (pOk<0.5)", p["gate_decision_accuracy"], n["gate_decision_accuracy"], "%-9.4f"),
                    ("direction accuracy (argmax)", p["p_ok_accuracy"], n["p_ok_accuracy"], "%-9.4f"),
                    ("median |err| (gated)", p["median_gated"], n["median_gated"], "%-9.3f"),
                    ("net helped (pOk<0.5)", p["helped_lt"] - p["hurt_lt"], n["helped_lt"] - n["hurt_lt"], "%-+9d")):
                print("     %-34s prev %s new %s" % (label, fmt % pv, fmt % nv))
    verdict = ("NEW is better" if wins["new"] and not wins["prev"]
               else "PREVIOUS is better" if wins["prev"] and not wins["new"]
               else "MIXED: each model is better in some strata" if wins["new"] and wins["prev"]
               else "NO CLEAR DIFFERENCE")
    print("\n  VERDICT: %s  (strata won -- new: %d, prev: %d, no clear difference: %d, too few samples "
          "(< %d) to judge: %d)" % (verdict, wins["new"], wins["prev"], wins["tie"], MIN_SAMPLES_FOR_VERDICT,
                                    wins["too few samples"]))
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
