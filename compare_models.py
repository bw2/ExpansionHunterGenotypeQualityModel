"""Compares a newly-trained genotype-quality model against the previous/deployed model on the
held-out HPRC benchmark, printing per-regime metric deltas. Run at the end of ``train_model.sh``.

Both models are applied UNCHANGED (no fitting) to the same held-out parquets -- the samples in
``heldout.SAMPLES``, absent from either model's training pool -- and to the same rows, via
``heldout.run_eval`` with the per-sample allele cap the report uses (``gen_datasets.EVAL_CAP_DEFAULT``).

Results are reported per genotyping regime AND per motif stratum (homopolymer vs non-homopolymer;
``run_eval`` keeps them apart, and homopolymers are over half of the alleles EH scores). The verdict
counts only gated MAE, as a per-sample paired difference with a bootstrap interval over samples; gate
decision accuracy, 3-class direction accuracy, the median and the net-helped count are printed as
context. The per-sample statistic and its interval are also given without
``heldout.FASTPATH_DIAGNOSIS_SAMPLES``, which were used during model development. The evaluated
population is variant loci only (truth differs from the reference) and one row per emitted
quality-metric entry; the printout says so. Held-out parquets built with
``--keep-hom-ref-loci`` also get the all-loci early-warning section (``heldout.all_loci_early_warning``),
printed after the verdict and not counted in it.

The "previous model" defaults to whichever model is deployed in the local ExpansionHunter-bw2 checkout
(``$EXPANSIONHUNTER_BW2_REPO/ehunter/data/genotype_quality_model_*.json.gz`` -- i.e. the "should we
ship the new one?" comparison), falling back to the most recent other ``model/*.json.gz`` in this repo.
Override with ``--prev-model``.

Self-skips (prints a note, exit 0) when ``--data-dir`` holds no held-out parquets or no previous model
exists, so it is safe to always invoke.
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


def _per_sample_gated_maes(prev_per_sample, new_per_sample, n_key, sum_key, samples):
    """Returns ``(prev, new)`` arrays of the gated MAE of each of ``samples`` that has alleles in this stratum."""
    kept = [s for s in samples if s in prev_per_sample and prev_per_sample[s][n_key] > 0]
    return tuple(np.array([per_sample[s][sum_key] / per_sample[s][n_key] for s in kept])
                 for per_sample in (prev_per_sample, new_per_sample))


def _bootstrap_mean_interval(values):
    """Returns the 95% percentile-bootstrap interval of the mean of ``values`` (resampling samples)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _paired_statistics(prev_per_sample, new_per_sample, n_key, sum_key, samples):
    """Returns the per-sample paired comparison of one stratum: ``prev_mean_over_samples``,
    ``new_mean_over_samples``, ``deltas`` (new - prev, one per sample with alleles), ``interval`` (bootstrap 95% of
    the deltas' mean) and ``winner`` ("new", "prev", "tie", or "too few samples" below ``MIN_SAMPLES_FOR_VERDICT``)."""
    prev_maes, new_maes = _per_sample_gated_maes(prev_per_sample, new_per_sample, n_key, sum_key, samples)
    deltas = new_maes - prev_maes
    low, high = _bootstrap_mean_interval(deltas)
    # A bootstrap over a handful of samples gives an interval far too narrow to mean anything
    # (one sample gives [d, d]), so such a stratum is reported but never decides the verdict.
    winner = ("too few samples" if len(deltas) < MIN_SAMPLES_FOR_VERDICT
              else "new" if high < 0 else "prev" if low > 0 else "tie")
    return {"prev_mean_over_samples": prev_maes.mean(), "new_mean_over_samples": new_maes.mean(), "deltas": deltas,
            "interval": (low, high), "winner": winner}


def _assert_same_alleles(prev, new, where):
    """Raises ValueError unless the two run_eval metrics ``prev`` and ``new`` scored the same alleles."""
    if (prev["n"], prev["mae_raw"]) != (new.get("n"), new.get("mae_raw")):
        raise ValueError("%s: the two results did not score the same alleles (n %s vs %s, raw MAE %s vs %s), so they "
                         "cannot be paired" % (where, prev["n"], new.get("n"), prev["mae_raw"], new.get("mae_raw")))


def paired_gated_mae_comparison(prev, new, samples):
    """Compares two ``heldout.run_eval`` results on the same rows, per genotyping regime and motif stratum.

    The statistic is the per-sample paired difference in gated MAE (new - prev) with a bootstrap 95% interval over
    samples. A stratum is won only when the interval excludes 0, and one with fewer than ``MIN_SAMPLES_FOR_VERDICT``
    samples cannot be won or lost. Every sample counts equally, however many alleles it has, so the winner follows
    the mean of the per-sample gated MAEs (``prev_mean_over_samples`` / ``new_mean_over_samples``), which can differ
    from the run_eval ``mae_gated`` pooled over alleles.

    Raises:
        ValueError: If a stratum's allele count or raw MAE differs between the two results, which happens only when
            they did not score the same rows.

    Returns:
        One dict per regime and stratum that has alleles: ``regime``, ``stratum``, ``prev`` and ``new`` (that
        stratum's run_eval metrics), ``prev_mean_over_samples`` / ``new_mean_over_samples``, ``deltas`` (one per
        sample), ``interval``, ``winner`` ("new", "prev", "tie" or "too few samples"), and ``prev_without_diagnosis``
        / ``new_without_diagnosis`` / ``interval_without_diagnosis``, the same per-sample mean and paired interval
        without ``heldout.FASTPATH_DIAGNOSIS_SAMPLES``.
    """
    without_diagnosis = [s for s in samples if s not in heldout.FASTPATH_DIAGNOSIS_SAMPLES]
    rows = []
    for rk in features.GENOTYPING_REGIMES:
        for stratum, key, n_key, sum_key in MOTIF_STRATA:
            p_regime, n_regime = prev["genotyping_regimes"][rk], new["genotyping_regimes"][rk]
            p, n = (p_regime, n_regime) if key is None else (p_regime[key], n_regime[key])
            if not p.get("n"):
                continue
            _assert_same_alleles(p, n, "%s, %s" % (rk, stratum))
            prev_kept, new_kept = _per_sample_gated_maes(p_regime["per_sample"], n_regime["per_sample"],
                                                         n_key, sum_key, without_diagnosis)
            rows.append({
                "regime": rk, "stratum": stratum, "prev": p, "new": n,
                **_paired_statistics(p_regime["per_sample"], n_regime["per_sample"], n_key, sum_key, samples),
                "prev_without_diagnosis": prev_kept.mean(), "new_without_diagnosis": new_kept.mean(),
                "interval_without_diagnosis": tuple(_bootstrap_mean_interval(new_kept - prev_kept))})
    return rows


def paired_gated_mae_by_motif_size_category(prev, new, samples):
    """Compares two ``heldout.run_eval`` results per genotyping regime and motif-size category
    (``features.MOTIF_SIZE_CATEGORY_LABELS``; homopolymers are "1"), with ``paired_gated_mae_comparison``'s
    per-sample paired statistic. Context only: these rows never enter the verdict.

    Raises:
        ValueError: If a category's allele count or raw MAE differs between the two results.

    Returns:
        One dict per regime and category that both results report: ``regime``, ``category``, ``prev`` and ``new``
        (that category's run_eval metrics), and ``_paired_statistics``' keys. None when either result was scored
        before run_eval reported the categories.
    """
    rows = []
    for rk in features.GENOTYPING_REGIMES:
        p_regime, n_regime = prev["genotyping_regimes"][rk], new["genotyping_regimes"][rk]
        if "by_motif_size_category" not in p_regime or "by_motif_size_category" not in n_regime:
            return None
        for category in features.MOTIF_SIZE_CATEGORY_LABELS:
            p = p_regime["by_motif_size_category"].get(category)
            if not p:
                continue
            n = n_regime["by_motif_size_category"].get(category, {})
            _assert_same_alleles(p, n, "%s, motif size %s" % (rk, category))
            rows.append({"regime": rk, "category": category, "prev": p, "new": n,
                         **_paired_statistics(p["per_sample"], n["per_sample"], "n", "sum_gated_abs_error", samples)})
    return rows


def verdict_of(rows):
    """Returns ``(verdict sentence, {winner: number of strata})`` for ``paired_gated_mae_comparison`` rows.

    Each stratum is judged by its own 95% interval, not adjusted for the number of strata, so with up to 6 strata a
    single win or loss can come from chance alone. A one-sided verdict therefore says how many of the judged strata
    (those with enough samples) the better model won.
    """
    wins = {"new": 0, "prev": 0, "tie": 0, "too few samples": 0}
    for row in rows:
        wins[row["winner"]] += 1
    judged = wins["new"] + wins["prev"] + wins["tie"]
    verdict = ("NEW is better in %d of %d judged strata, worse in none" % (wins["new"], judged)
               if wins["new"] and not wins["prev"]
               else "PREVIOUS is better in %d of %d judged strata, worse in none" % (wins["prev"], judged)
               if wins["prev"] and not wins["new"]
               else "MIXED: each model is better in some strata" if wins["new"] and wins["prev"]
               else "NO CLEAR DIFFERENCE")
    return verdict, wins


def compare(new_model, prev_model, data_dir, max_alleles):
    """Applies both models to the held-out parquets and prints the per-regime comparison."""
    paths = sorted(p for p in (os.path.join(data_dir, "real_43", "%s.parquet" % s)
                               for s in heldout.SAMPLES) if os.path.exists(p))
    if not paths:
        print("\n[compare_models] skipped: no held-out parquets under %s/real_43/ -- build them with "
              "heldout.py --build-only, or pass the --data-dir that holds them." % data_dir)
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
    # Every row, and the rows whose truth the person's own short reads don't contradict (the rows training uses).
    # Neither score alone can be trusted (the first penalizes a model that is right where the truth is wrong, the
    # second can't see a model that trusts misleading reads), so a new model should be no worse on either.
    row_sets = [("every held-out row", False)]
    unmarked = dataset.parquets_without_the_short_read_truth_check(paths)
    if unmarked:
        print("  note: %d of %d held-out parquets were never checked against short reads, so only every row is scored; "
              "run check_truth_genotypes_against_short_reads.py --mark to also score without the rows whose truth the "
              "reads contradict" % (len(unmarked), len(paths)))
    else:
        row_sets.append(("rows whose truth the short reads don't contradict", True))
    verdicts = []
    for row_set, drop in row_sets:
        verdicts.append((row_set, _compare_on_row_set(paths, prev_model, new_model, max_alleles, row_set, drop)))
    if len(verdicts) > 1:
        lost = [row_set for row_set, wins in verdicts if wins["prev"]]
        n_judged = sum(wins["new"] + wins["prev"] + wins["tie"] for _, wins in verdicts)
        n_too_few_samples = sum(wins["too few samples"] for _, wins in verdicts)
        print("\n  COMBINED VERDICT: %s" % (
            "the new model loses strata in: %s" % "; ".join(lost) if lost
            else "no stratum in either row set had enough samples to judge" if not n_judged
            else "the new model loses no judged stratum in either row set (%d strata had too few samples to judge)"
                 % n_too_few_samples))
    print("=" * 108)


def _compare_on_row_set(paths, prev_model, new_model, max_alleles, row_set, drop_truth_contradicted_by_short_reads):
    """Scores both models on one set of held-out rows, prints the comparison and returns ``verdict_of``'s wins."""
    print("\n---------------- scored on %s ----------------" % row_set)
    # run_eval writes an out_json + violin npz; both are throwaway here, so use a temp dir.
    with tempfile.TemporaryDirectory() as td:
        prev, new = (heldout.run_eval(paths, model, os.path.join(td, name), max_alleles,
                                      drop_truth_contradicted_by_short_reads=drop_truth_contradicted_by_short_reads)
                     for model, name in ((prev_model, "prev.json"), (new_model, "new.json")))

    print("  scope: alleles at loci whose truth genotype differs from the reference (run_eval leaves out truth "
          "hom-ref loci; the all-loci early-warning section below scores them), and one row per emitted "
          "quality-metric entry (the second copy of a homozygous call is not scored, so a missed heterozygous "
          "allele is not counted).")
    print("  verdict: per genotyping regime and motif stratum, the per-sample paired difference in gated MAE "
          "(new - prev) with a %d-resample bootstrap 95%% interval over samples (every sample weighted equally); "
          "a stratum is won only when the interval excludes 0. The intervals are not adjusted for the number of "
          "strata, so a single win or loss can be chance. The other rows are context and are not counted."
          % BOOTSTRAP_RESAMPLES)

    samples = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    wins = print_paired_comparison(paired_gated_mae_comparison(prev, new, samples), samples)
    print_motif_size_category_comparison(paired_gated_mae_by_motif_size_category(prev, new, samples))
    _print_all_loci_early_warning(paths, prev_model, new_model, drop_truth_contradicted_by_short_reads)
    return wins


def print_paired_comparison(rows, samples):
    """Prints ``paired_gated_mae_comparison`` rows stratum by stratum, then the verdict, and returns
    ``verdict_of``'s wins."""
    n_diagnosis = sum(s in heldout.FASTPATH_DIAGNOSIS_SAMPLES for s in samples)
    for row in rows:
        p, n, deltas, (low, high) = row["prev"], row["new"], row["deltas"], row["interval"]
        print("\n  [%s, %s] n=%d alleles in %d samples  (raw MAE %.4f, identical input for both)"
              % (row["regime"], row["stratum"], p["n"], len(deltas), p["mae_raw"]))
        print("     %-34s prev %-9.4f new %-9.4f" % ("gated MAE, pooled over alleles", p["mae_gated"], n["mae_gated"]))
        print("     %-34s prev %-9.4f new %-9.4f -> %s"
              % ("gated MAE, mean over samples", row["prev_mean_over_samples"], row["new_mean_over_samples"],
                 row["winner"]))
        print("     %-34s mean %+.4f, 95%% interval [%+.4f, %+.4f]; new better in %d of %d samples"
              % ("paired per-sample delta", deltas.mean(), low, high, int((deltas < 0).sum()), len(deltas)))
        print("     %-34s prev %-9.4f new %-9.4f paired delta 95%% interval [%+.4f, %+.4f]"
              % ("mean over samples, without %d diagnosis" % n_diagnosis, row["prev_without_diagnosis"],
                 row["new_without_diagnosis"], *row["interval_without_diagnosis"]))
        for label, pv, nv, fmt in (
                ("gate decision accuracy (pOk<0.5)", p["gate_decision_accuracy"], n["gate_decision_accuracy"], "%-9.4f"),
                ("direction accuracy (argmax)", p["p_ok_accuracy"], n["p_ok_accuracy"], "%-9.4f"),
                ("median |err| (gated)", p["median_gated"], n["median_gated"], "%-9.3f"),
                ("net helped (pOk<0.5)", p["helped_lt"] - p["hurt_lt"], n["helped_lt"] - n["hurt_lt"], "%-+9d")):
            print("     %-34s prev %s new %s" % (label, fmt % pv, fmt % nv))
    verdict, wins = verdict_of(rows)
    print("\n  VERDICT: %s  (strata won -- new: %d, prev: %d, no clear difference: %d, too few samples "
          "(< %d) to judge: %d)" % (verdict, wins["new"], wins["prev"], wins["tie"], MIN_SAMPLES_FOR_VERDICT,
                                    wins["too few samples"]))
    return wins


def print_motif_size_category_comparison(rows):
    """Prints ``paired_gated_mae_by_motif_size_category`` rows (None: a result predates the categories)."""
    if rows is None:
        print("\n  by motif size category: skipped (a result was scored before run_eval reported the categories)")
        return
    print("\n  gated MAE by motif size category (bp), mean over samples; context, not counted in the verdict")
    print("     %-17s %-6s %9s %9s %9s %8s %-28s %s" % ("regime", "motif", "alleles", "prev", "new", "change",
                                                       "paired delta [95% interval]", "winner"))
    for row in rows:
        print("     %-17s %-6s %9d %9.4f %9.4f %+7.1f%% %+.4f [%+.4f, %+.4f]  %s"
              % (row["regime"], row["category"], row["prev"]["n"], row["prev_mean_over_samples"],
                 row["new_mean_over_samples"], 100 * (row["new_mean_over_samples"] / row["prev_mean_over_samples"] - 1),
                 row["deltas"].mean(), row["interval"][0], row["interval"][1], row["winner"]))


def _print_all_loci_early_warning(paths, prev_model, new_model, drop_truth_contradicted_by_short_reads=False):
    """Prints ``heldout.all_loci_early_warning`` for both models, when the held-out parquets have hom-ref loci."""
    print_early_warning_comparison(*(heldout.all_loci_early_warning(
        paths, p, drop_truth_contradicted_by_short_reads=drop_truth_contradicted_by_short_reads)
        for p in (prev_model, new_model)))


def print_early_warning_comparison(prev, new):
    """Prints two models' ``heldout.all_loci_early_warning`` results (None: no parquet had hom-ref loci)."""
    if prev is None:
        print("\n  all-loci early warning: skipped (no held-out parquet was built with --keep-hom-ref-loci)")
        return
    print("\n  ALL-LOCI EARLY WARNING (fast path, %d samples with hom-ref loci; a miss is a reference-length call "
          "whose truth is more than 2 repeats from the reference)" % len(prev["per_sample"]))
    print("     %-44s prev %-9.3f new %-9.3f" % ("ROC-AUC of 1 - pOk for misses, mean over samples",
                                                 prev["auc_miss_mean"], new["auc_miss_mean"]))
    print("     %-44s prev %-9.3f new %-9.3f" % ("share of fast-path calls below pOk 0.5",
                                                 prev["share_below_half_mean"], new["share_below_half_mean"]))
    for subset_name in prev["recall"]:
        for rate in heldout.EARLY_WARNING_FLAG_RATES:
            print("     %-44s prev %-9.3f new %-9.3f"
                  % ("misses in the lowest %d%% of pOk (%s)" % (round(100 * rate), subset_name),
                     prev["recall"][subset_name][rate], new["recall"][subset_name][rate]))


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
