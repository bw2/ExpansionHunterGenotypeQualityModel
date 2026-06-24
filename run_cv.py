"""Phase-4 orchestrator: 10-fold chromosome CV + cross splits + diagnostics/ablation/explain.

Ties together splits / features / model_q / model_direction / evaluate / diagnostics / ablation /
explain for one genotyping branch (``--branch full|fast``) over its parquet
(``data/parquet/{branch}.parquet``). For each Monte-Carlo fold it trains the q quantile models
and the 3-class direction model on the 17 train chromosomes (early-stopped on the 2 calib
chromosomes) and scores the 5 test chromosomes, writing per-fold results JSON. It then runs the
cross-sample / cross-coverage / cross-domain generalization splits and, on one representative
fold, the training-curve diagnostics, the feature ablation -> minimal set, and the permutation-
importance / PDP explainability (SPEC sec 7-8).

Run (per branch, parallelizable as separate processes):
  python3 run_cv.py --branch full
  python3 run_cv.py --branch fast
  python3 run_cv.py --branch full --max-rows 200000 --folds 2   # fast integration smoke

Determinism: global SEED; fold definitions persisted to results/folds.json. Coding rules: no
type hints, Google docstrings, print(). Heavy fits run single-process; run the two branches
concurrently for wall-clock.
"""

import argparse
import json
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import ablation
import diagnostics
import evaluate
import explain
import features
import model_direction as MD
import model_q as MQ
import size_bins
import splits

SEED = 20260616
QUANTILE_COLS = {0.05: "q05", 0.1: "q10", 0.5: "q50", 0.9: "q90", 0.95: "q95"}

# Scalar keys macro-averaged across size bins for the equal-per-bin headline.
_Q_MACRO_KEYS = list(evaluate._Q_METRIC_KEYS)
_DIR_MACRO_KEYS = ["log_loss", "too_long_auc", "too_short_auc", "too_long_ap", "too_short_ap"]

# Size bins with fewer than this many alleles are excluded from the equal-per-bin headline macro:
# a 1-2 allele tail bin would otherwise swing the equal-weighted mean as much as a dense bin,
# making the headline high-variance across folds. The per-bin breakdown still reports every bin.
_MACRO_MIN_N = 5


def _finite_or_none(obj):
    """Recursively replaces non-finite floats (NaN/inf) with None for strict JSON output."""
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _write_json(obj, path):
    """Writes ``obj`` as strict JSON (NaN/inf -> null) so downstream strict parsers don't choke.

    ``json.dump`` defaults to ``allow_nan=True``, which emits bare ``NaN`` tokens that are invalid
    JSON; empty stratified slices legitimately yield NaN metrics (evaluate._stratify on a zero-row
    source slice), so coerce every non-finite float to ``null`` first and dump with allow_nan=False.
    """
    with open(path, "w") as f:
        json.dump(_finite_or_none(obj), f, indent=1, allow_nan=False, default=float)


# Run configuration -- the fit-controlling caps, passed EXPLICITLY through run_regime / run_fold /
# run_cross / _bin_weights as one immutable value rather than via mutable module globals, so those
# functions stay pure (no hidden state shared across calls or reached in by external scripts).
#
#   train_cap      -- max train rows per fold fit (None = no cap). HistGBM on ~1M rows is
#                     statistically equivalent to the full pool here (min_samples_leaf=300) but far
#                     faster; TEST sets are never capped (honest held-out eval).
#   extras_cap     -- bounds the secondary cross-split / ablation / explain fits.
#   bin_weight_cap -- capped inverse-frequency size-bin TRAINING weighting (size_bins.bin_weights).
#                     SHELVED: the regime split + size-dependent tolerance are the structural fix, so
#                     training is UNWEIGHTED by default (cap < 0). Pass --bin-weight-cap >= 0 to
#                     re-enable; 0 = pure inverse-frequency (no cap). Per-bin REPORTING is unaffected.
RunConfig = namedtuple("RunConfig", ["train_cap", "extras_cap", "bin_weight_cap"])


def _cap_idx(mask, cap, seed):
    """Returns the True-row indices of ``mask``, randomly subsampled to ``cap`` if larger (seeded)."""
    idx = np.where(np.asarray(mask))[0]
    if cap and idx.size > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def _bin_weights(bins_subset, cfg):
    """Returns capped inverse-frequency size-bin weights for ``bins_subset`` (or None if disabled).

    ``cfg.bin_weight_cap < 0`` disables weighting (unweighted legacy fit); ``0`` is pure
    inverse-frequency (no cap); ``> 0`` caps per-allele weight at that many mean-1 units.
    """
    cap = cfg.bin_weight_cap
    if cap is not None and cap < 0:
        return None
    return size_bins.bin_weights(bins_subset, cap=(cap or 0.0))


def _prep(df, branch):
    """Returns (X DataFrame, feature_names, t array, dir_code array) for the branch."""
    X, names = features.build_matrix(df, branch)
    return X, names, df["t"].to_numpy(dtype=float), df["dir_code"].to_numpy(dtype=int)


def _fit_predict(X_tr, t_tr, y_tr, X_ca, t_ca, y_ca, X_te, w_tr=None, w_ca=None):
    """Trains q + direction on train (early-stopped on calib) and predicts on test.

    ``w_tr`` / ``w_ca`` are optional per-row size-bin weights for the train and
    calib sets (equalize the expansion/contraction-size bins). Returns a dict
    with the q models, direction model, test quantile preds and test direction
    proba, plus the two warm-start histories for diagnostics.
    """
    q = MQ.train_q(X_tr, t_tr, X_ca, t_ca, sample_weight=w_tr, sample_weight_calib=w_ca)
    qpreds = MQ.predict_q(q["models"], X_te)
    d = MD.train_direction(X_tr, y_tr, X_ca, y_ca, sample_weight=w_tr, sample_weight_calib=w_ca)
    proba = MD.predict_proba(d, X_te)
    return {"q_models": q["models"], "q_hist": q["histories"],
            "dir_model": d, "dir_hist": d["history"],
            "qpreds": qpreds, "proba": proba}


def _eval_frame(df_te, qpreds, proba, bins_te):
    """Builds the df_eval frame evaluate.stratified_* expects from test rows + predictions."""
    fr = pd.DataFrame({
        "t_true": df_te["t"].to_numpy(dtype=float),
        "eh": df_te["eh"].to_numpy(dtype=float),
        "true": df_te["true"].to_numpy(dtype=float),
        "dir_code": df_te["dir_code"].to_numpy(dtype=int),
        "source": df_te["source"].to_numpy(),
        "motif_size": pd.to_numeric(df_te["motif_size"], errors="coerce").to_numpy(),
        "coverage": pd.to_numeric(df_te["coverage"], errors="coerce").to_numpy(),
        "sample": df_te["sample"].to_numpy(),
        "size_bin": np.asarray(bins_te, dtype=int),
        "tol_repeats": pd.to_numeric(df_te.get("tol_repeats"), errors="coerce").to_numpy()
        if "tol_repeats" in df_te else np.nan,
        "ci_width": pd.to_numeric(df_te.get("ci_width"), errors="coerce").to_numpy()
        if "ci_width" in df_te else np.nan,
        "p_ok": proba[:, 0], "p_too_long": proba[:, 1], "p_too_short": proba[:, 2],
    })
    if "eh_q" in df_te.columns:
        fr["eh_q"] = pd.to_numeric(df_te["eh_q"], errors="coerce").to_numpy()
    for a, col in QUANTILE_COLS.items():
        fr[col] = np.asarray(qpreds[a], dtype=float)
    return fr


def _score(df_te, branch, qpreds, proba, bins_te):
    """Returns pooled + stratified q and direction metrics for one held-out test set.

    Adds the equal-per-bin (macro over size bins) headline ``q_bin_macro`` /
    ``direction_bin_macro`` alongside the data-frequency ``*_pooled`` metrics.
    """
    fr = _eval_frame(df_te, qpreds, proba, bins_te)
    q_pooled = evaluate.evaluate_q(fr["t_true"].to_numpy(), fr["q50"].to_numpy(),
                                   fr["eh"].to_numpy(), fr["true"].to_numpy(),
                                   {a: fr[c].to_numpy() for a, c in QUANTILE_COLS.items()},
                                   tol=fr["tol_repeats"].to_numpy(dtype=float)
                                   if "tol_repeats" in fr else None)
    d_pooled = evaluate.evaluate_direction(fr["dir_code"].to_numpy(),
                                           proba, branch,
                                           baseline_cols=None)
    d_pooled["baselines"] = evaluate.direction_baseline(fr, branch)
    q_strat = evaluate.stratified_q(fr, quantile_cols=QUANTILE_COLS)
    d_strat = evaluate.stratified_direction(fr, branch)
    return {
        "q_pooled": q_pooled,
        "q_bin_macro": evaluate.macro_over_bins(q_strat.get("size_bin", {}), _Q_MACRO_KEYS,
                                                min_n=_MACRO_MIN_N),
        "q_stratified": q_strat,
        "direction_pooled": d_pooled,
        "direction_bin_macro": evaluate.macro_over_bins(
            d_strat.get("size_bin", {}), _DIR_MACRO_KEYS, min_n=_MACRO_MIN_N),
        "direction_stratified": d_strat,
    }


def run_fold(df, branch, X, t, y, bins, fold, fold_i, cfg):
    """Trains + scores one CV fold; returns its results dict (+ train-gap q metrics).

    ``cfg`` (``RunConfig``) supplies ``train_cap`` and ``bin_weight_cap`` for this fit.
    """
    tr, ca, te = splits.fold_masks(df, fold)
    tr_idx = _cap_idx(tr, cfg.train_cap, SEED + fold_i)
    print("  fold %d: train=%d (capped to %d) calib=%d test=%d"
          % (fold_i, tr.sum(), tr_idx.size, ca.sum(), te.sum()))
    fit = _fit_predict(X.iloc[tr_idx], t[tr_idx], y[tr_idx], X[ca], t[ca], y[ca], X[te],
                       w_tr=_bin_weights(bins[tr_idx], cfg), w_ca=_bin_weights(bins[ca], cfg))
    res = _score(df[te], branch, fit["qpreds"], fit["proba"], bins[te])
    # train-set q score (overfitting gap): predict on a capped train sample
    cap = tr_idx if tr_idx.size <= 200000 else np.random.default_rng(SEED + fold_i).choice(
        tr_idx, 200000, replace=False)
    qpreds_tr = MQ.predict_q(fit["q_models"], X.iloc[cap])
    res["q_train"] = evaluate.evaluate_q(
        t[cap], qpreds_tr[0.5], df["eh"].to_numpy(dtype=float)[cap],
        df["true"].to_numpy(dtype=float)[cap], qpreds_tr,
        tol=pd.to_numeric(df["tol_repeats"], errors="coerce").to_numpy()[cap]
        if "tol_repeats" in df else None)
    res["fold_i"] = fold_i
    res["sizes"] = {"train": int(tr.sum()), "calib": int(ca.sum()), "test": int(te.sum())}
    return res, fit


def run_cross(df, branch, X, t, y, bins, name, train_mask, test_mask, cfg):
    """Trains on ``train_mask`` (calib held out by chromosome) and scores ``test_mask``.

    ``cfg`` (``RunConfig``) supplies ``extras_cap`` and ``bin_weight_cap`` for this fit.
    """
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None
    # Hold out whole CHROMOSOMES (not random rows) from the training side for the early-stop /
    # calibration set, so calibration never shares a locus/chromosome with the fit rows -- a random
    # row split would leak loci across the fit/calib boundary. Falls back to a row split only in the
    # degenerate case of a single training chromosome.
    rng = np.random.default_rng(SEED)
    chrom = df["chrom"].astype(str).to_numpy()
    train_chroms = sorted(set(chrom[train_mask]) - {"nan"})
    ca = np.zeros(len(df), bool)
    if len(train_chroms) > 1:
        n_calib = max(1, int(round(0.1 * len(train_chroms))))
        calib_chroms = set(rng.choice(np.array(train_chroms), n_calib, replace=False).tolist())
        ca = train_mask & np.isin(chrom, list(calib_chroms))
    else:
        tr_all = np.where(train_mask)[0]
        ca[rng.choice(tr_all, max(1, int(0.1 * tr_all.size)), replace=False)] = True
    tr_idx = _cap_idx(train_mask & ~ca, cfg.extras_cap, SEED)
    fit = _fit_predict(X.iloc[tr_idx], t[tr_idx], y[tr_idx], X[ca], t[ca], y[ca], X[test_mask],
                       w_tr=_bin_weights(bins[tr_idx], cfg), w_ca=_bin_weights(bins[ca], cfg))
    print("  cross[%s]: train=%d test=%d" % (name, tr_idx.size, test_mask.sum()))
    return _score(df[test_mask], branch, fit["qpreds"], fit["proba"], bins[test_mask])


def run_regime(df, branch, regime, folds, args, cfg):
    """Trains + evaluates the q + direction models for ONE regime; writes results/<regime>_*.json
    and plots/<regime>_*.png. ``cfg`` (``RunConfig``) carries the fit caps + bin-weight cap.

    ``branch`` still selects the FEATURE set (full vs fast); ``regime`` is the output label and the
    row subset (fast / full_spanning / full_nonspanning). Folds are shared across regimes (the
    chromosome partition is data-independent), so the per-regime numbers are directly comparable.
    """
    print("\n=== regime %s (branch %s): %d rows, %d chroms ===" % (
        regime, branch, len(df), df["chrom"].nunique()))
    bins = size_bins.assign_bins(df["true"], df["num_repeats_in_reference"])
    X, names, t, y = _prep(df, branch)

    fold_results = []
    fold0_fit = None
    for fold_i, fold in enumerate(folds):
        res, fit = run_fold(df, branch, X, t, y, bins, fold, fold_i, cfg)
        fold_results.append(res)
        _write_json(res, os.path.join(args.out_dir, "%s_fold%d.json" % (regime, fold_i)))
        if fold_i == 0:
            fold0_fit = fit

    # across-fold means of the headline pooled metrics (descriptive spread only)
    summary = {"branch": branch, "regime": regime, "n_rows": len(df), "n_folds": len(folds),
               "bin_weight_cap": cfg.bin_weight_cap}
    for head in ("q_pooled", "q_train", "q_bin_macro"):
        keys = [k for k in fold_results[0][head] if k != "n"]
        summary[head] = {k: float(np.nanmean([fr[head][k] for fr in fold_results])) for k in keys}
    for k in ("log_loss", "too_long_auc", "too_short_auc", "too_long_ap", "too_short_ap"):
        summary.setdefault("direction_pooled", {})[k] = float(np.nanmean(
            [fr["direction_pooled"][k] for fr in fold_results]))
        summary.setdefault("direction_bin_macro", {})[k] = float(np.nanmean(
            [fr["direction_bin_macro"][k] for fr in fold_results]))
    _write_json(summary, os.path.join(args.out_dir, "%s_summary.json" % regime))
    print("  SUMMARY q:", summary["q_pooled"])
    print("  SUMMARY direction:", summary["direction_pooled"])

    if args.no_extras:
        return

    # --- cross-split generalization (each trained once) ---
    cross = {}
    tr_m, te_m = splits.cross_sample_split(df)
    cross["cross_sample_HG002_to_CHM"] = run_cross(df, branch, X, t, y, bins, "sample", tr_m, te_m,
                                                   cfg)
    # NOTE: leave-one-coverage-out tests the SAME loci at a held-out coverage (HG002 is sequenced at
    # 10x/20x/31x), so every test locus is also in the training set at the other coverages. This
    # measures coverage robustness on SEEN loci, NOT held-out-locus generalization -- the key name
    # says so explicitly so the report cannot be misread as a leak-free generalization number.
    for cov, trm, tem in splits.cross_coverage_splits(df):
        cross["coverage_robustness_seen_loci_%s" % cov] = run_cross(
            df, branch, X, t, y, bins, "cov%s" % cov, trm, tem, cfg)
    for nm, trm, tem in splits.cross_domain_splits(df):
        cross["cross_domain_%s" % nm] = run_cross(df, branch, X, t, y, bins, nm, trm, tem, cfg)
    _write_json(cross, os.path.join(args.out_dir, "%s_cross.json" % regime))

    # --- diagnostics: loss-vs-iteration from fold 0's warm-start histories ---
    diagnostics.plot_loss_vs_iteration(
        fold0_fit["q_hist"][0.5], os.path.join(args.plots_dir, "%s_qloss_iter" % regime),
        title="%s q (median) loss vs iteration" % regime)
    diagnostics.plot_loss_vs_iteration(
        fold0_fit["dir_hist"], os.path.join(args.plots_dir, "%s_dirloss_iter" % regime),
        title="%s direction loss vs iteration" % regime)

    # --- explainability + ablation on fold 0 (capped to cfg.extras_cap for tractability) ---
    tr0, ca0, te0 = splits.fold_masks(df, folds[0])
    te0_idx = _cap_idx(te0, cfg.extras_cap, SEED)
    tr0_idx = _cap_idx(tr0, cfg.extras_cap, SEED)
    ca0_idx = _cap_idx(ca0, cfg.extras_cap, SEED + 1)

    explain.permutation_importance_plot(
        explain.ProbaEstimator(lambda X_: MD.predict_proba(fold0_fit["dir_model"], X_),
                               MD.N_CLASSES),
        X.iloc[te0_idx], y[te0_idx], names,
        os.path.join(args.plots_dir, "%s_perm_importance" % regime),
        scoring="neg_log_loss", n_repeats=5, seed=SEED)
    ranked = explain.permutation_importance_plot(
        fold0_fit["q_models"][0.5], X.iloc[te0_idx], t[te0_idx], names,
        os.path.join(args.plots_dir, "%s_q_perm_importance" % regime),
        scoring=explain.pinball_scorer(0.5), n_repeats=5, seed=SEED)
    top = [f for f, _, _ in ranked[:5]]
    explain.partial_dependence_plots(
        fold0_fit["q_models"][0.5], X.iloc[te0_idx], names, top,
        os.path.join(args.plots_dir, "%s_pdp" % regime))

    # --- ablation -> minimal set on fold 0 (rank + score on the calib chroms) ---
    w_tr0, w_ca0 = _bin_weights(bins[tr0_idx], cfg), _bin_weights(bins[ca0_idx], cfg)

    def score_fn(feat_subset):
        cols = list(feat_subset)
        m = MQ.fit_with_early_stop(lambda: MQ.make_quantile_estimator(0.5),
                                   X.iloc[tr0_idx][cols], t[tr0_idx],
                                   X.iloc[ca0_idx][cols], t[ca0_idx],
                                   sample_weight=w_tr0, sample_weight_calib=w_ca0)[0]
        return -MQ.pinball_loss(t[ca0_idx], m.predict(X.iloc[ca0_idx][cols]), 0.5, w_ca0)

    def rank_fn():
        return [f for f, _, _ in explain.permutation_importance_plot(
            fold0_fit["q_models"][0.5], X.iloc[ca0_idx], t[ca0_idx], names,
            os.path.join(args.plots_dir, "%s_ablation_rank" % regime),
            scoring=explain.pinball_scorer(0.5), n_repeats=3, seed=SEED)]

    k, minimal = ablation.add_one_curve(rank_fn, score_fn, names,
                                        os.path.join(args.plots_dir, "%s_addone" % regime))
    grouped = ablation.grouped_ablation(score_fn, features.feature_families(branch), names,
                                        os.path.join(args.plots_dir, "%s_grouped_ablation" % regime))
    _write_json({"minimal_k": k, "minimal_set": minimal, "grouped": grouped},
                os.path.join(args.out_dir, "%s_ablation.json" % regime))
    print("  minimal set (k=%d): %s" % (k, minimal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=["full", "fast"])
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--plots-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None, help="subsample whole dataset (smoke)")
    parser.add_argument("--train-cap", type=int, default=1000000,
                        help="max train rows per fold fit (test sets are never capped); 0 = no cap")
    parser.add_argument("--extras-cap", type=int, default=300000,
                        help="max train rows for cross / ablation / explain fits")
    parser.add_argument("--bin-weight-cap", type=float, default=-1.0,
                        help="equal-per-size-bin TRAINING weighting (SHELVED, default off): "
                             "<0 = unweighted (default); >0 caps per-allele weight at this many "
                             "mean-1 units; 0 = pure inverse-frequency (no cap)")
    parser.add_argument("--no-extras", action="store_true",
                        help="skip cross splits / diagnostics / ablation / explain")
    parser.add_argument("--regime", default=None,
                        help="restrict to one regime (fast / full_spanning / full_nonspanning); "
                             "default = every regime present in the branch parquet")
    args = parser.parse_args()

    cfg = RunConfig(train_cap=args.train_cap or None,
                    extras_cap=args.extras_cap or None,
                    bin_weight_cap=args.bin_weight_cap)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    branch = args.branch

    # Load ONLY the columns the models / evaluator need (drops the heavy string/audit columns
    # like locus_id, repeat_unit, consensus, tsv_eh) to keep peak RAM bounded on this machine.
    path = os.path.join(args.data_dir, "parquet", "%s.parquet" % branch)
    need = (set(features.FULL_FEATURES) | set(features.FAST_FEATURES)
            | {"ci_start", "ci_end", "ci_width", "eh", "eh_q", "true", "t", "dir_code", "direction",
               "source", "motif_size", "coverage", "sample", "chrom", "genotyping_branch",
               "num_repeats_in_reference", "regime", "tol_repeats"})
    cols = [c for c in pq.ParquetFile(path).schema.names if c in need]
    df = pd.read_parquet(path, columns=cols)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=SEED).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    folds = splits.make_cv_folds(n_folds=args.folds, seed=SEED)
    splits.save_folds(folds, os.path.join(args.out_dir, "folds.json"))

    regimes = sorted(df["regime"].dropna().unique())
    if args.regime:
        regimes = [r for r in regimes if r == args.regime]
    print("=== %s branch: %d rows -> regimes %s ===" % (branch, len(df), regimes))
    for regime in regimes:
        dfr = df[df["regime"] == regime].reset_index(drop=True)
        if dfr.empty:
            print("  regime %s: 0 rows -- skipping" % regime)
            continue
        run_regime(dfr, branch, regime, folds, args, cfg)


if __name__ == "__main__":
    main()
