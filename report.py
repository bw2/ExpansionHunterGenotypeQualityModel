"""5-fold chromosome-clean evaluation + the HTML training report.

The deployable model (``train.py``) is fit on all real data, so it has no held-out
set of its own. This module measures held-out accuracy honestly with 5-fold
chromosome-clean cross-validation: the 24 chromosomes are partitioned into 5
disjoint test groups, each genotyping_regime's heads are trained out-of-fold on the other
chromosomes with the exported model's fitting steps (``train.fit_heads`` at the regime's fixed
``train.ITERATIONS_BY_REGIME``, calibrated on a held-out chromosome subset; fit separately per motif
panel, unlike the exported model), and the pooled
out-of-fold predictions feed the report. Splitting by chromosome group -- never by
row -- ensures no locus leaks between train and test.

The report (a single standalone ``.html`` with embedded plots) shows:
  - the raw-EH vs gated-LCF MAE chart (apply the LCF only where ``pOk < 0.5``),
    per genotyping_regime, on a broken linear axis;
  - direction-head confusion matrices, ROC / precision-recall curves and probability violins;
  - per-genotyping_regime permutation feature importance (relative), for the q head
    and for the direction head separately;
  - add-one-feature ablation curves for both heads (q head scored by corrected-size
    MAE, direction head by calibrated multinomial log-loss).

Coding rules: no type hints, Google docstrings, ``print()``. Determinism: ``SEED``.
"""

import argparse
import base64
import html
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, log_loss, make_scorer, mean_pinball_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

import accuracy_by_size as ABS
import dataset
import eh_json
import features
import heldout
import metrics
import model as M
import train

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
ALL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
GRAY, ORANGE = "#888888", "#F58518"
IMPORTANCE_CAP = 30_000  # held-out rows used for permutation importance
IMPORTANCE_REPEATS = 5   # permutation repeats per feature (both heads)
TOP_N = 15               # features shown per importance panel
# Max # of top features in the add-one ablation curve. Derived from the feature lists so the curve
# always runs to "all features" as the report text claims, instead of silently truncating when a
# feature is added.
ABLATION_KMAX = len(features.FULL_FEATURES)
ABLATION_TRAIN_CAP = 120_000
ABLATION_TEST_CAP = 150_000
# The ablations refit a head once per prefix length (up to ABLATION_KMAX times per head), so they use
# the simpler early-stopped recipe (model.EARLY_STOP_MAX_ITERATIONS ceiling, one calib set for stopping
# and calibration, an ungated q-median head): they rank how much each added feature helps, not the
# shipped model's accuracy.
REGIME_COLORS = {"quick": "#4c72b0", "full_spanning": "#dd8452", "full_nonspanning": "#55a868"}


def make_folds(n_folds=5, n_calib=2, seed=SEED):
    """Partitions the chromosomes into ``n_folds`` disjoint OOF test groups.

    Each fold's test group is one partition block; ``n_calib`` chromosomes are drawn
    (seeded) from the remaining chromosomes for the isotonic-calibration set and the rest are
    the training chromosomes. Every chromosome is a test chromosome in exactly one fold, so the
    pooled out-of-fold predictions cover the data once.

    Returns:
        A list of ``{"train": [...], "calib": [...], "test": [...]}`` dicts.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ALL_CHROMS))
    groups = [list(g) for g in np.array_split(shuffled, n_folds)]
    folds = []
    for test in groups:
        rest = [c for c in ALL_CHROMS if c not in test]
        calib = list(rng.choice(rest, size=min(n_calib, len(rest)), replace=False))
        train = [c for c in rest if c not in calib]
        folds.append({"train": train, "calib": calib, "test": [str(c) for c in test]})
    return folds


def _cap_rows(idx, cap, seed):
    if cap and idx.size > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def collect_oof(df, genotyping_regime, branch, folds, train_cap):
    """Runs 5-fold OOF training for one genotyping_regime; returns pooled per-row arrays + importance.

    Each fold fits with ``train.fit_heads`` at the regime's ``train.ITERATIONS_BY_REGIME``, the exported
    model's recipe.

    Returns:
        ``(oof, ranked, ranked_dir)`` where ``oof`` is a dict of concatenated arrays
        (``eh``, ``true``, ``true_pred``, ``t``, ``t_pred``, ``p_ok``, ``dir_code``,
        ``tol_repeats``), ``ranked`` is the fold-0 q-head permutation importance list
        ``[(feature, mean, std), ...]`` and ``ranked_dir`` the fold-0 direction-head one
        (either is ``None`` if it could not be run). The two heads get separate rankings
        because they answer different questions, and each head's ablation adds features
        in its own order.
    """
    chrom = df["chrom"].astype(str)
    acc = {k: [] for k in ("eh", "true", "true_pred", "t", "t_pred", "p_ok", "p_long", "p_short",
                           "dir_code", "tol_repeats")}
    ranked = ranked_dir = None
    held_cap = max(1, train_cap // 5) if train_cap else None
    # Test and calibration rows estimate frequencies, so only the representative rows qualify; the
    # training-only quota top-up rows (dataset._cap_rows_per_genotyping_regime) only join training.
    rep = df["representative"].to_numpy(bool)
    for i, f in enumerate(folds):
        m_tr = chrom.isin(set(f["train"])).to_numpy()
        m_ca = chrom.isin(set(f["calib"])).to_numpy() & rep
        m_te = chrom.isin(set(f["test"])).to_numpy() & rep
        if not m_te.any() or not m_tr.any() or not m_ca.any():
            continue
        # The exported model's recipe (train.fit_heads), so these metrics describe what ships.
        ca_idx = _cap_rows(np.where(m_ca)[0], held_cap, SEED + 2)
        # Gating folds split by chromosome too, so no test-chromosome locus is learned anywhere.
        qreg, dmodel, names, Xca = train.fit_heads(
            df, np.where(m_tr)[0], ca_idx, chrom.to_numpy(), branch, train_cap,
            train.ITERATIONS_BY_REGIME[genotyping_regime], "fold %d" % i)
        Xte, _ = features.build_matrix(df[m_te], branch)
        tca, yca = df["t"].to_numpy(float)[ca_idx], df["dir_code"].to_numpy(int)[ca_idx]
        eh_te = df.loc[m_te, "eh"].to_numpy(float)

        t_pred = qreg.predict(Xte)
        proba = M.predict_proba(dmodel, Xte)
        acc["eh"].append(eh_te)
        acc["true"].append(df.loc[m_te, "true"].to_numpy(float))
        acc["true_pred"].append(eh_te / np.exp(t_pred))
        acc["t"].append(df.loc[m_te, "t"].to_numpy(float))
        acc["t_pred"].append(t_pred)
        acc["p_ok"].append(proba[:, 0])
        acc["p_long"].append(proba[:, 1])
        acc["p_short"].append(proba[:, 2])
        acc["dir_code"].append(df.loc[m_te, "dir_code"].to_numpy(int))
        acc["tol_repeats"].append(df.loc[m_te, "tol_repeats"].to_numpy(float))
        print("    fold %d: train pool=%d test=%d" % (i, int(m_tr.sum()), int(m_te.sum())), flush=True)
        if ranked is None:
            # Both rankings are measured on the CALIBRATION chromosomes, not the test ones. The
            # ablation curves add features in these orders and then score each prefix on the fold's
            # TEST chromosomes; ranking on those same test rows would let their labels pick the
            # feature subsets whose held-out score is then reported, biasing both curves optimistically.
            # The calib chromosomes are disjoint from both train and test, so the ordering is chosen
            # without seeing a single ablation-scoring row. They are not pristine either: the direction
            # head's isotonic calibrators are fit on them, so these bars are out-of-training but not
            # out-of-sample. The report's importance note says so. The LCF head is fit and applied only
            # where the gate fires (train.fit_heads), so its importance is scored on those rows too.
            fires = M.round_like_emitted(M.predict_proba(dmodel, Xca)[:, 0]) < 0.5
            ranked = _importance(qreg, Xca[fires], tca[fires], names)
            ranked_dir = _dir_importance(dmodel, Xca, yca, names)
    return {k: np.concatenate(v) for k, v in acc.items()}, ranked, ranked_dir


def _importance(qreg, X, t_true, names):
    """Returns fold-0 q-head permutation importance (pinball-scored), most important first.

    ``X`` / ``t_true`` are the fold's calibration rows (see ``collect_oof``). Positive = the loss ROSE
    when the feature was permuted, so bigger is more important.
    """
    idx = _cap_rows(np.arange(len(X)), IMPORTANCE_CAP, SEED)
    res = permutation_importance(
        qreg, X.iloc[idx], np.asarray(t_true, dtype=float)[idx],
        scoring=make_scorer(mean_pinball_loss, alpha=0.5, greater_is_better=False),
        n_repeats=IMPORTANCE_REPEATS, random_state=SEED)
    order = np.argsort(res.importances_mean)[::-1]
    return [(names[i], float(res.importances_mean[i]), float(res.importances_std[i])) for i in order]


def _dir_log_loss(y, proba):
    """Multinomial cross-entropy of ``proba`` (``(n, 3)`` in ``[pOk, pTooLong, pTooShort]``) against ``y``."""
    return float(log_loss(y, proba, labels=list(range(M.N_CLASSES))))


def _dir_importance(dmodel, X_in, y_true, names):
    """Returns fold-0 direction-head permutation importance (log-loss-scored), most important first.

    Hand-rolled rather than ``permutation_importance`` because the deployed direction predictor is a
    classifier PLUS per-class isotonic calibrators (``model.predict_proba``), not a bare sklearn
    estimator -- scoring the classifier alone would rank features against a predictor that is not the
    one shipped. Each feature's column is shuffled ``IMPORTANCE_REPEATS`` times in place and the
    importance is the mean RISE in log-loss, so the sign convention matches ``_importance``'s
    loss-increase-when-destroyed and the two panels read the same way.

    ``X_in`` / ``y_true`` are the fold's calibration rows (see ``collect_oof``); ``X_in`` is copied
    before any shuffling, so the caller's frame is never mutated.
    """
    idx = _cap_rows(np.arange(len(X_in)), IMPORTANCE_CAP, SEED)
    X = X_in.iloc[idx].copy()
    y = np.asarray(y_true, dtype=int)[idx]
    base = _dir_log_loss(y, M.predict_proba(dmodel, X))
    rng = np.random.default_rng(SEED)
    ranked = []
    for j, name in enumerate(names):
        # Shuffle the column in place and restore it, rather than copying the whole (up to
        # IMPORTANCE_CAP x 29) frame once per repeat per feature.
        original = X.iloc[:, j].to_numpy(copy=True)
        rises = []
        for _ in range(IMPORTANCE_REPEATS):
            X.iloc[:, j] = rng.permutation(original)
            rises.append(_dir_log_loss(y, M.predict_proba(dmodel, X)) - base)
        X.iloc[:, j] = original
        ranked.append((name, float(np.mean(rises)), float(np.std(rises))))
    return sorted(ranked, key=lambda e: e[1], reverse=True)


def _ablation_split(df, branch, fold):
    """Builds the one-fold train/calib/test slices both ablation curves are computed on.

    Shared so the q-head and direction-head curves are always fit and scored on exactly the same
    rows (same fold, same seeded caps), which is what makes them comparable side by side. The caps
    are smaller than the headline CV's because an ablation re-fits a head once per prefix length.

    Returns:
        ``(Xtr, Xca, Xte, tr_idx, ca_mask, te_idx)`` -- three feature matrices carrying every
        feature column, plus the row selectors the caller uses to slice its own labels.
    """
    chrom = df["chrom"].astype(str)
    # Test and calibration rows must be representative (see collect_oof); training may use every row.
    rep = df["representative"].to_numpy(bool)
    tr_idx = _cap_rows(np.where(chrom.isin(set(fold["train"])).to_numpy())[0], ABLATION_TRAIN_CAP, SEED)
    te_idx = _cap_rows(np.where(chrom.isin(set(fold["test"])).to_numpy() & rep)[0], ABLATION_TEST_CAP, SEED)
    ca_mask = chrom.isin(set(fold["calib"])).to_numpy() & rep
    Xtr, _ = features.build_matrix(df.iloc[tr_idx], branch)
    Xca, _ = features.build_matrix(df[ca_mask], branch)
    Xte, _ = features.build_matrix(df.iloc[te_idx], branch)
    return Xtr, Xca, Xte, tr_idx, ca_mask, te_idx


def _ablation_curve(df, branch, fold, order):
    """Add-one-feature held-out MAE curve: fit the q-head on the top-1, top-2, ... features.

    Features are added in ``order`` (this genotyping regime's own q-head importance ranking). Each
    q-head is fit on one fold's training chromosomes and scored on its test chromosomes by the MAE
    of the corrected call ``eh/LCF`` against the truth (repeat units) -- i.e. how close the
    corrected call is to the true size.

    Returns:
        A list of ``{"k", "feature", "mae"}`` dicts: ``k=0`` is the raw-EH baseline (no correction)
        on the fold-0 test rows, then one dict per prefix length ``k=1, 2, ...``.
    """
    Xtr, Xca, Xte, tr_idx, ca_mask, te_idx = _ablation_split(df, branch, fold)
    ttr, tca = df["t"].to_numpy(float)[tr_idx], df.loc[ca_mask, "t"].to_numpy(float)
    eh_te = df["eh"].to_numpy(float)[te_idx]
    true_te = df["true"].to_numpy(float)[te_idx]
    # k=0 baseline: raw-EH MAE on the SAME fold-0 capped test rows as the k>=1 points, so the curve's
    # x=0 anchor and its corrected points are on one population (not the full-pool q.mae_eh).
    curve = [{"k": 0, "feature": "(raw EH)", "mae": float(np.mean(np.abs(true_te - eh_te)))}]
    for k in range(1, min(ABLATION_KMAX, len(order)) + 1):
        cols = order[:k]
        qreg = M.train_q_median(Xtr[cols], ttr, Xca[cols], tca)
        true_pred = eh_te / np.exp(qreg.predict(Xte[cols]))
        mae = float(np.mean(np.abs(true_te - true_pred)))
        curve.append({"k": k, "feature": order[k - 1], "mae": mae})
        print("    ablation k=%2d (+%-24s) MAE=%.4f" % (k, order[k - 1], mae), flush=True)
    return curve


def _ranking_contract_gap(ranking, branch):
    """Returns how a cached importance ranking differs from the branch's current feature list, or None.

    An ablation loop runs to ``min(ABLATION_KMAX, len(order))``, so a ranking written before a feature
    was added is SHORT and its curve stops early while the report's text says it runs up to all
    features. Both consumers of a cached ranking have to know: ``--ablation-only`` recomputes curves
    from it, and ``--render-only`` re-plots the curve it already produced.
    """
    want, got = features.feature_names(branch), [f for f, _, _ in ranking]
    if set(want) == set(got):
        return None
    return ("covers %d feature(s) but the %s branch now has %d (only in the cache: %s; only in "
            "features.py: %s)" % (len(got), branch, len(want),
                                  sorted(set(got) - set(want)) or "(none)",
                                  sorted(set(want) - set(got)) or "(none)"))


def _assert_ranking_covers_contract(ranking, branch, genotyping_regime, key):
    """Raises if a cached importance ranking does not cover the branch's current feature list.

    Used by ``--ablation-only``, which RECOMPUTES curves from the cached ranking: a short ranking
    would silently produce a truncated curve, so refusing is better than warning.
    """
    gap = _ranking_contract_gap(ranking, branch)
    if gap:
        raise SystemExit(
            "ERROR: results.json's %r ranking for %s %s.\nThe ablation curve would stop short of "
            "'all features' without saying so. Re-run `python3 report.py` without --ablation-only "
            "to recompute the rankings first." % (key, genotyping_regime, gap))


def _stale_contract_warning(results, homo_results=None):
    """Returns a reader-facing note when a cached result set predates the current code.

    The render paths must still work from cached artifacts (that is what they are for), so this
    warns rather than refusing -- but the warning goes into the HTML as well as stdout, because two
    sections make claims a stale cache breaks: the ablation note promises a curve running "up to all
    features", and the importance note says the bars were measured on the CALIBRATION chromosomes.

    Both caches are checked. The homopolymer panels come from ``results_homopolymer.json``, whose
    only other guard is a per-regime row count -- which passes unchanged when the code, not the data,
    is what moved on.

    ``importance_direction`` doubles as the provenance marker for the second claim: it was added by
    the same change that moved importance off the test chromosomes, so a ranking without it was
    ranked on the test chromosomes.
    """
    gaps = {}
    for label, bundle in (("results.json", results), ("results_homopolymer.json", homo_results)):
        for r in bundle or []:
            branch = features.GENOTYPING_REGIME_BRANCH[r["genotyping_regime"]]
            ranking = r.get("importance") or []
            reasons = []
            if not ranking:
                reasons.append("carries no ranking at all")
            else:
                gap = _ranking_contract_gap(ranking, branch)
                if gap:
                    reasons.append(gap)
                if not r.get("importance_direction"):
                    reasons.append("was ranked on the TEST chromosomes, not the calibration ones")
            if reasons:
                gaps["%s / %s" % (label, r["genotyping_regime"])] = "; ".join(reasons)
    if not gaps:
        return ""
    for where, why in sorted(gaps.items()):
        print("  WARNING: cached %s %s" % (where, why), flush=True)
    return ("<p class='note' style='color:#a33'><b>Stale cache:</b> this report reuses cached "
            "cross-validation results that predate the current code, so the importance and ablation "
            "panels below do not match the descriptions beside them (%s). Re-run "
            "<code>python3 report.py --homopolymer-cv</code> and <code>python3 report.py</code> to "
            "recompute them.</p>"
            % html.escape("; ".join("%s: %s" % (k, v) for k, v in sorted(gaps.items()))))


def _dir_ablation_scores(y_te, proba):
    """Returns the per-``k`` direction-head scores stored in a ``dir_ablation`` curve entry.

    ``log_loss`` is the plotted one (the head's own training objective, and a proper scoring rule,
    so it rewards calibration and not just ranking); the AUC / average-precision / calibration-error
    columns ride along so the curve can be re-plotted against a different metric without re-fitting
    every prefix. Mirrors ``metrics.direction_metrics`` minus the confusion matrix, which is not
    meaningful to plot against ``k``.
    """
    scored = metrics.direction_metrics(y_te, proba)
    return {k: v for k, v in scored.items() if k not in ("n", "confusion")}


def _dir_ablation_curve(df, branch, fold, order):
    """Add-one-feature held-out log-loss curve: fit the direction head on the top-1, top-2, ... features.

    The direction-head counterpart of ``_ablation_curve``: same fold, same rows (``_ablation_split``),
    features added in this genotyping regime's own DIRECTION-head importance order, and each prefix
    scored on the calibrated ``[pOk, pTooLong, pTooShort]`` probabilities the deployed predictor emits
    (classifier + isotonic, i.e. ``model.predict_proba``).

    Returns:
        A list of ``{"k", "feature", <scores from _dir_ablation_scores>}`` dicts. ``k=0`` is the
        feature-free baseline: the class prior measured on this fold's representative training rows, predicted
        constantly for every test row -- the log-loss any feature has to beat.
    """
    Xtr, Xca, Xte, tr_idx, ca_mask, te_idx = _ablation_split(df, branch, fold)
    ytr = df["dir_code"].to_numpy(int)[tr_idx]
    yca = df.loc[ca_mask, "dir_code"].to_numpy(int)
    yte = df["dir_code"].to_numpy(int)[te_idx]

    # The prior is a frequency, so it is measured on the representative training rows only; the
    # training-only quota top-up over-represents rare, error-prone calls.
    rep_tr = df["representative"].to_numpy(bool)[tr_idx]
    prior = np.bincount(ytr[rep_tr], minlength=M.N_CLASSES) / rep_tr.sum()
    curve = [{"k": 0, "feature": "(class prior)",
              **_dir_ablation_scores(yte, np.tile(prior, (yte.size, 1)))}]
    print("    dir ablation k= 0 (%-25s) log_loss=%.4f" % ("class prior", curve[0]["log_loss"]),
          flush=True)
    for k in range(1, min(ABLATION_KMAX, len(order)) + 1):
        cols = order[:k]
        dmodel = M.train_direction(Xtr[cols], ytr, Xca[cols], yca)
        scores = _dir_ablation_scores(yte, M.predict_proba(dmodel, Xte[cols]))
        curve.append({"k": k, "feature": order[k - 1], **scores})
        print("    dir ablation k=%2d (+%-24s) log_loss=%.4f"
              % (k, order[k - 1], scores["log_loss"]), flush=True)
    return curve


def _load_genotyping_regime_df(genotyping_regime, data_dir, homopolymers_only=False):
    """Loads one genotyping regime's rows (only the columns the eval needs) + its branch.

    By default homopolymer (1 bp motif) loci are excluded (the main report); ``homopolymers_only``
    flips that to keep ONLY homopolymers (the separate side-by-side homopolymer charts).
    """
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"eh", "true", "t", "dir_code", "chrom", "genotyping_regime", "tol_repeats", "representative"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    if "representative" not in cols:
        raise SystemExit("ERROR: %s has no 'representative' column; re-assemble it with dataset.py" % parquet)
    df = pd.read_parquet(parquet, columns=cols)
    df = df[df["genotyping_regime"] == genotyping_regime]
    df = df[df["motif_size"] == 1] if homopolymers_only else df[df["motif_size"] != 1]
    return df.reset_index(drop=True), branch


def _homopolymer_row_counts(data_dir):
    """Returns the current homopolymer (1 bp motif) row count per genotyping_regime in the parquets.

    Mirrors the filtering ``_load_genotyping_regime_df(homopolymers_only=True)`` applies (motif_size
    == 1, grouped by genotyping_regime), so the counts can be compared against a cached
    ``results_homopolymer.json``'s ``n_rows`` to detect that it was computed from a different (stale)
    data pool. Returns an empty dict when no parquets are present (freshness then cannot be checked).
    """
    counts = {}
    for src in ("quick", "full"):
        parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
        if not os.path.exists(parquet):
            continue
        df = pd.read_parquet(parquet, columns=["motif_size", "genotyping_regime"])
        for regime, n in df[df["motif_size"] == 1]["genotyping_regime"].value_counts().items():
            counts[regime] = counts.get(regime, 0) + int(n)
    return counts


def evaluate_genotyping_regime(genotyping_regime, data_dir, folds, train_cap, homopolymers_only=False):
    """Loads one genotyping_regime's rows, runs OOF, returns ``(metrics_bundle, oof)``.

    The ``oof`` dict (per-allele OOF arrays from ``collect_oof``) is returned alongside the
    JSON-serializable metrics bundle so the caller can persist the direction-head probabilities for
    the precision-recall plots (see ``_save_dir_oof``); it is NOT written to
    ``results.json``.
    """
    df, branch = _load_genotyping_regime_df(genotyping_regime, data_dir, homopolymers_only)
    print("=== %s (branch %s): %d rows ===" % (genotyping_regime, branch, len(df)), flush=True)

    oof, ranked, ranked_dir = collect_oof(df, genotyping_regime, branch, folds, train_cap)
    order = [f for f, _, _ in (ranked or [])]
    order_dir = [f for f, _, _ in (ranked_dir or [])]
    print("  ablation (add-one curve, LCF head) ...", flush=True)
    ablation = _ablation_curve(df, branch, folds[0], order) if order else []
    print("  ablation (add-one curve, direction head) ...", flush=True)
    dir_ablation = _dir_ablation_curve(df, branch, folds[0], order_dir) if order_dir else []
    return {
        "genotyping_regime": genotyping_regime,
        "n_rows": len(df),
        "q": metrics.q_metrics(oof["eh"], oof["true"], oof["true_pred"], oof["t"], oof["t_pred"],
                               oof["tol_repeats"]),
        "direction": metrics.direction_metrics(
            oof["dir_code"], np.column_stack([oof["p_ok"], oof["p_long"], oof["p_short"]])),
        # The deployment metric: LCF and pOk rounded to the 3 decimals ExpansionHunter emits and the
        # corrected call rounded to whole repeats, as the held-out evaluation does, so both score the
        # same corrected call under the same gate.
        "gated": metrics.gated_mae(oof["eh"], oof["true"],
                                   np.round(oof["eh"] / M.round_like_emitted(np.exp(oof["t_pred"]))),
                                   M.round_like_emitted(oof["p_ok"])),
        "importance": ranked,
        "importance_direction": ranked_dir,
        "ablation": ablation,
        "dir_ablation": dir_ablation,
    }, oof


def _save_dir_oof(evals, path):
    """Persists each regime's OOF direction-head arrays (``dir_code``, the 3 class probabilities, and
    the signed call-error ``delta``).

    Keyed ``<genotyping_regime>__{dir_code,p_ok,p_long,p_short,delta}`` so
    ``plot_pr`` / ``plot_roc`` / ``plot_prob_violins`` can be regenerated in ``--render-only`` mode
    without re-running the cross-validation. ``delta = round(eh) - round(true)`` is the per-allele EH
    call error in repeat units (the size-bin x-axis).

    Args:
        evals: List of ``(metrics_bundle, oof)`` tuples from ``evaluate_genotyping_regime``.
        path: Output ``.npz`` path.
    """
    out = {}
    for res, oof in evals:
        r = res["genotyping_regime"]
        for k in ("dir_code", "p_ok", "p_long", "p_short"):
            out["%s__%s" % (r, k)] = np.asarray(oof[k])
        out["%s__delta" % r] = np.round(oof["eh"]) - np.round(oof["true"])
    np.savez_compressed(path, **out)


# --- plots ----------------------------------------------------------------

def plot_mae(results, out_png, title_tag=""):
    """Draws the raw-EH vs gated-LCF MAE bar chart (broken linear axis), per genotyping_regime.

    ``title_tag`` (the dataset and loci set) is added as a second title line, so charts of different
    datasets cannot be mistaken for one another.
    """
    labels = [features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]] for r in results]
    raw = [r["gated"]["mae_raw"] for r in results]
    gat = [r["gated"]["mae_gated"] for r in results]
    x = np.arange(len(labels))

    biggest = max(raw + gat)
    rest = max([v for v in raw + gat if v < biggest] or [biggest])
    broken = biggest > 2.5 * rest  # a lone outlier bar -> break the axis to keep small bars readable

    def draw_bars(ax):
        ax.bar(x - 0.2, raw, 0.4, label="raw EH", color=GRAY)
        ax.bar(x + 0.2, gat, 0.4, label="LCF-corrected (pOk<0.5 threshold)", color=ORANGE)
        ax.grid(axis="y", alpha=0.3)

    title = "Mean absolute error: raw EH allele size vs allele size after LCF correction"
    if title_tag:
        title += "\n" + title_tag
    ylabel = "MAE |true − call|  (repeat units)"
    if not broken:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        draw_bars(ax)
        for i, (rr, gg) in enumerate(zip(raw, gat)):
            ax.text(i - 0.2, rr, "%.2f" % rr, ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.2, gg, "%.2f" % gg, ha="center", va="bottom", fontsize=8, weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=8, loc="upper left")
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        return

    lo = (0, rest * 1.15)
    hi = (biggest * 0.93, biggest * 1.04)
    fig, (ax_hi, ax_lo) = plt.subplots(2, 1, sharex=True, figsize=(7, 4.6),
                                       gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
    for ax in (ax_hi, ax_lo):
        draw_bars(ax)
    ax_hi.set_ylim(*hi); ax_lo.set_ylim(*lo)

    def label(ax, xi, val, bold):
        rng = (hi[1] - hi[0]) if ax is ax_hi else (lo[1] - lo[0])
        ax.text(xi, val + rng * 0.03, "%.2f" % val, ha="center", fontsize=8,
                weight="bold" if bold else "normal")
    for i, (rr, gg) in enumerate(zip(raw, gat)):
        label(ax_hi if rr > lo[1] else ax_lo, i - 0.2, rr, False)
        label(ax_hi if gg > lo[1] else ax_lo, i + 0.2, gg, True)

    ax_hi.spines["bottom"].set_visible(False); ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False)
    d = 0.012
    kw = dict(transform=ax_hi.transAxes, color="k", clip_on=False, lw=1)
    ax_hi.plot((-d, +d), (-d, +d), **kw); ax_hi.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (1 - d * 3, 1 + d * 3), **kw)
    ax_lo.plot((1 - d, 1 + d), (1 - d * 3, 1 + d * 3), **kw)

    ax_lo.set_xticks(x); ax_lo.set_xticklabels(labels)
    ax_hi.set_title(title); ax_hi.legend(fontsize=8, loc="upper left")
    fig.supylabel(ylabel, fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def _panel_rows(results, key, top_n):
    """Selects the bars each importance panel draws, reading ranking ``key`` and nothing else.

    Split out of ``plot_importance_panel`` so the choice of ranking is testable without rendering:
    the panel is parameterized by ``key`` (q head vs direction head) and drawing the wrong one under
    the right title is a silent, plausible mistake that a "did a PNG appear" test cannot catch.

    Returns:
        ``(rank, by_regime)`` where ``rank`` maps a feature to its 1-based position in the
        ``full_nonspanning`` ranking (the cross-panel ``#n`` suffix) and ``by_regime`` maps each
        genotyping regime to its own top-``top_n`` ``[(feature, mean, std), ...]``. Both are empty
        when any regime is missing that ranking (a cached ``results.json`` predating it).
    """
    by_reg = {r["genotyping_regime"]: r for r in results}
    if any(not (by_reg.get(reg) or {}).get(key) for reg in features.GENOTYPING_REGIMES):
        return {}, {}
    rank = {f: i + 1 for i, (f, _, _) in
            enumerate(by_reg[features.GENOTYPING_REGIME_FULL_NONSPANNING][key])}
    return rank, {reg: by_reg[reg][key][:top_n] for reg in features.GENOTYPING_REGIMES}


def plot_importance_panel(results, out_png, top_n=TOP_N, loci_label="non-homopolymer loci",
                          key="importance", head_label="LCF prediction",
                          score_label="mean pinball-loss rise"):
    """Draws the three per-regime importance charts in one horizontal row.

    Charts run quick, full_spanning, full_nonspanning left to right; the ``#n`` ranks are set by
    full_nonspanning.

    The ``(#rank)`` suffix on every feature label is that feature's rank in the
    ``full_nonspanning`` importance order (1 = most important there); the same suffix is reused on
    the other two charts so a feature can be cross-referenced across genotyping regimes. Bars within
    each chart are still sorted by that chart's own importance.

    Args:
        results: The per-regime metrics bundles.
        out_png: Output path.
        top_n: How many features to show per chart.
        loci_label: Which loci the panel was computed on (title only).
        key: Which ranking to read -- ``"importance"`` (q head) or ``"importance_direction"``.
        head_label: Which head the ranking scores (title only).
        score_label: What the bar length means (x-axis label).

    Returns:
        True if the panel was drawn, False if ``key`` is absent/empty for some regime (a cached
        ``results.json`` predating that ranking), in which case no file is written.
    """
    rank, by_reg = _panel_rows(results, key, top_n)
    if not by_reg:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, reg in zip(axes, features.GENOTYPING_REGIMES):
        ranked = by_reg[reg]
        peak = max((m for _, m, _ in ranked), default=1.0) or 1.0
        pos = np.arange(len(ranked))[::-1]
        ax.barh(pos, [m / peak for _, m, _ in ranked], xerr=[s / peak for _, _, s in ranked],
                color="#4c72b0", ecolor="gray", capsize=3)
        ax.set_yticks(pos)
        ax.set_yticklabels(["%s (#%d)" % (f, rank.get(f, 0)) for f, _, _ in ranked])
        ax.set_xlabel("relative importance\n(%s, divided by the bucket's largest)" % score_label)
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[reg])
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Relative feature importance (%s, calibration chromosomes; %s) — rank # set by "
                 "full_nonspanning" % (head_label, loci_label), fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


def plot_ablation(results, out_png):
    """Add-one-feature held-out MAE curve per genotyping regime, with a k=0 raw-EH point.

    k=0 is the raw-EH MAE (no correction) on the same capped fold-0 test rows as the k>=1 points, as
    carried in the curve itself; ``q.mae_eh`` is only a fallback for cached curves that predate that
    k=0 row and is measured on the whole pool, so the two are not interchangeable.
    """
    series = []  # (display, color, xs, ys)
    for r in results:
        ablation = r.get("ablation") or []
        if not ablation:
            continue
        xs = [d["k"] for d in ablation]
        ys = [d["mae"] for d in ablation]
        if xs[0] != 0:  # old cached curve w/o a k=0 row: fall back to the full-pool raw-EH baseline
            xs = [0] + xs
            ys = [r["q"]["mae_eh"]] + ys
        series.append((features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]],
                       REGIME_COLORS.get(r["genotyping_regime"]), xs, ys))
    return _plot_ablation_series(
        series, out_png,
        title="Add-one-feature ablation (LCF prediction)",
        xlabel="number of top features included (LCF prediction importance order; 0 = raw EH)",
        ylabel="MAE |true − corrected|  (repeat units, held-out)")


def plot_dir_ablation(results, out_png, title_tag=""):
    """Add-one-feature held-out log-loss curve per genotyping regime, with a k=0 class-prior point.

    The direction-head counterpart of ``plot_ablation``: k=0 is the feature-free class-prior
    predictor and k>=1 the calibrated 3-class head fit on that many top features.

    Returns:
        ``"single"`` or ``"broken"`` (which y-axis treatment was used) once the plot is drawn, or
        False when no regime carries a ``dir_ablation`` curve (a cached ``results.json`` predating
        it), in which case no file is written. Both strings are truthy, so callers that only ask
        "was anything drawn?" still read correctly.
    """
    series = []
    for r in results:
        curve = r.get("dir_ablation") or []
        if not curve:
            continue
        series.append((features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]],
                       REGIME_COLORS.get(r["genotyping_regime"]),
                       [d["k"] for d in curve], [d["log_loss"] for d in curve]))
    if not series:
        return False
    return _plot_ablation_series(
        series, out_png,
        title="Add-one-feature ablation (direction prediction)%s" % (" — %s" % title_tag if title_tag else ""),
        xlabel="number of top features included (direction importance order; 0 = class prior)",
        ylabel="multinomial log-loss  (held-out, calibrated)")


def _plot_ablation_series(series, out_png, title, xlabel, ylabel):
    """Draws add-one-feature curves (one line per genotyping regime) and writes ``out_png``.

    ``series`` is a list of ``(display_name, color, xs, ys)``. ONLY when the k=0 anchor of the worst
    regime towers over everything else (more than 2.5x the next value) is the y-axis broken, so the
    baseline and the post-correction detail are both readable in one plot; otherwise a single
    continuous axis is used. Whether that happened is returned, because the report's prose describes
    the axis and must not claim a break that is not there.

    Returns:
        ``"broken"`` or ``"single"`` once the figure is written; False when there is nothing to draw
        (both truthy strings, so callers can keep treating the result as a drew-anything flag).
    """
    if not series:
        return False
    biggest = max(ys[0] for _, _, _, ys in series)                            # worst regime's k=0 anchor
    below = [v for _, _, _, ys in series for v in ys if v < biggest]
    rest = max(below) if below else biggest
    broken = biggest > 2.5 * rest

    def draw(ax):
        for disp, color, xs, ys in series:
            ax.plot(xs, ys, "-o", ms=4, label=disp, color=color)
        ax.grid(True, alpha=0.3)

    if not broken:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax); ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        return "single"

    lo = (0, rest * 1.15)
    hi = (biggest * 0.95, biggest * 1.04)
    fig, (ax_hi, ax_lo) = plt.subplots(2, 1, sharex=True, figsize=(8, 5.4),
                                       gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
    draw(ax_hi); draw(ax_lo)
    ax_hi.set_ylim(*hi); ax_lo.set_ylim(*lo)
    ax_hi.spines["bottom"].set_visible(False); ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False)
    d = 0.012
    kw = dict(transform=ax_hi.transAxes, color="k", clip_on=False, lw=1)
    ax_hi.plot((-d, +d), (-d, +d), **kw); ax_hi.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (1 - d * 3, 1 + d * 3), **kw)
    ax_lo.plot((1 - d, 1 + d), (1 - d * 3, 1 + d * 3), **kw)
    ax_lo.set_xlabel(xlabel); ax_hi.set_title(title); ax_hi.legend(fontsize=9)
    fig.supylabel(ylabel, fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return "broken"


# --- direction-head accuracy plots (pOk / pTooLong / pTooShort) -----------
# 1×2 precision-recall + 1×3 confusion, on the pooled 5-fold OOF probabilities. Regimes are
# overlaid (PR) or paneled (confusion) and read straight from the dir_oof npz / results.json, so
# they regenerate in --render-only.


def plot_pr(oof, out_png, title_tag="non-homopolymer"):
    """One-vs-rest precision-recall curves for TOO_LONG and TOO_SHORT, regimes overlaid.

    These are the two error directions the ``pOk < 0.5`` gate must catch, and PR (unlike ROC) stays
    honest under the heavy OK-class imbalance. The legend reports each regime's average precision
    (AP) and class prevalence; the dotted horizontal is that prevalence (a random classifier's
    precision floor).

    Args:
        oof: Dict of ``<genotyping_regime>__{dir_code,p_ok,p_long,p_short}`` arrays (the dir_oof npz).
        out_png: Output path.
        title_tag: Loci-set label for the suptitle.
    """
    panels = (("TOO_LONG", "p_long", features.TOO_LONG), ("TOO_SHORT", "p_short", features.TOO_SHORT))
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__dir_code" % r) in oof]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (cname, pkey, code) in zip(axes, panels):
        for r in regs:
            y = (np.asarray(oof["%s__dir_code" % r], dtype=int) == code).astype(int)
            p = np.asarray(oof["%s__%s" % (r, pkey)], dtype=float)
            if y.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, color=REGIME_COLORS[r], lw=1.6, label="%s (AP=%.3f, base=%.3f)"
                    % (features.GENOTYPING_REGIME_DISPLAY[r], average_precision_score(y, p), y.mean()))
            ax.axhline(y.mean(), color=REGIME_COLORS[r], lw=0.9, ls=":", alpha=0.6)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("recall"); ax.set_title("%s vs rest" % cname)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("precision")
    fig.suptitle("PR-ROC: detecting TOO_LONG / TOO_SHORT calls — %s (5-fold held-out)"
                 % title_tag, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_roc(oof, out_png, title_tag="non-homopolymer"):
    """One-vs-rest ROC curves for TOO_LONG and TOO_SHORT, regimes overlaid.

    The same two error directions as the PR-ROC plot, shown as the TPR/FPR trade-off; the legend shows
    each curve's AUC and the dashed diagonal is chance.

    Args:
        oof: Dict of ``<genotyping_regime>__{dir_code,p_ok,p_long,p_short}`` arrays (the dir_oof npz).
        out_png: Output path.
        title_tag: Loci-set label for the suptitle.
    """
    panels = (("TOO_LONG", "p_long", features.TOO_LONG), ("TOO_SHORT", "p_short", features.TOO_SHORT))
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__dir_code" % r) in oof]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (cname, pkey, code) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], ls="--", color="k", lw=1, alpha=0.6)
        for r in regs:
            y = (np.asarray(oof["%s__dir_code" % r], dtype=int) == code).astype(int)
            p = np.asarray(oof["%s__%s" % (r, pkey)], dtype=float)
            if y.sum() == 0 or y.sum() == y.size:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=REGIME_COLORS[r], lw=1.6, label="%s (AUC=%.3f)"
                    % (features.GENOTYPING_REGIME_DISPLAY[r], roc_auc_score(y, p)))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("false positive rate"); ax.set_title("%s vs rest" % cname)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    fig.suptitle("ROC: detecting TOO_LONG / TOO_SHORT calls — %s (5-fold held-out)"
                 % title_tag, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


# Signed call-error bins for the prob-vs-delta violins: delta = round(eh) - round(true).
# The 27 bins of the truth-vs-reference figure (mirrors original/size_bins.py); searchsorted on the
# inclusive upper edges (len 26) maps delta -> bin index 0..26 (bin 13 = the no-change "0" bin).
_DELTA_UPPER_EDGES = np.array([-31, -26, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 0,
                               2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30], dtype=float)
_DELTA_BIN_LABELS = ("≤-31", "-30:-26", "-25:-21", "-20:-19", "-18:-17", "-16:-15", "-14:-13",
                     "-12:-11", "-10:-9", "-8:-7", "-6:-5", "-4:-3", "-2:-1", "0", "1:2", "3:4",
                     "5:6", "7:8", "9:10", "11:12", "13:14", "15:16", "17:18", "19:20", "21:25",
                     "26:30", "≥31")
_PROB_ROWS = (("pOk", "p_ok"), ("pTooLong", "p_long"), ("pTooShort", "p_short"))
_DELTA_VIOLIN_CAP = 20_000  # per-bin alleles sampled for the KDE only (determinism via SEED)


def plot_prob_violins(oof, out_png, title_tag="non-homopolymer"):
    """Predicted pOk / pTooLong / pTooShort distributions vs the signed event-size bin.

    3x3 grid: rows are the three direction-head probabilities, columns the three genotyping regimes.
    Each cell holds one violin per delta bin (``delta = round(eh) - round(true)``, the per-allele EH
    call error in repeat units), so a column reads left (EH undercalls, call too short) to right (EH
    overcalls, call too long), with the dashed line at the no-error ``0`` bin. Violins are colored by
    delta (undercall -> overcall); per-bin allele counts are capped for the KDE only.

    Args:
        oof: Dict of ``<genotyping_regime>__{p_ok,p_long,p_short,delta}`` arrays (the dir_oof npz).
        out_png: Output path.
        title_tag: Loci-set label for the suptitle.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__delta" % r) in oof]
    nb = len(_DELTA_BIN_LABELS)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nb))
    rng = np.random.RandomState(SEED)
    fig, axes = plt.subplots(len(_PROB_ROWS), len(regs), figsize=(7.5 * len(regs), 10.5),
                             sharex="col", sharey=True, squeeze=False)
    for ci, r in enumerate(regs):
        delta = np.asarray(oof["%s__delta" % r], dtype=float)
        bin_idx = np.searchsorted(_DELTA_UPPER_EDGES, delta, side="left")
        bin_idx[~np.isfinite(delta)] = -1  # keep NaN deltas out of the extreme-expansion bin
        counts = [int((bin_idx == b).sum()) for b in range(nb)]  # true (pre-cap) alleles per delta bin
        for ri, (pname, pkey) in enumerate(_PROB_ROWS):
            ax = axes[ri][ci]
            p = np.asarray(oof["%s__%s" % (r, pkey)], dtype=float)
            data = []
            for b in range(nb):
                vals = p[bin_idx == b]
                if vals.size > _DELTA_VIOLIN_CAP:
                    vals = vals[rng.choice(vals.size, _DELTA_VIOLIN_CAP, replace=False)]
                data.append(vals)
            vp = _violins_skipping_empty_bins(ax, data, colors, 0.75, widths=0.9, showextrema=False)
            if vp and "cmedians" in vp:
                vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(0.8)
            ax.axvline(14, color="k", lw=0.8, ls="--", alpha=0.5)  # the "0" (no-change) bin
            ax.set_ylim(-0.02, 1.02); ax.grid(axis="y", alpha=0.3)
            if ri == 0:
                _violin_counts(ax, counts, header=(ci == 0))
                ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r], fontsize=11, weight="bold",
                             pad=_VIOLIN_COUNT_PAD)
            if ci == 0:
                ax.set_ylabel("%s\n(predicted probability)" % pname, fontsize=10)
            ax.set_xticks(np.arange(1, nb + 1))
            if ri == len(_PROB_ROWS) - 1:
                ax.set_xticklabels(_DELTA_BIN_LABELS, rotation=90, fontsize=9)
                ax.set_xlabel("EH call − True allele size (repeats)", fontsize=13)
            else:
                ax.set_xticklabels([])
    fig.suptitle("Predicted direction probabilities vs EH call error (delta bin) — %s (5-fold held-out)"
                 % title_tag, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close(fig)


def plot_confusion(results, out_png, title_tag="non-homopolymer"):
    """Row-normalized 3×3 confusion matrices (true rows × argmax-predicted columns), one per regime.

    Each cell is annotated with its raw count and row-percent (the diagonal is per-class recall);
    color encodes the row-normalized value. Reads ``direction.confusion`` straight from the metrics
    bundle, so it needs no OOF arrays.

    Args:
        results: List of per-regime metrics bundles (each with ``direction.confusion``).
        out_png: Output path.
        title_tag: Loci-set label for the suptitle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False, layout="constrained")
    for ax, r in zip(axes[0], results):
        cm = np.asarray(r["direction"]["confusion"], dtype=float)
        rowsum = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, rowsum, out=np.zeros_like(cm), where=rowsum > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "%d\n%.0f%%" % (int(cm[i, j]), 100 * norm[i, j]), ha="center",
                        va="center", fontsize=9, color="white" if norm[i, j] > 0.5 else "#222")
        ax.set_xticks(range(3)); ax.set_xticklabels(features.DIR_CLASS_NAMES, fontsize=8, rotation=20, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(features.DIR_CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted (argmax)"); ax.set_ylabel("true")
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]])
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="row-normalized (recall)")
    fig.suptitle("Direction-predictor confusion (argmax prediction, row-normalized) — %s (5-fold held-out)"
                 % title_tag, fontsize=13, weight="bold")
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_helped_hurt(holdout, out_png, homopolymer=False, title_tag="non-homopolymer",
                     dataset_label="held-out HPRC"):
    """Loci the LCF would move closer (green) vs further (red) from truth, split by pOk stratum.

    Left panel = pOk<0.5 (where the gate APPLIES the LCF): helped should dominate. Right panel =
    pOk>=0.5 (where the gate KEEPS raw EH): hurt dominating is exactly why those calls are left alone.
    ``homopolymer`` reads the per-regime ``homopolymer`` sub-dict counts instead of the top-level
    (non-homopolymer) counts; ``title_tag`` labels the suptitle; ``dataset_label`` names the dataset.
    """
    gr = holdout["genotyping_regimes"]
    def _c(r):
        return gr[r].get("homopolymer", {}) if homopolymer else gr[r]
    regs = [r for r in features.GENOTYPING_REGIMES if _c(r).get("helped_lt") is not None]
    labels = [features.GENOTYPING_REGIME_DISPLAY[r] for r in regs]
    x = np.arange(len(regs))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (tag, title) in zip(axes, (("lt", "pOk < 0.5"),
                                       ("ge", "pOk ≥ 0.5"))):
        helped = [_c(r)["helped_%s" % tag] for r in regs]
        hurt = [_c(r)["hurt_%s" % tag] for r in regs]
        ax.bar(x - 0.2, helped, 0.4, label="helped (corrected closer)", color="#2ca02c")
        ax.bar(x + 0.2, hurt, 0.4, label="hurt (corrected further)", color="#d62728")
        for xi, v in list(zip(x - 0.2, helped)) + list(zip(x + 0.2, hurt)):
            ax.text(xi, v, "{:,}".format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("number of alleles")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)  # headroom so legend clears tall bars
        ax.set_title(title); ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.suptitle("Would LCF correction help or hurt? (%s, by pOk stratum) — %s" % (dataset_label, title_tag),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


# Per-allele error reduction (the violin y-axis): how much closer the LCF-corrected
# call lands to truth than raw EH, in repeat units. >0 = correction helped.
_REDUCTION_LABEL = "error reduction (repeats)\n>0 = better, <0 = worse than EH original allele size"


def _violin_ylim(ax, data):
    """Sets a 1-99th-percentile y-limit (with padding) so a few extreme alleles don't flatten violins."""
    nonempty = [d for d in data if len(d)]
    if not nonempty:
        return
    allv = np.concatenate(nonempty)
    if allv.size > 1:
        lo, hi = np.percentile(allv, [1, 99])
        pad = max(0.5, 0.1 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)


_VIOLIN_COUNT_PAD = 26  # title pad (points) leaving room for the angled per-violin count row


def _violin_counts(ax, counts, header=False):
    """Angled per-violin allele counts just above the top axis border, matching the accuracy-by-size
    plot's 'Alleles Per Bin' row.

    ``counts`` is one int per violin in x-order (violin positions 1..N); zeros are skipped. Draw AFTER
    plotting and pair with ``set_title(..., pad=_VIOLIN_COUNT_PAD)`` so the panel title clears the
    numbers. ``header`` adds the grey 'Alleles per violin' caption (use on one panel per figure).
    """
    trans = ax.get_xaxis_transform()  # x in data coords (violin positions), y in axes fraction
    for i, n in enumerate(counts):
        if n:
            ax.text(i + 1, 1.01, "{:,}".format(int(n)), transform=trans, ha="center", va="bottom",
                    rotation=45, fontsize=6, color="#777777", clip_on=False)
    if header:
        ax.text(0.0, 1.16, "Alleles per violin", transform=ax.transAxes, fontsize=8, color="#777777")


def _style_violin_extrema(vp):
    """Thin grey min/max whiskers + connector so each violin's true extremes read as caps.

    ``violinplot`` clips a bin's KDE to its observed min/max, ending in a flat edge; these caps mark
    that the edge IS the data extreme (not an axis truncation). Medians keep their default styling.
    """
    for key, lw in (("cmins", 0.9), ("cmaxes", 0.9), ("cbars", 0.5)):
        if key in vp:
            vp[key].set_edgecolor("#666666")
            vp[key].set_linewidth(lw)


def _violins_skipping_empty_bins(ax, data, colors, alpha, widths=0.85, showextrema=True):
    """Draws one violin per non-empty array in ``data`` at x position ``i + 1``, colored ``colors[i]``.

    An empty bin keeps its x slot (callers set every tick) but gets no violin, rather than a made-up
    zero-valued one whose median and extrema would read as observed data. Returns the
    ``violinplot`` dict, or None when every bin is empty.
    """
    positions = [i + 1 for i, d in enumerate(data) if len(d)]
    if not positions:
        return None
    vp = ax.violinplot([data[p - 1] for p in positions], positions=positions, showmedians=True,
                       showextrema=showextrema, widths=widths)
    _style_violin_extrema(vp)
    for body, p in zip(vp["bodies"], positions):
        body.set_facecolor(colors[p - 1]); body.set_alpha(alpha)
    return vp


def plot_violins(violin, out_png, thresholds=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                 pfx="", title_tag="non-homopolymer", dataset_label="held-out HPRC"):
    """Violins of the per-allele signed error reduction, one panel per genotyping regime.

    reduction = |true - eh| - |true - round(eh/LCF)| (>0 = correction moved the call closer to truth). Per
    panel: one violin for each tightening gate pOk < t (blue, nested subsets) plus pOk >= 0.5 (orange,
    where the gate keeps raw EH). Per-regime y-scale; y clipped to the 1-99th percentile.

    ``pfx`` selects the per-allele arrays: ``""`` -> non-homopolymer ``{r}__red/pok``, ``"h"`` ->
    homopolymer-only ``{r}__hred/hpok``. ``title_tag`` labels the suptitle; ``dataset_label`` names
    the dataset.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__%sred" % (r, pfx)) in violin]
    fig, axes = plt.subplots(1, len(regs), figsize=(5.0 * len(regs), 5.8))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regs):
        red = np.asarray(violin["%s__%sred" % (r, pfx)], dtype=float)
        pok = np.asarray(violin["%s__%spok" % (r, pfx)], dtype=float)
        data = [red[pok < t] for t in thresholds] + [red[pok >= 0.5]]
        counts = [int(d.size) for d in data]
        _violin_ylim(ax, data)
        _violins_skipping_empty_bins(ax, data, ["#4c72b0"] * len(thresholds) + ["#dd8452"], 0.65)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        ax.set_xticks(np.arange(1, len(data) + 1))
        ax.set_xticklabels(["pOk<%g" % t for t in thresholds] + ["pOk≥0.5"], fontsize=8, rotation=45)
        _violin_counts(ax, counts, header=(ax is axes[0]))
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r], pad=_VIOLIN_COUNT_PAD)
        ax.set_ylabel(_REDUCTION_LABEL)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-allele error reduction from LCF correction (%s), by pOk stratum — %s"
                 % (dataset_label, title_tag), fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# Raw predicted-LCF bins (correction factor; corrected = eh/LCF, so LCF=1 = no correction).
# LCF<1 grows the call, LCF>1 shrinks it; bins farther from 1 are larger corrections.
_LCF_BIN_EDGES = (0.5, 0.8, 1.25, 2.0, 3.0, 4.0, 5.0)
_LCF_BIN_LABELS = ("LCF<0.5\nbig grow", "0.5–0.8\ngrow", "0.8–1.25\n~none",
                   "1.25–2\nshrink", "2–3\nshrink", "3–4\nshrink", "4–5\nshrink",
                   "LCF>5\nhuge shrink")
_LCF_BIN_COLORS = ("#3b5b8c", "#6f9bd1", "#bbbbbb", "#f0a868", "#dd8452", "#c44e52",
                   "#9e3036", "#7a1f1f")

# What the alleles in each genotyping regime are (for the violin panel titles).
_REGIME_ALLELE_DESC = {
    features.GENOTYPING_REGIME_QUICK: "Alleles: quick genotype",
    features.GENOTYPING_REGIME_FULL_SPANNING: "Alleles: full_spanning genotype",
    features.GENOTYPING_REGIME_FULL_NONSPANNING: "Alleles: full_nonspanning genotype",
}


def plot_violins_lcf(violin, out_png, pfx="", title_tag="non-homopolymer", dataset_label="held-out HPRC"):
    """Second violin plot: error reduction by pOk(<0.5 / >=0.5) x predicted-LCF bin.

    Grid of regime (rows) x pOk stratum (cols, pOk<0.5 = gate APPLIES the LCF, pOk>=0.5 = gate keeps
    raw EH); within each cell one violin per raw-LCF bin, so the x-axis sweeps correction size and
    direction together (LCF<1 grows the call, ~1 leaves it, >1 shrinks it; farther from 1 = bigger
    correction). Allele count angled above each violin.

    ``pfx`` selects the per-allele arrays: ``""`` -> the non-homopolymer ``{r}__red/pok/lcf`` sample,
    ``"h"`` -> the homopolymer-only ``{r}__hred/hpok/hlcf`` sample. ``title_tag`` labels the suptitle;
    ``dataset_label`` names the dataset.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__%sred" % (r, pfx)) in violin]
    cols = [("p < 0.5", lambda p: p < 0.5), ("p ≥ 0.5", lambda p: p >= 0.5)]
    fig, axes = plt.subplots(len(regs), 2, figsize=(16, 4.2 * len(regs)), sharey="row", squeeze=False)
    for i, r in enumerate(regs):
        red = np.asarray(violin["%s__%sred" % (r, pfx)], dtype=float)
        pok = np.asarray(violin["%s__%spok" % (r, pfx)], dtype=float)
        bin_idx = np.digitize(np.asarray(violin["%s__%slcf" % (r, pfx)], dtype=float), _LCF_BIN_EDGES)
        row_ylim = []
        for j, (ctitle, cmask) in enumerate(cols):
            ax = axes[i][j]
            cm = cmask(pok)
            data = [red[cm & (bin_idx == b)] for b in range(len(_LCF_BIN_LABELS))]
            counts = [int(d.size) for d in data]
            row_ylim += [d for d in data if len(d)]
            _violins_skipping_empty_bins(ax, data, _LCF_BIN_COLORS, 0.65)
            ax.axhline(0, color="k", lw=0.9, ls="--")
            ax.set_xticks(np.arange(1, len(_LCF_BIN_LABELS) + 1))
            ax.set_xticklabels(_LCF_BIN_LABELS, fontsize=7)
            _violin_counts(ax, counts, header=(i == 0 and j == 0))
            ax.set_title("%s (%s)" % (_REGIME_ALLELE_DESC[r], ctitle), fontsize=10, pad=_VIOLIN_COUNT_PAD)
            if j == 0:
                ax.set_ylabel(_REDUCTION_LABEL)
            ax.grid(axis="y", alpha=0.3)
        if row_ylim:
            allv = np.concatenate(row_ylim)
            lo, hi = allv.min(), allv.max(); pad = max(0.5, 0.05 * (hi - lo))
            # cap the long full_nonspanning shrink tail at 750; other regimes use the full data range
            top = min(hi + pad, 750) if r == features.GENOTYPING_REGIME_FULL_NONSPANNING else hi + pad
            for j in range(2):
                axes[i][j].set_ylim(lo - pad, top)
    fig.suptitle("Error reduction by predicted-LCF bin (correction size + direction), by pOk stratum "
                 "— %s (%s)" % (title_tag, dataset_label), fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# Motif-size bins (repeat-unit length in bp): homopolymer..hexamer singly, then 7-24 bp and 25+ bp.
_MOTIF_BIN_LABELS = ("1bp", "2bp", "3bp", "4bp", "5bp", "6bp", "7–24bp", "25+bp")
_MOTIF_BIN_COLORS = ("#3b5b8c", "#4c72b0", "#6f9bd1", "#55a868", "#8cc474", "#dd8452", "#d1693a", "#c44e52")


def _motif_bin(motif):
    """Maps each motif size (bp) to a bin index in ``[0, 8)``; unmatched (NaN/0) -> -1."""
    m = np.asarray(motif, dtype=float)
    idx = np.full(m.shape, -1, dtype=int)
    for k in range(1, 7):
        idx[m == k] = k - 1
    idx[(m >= 7) & (m <= 24)] = 6
    idx[m >= 25] = 7
    return idx


def plot_violins_motif(violin, out_png, dataset_label="held-out HPRC"):
    """Violins of per-allele signed error reduction at the deployment gate (pOk<0.5), by motif size.

    One panel per genotyping regime; within each, one violin per motif-size bin (1..6 bp, 7-24 bp,
    25+ bp). Only ``pOk < 0.5`` alleles -- the stratum where the gate actually applies the LCF. Allele
    count angled above each violin; per-regime y-scale clipped to the 1-99th percentile.
    ``dataset_label`` names the dataset.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__motif" % r) in violin]
    fig, axes = plt.subplots(1, len(regs), figsize=(5.0 * len(regs), 5.8), squeeze=False)
    for ax, r in zip(axes[0], regs):
        red = np.asarray(violin["%s__mred" % r], dtype=float)
        pok = np.asarray(violin["%s__mpok" % r], dtype=float)
        mbin = _motif_bin(violin["%s__motif" % r])
        gate = pok < 0.5
        data = [red[gate & (mbin == b)] for b in range(len(_MOTIF_BIN_LABELS))]
        counts = [int(d.size) for d in data]
        nonempty = [d for d in data if d.size]
        _violins_skipping_empty_bins(ax, data, _MOTIF_BIN_COLORS, 0.65)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        if nonempty:
            lo, hi = np.percentile(np.concatenate(nonempty), [1, 99]); pad = max(0.5, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(np.arange(1, len(data) + 1))
        ax.set_xticklabels(_MOTIF_BIN_LABELS, fontsize=7)
        _violin_counts(ax, counts, header=(ax is axes[0][0]))
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r], pad=_VIOLIN_COUNT_PAD)
        ax.set_ylabel(_REDUCTION_LABEL)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-allele error reduction at the pOk < 0.5 threshold, by motif size (%s)" % dataset_label,
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# Direction-head lean bins for the by-lean violins: pTooLong - pTooShort, from -1 (certain TOO_SHORT)
# through 0 (balanced) to +1 (certain TOO_LONG). 20 bins of width 0.1; colored blue (short) -> red (long).
_PDIFF_BIN_EDGES = np.round(np.linspace(-1.0, 1.0, 21), 2)
_PDIFF_BIN_LABELS = tuple("%.1f:%.1f" % (_PDIFF_BIN_EDGES[i], _PDIFF_BIN_EDGES[i + 1])
                          for i in range(len(_PDIFF_BIN_EDGES) - 1))


def plot_violins_pdiff(violin, out_png, pfx="", title_tag="non-homopolymer", dataset_label="held-out HPRC"):
    """Violins of per-allele signed error reduction, one panel per genotyping regime, binned by the
    direction-head lean ``pTooLong - pTooShort``.

    reduction = |true - eh| - |true - round(eh/LCF)| (>0 = correction moved the call closer to truth).
    Within each panel one violin per 0.1-wide ``pTooLong - pTooShort`` bin from -1.0 (certain TOO_SHORT)
    through 0 (balanced) to +1.0 (certain TOO_LONG); violins colored blue (short) -> red (long).
    Per-regime y-scale; y clipped to the 1-99th percentile.

    ``pfx`` selects the per-allele arrays: ``""`` -> non-homopolymer ``{r}__red`` + ``{r}__pdiff``,
    ``"h"`` -> homopolymer-only ``{r}__hred`` + ``{r}__hpdiff``. ``title_tag`` labels the suptitle;
    ``dataset_label`` names the dataset.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__%spdiff" % (r, pfx)) in violin]
    nb = len(_PDIFF_BIN_LABELS)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nb))
    fig, axes = plt.subplots(1, len(regs), figsize=(6.0 * len(regs), 5.8), squeeze=False)
    for ax, r in zip(axes[0], regs):
        red = np.asarray(violin["%s__%sred" % (r, pfx)], dtype=float)
        pdiff = np.asarray(violin["%s__%spdiff" % (r, pfx)], dtype=float)
        bin_idx = np.clip(np.digitize(pdiff, _PDIFF_BIN_EDGES[1:-1]), 0, nb - 1)
        data = [red[bin_idx == b] for b in range(nb)]
        counts = [int(d.size) for d in data]
        nonempty = [d for d in data if d.size]
        _violins_skipping_empty_bins(ax, data, colors, 0.7)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        if nonempty:
            lo, hi = np.percentile(np.concatenate(nonempty), [1, 99]); pad = max(0.5, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(np.arange(1, nb + 1))
        ax.set_xticklabels(_PDIFF_BIN_LABELS, fontsize=6, rotation=90)
        _violin_counts(ax, counts, header=(ax is axes[0][0]))
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r], pad=_VIOLIN_COUNT_PAD)
        ax.set_ylabel(_REDUCTION_LABEL)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-allele error reduction from LCF correction (%s), by direction lean "
                 "(pTooLong − pTooShort) — %s" % (dataset_label, title_tag), fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# Predicted-LCF strata for the by-LCF-bin violins (corrected = eh/LCF): LCF<1 grows the call, LCF>1
# shrinks it, LCF~1 leaves it ~unchanged; bins farther from 1 are larger corrections. 8 irregular bins;
# colored blue (grow) -> red (shrink). np.digitize on the 7 interior edges maps LCF -> bin 0..7.
_LCF_STRATA_EDGES = (0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0)
_LCF_STRATA_LABELS = ("0–0.2", "0.2–0.25", "0.25–0.5", "0.5–1", "1–2", "2–4", "4–5", ">5")


def plot_violins_lcf_bins(violin, out_png, pfx="", title_tag="non-homopolymer",
                          dataset_label="held-out HPRC"):
    """Violins of per-allele signed error reduction, one panel per genotyping regime, binned by the
    predicted ``LCF`` (correction size + direction).

    reduction = |true - eh| - |true - round(eh/LCF)| (>0 = correction moved the call closer to truth).
    Within each panel one violin per predicted-LCF stratum (0-0.2, 0.2-0.25, 0.25-0.5, 0.5-1, 1-2, 2-4,
    4-5, >5); corrected = round(eh/LCF), so LCF<1 grows the call, LCF>1 shrinks it, LCF~1 (the 0.5-1 / 1-2 bins)
    leaves it ~unchanged. Violins colored blue (grow) -> red (shrink); per-regime y-scale, y clipped to
    the 1-99th percentile.

    ``pfx`` selects the per-allele arrays: ``""`` -> non-homopolymer ``{r}__red`` + ``{r}__lcf``,
    ``"h"`` -> homopolymer-only ``{r}__hred`` + ``{r}__hlcf``. ``title_tag`` labels the suptitle;
    ``dataset_label`` names the dataset.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__%slcf" % (r, pfx)) in violin]
    nb = len(_LCF_STRATA_LABELS)
    colors = plt.cm.coolwarm(np.linspace(0, 1, nb))
    fig, axes = plt.subplots(1, len(regs), figsize=(6.0 * len(regs), 5.8), squeeze=False)
    for ax, r in zip(axes[0], regs):
        red = np.asarray(violin["%s__%sred" % (r, pfx)], dtype=float)
        lcf = np.asarray(violin["%s__%slcf" % (r, pfx)], dtype=float)
        bin_idx = np.clip(np.digitize(lcf, _LCF_STRATA_EDGES), 0, nb - 1)
        data = [red[bin_idx == b] for b in range(nb)]
        counts = [int(d.size) for d in data]
        nonempty = [d for d in data if d.size]
        _violins_skipping_empty_bins(ax, data, colors, 0.7)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        if nonempty:
            lo, hi = np.percentile(np.concatenate(nonempty), [1, 99]); pad = max(0.5, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(np.arange(1, nb + 1))
        ax.set_xticklabels(_LCF_STRATA_LABELS, fontsize=7, rotation=90)
        _violin_counts(ax, counts, header=(ax is axes[0][0]))
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r], pad=_VIOLIN_COUNT_PAD)
        ax.set_ylabel(_REDUCTION_LABEL)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-allele error reduction from LCF correction (%s), by predicted LCF "
                 "(corrected = round(eh/LCF)) — %s" % (dataset_label, title_tag), fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# --- HTML -----------------------------------------------------------------

def _fixed_palette(hex_colors, n_grays=17):
    """Returns a PIL palette image holding exactly ``hex_colors`` plus ``n_grays`` evenly spaced grays.

    For the accuracy-by-size stacked-bar charts, the report's largest image family (one chart per
    dataset x motif x correction x purity x pOk-filter pill combination): quantizing them onto their own
    category colors plus grays for text, axes and edges keeps every category exactly its color, at
    ~23% fewer bytes than the adaptive 256-color palette. A small adaptive palette would be smaller
    still, but median-cut gives scarce category colors no slot of their own and merges them.
    """
    rgb = [tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for h in hex_colors]
    rgb += [(v, v, v) for v in np.linspace(0, 255, n_grays).round().astype(int)]
    palette = Image.new("P", (1, 1))
    palette.putpalette([x for c in rgb for x in c] + [0] * (768 - 3 * len(rgb)))
    return palette


def _img(path, palette=None):
    """Embeds a PNG as a base64 data URI, palette-quantized to keep the standalone HTML small enough
    to commit + serve on GitHub Pages.

    matplotlib charts use few distinct colors, so an adaptive 256-color palette is visually
    near-lossless yet ~2-3x smaller than the truecolor PNG; the on-disk PNG (a gitignored build
    artifact) is left untouched -- only the embedded copy is quantized. ``palette`` (from
    ``_fixed_palette``) maps the image onto a fixed set of colors instead, without dithering.
    """
    rgb = Image.open(path).convert("RGB")
    quantized = (rgb.quantize(palette=palette, dither=Image.Dither.NONE) if palette is not None
                 else rgb.convert("P", palette=Image.ADAPTIVE, colors=256))
    buf = io.BytesIO()
    quantized.save(buf, format="PNG", optimize=True)
    return '<img src="data:image/png;base64,%s" />' % base64.b64encode(buf.getvalue()).decode()


def _img_toggle(excluded_png, homopolymer_png, gid):
    """A 'Homopolymers: [Excluded | Only Homopolymers]' pill toggle showing one plot at a time.

    Pure CSS (radio + sibling selectors; no JS), so it works in a saved standalone file. ``gid`` must be
    unique per page (it names the radio group + element ids). Falls back to the excluded image alone when
    ``homopolymer_png`` is missing (e.g. the homopolymer benchmark has not been run yet), and to an empty
    string when even ``excluded_png`` is missing (e.g. ``--render-text-only`` with no cached PNG yet).
    """
    if not excluded_png:
        return ""
    if not homopolymer_png:
        return _img(excluded_png)
    return (
        "<div class='htgroup'>"
        "<span class='htlabel'>Homopolymers:</span>"
        "<input type='radio' class='htr htr-ex' id='%s-ex' name='%s' checked>"
        "<label class='pillseg' for='%s-ex'>Excluded</label>"
        "<input type='radio' class='htr htr-ho' id='%s-ho' name='%s'>"
        "<label class='pillseg' for='%s-ho'>Only Homopolymers</label>"
        "<div class='htpanel htpanel-ex'>%s</div>"
        "<div class='htpanel htpanel-ho'>%s</div>"
        "</div>"
    ) % (gid, gid, gid, gid, gid, gid, _img(excluded_png), _img(homopolymer_png))


def _feature_glossary(results):
    """Feature definition table: EVERY current feature, ordered by full_nonspanning importance rank.

    Driven by ``features.feature_names`` rather than by the cached ranking. Enumerating the ranking
    instead silently dropped any feature added after ``results.json`` was written -- the published
    report shipped a 24-row table while the contract already had 29 features, with no warning.
    Features the cached ranking does not cover sort last and show ``#-`` instead of a rank.
    """
    ranked = {r["genotyping_regime"]: r for r in
              results}[features.GENOTYPING_REGIME_FULL_NONSPANNING]["importance"]
    rank = {feat: i for i, (feat, _, _) in enumerate(ranked, start=1)}
    branch = features.GENOTYPING_REGIME_BRANCH[features.GENOTYPING_REGIME_FULL_NONSPANNING]
    names = sorted(features.feature_names(branch), key=lambda f: rank.get(f, len(rank) + 1))
    rows = ["<tr><th>rank</th><th>feature</th><th>definition</th></tr>"]
    for feat in names:
        rows.append("<tr><td>%s</td><td><code>%s</code></td><td>%s</td></tr>" % (
            "#%d" % rank[feat] if feat in rank else "#-", html.escape(feat),
            html.escape(features.FEATURE_DEFINITIONS.get(feat, ""))))
    return "<table>%s</table>" % "".join(rows)


def _eh_output_glossary():
    """Table of the raw ExpansionHunter output JSON fields the parser consumes (eh_json.py).

    The field column is HTML-escaped (it contains ``<locus>``/``<v>`` placeholders); the definition
    column is emitted verbatim because it carries intentional ``<code>`` markup naming the features.
    """
    rows = ["<tr><th>ExpansionHunter output field</th><th>definition</th></tr>"]
    for field, definition in eh_json.EH_OUTPUT_FIELDS:
        rows.append("<tr><td><code>%s</code></td><td>%s</td></tr>" % (html.escape(field), definition))
    return "<table>%s</table>" % "".join(rows)


# The per-allele quantities each regime model predicts: (output, head, meaning, range, use).
# The tolerance tiers below restate features.tol_repeats, which defines OK / TOO_LONG / TOO_SHORT.
_MODEL_OUTPUTS = (
    ("pOk", "direction predictor (3-class classifier + per-class isotonic calibration)",
     "probability that the EH call is correct, i.e. within a tolerance of the true allele size that "
     "depends on the true allele's length: exact (0 repeats) under 50 bp, 1 repeat for 50&ndash;120 bp, "
     "2 for 121&ndash;269 bp, 4 for 270&ndash;599 bp, 8 at 600 bp or more",
     "0&ndash;1"),
    ("pTooLong", "direction predictor (3-class classifier + per-class isotonic calibration)",
     "probability that EH called the allele longer than its true size by more than the pOk tolerance",
     "0&ndash;1"),
    ("pTooShort", "direction predictor (3-class classifier + per-class isotonic calibration)",
     "probability that EH called the allele shorter than its true size by more than the pOk tolerance",
     "0&ndash;1"),
    ("LCF", "length predictor (regression)",
     "<span style='white-space:nowrap'>predicted length-correction factor = "
     "(EH called allele size)/(true allele size)</span><br>"
     "Predicts that the true allele size = <code>round(eh / LCF)</code>",
     "1 = no correction; &gt;1 predicts the EH allele size is too long,<br>&lt;1 predicts the EH allele "
     "size is too short; (always &gt; 0)"),
)


def _model_outputs_table():
    """Table of the per-allele quantities the model predicts (length head + 3-class direction head)."""
    rows = ["<tr><th>output</th><th>meaning</th><th>range</th><th>predictor</th></tr>"]
    for name, head, meaning, rng in _MODEL_OUTPUTS:
        rows.append("<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (name, meaning, rng, head))
    return "<table>%s</table>" % "".join(rows)


def _holdout_results(holdout):
    """Builds a plot_mae-compatible results list from the 43-sample benchmark JSON."""
    gr = holdout["genotyping_regimes"]
    return [{"genotyping_regime": r,
             "gated": {"mae_raw": gr[r]["mae_raw"], "mae_gated": gr[r]["mae_gated"]}}
            for r in features.GENOTYPING_REGIMES if gr[r].get("n")]


def _holdout_homopolymer_results(holdout):
    """plot_mae-compatible list from the homopolymer-only sub-summary of the 43-sample benchmark."""
    gr = holdout["genotyping_regimes"]
    return [{"genotyping_regime": r,
             "gated": {"mae_raw": gr[r]["homopolymer"]["mae_raw"],
                       "mae_gated": gr[r]["homopolymer"]["mae_gated"]}}
            for r in features.GENOTYPING_REGIMES if gr[r].get("homopolymer", {}).get("n")]


def _holdout_table(holdout):
    rows = ["<tr><th>allele size bucket</th><th>n</th><th>raw EH MAE</th><th>gated MAE</th>"
            "<th>dist. reduction</th><th>median |err|: EH&rarr;gated</th><th>exact: EH&rarr;gated</th>"
            "<th>3-class direction argmax acc.</th></tr>"]
    for r in features.GENOTYPING_REGIMES:
        m = holdout["genotyping_regimes"][r]
        if not m.get("n"):
            continue
        rows.append("<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td><td>%+.1f%%</td>"
                    "<td>%.3f &rarr; %.3f</td><td>%.3f &rarr; %.3f</td><td>%.3f</td></tr>" % (
                        features.GENOTYPING_REGIME_DISPLAY[r], m["n"], m["mae_raw"], m["mae_gated"],
                        100 * m["dist_reduction"], m["median_raw"], m["median_gated"],
                        m["exact_eh"], m["exact_gated"], m["p_ok_accuracy"]))
    return "<table>%s</table>" % "".join(rows)


# Datasets the per-plot Dataset pill switches between (key -> display label).
# The held-out panel comes first (the default pill): it is the only dataset absent from training. Its
# label gets the sample count of the artifact actually loaded (build_dataset_sections), not of
# heldout.SAMPLES, since a partial build scores fewer samples.
DATASETS = [("heldout_hprc", "held-out HPRC"),
            ("hg002_genome", "HG002 genome (31x, a training sample)"),
            # ("hg002_exome", "HG002 exome (3x)"),  # disabled: input parquet unavailable, can't refresh
            ]


def _pills(gid, dims, contents):
    """Pure-CSS N-pill switcher: one labeled radio group per dim, showing the single content whose
    combo matches all checked radios. Works in a saved standalone file (no JS).

    Args:
        gid: Unique id prefix for this switcher.
        dims: List of ``(dim_key, label, [(val_key, val_label), ...])``; the first value of each dim
            is the default selection.
        contents: ``{(val_key per dim, in dim order): html_string}`` -- the content shown for each combo.
    """
    import html
    import itertools

    def rid(dk, vk):
        return "%s-%s-%s" % (gid, dk, vk)

    out = ["<div class='pillbox' id='%s'>" % gid]
    for di, (dk, label, vals) in enumerate(dims):
        if di:
            out.append("<br>")
        out.append("<span class='pilldim'>%s</span>" % html.escape(label))
        for vi, (vk, vlabel) in enumerate(vals):
            out.append("<input type='radio' class='pr' id='%s' name='%s-%s'%s>"
                       % (rid(dk, vk), gid, dk, " checked" if vi == 0 else ""))
            out.append("<label class='pillseg' for='%s'>%s</label>"
                       % (rid(dk, vk), html.escape(vlabel)))
    out.append("<div class='pillimgs'>")
    css = []
    dimkeys = [dk for dk, _, _ in dims]
    for combo in itertools.product(*[[vk for vk, _ in vals] for _, _, vals in dims]):
        if combo not in contents:
            continue
        cls = "c-" + "-".join(combo)
        out.append("<span class='pc %s'>%s</span>" % (cls, contents[combo]))
        chain = " ~ ".join("#%s:checked" % rid(dimkeys[i], combo[i]) for i in range(len(dims)))
        css.append("%s ~ .pillimgs .%s{display:block}" % (chain, cls))
    out.append("</div></div><style>%s</style>" % "".join(css))
    return "".join(out)


def _ds_dim_for(contents, present):
    """Dataset pill dimension restricted to the datasets actually populated in ``contents`` (a dataset
    missing this section's artifact would otherwise render a blank panel).

    Args:
        contents: ``{(val_key per dim, in dim order): html_string}`` -- the section's own pill contents.
        present: ``[(key, label), ...]`` datasets with at least one artifact loaded this report run.
    """
    keys = {combo[0] for combo in contents}
    return ("ds", "Dataset", [(k, l) for k, l in present if k in keys])


def _stacked_has_every_pill_option(sd):
    """True if a loaded ``stacked_<key>.json`` has counts for every accuracy-by-size pill combination."""
    return all(kk in sd[h].get(vk, {}).get(pk, {}) for h in ("nonhomo", "homo")
               for vk, _, _, _ in ABS.CORRECTION_VARIANTS for pk, _, _ in ABS.PURITY_VARIANTS
               for kk, _, _ in ABS.POK_VARIANTS)


def build_dataset_sections(out_dir, model_path="", regenerate=True):
    """Renders the per-dataset evaluation + accuracy-by-size plots and returns the report HTML.

    When ``regenerate`` is False (``report.py --render-text-only``) the PNGs are not re-plotted; the
    existing on-disk PNGs are embedded as-is and any that are missing are skipped.

    Every plot carries a <b>Dataset</b> pill (the held-out HPRC samples, counted from the loaded
    artifact, / HG002 genome 31x) and the <b>Homopolymers</b> pill; the accuracy-by-size stacked-bar additionally gets an
    <b>LCF correction</b> on/off pill. Reads ``eval_<key>.json`` + ``eval_<key>_violin.npz`` (the
    apply-based eval) and ``stacked_<key>.json`` (the accuracy-by-size counts) for each dataset.
    """
    evals, violins, stacked = {}, {}, {}
    for key, _ in DATASETS:
        if os.path.exists(os.path.join(out_dir, "eval_%s.json" % key)):
            with open(os.path.join(out_dir, "eval_%s.json" % key)) as f:
                evals[key] = json.load(f)
        if os.path.exists(os.path.join(out_dir, "eval_%s_violin.npz" % key)):
            violins[key] = dict(np.load(os.path.join(out_dir, "eval_%s_violin.npz" % key)))
        if os.path.exists(os.path.join(out_dir, "stacked_%s.json" % key)):
            with open(os.path.join(out_dir, "stacked_%s.json" % key)) as f:
                stacked[key] = json.load(f)
    # Artifacts record the model that produced them (heldout.run_eval / gen_datasets.gen_stacked).
    # Drop any that came from a different model than the one this report describes, rather than
    # showing an older model's numbers under this model's name.
    stale = []
    if model_path:
        # The content fingerprint (model.fingerprint), not the file name: names are dated by day, so a
        # same-day retrain reuses the name.
        want = M.fingerprint(model_path)
        for key, _ in DATASETS:
            for store in (evals, stacked):
                if key in store and store[key].get("model") != want:
                    stale.append("%s (made by a different model: %s)"
                                 % (key, store.pop(key).get("model") or "not recorded"))
            if key not in evals:
                violins.pop(key, None)  # the violin npz is written alongside its eval JSON
    # A stacked JSON written before a pill option existed (e.g. a pOk threshold added later) has no counts
    # for it, so that button would show an empty panel; skip the JSON like a different model's.
    for key in [k for k, sd in stacked.items() if not _stacked_has_every_pill_option(sd)]:
        stale.append("%s (made before the current accuracy-by-size pill options)" % key)
        stacked.pop(key)
    for s in sorted(set(stale)):
        print("  WARNING: skipping out-of-date dataset artifact: %s" % s, flush=True)

    def label_of(key, label):
        n = (evals.get(key) or {}).get("n_samples")
        return "%d %s" % (n, label) if key == "heldout_hprc" and n else label
    present = [(k, label_of(k, l)) for k, l in DATASETS if k in evals or k in stacked]
    if not present:
        return ""
    homo_dim = ("homo", "Homopolymers", [("nh", "Non-homopolymer"), ("ho", "Homopolymer")])

    def P(name):
        return os.path.join(out_dir, name)

    import gen_datasets  # imported here, like _maybe_regenerate_dataset_artifacts does
    # The apply-based eval scores a seeded sample of at most this many alleles per sample (heldout.run_eval).
    eval_caps = sorted({e.get("max_alleles_per_sample") for e in evals.values()} - {None})
    parts = ["<h2>Accuracy by dataset</h2>",
             "<p class='note'>The exported model is applied unchanged (no fitting) to each dataset, "
             "scored against truth. Use the <b>Dataset</b> pill on every plot below to switch between "
             "datasets: %s. Only the held-out HPRC samples are absent from training (a further %d HPRC "
             "samples were promoted into the training pool for ancestry/sex diversity); HG002 is a "
             "training sample, so its numbers are in-sample and show fit, not generalization. %d of the "
             "held-out HPRC samples (%s) were never trained on but were used during model development to "
             "diagnose defects and compare candidate features, and all of the held-out samples were used "
             "to compare candidate models and training recipes (for example, how calibration data is held "
             "out), so none of them is entirely untouched by development.%s%s</p>"
             % (", ".join("<b>%s</b>" % l for _, l in present), len(dataset.PROMOTED_HELDOUT_SAMPLES),
                len(heldout.FASTPATH_DIAGNOSIS_SAMPLES), ", ".join(heldout.FASTPATH_DIAGNOSIS_SAMPLES),
                "" if not eval_caps else
                " The accuracy table and the MAE, help-vs-hurt and error-reduction charts score a seeded "
                "random sample of at most %s alleles per sample (homopolymers included, split out "
                "afterwards), and the error-reduction violins keep an evenly spaced subset of at most %s "
                "of those per sample and allele size bucket, so their counts can be smaller than the "
                "table's; the accuracy-by-size charts use a separate, larger seeded sample of whole loci "
                "holding at most %s alleles per sample that ExpansionHunter scores (one per homozygous "
                "call; the cap report.py regenerates them with), and count both copies of each "
                "homozygous call plus the no-calls of those loci, so they show more alleles than that "
                "and their allele counts are not comparable."
                % (" / ".join(format(c, ",") for c in eval_caps), format(heldout.VIOLIN_PER_SAMPLE, ","),
                   format(gen_datasets.CORRECTED_CAP_DEFAULT, ",")),
                "" if not stale else " Skipped because they are out of date: %s."
                % html.escape(", ".join(sorted(set(stale)))))]

    # --- accuracy by true allele size (str-truth-set-v2 stacked-bar replica) ---
    abs_imgs = {}
    abs_palette = _fixed_palette(ABS.CATEGORY_COLORS.values())
    for key, _ in present:
        if key not in stacked:
            continue
        sd = stacked[key]
        for hk, hcol, hdesc0 in (("nh", "nonhomo", "non-homopolymer, motif > 1 bp"),
                                 ("ho", "homo", "homopolymer, 1 bp motif")):
            for vkey, _, vnote, _ in ABS.CORRECTION_VARIANTS:
                for pkey, _, pmin in ABS.PURITY_VARIANTS:
                    for kkey, klabel, _ in ABS.POK_VARIANTS:
                        data = sd[hcol][vkey][pkey][kkey]  # present: _stacked_has_every_pill_option
                        hdesc = (hdesc0 + ("" if pmin is None else ", repeat purity > %g" % pmin)
                                 + ("" if kkey == "all" else ", %s" % klabel))
                        png = P("ds_abs_%s_%s_%s_%s_%s.png" % (key, hk, vkey, pkey, kkey))
                        try:
                            if regenerate:
                                # Append the per-dataset no-call caveat (set only for datasets whose
                                # parquet lacks no-call rows, e.g. the legacy exome one) to the title.
                                ABS.plot_accuracy_by_size(data, png, sd["tool_label"],
                                                          sd["coverage_label"] + sd.get("no_call_note", ""),
                                                          hdesc, vnote)
                            abs_imgs[(key, hk, vkey, pkey, kkey)] = _img(png, palette=abs_palette)
                        except Exception as e:
                            print("  [%s] accuracy-by-size %s/%s/%s/%s skipped: %s"
                                  % (key, hk, vkey, pkey, kkey, e), flush=True)
    # The caption names each LCF correction option by its own pill label, so the two cannot drift apart.
    correction_notes = {"raw": "no correction",
                        "p050": "the LCF applied to alleles with <code>pOk&lt;0.5</code>",
                        "p025": "a stricter <code>pOk&lt;0.25</code> threshold",
                        "p050ns": "the <code>pOk&lt;0.5</code> threshold restricted to full_nonspanning-bucket "
                                  "alleles"}
    correction_options = ", ".join(
        "<b>%s</b>%s" % (html.escape(label), " (%s)" % correction_notes[k] if k in correction_notes else "")
        for k, label, _, _ in ABS.CORRECTION_VARIANTS)
    if abs_imgs:
        parts += [
            "<h3>Accuracy by true allele size (per-allele call vs truth)</h3>",
            _pills("absp", [_ds_dim_for(abs_imgs, present), homo_dim,
                            ("lcf", "LCF correction",
                             [(k, lbl) for k, lbl, _, _ in ABS.CORRECTION_VARIANTS]),
                            ("pur", "Repeat Purity Filter",
                             [(k, lbl) for k, lbl, _ in ABS.PURITY_VARIANTS]),
                            ("pok", "pOk Filter",
                             [(k, lbl) for k, lbl, _ in ABS.POK_VARIANTS])],
                   abs_imgs),
            "<p class='note'>Each allele is colored by how the "
            "ExpansionHunter call compares to its true size (<b>Same</b> = within &plusmn;1 repeat, "
            "widened to &plusmn;2 / &plusmn;3 / &plusmn;4 for calls longer than 120 / 240 / 360 repeats; "
            "blue = under-call, orange = over-call, cyan = wrong direction, plus "
            "No&nbsp;Call / Hom&nbsp;Ref / Het&nbsp;Ref); the x-axis is true allele size minus the "
            "reference. Left panel = allele counts, right panel = fractions. <b>Same</b> is not the "
            "tolerance behind <code>pOk</code> (see the model outputs table): for true alleles under 50 bp "
            "the model counts only an exact call as correct, so a call one repeat off can be <b>Same</b> "
            "here yet TOO_LONG or TOO_SHORT for the model. The <b>LCF correction</b> "
            "pill replaces each gated call with <code>round(eh/LCF)</code>, growing the green "
            "<b>Same</b> band where it helps: " + correction_options + ". The <b>Repeat Purity Filter</b> pill "
            "(<b>Off</b> vs <b>&gt; 0.95 pure</b>) restricts the plot to alleles whose truth repeat "
            "purity exceeds 0.95 (purity is on a 0&ndash;1 scale), i.e. near-perfect tandem repeats "
            "without interruptions. The <b>pOk Filter</b> pill keeps only alleles whose own predicted "
            "<code>pOk</code> is below 0.2, 0.3, 0.4 or 0.5, or at least 0.5, independent of the "
            "correction selected, so the lower thresholds show how the calls look as the gate tightens. "
            "No-call alleles have no <code>pOk</code>, so every option other than <b>All</b> leaves them "
            "out (the No Call band is empty there), and <b>pOk &lt; 0.5</b> plus <b>pOk &ge; 0.5</b> add up "
            "to <b>All</b> minus the no-calls. "
            "(Every dataset is categorized from its per-allele parquet; the raw "
            "and corrected bands both cover the same capped, whole-locus sample per parquet, so with the "
            "pOk Filter at All, switching the correction changes only the calls. With a pOk threshold, "
            "each corrected call keeps its own <code>pOk</code> after the calls are re-paired with the true "
            "alleles by size, so which true alleles fall below the threshold can shift slightly between "
            "corrections. A dataset whose parquet lacks "
            "no-call rows notes so in its panel title.)</p>"]

    # --- per-dataset held-out accuracy table ---
    etab = {(k,): _holdout_table(evals[k]) for k, _ in present if k in evals}
    parts += ["<h3>Accuracy table, non-homopolymer loci (raw EH vs pOk&lt;0.5-gated LCF)</h3>",
              _pills("etab", [_ds_dim_for(etab, present)], etab),
              "<p class='note'>Homopolymer (1 bp motif) alleles are left out of this table; the charts below "
              "show them under their <b>Homopolymers</b> pill.</p>"]

    def homo_family(gid, h3, prefix, render, desc, datasets_with):
        imgs = {}
        for key, label in present:
            if key not in datasets_with:
                continue
            pngs = {}
            for hk, pfx, tag in (("nh", "", "non-homopolymer"), ("ho", "h", "homopolymer (1 bp motif)")):
                png = P("ds_%s_%s_%s.png" % (prefix, key, hk))
                try:
                    if regenerate:
                        render(key, png, pfx, tag, label)
                    if os.path.exists(png):
                        pngs[hk] = png
                except Exception as e:
                    print("  [%s] %s/%s skipped: %s" % (key, prefix, hk, e), flush=True)
            if "nh" in pngs:
                imgs[(key, "nh")] = _img(pngs["nh"])
                imgs[(key, "ho")] = _img(pngs.get("ho", pngs["nh"]))
        return ["<h3>%s</h3>" % h3, _pills(gid, [_ds_dim_for(imgs, present), homo_dim], imgs), desc] if imgs else []

    parts += homo_family(
        "dmae", "Mean absolute error: raw EH vs gated-LCF", "mae",
        lambda k, png, pfx, tag, label: plot_mae(
            _holdout_homopolymer_results(evals[k]) if pfx == "h" else _holdout_results(evals[k]), png,
            title_tag="%s, %s" % (label, tag)),
        "<p class='note'>Raw-EH MAE vs the MAE after applying the LCF only where <code>pOk&lt;0.5</code>, "
        "per allele size bucket (repeat units).</p>", [k for k, _ in present if k in evals])
    parts += homo_family(
        "dhh", "Allele calls the LCF would help vs hurt, by pOk stratum", "hh",
        lambda k, png, pfx, tag, label: plot_helped_hurt(evals[k], png, homopolymer=(pfx == "h"),
                                                         title_tag=tag, dataset_label=label),
        "<p class='note'>Would the LCF move each call <b>closer</b> (green) or <b>further</b> (red) from "
        "truth? Left = <code>pOk&lt;0.5</code> (threshold applies the LCF); right = <code>pOk&ge;0.5</code> "
        "(threshold keeps raw EH).</p>", [k for k, _ in present if k in evals])
    parts += homo_family(
        "dpokv", "Per-allele error reduction (signed), by pOk stratum", "pokv",
        lambda k, png, pfx, tag, label: plot_violins(violins[k], png, pfx=pfx, title_tag=tag,
                                                      dataset_label=label),
        "<p class='note'>Signed error reduction <code>|true&minus;eh| &minus; |true&minus;round(eh/LCF)|</code> "
        "(above 0 = closer to truth) as the pOk threshold tightens.</p>",
        [k for k, _ in present if k in violins])
    parts += homo_family(
        "dlcfv", "Error reduction by predicted-LCF bin x pOk stratum", "lcfv",
        lambda k, png, pfx, tag, label: plot_violins_lcf(violins[k], png, pfx=pfx, title_tag=tag,
                                                          dataset_label=label),
        "<p class='note'>Signed error reduction split by pOk stratum (columns) and predicted-LCF bin.</p>",
        [k for k, _ in present if k in violins])
    parts += homo_family(
        "dpdv", "Per-allele error reduction by direction lean (pTooLong &minus; pTooShort)", "pdv",
        lambda k, png, pfx, tag, label: plot_violins_pdiff(violins[k], png, pfx=pfx, title_tag=tag,
                                                            dataset_label=label),
        "<p class='note'>Signed error reduction binned by the direction predictor's lean "
        "<code>pTooLong &minus; pTooShort</code> (&minus;1 = TOO_SHORT .. +1 = TOO_LONG).</p>",
        [k for k, _ in present if k in violins])
    parts += homo_family(
        "dlcfb", "Per-allele error reduction by predicted LCF (correction size)", "lcfb",
        lambda k, png, pfx, tag, label: plot_violins_lcf_bins(violins[k], png, pfx=pfx, title_tag=tag,
                                                               dataset_label=label),
        "<p class='note'>Signed error reduction binned by the predicted LCF "
        "(0&ndash;0.2 .. &gt;5; corrected = <code>round(eh/LCF)</code>).</p>",
        [k for k, _ in present if k in violins])

    motif_imgs = {}
    for key, label in present:
        if key in violins:
            png = P("ds_motif_%s.png" % key)
            try:
                if regenerate:
                    plot_violins_motif(violins[key], png, dataset_label=label)
                motif_imgs[(key,)] = _img(png)
            except Exception as e:
                print("  [%s] motif violins skipped: %s" % (key, e), flush=True)
    if motif_imgs:
        parts += ["<h3>Error reduction at the pOk &lt; 0.5 threshold, by motif size</h3>",
                  _pills("dmotif", [_ds_dim_for(motif_imgs, present)], motif_imgs),
                  "<p class='note'>Signed error reduction for <code>pOk&lt;0.5</code> alleles, by motif "
                  "size (1bp .. 25+bp).</p>"]
    return "".join(parts)


def report_html_name(model_path):
    """Returns the report's file name, suffixed with the model it describes so that one model's report
    never overwrites another's: ``..._plus50.20261008.json.gz`` -> ``model_report.2026-10-08_model.html``.

    A model file name that does not end in an 8-digit date (e.g. ``....20261006.quick500.json.gz``) is
    used whole, minus ``.json.gz``, as the suffix. Without a model the name is ``model_report.html``.
    """
    if not model_path:
        return "model_report.html"
    stem = os.path.basename(model_path)
    for extension in (".gz", ".json"):
        if stem.endswith(extension):
            stem = stem[:-len(extension)]
    date = stem.rsplit(".", 1)[-1]
    if len(date) == 8 and date.isdigit():
        return "model_report.%s-%s-%s_model.html" % (date[:4], date[4:6], date[6:])
    return "model_report.%s_model.html" % stem


def render_html(results, mae_png, importance_png, ablation_png, model_path, out_html,
                mae_homopolymer_png=None, ablation_homopolymer_png=None, importance_homopolymer_png=None,
                pr_png=None, roc_png=None, confusion_png=None, prob_violins_png=None,
                pr_homopolymer_png=None, roc_homopolymer_png=None,
                confusion_homopolymer_png=None, prob_violins_homopolymer_png=None,
                dir_importance_png=None, dir_importance_homopolymer_png=None,
                dir_ablation_png=None, dir_ablation_homopolymer_png=None,
                stale_contract_html="", ablation_axis="single", dataset_sections_html=""):
    """Writes the standalone HTML report embedding every plot + metric table."""
    css = ("body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;"
           "color:#222;max-width:1180px}h1{font-size:22px}h2{font-size:17px;margin-top:34px;"
           "border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;"
           "margin:10px 0;font-size:13px}th,td{border:1px solid #ccc;padding:5px 9px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}th{background:#f4f4f4}"
           "img{max-width:100%;height:auto}.note{color:#555;font-size:14px}code{background:#f4f4f4;"
           "padding:1px 4px;border-radius:3px}"
           # Pure-CSS pill toggle: 'Homopolymers: [Excluded | Only Homopolymers]', one plot shown at a time.
           ".htgroup{margin:16px 0}.htr{position:absolute;left:-9999px}"
           ".htlabel{font-size:14px;color:#555;font-weight:600;margin-right:8px;vertical-align:middle}"
           ".pillseg{display:inline-block;padding:4px 13px;border:1px solid #bbb;border-left:none;cursor:pointer;"
           "font-size:13px;color:#444;background:#fff;vertical-align:middle;user-select:none}"
           ".htr-ex+label.pillseg{border-radius:14px 0 0 14px;border-left:1px solid #bbb}"
           ".htr-ho+label.pillseg{border-radius:0 14px 14px 0}"
           ".htr:checked+label.pillseg{background:#1457b8;color:#fff;border-color:#1457b8}"
           ".htpanel{margin-top:8px}.htpanel-ho{display:none}"
           ".htr-ho:checked~.htpanel-ho{display:block}.htr-ho:checked~.htpanel-ex{display:none}"
           # Generic N-pill switcher (Dataset / Homopolymers / LCF correction).
           ".pillbox{margin:18px 0}.pilldim{display:inline-block;font-size:13px;color:#555;"
           "font-weight:600;margin:6px 10px 4px 0}.pr{position:absolute;left:-9999px}"
           ".pr+label.pillseg{border-radius:13px;border-left:1px solid #bbb;margin:0 4px 4px 0}"
           ".pr:checked+label.pillseg{background:#1457b8;color:#fff;border-color:#1457b8}"
           ".pillimgs .pc{display:none}")
    pool = " + ".join("%s %s" % (s, c) for s, c in (
        ("HG002", "(10x, 20x, and 31x coverage)"),
        ("CHM1_CHM13", "(46x coverage)"),
        ("%d HPRC samples" % len(dataset.PROMOTED_HELDOUT_SAMPLES),
         "(promoted from the held-out panel for ancestry/sex diversity)")))
    # Only promise the per-plot Excluded/Only-Homopolymers toggle when homopolymer panels were
    # actually generated (i.e. report.py --homopolymer-cv was run); otherwise _img_toggle falls back
    # to a single plot with no toggle, so the text must not claim one.
    homo_shown = any((mae_homopolymer_png, importance_homopolymer_png, ablation_homopolymer_png,
                      confusion_homopolymer_png, pr_homopolymer_png,
                      roc_homopolymer_png, prob_violins_homopolymer_png,
                      dir_importance_homopolymer_png, dir_ablation_homopolymer_png))
    intro_homo = (
        ""
        if homo_shown else
        "<p class='note'><b>Homopolymer (1&nbsp;bp motif) loci are excluded</b> from the plots below; only "
        "non-homopolymer (motif &gt; 1&nbsp;bp) loci are shown. Run <code>report.py --homopolymer-cv</code> "
        "first to add the per-plot <b>Homopolymers: [Excluded | Only Homopolymers]</b> toggle. Homopolymers "
        "still appear in the by-motif-size violins (<code>1bp</code> bin).</p>")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Genotype-quality model report</title>",
        "<style>%s</style></head><body>" % css,
        "<h1>ExpansionHunter genotype-quality model</h1>",
        intro_homo,
        "<p class='note'>Model file: <code>%s</code></p>" % html.escape(os.path.basename(model_path)),
        "<h2>Model Overview</h2>",
        "<p class='note'>The model scores each ExpansionHunter (EH) allele call vs the probable true "
        "allele size. Each called allele gets a set of 4 scores (a homozygous or hemizygous call gets one "
        "set, shared by both copies):</p>",
        _model_outputs_table(),
        "<p class='note'>ExpansionHunter genotyped allele sizes are split into separate buckets, each with "
        "its own predictor. In <code>--analysis-mode optimized-streaming</code>, the <b>quick</b> bucket "
        "holds alleles that could be confidently and quickly genotyped using only spanning reads, without "
        "running the full, computationally-expensive ExpansionHunter genotyping algorithm; alleles at the "
        "other loci are genotyped by the full algorithm and split into the <b>full_spanning</b> bucket "
        "(&ge;1 spanning read at the called size) and the <b>full_nonspanning</b> bucket (no spanning read "
        "at the called size; the locus may still have spanning reads supporting other sizes). All training "
        "and evaluation data come from optimized-streaming runs, so the full_spanning and full_nonspanning "
        "predictors were trained and scored only on the loci the quick path left to the full algorithm; "
        "other analysis modes, which send every locus through the full algorithm, give them a different "
        "population of loci than they were trained on.</p>",
        "<p class='note'><b>Training data:</b> %s &mdash; Illumina whole-genome sequencing, genotyped by "
        "ExpansionHunter-bw2 with <code>--analysis-mode optimized-streaming</code>.</p>" % html.escape(pool),
        "<p class='note'><b>Truth set:</b> <code>%s</code> (TRExplorer v2.1 catalog). Per-allele true "
        "repeat counts (and repeat purity) are derived from haplotype-resolved long-read genome "
        "assemblies of the same samples (HG002, CHM1&ndash;CHM13 and the HPRC samples) aligned with "
        "DipCall, giving the true repeat number at each tandem-repeat locus; a repeat insertion that "
        "DipCall placed up to one motif length before a locus is counted in that locus. Training and "
        "every evaluation below use only loci inside each sample's DipCall high-confidence regions (from "
        "<a href='https://github.com/broadinstitute/str-truth-set-v2'><code>str-truth-set-v2</code></a>) "
        "where at least one true allele differs from the reference, so "
        "loci where the sample is homozygous for the reference allele are left out. Truth repeat purity "
        "is not used as a training filter, but is available below as an opt-in <b>Repeat Purity "
        "Filter</b> stratification pill.</p>" % html.escape(os.path.basename(dataset.TRUTH_GENOTYPES_ROOT)),
        "<p class='note'>Held-out accuracy is measured by 5-fold cross validation: "
        "each fold trains on 17-18 chromosomes with the exported model's fitting steps (quota "
        "sampling, gated length-correction fit, the same fixed number of boosting iterations per "
        "regime), calibrates on 2 other chromosomes, and tests on the remaining 4-5, so it measures "
        "accuracy on loci the model never saw. It approximates rather than reproduces the exported "
        "model, which trains one model per regime on homopolymer and non-homopolymer loci together "
        "(the cross-validation fits the two panels separately) and fits its isotonic calibration on "
        "%d of its training HPRC samples, kept out of tree fitting, plus one autosome each of %s "
        "(the cross-validation calibrates on chromosomes). Each cross-validation head is also fit on "
        "fewer rows: at most <code>report.py --cv-train-cap</code> rows (400,000 in "
        "<code>train_model.sh</code>, with a fifth as many calibration rows), where the exported heads "
        "use up to <code>train.py --train-cap</code> rows (1,000,000), with the same iteration counts. "
        "None of the held-out HPRC samples below is used.</p>"
        % (train.N_CALIB_INDIVIDUALS, " and ".join(train.ALWAYS_TRAIN_INDIVIDUALS)),
        "<h2>ExpansionHunter output fields used for model training</h2>",
        "<p class='note'>The fields of each ExpansionHunter output JSON that the model features are "
        "computed from: per-locus, per-variant and per-allele "
        "(<code>AlleleQualityMetrics.Alleles[]</code>).</p>",
        _eh_output_glossary(),
        "<h2>Feature definitions (derived from ExpansionHunter output fields)</h2>",
        "<p class='note'>The model features, with their <code>full_nonspanning</code> importance rank for "
        "the LCF (length) prediction (<code>#1</code> = most important there; the direction prediction "
        "that produces <code>pOk</code> ranks them differently, see its importance panel below). These "
        "are derived (and two engineered: "
        "<code>ci_asymmetry</code>, <code>ci_over_eh</code>) from the ExpansionHunter output fields "
        "above.</p>",
        stale_contract_html,
        _feature_glossary(results),
        "<h2>Mean absolute error: raw EH allele size vs allele size after LCF correction</h2>",
        _img_toggle(mae_png, mae_homopolymer_png, "mae"),
        "<p class='note'>The length-correction factor "
        "<code>LCF = (ExpansionHunter called allele size)/(True allele size)</code> is "
        "applied only where <code>pOk &lt; 0.5</code>; calls with <code>pOk &ge; 0.5</code> keep the raw "
        "EH allele size. "
        "MAE is computed over held-out alleles, and is measured in repeat units. The pOk &lt; 0.5 threshold "
        "fires on few <code>quick</code> alleles, so the correction is concentrated on the two "
        "full-genotyper buckets (most <code>full_spanning</code> and <code>full_nonspanning</code> alleles "
        "in the default panel; among homopolymers, nearly all <code>full_nonspanning</code> alleles but "
        "under a quarter of <code>full_spanning</code> ones).</p>",
        "<h3>Confusion matrix (argmax prediction, row-normalized)</h3>",
        _img_toggle(confusion_png, confusion_homopolymer_png, "cm"),
        "<p class='note'>Rows are the <b>true</b> direction, columns the <b>argmax</b>-predicted "
        "direction; each cell shows the allele count and its row-percent, and the diagonal is "
        "per-class <b>recall</b>. Row-normalizing (rather than showing raw counts) keeps every true "
        "direction readable however many alleles it has (the OK / TOO_LONG / TOO_SHORT mix differs a lot "
        "between buckets; the PR-ROC legend below gives each class's share), and exposes the asymmetry "
        "in <i>where</i> the predictor misroutes calls. It is the hard-decision (argmax) view of the "
        "calibrated probabilities.</p>",
        ("<h3>ROC curves: detecting TOO_LONG / TOO_SHORT calls</h3>" if roc_png else ""),
        (_img_toggle(roc_png, roc_homopolymer_png, "roc") if roc_png else ""),
        ("<p class='note'>One-vs-rest ROC for the two error directions the pOk &lt; 0.5 threshold must "
         "catch: true "
         "positive rate vs false positive rate as the probability threshold sweeps, allele size buckets "
         "overlaid. The legend gives each curve's <code>AUC</code>; the dashed diagonal is chance. ROC is "
         "threshold-independent but ignores how common each class is, so it reads optimistically for a "
         "rare class &mdash; see the PR-ROC curves below for precision at each class's actual "
         "prevalence.</p>" if roc_png else ""),
        ("<h3>PR-ROC curves: detecting TOO_LONG / TOO_SHORT calls</h3>" if pr_png else ""),
        (_img_toggle(pr_png, pr_homopolymer_png, "pr") if pr_png else ""),
        ("<p class='note'>One-vs-rest precision&ndash;recall for the two error directions the pOk &lt; 0.5 "
         "threshold must catch. Unlike ROC, PR-ROC depends on how common the class is, so it shows the "
         "real precision trade-off where a class is rare. <code>AP</code> is the average precision (area "
         "under the curve); <code>base</code> (the dotted line) is the class prevalence &mdash; the "
         "precision a random classifier would get, so the gap above it is the model's lift. "
         "<code>full_nonspanning</code>, where raw EH mis-sizes most, is where detection matters "
         "most.</p>" if pr_png else ""),
        ("<h3>Predicted probabilities vs EH call error (delta bin)</h3>" if prob_violins_png else ""),
        (_img_toggle(prob_violins_png, prob_violins_homopolymer_png, "pv") if prob_violins_png else ""),
        ("<p class='note'>Distribution of each predicted direction probability "
         "(<code>pOk</code> / <code>pTooLong</code> / <code>pTooShort</code>, rows) as a function of "
         "the EH call error, per allele size bucket (columns). The x-axis is the signed bin "
         "<code>delta = round(eh) &minus; round(true)</code> in repeat units "
         "&mdash; left is EH <b>undercalling</b> (call too short), right is EH <b>overcalling</b> "
         "(call too long), and the dashed line marks the no-error <code>0</code> bin (violins colored "
         "blue&rarr;red by delta). One violin per bin shows how the predictor's probability shifts with the "
         "call error; e.g. <code>pTooShort</code> should rise where EH undercalls and "
         "<code>pTooLong</code> where it overcalls, while <code>pOk</code> peaks near <code>0</code>. "
         "Per-bin allele counts are capped at %s for the kernel-density estimate only.</p>"
         % format(_DELTA_VIOLIN_CAP, ",") if prob_violins_png else ""),
        "<h2>Relative feature importance (per allele size bucket, LCF prediction)</h2>",
        _img_toggle(importance_png, importance_homopolymer_png, "imp"),
        "<p class='note'>Each feature's <code>(#n)</code> suffix is its importance rank in the "
        "<code>full_nonspanning</code> allele size bucket; the same number is reused across all three "
        "charts so a feature can be tracked between allele size buckets. Toggle between "
        "<b>non-homopolymer</b> loci and <b>homopolymer</b> (1&nbsp;bp motif) loci (the homopolymer panel's "
        "<code>#</code> ranks are set by its own homopolymer full_nonspanning order). "
        "Bars are the rise in loss when that feature's column is shuffled, divided by the largest rise in "
        "that bucket (so each bucket's top feature is 1, and bars compare features within a bucket, not "
        "across buckets), measured on fold&nbsp;0's "
        "<b>calibration</b> chromosomes &mdash; not its test chromosomes, so that the ablation curves "
        "below, which add features in this order and score on the test chromosomes, never have their "
        "feature subsets chosen using the rows they are scored on. Those calibration chromosomes are "
        "held out of <i>training</i> but are not untouched: the direction head's isotonic "
        "calibrators are fit on them, so read these bars as a feature "
        "ranking rather than as an out-of-sample effect size. The LCF prediction is fit and applied only "
        "where the <code>pOk&lt;0.5</code> gate fires, so its bars are measured on those calibration "
        "rows only.</p>",
        "<h2>Add-one-feature ablation (LCF prediction)</h2>",
        _img_toggle(ablation_png, ablation_homopolymer_png, "abl"),
        "<p class='note'>The LCF prediction (the regressor predicting the length-correction factor "
        "<code>LCF = exp(t)</code>, so the corrected size is <code>eh/LCF</code>) is re-fit using only "
        "its top-1 most-important feature, then top-2, ... up to all features (x-axis; added in each "
        "allele size bucket's own importance order). The y-axis is the <b>held-out MAE</b> "
        "<code>mean|true &minus; eh/LCF|</code> (repeat units, fold-0 test chromosomes, within-pool CV). "
        "<b>x = 0 is raw EH</b> (no correction); <b>x &ge; 1</b> apply the LCF fit on that many features"
        "%s. Ungated (the correction is applied to every allele, unlike the "
        % (". The y-axis is broken so the large raw-EH baseline and the corrected detail are both "
           "readable" if ablation_axis == "broken" else ""),
        "gated MAE bar chart above which compares raw EH vs the pOk&lt;0.5-gated correction). Each "
        "prefix fits a <b>simplified</b> LCF prediction, on a seeded random sample of at most %s "
        "training-chromosome rows (not only the rows the gate fires on), scored on at most %s test rows, "
        "and with at most %d boosting iterations, so the curve ranks how much each feature adds rather "
        "than reproducing the shipped LCF prediction.</p>"
        % (format(ABLATION_TRAIN_CAP, ","), format(ABLATION_TEST_CAP, ","), M.EARLY_STOP_MAX_ITERATIONS),
    ]
    if dir_importance_png:
        parts += [
            "<h2>Relative feature importance (per allele size bucket, direction prediction)</h2>",
            _img_toggle(dir_importance_png, dir_importance_homopolymer_png, "dirimp"),
            "<p class='note'>The same permutation analysis as the panel above, but scoring the "
            "<b>direction predictor</b> (<code>pOk</code> / <code>pTooLong</code> / "
            "<code>pTooShort</code>) instead of the LCF regressor: each feature's column is shuffled "
            "and the bar is the resulting <b>rise in log-loss</b> of the calibrated probabilities "
            "(classifier + isotonic, exactly what ExpansionHunter emits), divided by the largest rise in "
            "that bucket as in the panel above. The two heads are ranked "
            "separately because they answer different questions -- a feature that pins down "
            "<i>how far off</i> a call is need not be the one that says <i>whether</i> it is off. "
            "Because the rankings differ, the <code>(#n)</code> suffixes on THIS panel are "
            "direction-head ranks and do not match the <code>#n</code> used in the feature-definition "
            "table or in the LCF importance panel above.</p>",
        ]
    if dir_ablation_png:
        parts += [
            "<h2>Add-one-feature ablation (direction prediction)</h2>",
            _img_toggle(dir_ablation_png, dir_ablation_homopolymer_png, "dirabl"),
            "<p class='note'>The direction predictor is re-fit using only its top-1 most-important "
            "feature, then top-2, ... up to all features (x-axis; added in each allele size bucket's "
            "own <b>direction</b> importance order from the panel above, not the LCF order used by the "
            "LCF ablation further up). The "
            "y-axis is the <b>held-out multinomial log-loss</b> of the calibrated "
            "<code>[pOk, pTooLong, pTooShort]</code> probabilities -- the head's own training "
            "objective, so lower is strictly better and it rewards calibration, not just ranking. "
            "<b>x = 0 is the class prior</b> (the feature-free predictor: this fold's representative "
            "training-set class frequencies emitted for every allele), which is the log-loss any feature has to "
            "beat. Same fold, rows and caps as the LCF ablation above, so the two curves describe "
            "the same alleles.</p>",
        ]
    parts.append(dataset_sections_html)
    parts += ["</body></html>"]
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("wrote %s" % out_html)


def _maybe_regenerate_dataset_artifacts(args):
    """Regenerates every ``DATASETS`` entry's eval + stacked artifacts before the sections are built.

    Runs by default, so every dataset section describes the model being reported (artifacts made by
    another model are dropped by ``build_dataset_sections``). Skipped when ``--skip-heldout-samples``
    is set, in ``--render-text-only`` mode (a pure re-render that regenerates nothing), or when no model
    was given (the apply needs one). Downloads nothing -- ``gen_datasets.generate`` is a no-op for a
    dataset whose parquets haven't been built locally (build the held-out ones with
    ``heldout.py --build-only`` / ``RUN_HELDOUT_SAMPLES=1 ./train_model.sh``).
    """
    if args.skip_heldout_samples or args.render_text_only:
        return
    if not args.model:
        print("datasets: no --model given, skipping regeneration (pass --skip-heldout-samples to "
              "silence)", flush=True)
        return
    import gen_datasets
    for key, label in DATASETS:
        print("\n==== regenerate %s artifacts (default; --skip-heldout-samples to skip) ====" % label,
              flush=True)
        # This runs after the hours-long CV, so a dataset whose parquets fail the feature-contract
        # check (heldout.assert_parquets_carry_contract, e.g. held-out parquets not rebuilt after a
        # features.py change) is skipped with a warning instead of aborting the report. Its old
        # artifacts then carry a different model and are dropped by build_dataset_sections.
        try:
            if gen_datasets.generate(key, args.model, args.out_dir, data_dir=args.data_dir) == 0:
                print("  %s: no local parquets -- skipping" % label, flush=True)
        except RuntimeError as e:
            print("  WARNING: %s skipped -- %s" % (label, e), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "report"))
    parser.add_argument("--model", default="", help="model .json.gz path (for the report header)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-train-cap", type=int, default=400_000,
                        help="max train rows per fold fit (test sets never capped)")
    parser.add_argument("--render-only", action="store_true",
                        help="skip the CV; re-render plots + HTML from a cached results.json")
    parser.add_argument("--render-text-only", action="store_true",
                        help="re-render ONLY the HTML from a cached results.json + the existing PNGs; "
                             "skip all plot regeneration (fast path for text/layout-only edits)")
    parser.add_argument("--ablation-only", action="store_true",
                        help="recompute only the ablation curves (cheap), reusing cached CV + importance")
    parser.add_argument("--homopolymer-cv", action="store_true",
                        help="run the in-pool CV on ONLY homopolymer (1 bp motif) loci, write "
                             "results_homopolymer.json, and exit (feeds the side-by-side homopolymer charts)")
    parser.add_argument("--skip-heldout-samples", action="store_true",
                        help="skip regenerating the per-dataset eval/stacked artifacts (held-out HPRC and "
                             "HG002 genome; by default they are regenerated from any locally-built "
                             "parquets, needs --model)")
    args = parser.parse_args()
    if args.render_text_only:  # text-only reuses the cached CV results exactly like --render-only
        args.render_only = True

    # Guard the pool-reading paths (CV, homopolymer CV, ablation) against a stale/missing parquet whose
    # upstream JSON/TSV has moved on. Pure --render-only/--render-text-only re-render from cached
    # results.json without touching the pool, so they skip the check.
    if args.homopolymer_cv or args.ablation_only or not args.render_only:
        dataset.assert_parquets_up_to_date(args.data_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.homopolymer_cv:
        folds = make_folds(n_folds=args.folds)
        evals = [evaluate_genotyping_regime(r, args.data_dir, folds, args.cv_train_cap or None,
                                            homopolymers_only=True)
                 for r in features.GENOTYPING_REGIMES]
        with open(os.path.join(args.out_dir, "results_homopolymer.json"), "w") as f:
            json.dump([e[0] for e in evals], f)
        _save_dir_oof(evals, os.path.join(args.out_dir, "dir_oof_homopolymer.npz"))
        print("wrote %s/results_homopolymer.json" % args.out_dir)
        return
    results_json = os.path.join(args.out_dir, "results.json")
    if args.render_only or args.ablation_only:
        with open(results_json) as f:
            results = json.load(f)
        if args.ablation_only:
            folds = make_folds(n_folds=args.folds)
            for r in results:
                regime = r["genotyping_regime"]
                df, branch = _load_genotyping_regime_df(regime, args.data_dir)
                print("=== ablation %s (%d rows) ===" % (regime, len(df)), flush=True)
                # Both curves reuse the cached rankings rather than recomputing them, so a cache
                # written under an older feature contract would silently truncate the curve.
                _assert_ranking_covers_contract(r["importance"], branch, regime, "importance")
                r["ablation"] = _ablation_curve(df, branch, folds[0], [f for f, _, _ in r["importance"]])
                # The direction curve needs the direction-head ranking, which only a full CV run
                # produces; a results.json from before that ranking existed keeps its old curve.
                if r.get("importance_direction"):
                    _assert_ranking_covers_contract(r["importance_direction"], branch, regime,
                                                    "importance_direction")
                    r["dir_ablation"] = _dir_ablation_curve(
                        df, branch, folds[0], [f for f, _, _ in r["importance_direction"]])
                else:
                    print("  no importance_direction in results.json -- skipping the direction "
                          "ablation (re-run report.py without --ablation-only to compute it)",
                          flush=True)
            with open(results_json, "w") as f:
                json.dump(results, f)
    else:
        folds = make_folds(n_folds=args.folds)
        evals = [evaluate_genotyping_regime(r, args.data_dir, folds, args.cv_train_cap or None)
                 for r in features.GENOTYPING_REGIMES]
        results = [e[0] for e in evals]
        with open(results_json, "w") as f:
            json.dump(results, f)
        _save_dir_oof(evals, os.path.join(args.out_dir, "dir_oof.npz"))

    drawn = {}  # name -> what plot_fn returned, for the few callers that need it

    def _png(name, plot_fn):
        """Path to ``out_dir/name``, regenerating it via ``plot_fn(path)`` unless ``--render-text-only``
        (which reuses the cached PNG and never calls ``plot_fn``). Returns None when the PNG is absent,
        so its report section is skipped rather than embedding a missing file. Each plot function's
        own return value is recorded in ``drawn`` for callers whose prose depends on it."""
        path = os.path.join(args.out_dir, name)
        if not args.render_text_only:
            drawn[name] = plot_fn(path)
        return path if os.path.exists(path) else None

    mae_png = _png("mae_raw_vs_gated.png",
                   lambda p: plot_mae(results, p, title_tag="non-homopolymer, 5-fold held-out"))
    importance_png = _png("feature_importance.png", lambda p: plot_importance_panel(results, p))
    ablation_png = _png("ablation.png", lambda p: plot_ablation(results, p))
    # plot_ablation reports whether it actually broke the y-axis. --render-text-only never redraws,
    # so it falls back to the wording that claims nothing about the axis.
    # Direction-head counterparts of the two panels above. Both self-skip (write nothing, so _png
    # returns None and the section is omitted) on a cached results.json that predates them.
    dir_importance_png = _png("feature_importance_direction.png",
                              lambda p: plot_importance_panel(
                                  results, p, key="importance_direction",
                                  head_label="direction prediction",
                                  score_label="mean log-loss rise"))
    dir_ablation_png = _png("ablation_direction.png", lambda p: plot_dir_ablation(results, p))

    # Direction-predictor accuracy plots (PR-ROC / ROC from the dir_oof npz, confusion from
    # results.json).
    confusion_png = _png("confusion.png", lambda p: plot_confusion(results, p))
    pr_png = roc_png = prob_violins_png = None
    dir_oof_npz = os.path.join(args.out_dir, "dir_oof.npz")
    if args.render_text_only or os.path.exists(dir_oof_npz):
        oof_dir = None if args.render_text_only else dict(np.load(dir_oof_npz))
        pr_png = _png("pr_curves.png", lambda p: plot_pr(oof_dir, p))
        roc_png = _png("roc_curves.png", lambda p: plot_roc(oof_dir, p))
        # prob-violins only exists once the delta-aware CV has been run; in text-only its PNG presence
        # is the signal (the npz is not loaded), otherwise check the npz for the delta arrays.
        if args.render_text_only or any(("%s__delta" % r) in oof_dir for r in features.GENOTYPING_REGIMES):
            prob_violins_png = _png("prob_violins.png", lambda p: plot_prob_violins(oof_dir, p))

    # Homopolymer (1 bp motif) versions of the in-pool MAE + ablation + importance + direction-predictor
    # charts, if the homopolymer CV (report.py --homopolymer-cv) has been run.
    htag = "homopolymer (1 bp motif)"
    mae_homopolymer_png = ablation_homopolymer_png = importance_homopolymer_png = None
    pr_homopolymer_png = roc_homopolymer_png = None
    confusion_homopolymer_png = prob_violins_homopolymer_png = None
    dir_importance_homopolymer_png = dir_ablation_homopolymer_png = None
    homo_results = None  # also read by the stale-cache check at the end of main()
    homo_json = os.path.join(args.out_dir, "results_homopolymer.json")
    if args.render_text_only or os.path.exists(homo_json):
        # Loaded regardless of --render-text-only: that mode still EMBEDS the homopolymer panels, so
        # the stale-cache check at the end of main() has to be able to inspect their source.
        if os.path.exists(homo_json):
            with open(homo_json) as f:
                homo_results = json.load(f)
        # Guard against a stale results_homopolymer.json (computed from a different data pool) being
        # mixed into a freshly-rendered report: skip the homopolymer panels when the cached per-regime
        # row counts disagree with the current parquets. When the parquets are absent (e.g. a pure
        # --render-only with no data/ tree) or in --render-text-only mode (no regeneration) staleness
        # cannot / need not be checked, so the cached PNGs are used as-is.
        stale = {}
        if not args.render_text_only:
            cur_counts = _homopolymer_row_counts(args.data_dir)
            if cur_counts:
                stale = {r["genotyping_regime"]: (r["n_rows"], cur_counts.get(r["genotyping_regime"], 0))
                         for r in homo_results if cur_counts.get(r["genotyping_regime"], 0) != r["n_rows"]}
                if stale:
                    print("  WARNING: results_homopolymer.json row counts %s (cached, current) disagree "
                          "with the current parquets; skipping homopolymer panels -- re-run "
                          "report.py --homopolymer-cv" % stale, flush=True)
        if not stale:
            mae_homopolymer_png = _png("mae_raw_vs_gated_homopolymer.png",
                                       lambda p: plot_mae(homo_results, p,
                                                          title_tag=htag + ", 5-fold held-out"))
            ablation_homopolymer_png = _png("ablation_homopolymer.png",
                                            lambda p: plot_ablation(homo_results, p))
            importance_homopolymer_png = _png("feature_importance_homopolymer.png",
                                              lambda p: plot_importance_panel(homo_results, p,
                                                                              loci_label="homopolymer loci"))
            dir_importance_homopolymer_png = _png(
                "feature_importance_direction_homopolymer.png",
                lambda p: plot_importance_panel(homo_results, p, loci_label="homopolymer loci",
                                                key="importance_direction",
                                                head_label="direction prediction",
                                                score_label="mean log-loss rise"))
            dir_ablation_homopolymer_png = _png(
                "ablation_direction_homopolymer.png",
                lambda p: plot_dir_ablation(homo_results, p, title_tag=htag))
            confusion_homopolymer_png = _png("confusion_homopolymer.png",
                                             lambda p: plot_confusion(homo_results, p, title_tag=htag))
            dir_oof_homo_npz = os.path.join(args.out_dir, "dir_oof_homopolymer.npz")
            if args.render_text_only or os.path.exists(dir_oof_homo_npz):
                oof_dir_homo = None if args.render_text_only else dict(np.load(dir_oof_homo_npz))
                pr_homopolymer_png = _png("pr_curves_homopolymer.png",
                                          lambda p: plot_pr(oof_dir_homo, p, title_tag=htag))
                roc_homopolymer_png = _png("roc_curves_homopolymer.png",
                                           lambda p: plot_roc(oof_dir_homo, p, title_tag=htag))
                if args.render_text_only or any(("%s__delta" % r) in oof_dir_homo
                                                for r in features.GENOTYPING_REGIMES):
                    prob_violins_homopolymer_png = _png("prob_violins_homopolymer.png",
                                                        lambda p: plot_prob_violins(oof_dir_homo, p,
                                                                                    title_tag=htag))

    # Both panels sit under one toggle and each chooses its own axis, so the sentence is emitted only
    # when they agree (or only one exists); otherwise it would describe whichever panel is not shown.
    ablation_axes = {drawn.get(n) for n in ("ablation.png", "ablation_homopolymer.png")} - {None}
    ablation_axis = ablation_axes.pop() if len(ablation_axes) == 1 else "single"

    # By default, regenerate every dataset's eval/stacked artifacts from the locally-built parquets so
    # the sections reflect the current model (no download; --skip-heldout-samples opts out,
    # --render-text-only never regenerates data).
    _maybe_regenerate_dataset_artifacts(args)

    # Per-dataset sections (Dataset / Homopolymers / LCF pills). Rendered from the eval_<key>.json +
    # eval_<key>_violin.npz + stacked_<key>.json artifacts (gen_datasets.py); absent datasets, and
    # artifacts recorded for a different model than --model, are skipped.
    dataset_sections_html = build_dataset_sections(args.out_dir, args.model,
                                                   regenerate=not args.render_text_only)

    render_html(results, mae_png, importance_png, ablation_png,
                args.model or "(model not specified)",
                os.path.join(args.out_dir, report_html_name(args.model)),
                mae_homopolymer_png=mae_homopolymer_png,
                ablation_homopolymer_png=ablation_homopolymer_png,
                importance_homopolymer_png=importance_homopolymer_png,
                pr_png=pr_png, roc_png=roc_png, confusion_png=confusion_png,
                prob_violins_png=prob_violins_png,
                pr_homopolymer_png=pr_homopolymer_png,
                roc_homopolymer_png=roc_homopolymer_png,
                confusion_homopolymer_png=confusion_homopolymer_png,
                prob_violins_homopolymer_png=prob_violins_homopolymer_png,
                dir_importance_png=dir_importance_png,
                dir_importance_homopolymer_png=dir_importance_homopolymer_png,
                dir_ablation_png=dir_ablation_png,
                dir_ablation_homopolymer_png=dir_ablation_homopolymer_png,
                stale_contract_html=_stale_contract_warning(results, homo_results),
                ablation_axis=ablation_axis,
                dataset_sections_html=dataset_sections_html)


if __name__ == "__main__":
    main()
