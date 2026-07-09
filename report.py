"""5-fold chromosome-clean evaluation + the HTML training report.

The deployable model (``train.py``) is fit on all real data, so it has no held-out
set of its own. This module measures held-out accuracy honestly with 5-fold
chromosome-clean cross-validation: the 24 chromosomes are partitioned into 5
disjoint test groups, each genotyping_regime's heads are trained out-of-fold on the other
chromosomes (early-stopped on a held-out calib chromosome subset), and the pooled
out-of-fold predictions feed the report. Splitting by chromosome group -- never by
row -- ensures no locus leaks between train and test.

The report (a single standalone ``.html`` with embedded plots) shows:
  - the raw-EH vs gated-LCF MAE chart (apply the LCF only where ``pOk < 0.5``),
    per genotyping_regime, on a broken linear axis;
  - per-genotyping_regime held-out accuracy + direction-head metrics;
  - per-genotyping_regime q-head permutation feature importance (relative).

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
from sklearn.metrics import (average_precision_score, make_scorer, mean_pinball_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

import accuracy_by_size as ABS
import dataset
import eh_json
import features
import heldout
import metrics
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
ALL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
GRAY, ORANGE = "#888888", "#F58518"
IMPORTANCE_CAP = 30_000  # held-out rows used for permutation importance
TOP_N = 15               # features shown per importance panel
ABLATION_KMAX = 22       # max # of top features in the add-one ablation curve (full: 22 of 24; quick: all 22, never truncated)
ABLATION_TRAIN_CAP = 120_000
ABLATION_TEST_CAP = 150_000
REGIME_COLORS = {"quick": "#4c72b0", "full_spanning": "#dd8452", "full_nonspanning": "#55a868"}


def make_folds(n_folds=5, n_calib=2, seed=SEED):
    """Partitions the chromosomes into ``n_folds`` disjoint OOF test groups.

    Each fold's test group is one partition block; ``n_calib`` chromosomes are drawn
    (seeded) from the remaining chromosomes for the early-stop calib set and the rest
    are the training chromosomes. Every chromosome is a test chromosome in exactly
    one fold, so the pooled out-of-fold predictions cover the data once.

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


def collect_oof(df, branch, folds, train_cap):
    """Runs 5-fold OOF training for one genotyping_regime; returns pooled per-row arrays + importance.

    Returns:
        ``(oof, ranked)`` where ``oof`` is a dict of concatenated arrays (``eh``,
        ``true``, ``true_pred``, ``t``, ``t_pred``, ``p_ok``, ``dir_code``,
        ``tol_repeats``) and ``ranked`` is the fold-0 q-head permutation importance
        list ``[(feature, mean, std), ...]`` (or ``None`` if it could not be run).
    """
    chrom = df["chrom"].astype(str)
    acc = {k: [] for k in ("eh", "true", "true_pred", "t", "t_pred", "p_ok", "p_long", "p_short",
                           "dir_code", "tol_repeats")}
    ranked = None
    for i, f in enumerate(folds):
        m_tr = chrom.isin(set(f["train"])).to_numpy()
        m_ca = chrom.isin(set(f["calib"])).to_numpy()
        m_te = chrom.isin(set(f["test"])).to_numpy()
        if not m_te.any() or not m_tr.any() or not m_ca.any():
            continue
        tr_idx = _cap_rows(np.where(m_tr)[0], train_cap, SEED + i)
        Xtr, names = features.build_matrix(df.iloc[tr_idx], branch)
        Xca, _ = features.build_matrix(df[m_ca], branch)
        Xte, _ = features.build_matrix(df[m_te], branch)
        ttr, tca = df["t"].to_numpy(float)[tr_idx], df.loc[m_ca, "t"].to_numpy(float)
        ytr, yca = df["dir_code"].to_numpy(int)[tr_idx], df.loc[m_ca, "dir_code"].to_numpy(int)
        eh_te = df.loc[m_te, "eh"].to_numpy(float)

        qreg = M.train_q_median(Xtr, ttr, Xca, tca)
        dmodel = M.train_direction(Xtr, ytr, Xca, yca)
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
        print("    fold %d: train=%d test=%d" % (i, tr_idx.size, int(m_te.sum())), flush=True)
        if ranked is None:
            ranked = _importance(qreg, Xte, df.loc[m_te, "t"].to_numpy(float), names)
    return {k: np.concatenate(v) for k, v in acc.items()}, ranked


def _importance(qreg, Xte, t_te, names):
    """Returns fold-0 q-head permutation importance (pinball-scored), most important first."""
    idx = _cap_rows(np.arange(len(Xte)), IMPORTANCE_CAP, SEED)
    res = permutation_importance(
        qreg, Xte.iloc[idx], t_te[idx],
        scoring=make_scorer(mean_pinball_loss, alpha=0.5, greater_is_better=False),
        n_repeats=5, random_state=SEED)
    order = np.argsort(res.importances_mean)[::-1]
    return [(names[i], float(res.importances_mean[i]), float(res.importances_std[i])) for i in order]


def _ablation_curve(df, branch, fold, order):
    """Add-one-feature held-out MAE curve: fit the q-head on the top-1, top-2, ... features.

    Features are added in ``order`` (this genotyping regime's own importance ranking). Each q-head
    is fit on one fold's training chromosomes and scored on its test chromosomes by the MAE of the
    corrected call ``eh/LCF`` against the truth (repeat units) -- i.e. how close the corrected call
    is to the true size. Uses a single fold and smaller caps than the headline CV.

    Returns:
        A list of ``{"k", "feature", "mae"}`` dicts: ``k=0`` is the raw-EH baseline (no correction)
        on the fold-0 test rows, then one dict per prefix length ``k=1, 2, ...``.
    """
    chrom = df["chrom"].astype(str)
    tr_idx = _cap_rows(np.where(chrom.isin(set(fold["train"])).to_numpy())[0], ABLATION_TRAIN_CAP, SEED)
    te_idx = _cap_rows(np.where(chrom.isin(set(fold["test"])).to_numpy())[0], ABLATION_TEST_CAP, SEED)
    m_ca = chrom.isin(set(fold["calib"])).to_numpy()
    Xtr, _ = features.build_matrix(df.iloc[tr_idx], branch)
    Xca, _ = features.build_matrix(df[m_ca], branch)
    Xte, _ = features.build_matrix(df.iloc[te_idx], branch)
    ttr, tca = df["t"].to_numpy(float)[tr_idx], df.loc[m_ca, "t"].to_numpy(float)
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


def _load_genotyping_regime_df(genotyping_regime, data_dir, homopolymers_only=False):
    """Loads one genotyping regime's rows (only the columns the eval needs) + its branch.

    By default homopolymer (1 bp motif) loci are excluded (the main report); ``homopolymers_only``
    flips that to keep ONLY homopolymers (the separate side-by-side homopolymer charts).
    """
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"eh", "true", "t", "dir_code", "chrom", "genotyping_regime", "tol_repeats"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
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

    oof, ranked = collect_oof(df, branch, folds, train_cap)
    order = [f for f, _, _ in (ranked or [])]
    print("  ablation (add-one curve) ...", flush=True)
    ablation = _ablation_curve(df, branch, folds[0], order) if order else []
    return {
        "genotyping_regime": genotyping_regime,
        "n_rows": len(df),
        "q": metrics.q_metrics(oof["eh"], oof["true"], oof["true_pred"], oof["t"], oof["t_pred"],
                               oof["tol_repeats"]),
        "direction": metrics.direction_metrics(
            oof["dir_code"], np.column_stack([oof["p_ok"], oof["p_long"], oof["p_short"]])),
        "gated": metrics.gated_mae(oof["eh"], oof["true"], oof["true_pred"], oof["p_ok"]),
        "importance": ranked,
        "ablation": ablation,
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

def plot_mae(results, out_png):
    """Draws the raw-EH vs gated-LCF MAE bar chart (broken linear axis), per genotyping_regime."""
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


def plot_importance_panel(results, out_png, top_n=TOP_N, loci_label="non-homopolymer loci"):
    """Draws the three q-head importance charts in one horizontal row, full_nonspanning first.

    The ``(#rank)`` suffix on every feature label is that feature's rank in the
    ``full_nonspanning`` importance order (1 = most important there); the same suffix is reused on
    the other two charts so a feature can be cross-referenced across genotyping regimes. Bars within
    each chart are still sorted by that chart's own importance.
    """
    by_reg = {r["genotyping_regime"]: r for r in results}
    nonspan = features.GENOTYPING_REGIME_FULL_NONSPANNING
    rank = {f: i + 1 for i, (f, _, _) in enumerate(by_reg[nonspan]["importance"])}
    order_reg = list(features.GENOTYPING_REGIMES)  # quick, full_spanning, full_nonspanning (left to right)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, reg in zip(axes, order_reg):
        ranked = by_reg[reg]["importance"][:top_n]
        peak = max((m for _, m, _ in ranked), default=1.0) or 1.0
        pos = np.arange(len(ranked))[::-1]
        ax.barh(pos, [m / peak for _, m, _ in ranked], xerr=[s / peak for _, _, s in ranked],
                color="#4c72b0", ecolor="gray", capsize=3)
        ax.set_yticks(pos)
        ax.set_yticklabels(["%s (#%d)" % (f, rank.get(f, 0)) for f, _, _ in ranked])
        ax.set_xlabel("relative importance (mean pinball-loss drop)")
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[reg])
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Relative feature importance (LCF prediction, held-out; %s) — rank # set by full_nonspanning"
                 % loci_label, fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_ablation(results, out_png):
    """Add-one-feature held-out MAE curve per genotyping regime, with a k=0 raw-EH point.

    k=0 is the uncorrected raw-EH MAE (``q.mae_eh``); k>=1 are the corrected MAEs from the ablation.
    A broken y-axis keeps the large raw-EH baseline (e.g. ~53 for full_nonspanning) and the corrected
    detail (~0-7) both readable in one plot.
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

    biggest = max(ys[0] for _, _, _, ys in series)                       # raw-EH MAE of the worst regime
    rest = max(v for _, _, _, ys in series for v in ys if v < biggest)   # everything below it
    broken = biggest > 2.5 * rest
    title = "Add-one-feature ablation (LCF prediction)"
    xlabel = "number of top features included (LCF prediction importance order; 0 = raw EH)"
    ylabel = "MAE |true − corrected|  (repeat units, held-out)"

    def draw(ax):
        for disp, color, xs, ys in series:
            ax.plot(xs, ys, "-o", ms=4, label=disp, color=color)
        ax.grid(True, alpha=0.3)

    if not broken:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax); ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        return

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

    The same two error directions as the PR-ROC plot, shown as the TPR/FPR trade-off; the legend AUC
    matches ``too_long_auc`` / ``too_short_auc`` in the table and the dashed diagonal is chance.

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
                data.append(vals if vals.size else np.zeros(1))
            vp = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.9)
            for body, c in zip(vp["bodies"], colors):
                body.set_facecolor(c); body.set_alpha(0.75)
            if "cmedians" in vp:
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


def plot_violins(violin, out_png, thresholds=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                 pfx="", title_tag="non-homopolymer", dataset_label="held-out HPRC"):
    """Violins of the per-allele signed error reduction, one panel per genotyping regime.

    reduction = |true - eh| - |true - eh/LCF| (>0 = correction moved the call closer to truth). Per
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
        data = [d if d.size else np.zeros(1) for d in data]
        _violin_ylim(ax, data)
        vp = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
        _style_violin_extrema(vp)
        for body, c in zip(vp["bodies"], ["#4c72b0"] * len(thresholds) + ["#dd8452"]):
            body.set_facecolor(c); body.set_alpha(0.65)
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
            data = [d if d.size else np.zeros(1) for d in data]
            vp = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
            _style_violin_extrema(vp)
            for body, c in zip(vp["bodies"], _LCF_BIN_COLORS):
                body.set_facecolor(c); body.set_alpha(0.65)
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
        data = [d if d.size else np.zeros(1) for d in data]
        vp = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
        _style_violin_extrema(vp)
        for body, c in zip(vp["bodies"], _MOTIF_BIN_COLORS):
            body.set_facecolor(c); body.set_alpha(0.65)
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

    reduction = |true - eh| - |true - eh/LCF| (>0 = correction moved the call closer to truth). Within
    each panel one violin per 0.1-wide ``pTooLong - pTooShort`` bin from -1.0 (certain TOO_SHORT)
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
        data = [d if d.size else np.zeros(1) for d in data]
        vp = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
        _style_violin_extrema(vp)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.7)
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

    reduction = |true - eh| - |true - eh/LCF| (>0 = correction moved the call closer to truth). Within
    each panel one violin per predicted-LCF stratum (0-0.2, 0.2-0.25, 0.25-0.5, 0.5-1, 1-2, 2-4, 4-5,
    >5); corrected = eh/LCF, so LCF<1 grows the call, LCF>1 shrinks it, LCF~1 (the 0.5-1 / 1-2 bins)
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
        data = [d if d.size else np.zeros(1) for d in data]
        vp = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
        _style_violin_extrema(vp)
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.7)
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
                 "(corrected = eh/LCF) — %s" % (dataset_label, title_tag), fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# --- HTML -----------------------------------------------------------------

def _img(path):
    """Embeds a PNG as a base64 data URI, palette-quantized to keep the standalone HTML small enough
    to commit + serve on GitHub Pages.

    matplotlib charts use few distinct colors, so an adaptive 256-color palette is visually
    near-lossless yet ~2-3x smaller than the truecolor PNG; the on-disk PNG (a gitignored build
    artifact) is left untouched -- only the embedded copy is quantized.
    """
    quantized = Image.open(path).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
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
    """Feature definition table ordered by the full_nonspanning importance rank (#)."""
    ranked = {r["genotyping_regime"]: r for r in
              results}[features.GENOTYPING_REGIME_FULL_NONSPANNING]["importance"]
    rows = ["<tr><th>rank</th><th>feature</th><th>definition</th></tr>"]
    for i, (feat, _, _) in enumerate(ranked, start=1):
        rows.append("<tr><td>#%d</td><td><code>%s</code></td><td>%s</td></tr>" % (
            i, html.escape(feat),
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
_MODEL_OUTPUTS = (
    ("pOk", "direction predictor (3-class softmax)", "probability that the EH call is correct (within tolerance)",
     "0&ndash;1"),
    ("pTooLong", "direction predictor (3-class softmax)",
     "probability that the true allele size is shorter than what EH called", "0&ndash;1"),
    ("pTooShort", "direction predictor (3-class softmax)",
     "probability that the true allele size is longer than what EH called", "0&ndash;1"),
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
            "<th>pOk argmax acc.</th></tr>"]
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
DATASETS = [("hg002_genome", "HG002 genome (31x)"),
            ("heldout_hprc", "%d held-out HPRC" % len(heldout.SAMPLES)),
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


def build_dataset_sections(out_dir, regenerate=True):
    """Renders the per-dataset evaluation + accuracy-by-size plots and returns the report HTML.

    When ``regenerate`` is False (``report.py --render-text-only``) the PNGs are not re-plotted; the
    existing on-disk PNGs are embedded as-is and any that are missing are skipped.

    Every plot carries a <b>Dataset</b> pill (HG002 genome 31x / 43 held-out HPRC)
    and the <b>Homopolymers</b> pill; the accuracy-by-size stacked-bar additionally gets an
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
    present = [(k, l) for k, l in DATASETS if k in evals or k in stacked]
    if not present:
        return ""
    homo_dim = ("homo", "Homopolymers", [("nh", "Non-homopolymer"), ("ho", "Homopolymer")])

    def P(name):
        return os.path.join(out_dir, name)

    parts = ["<h2>External validation by dataset</h2>",
             "<p class='note'>The exported model is applied unchanged (no fitting) to each dataset, "
             "scored against truth. Use the <b>Dataset</b> pill on every plot below to switch between "
             "<b>HG002 genome (31x)</b> and the <b>%d held-out HPRC samples</b> (absent from training -- a "
             "further %d HPRC samples were promoted into the training pool below for ancestry/sex "
             "diversity).</p>"
             % (len(heldout.SAMPLES), len(dataset.PROMOTED_HELDOUT_SAMPLES))]

    # --- accuracy by true allele size (str-truth-set-v2 stacked-bar replica) ---
    abs_imgs = {}
    for key, _ in present:
        if key not in stacked:
            continue
        sd = stacked[key]
        for hk, hcol, hdesc0 in (("nh", "nonhomo", "non-homopolymer, motif > 1 bp"),
                                 ("ho", "homo", "homopolymer, 1 bp motif")):
            for vkey, _, vnote, _ in ABS.CORRECTION_VARIANTS:
                for pkey, _, pmin in ABS.PURITY_VARIANTS:
                    for kkey, klabel, _ in ABS.POK_VARIANTS:
                        data = sd[hcol].get(vkey, {}).get(pkey, {}).get(kkey)
                        if data is None:
                            continue
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
                            abs_imgs[(key, hk, vkey, pkey, kkey)] = _img(png)
                        except Exception as e:
                            print("  [%s] accuracy-by-size %s/%s/%s/%s skipped: %s"
                                  % (key, hk, vkey, pkey, kkey, e), flush=True)
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
            "ExpansionHunter call compares to its true size (<b>Same</b> = within &plusmn;1 repeat = "
            "exactly right; blue = under-call, orange = over-call, cyan = wrong direction, plus "
            "No&nbsp;Call / Hom&nbsp;Ref / Het&nbsp;Ref); the x-axis is true allele size minus the "
            "reference. Left panel = allele counts, right panel = fractions. The <b>LCF correction</b> "
            "pill replaces each gated call with <code>round(eh/LCF)</code>, growing the green "
            "<b>Same</b> band where it helps: <b>Raw EH</b> (no correction), <b>p&lt;0.5</b> "
            "(correct alleles with <code>pOk&lt;0.5</code>), <b>p&lt;0.25</b> (a stricter "
            "<code>pOk&lt;0.25</code> threshold), and <b>p&lt;0.5, non-spanning</b> (the <code>pOk&lt;0.5</code> "
            "threshold restricted to full_nonspanning-bucket alleles). The <b>Repeat Purity Filter</b> pill "
            "(<b>Off</b> vs <b>&gt; 0.95 pure</b>) restricts the plot to alleles whose truth repeat "
            "purity exceeds 0.95 (purity is on a 0&ndash;1 scale), i.e. near-perfect tandem repeats "
            "without interruptions. (Every dataset is categorized from its per-allele parquet; the "
            "corrected bands reflect a capped per-sample model apply. A dataset whose parquet lacks "
            "no-call rows notes so in its panel title.)</p>"]

    # --- per-dataset held-out accuracy table ---
    etab = {(k,): _holdout_table(evals[k]) for k, _ in present if k in evals}
    parts += ["<h3>Held-out accuracy (raw EH vs pOk&lt;0.5-gated LCF)</h3>",
              _pills("etab", [_ds_dim_for(etab, present)], etab)]

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
            _holdout_homopolymer_results(evals[k]) if pfx == "h" else _holdout_results(evals[k]), png),
        "<p class='note'>Raw-EH MAE vs the MAE after applying the LCF only where <code>pOk&lt;0.5</code>, "
        "per allele size bucket (repeat units).</p>", [k for k, _ in present if k in evals])
    parts += homo_family(
        "dhh", "Loci the LCF would help vs hurt, by pOk stratum", "hh",
        lambda k, png, pfx, tag, label: plot_helped_hurt(evals[k], png, homopolymer=(pfx == "h"),
                                                         title_tag=tag, dataset_label=label),
        "<p class='note'>Would the LCF move each call <b>closer</b> (green) or <b>further</b> (red) from "
        "truth? Left = <code>pOk&lt;0.5</code> (threshold applies the LCF); right = <code>pOk&ge;0.5</code> "
        "(threshold keeps raw EH).</p>", [k for k, _ in present if k in evals])
    parts += homo_family(
        "dpokv", "Per-allele error reduction (signed), by pOk stratum", "pokv",
        lambda k, png, pfx, tag, label: plot_violins(violins[k], png, pfx=pfx, title_tag=tag,
                                                      dataset_label=label),
        "<p class='note'>Signed error reduction <code>|true&minus;eh| &minus; |true&minus;eh/LCF|</code> "
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
        "(0&ndash;0.2 .. &gt;5; corrected = <code>eh/LCF</code>).</p>",
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


def render_html(results, mae_png, importance_png, ablation_png, model_path, out_html,
                mae_homopolymer_png=None, ablation_homopolymer_png=None, importance_homopolymer_png=None,
                pr_png=None, roc_png=None, confusion_png=None, prob_violins_png=None,
                pr_homopolymer_png=None, roc_homopolymer_png=None,
                confusion_homopolymer_png=None, prob_violins_homopolymer_png=None,
                dataset_sections_html=""):
    """Writes the standalone HTML report embedding every plot + metric table."""
    non_homo, homo = "All non-homopolymer loci", "Homopolymer loci (1 bp motif)"
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
                      roc_homopolymer_png, prob_violins_homopolymer_png))
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
        "allele size. Each allele gets its own set of 4 scores:</p>",
        _model_outputs_table(),
        "<p class='note'>During model training and prediction, ExpansionHunter genotyped allele sizes are "
        "split into two separate buckets: the <b>full_spanning</b> bucket for allele sizes supported by "
        "&ge;1 spanning read, and the <b>full_nonspanning</b> bucket for alleles with no spanning reads. "
        "Additionally, in <code>--analysis-mode optimized-streaming</code>, a 3rd <b>quick</b> bucket is "
        "added for alleles that could be confidently and quickly genotyped using only spanning reads "
        "without running the full, computationally-expensive ExpansionHunter genotyping algorithm.</p>",
        "<p class='note'><b>Training data:</b> %s &mdash; Illumina whole-genome sequencing.</p>" % html.escape(pool),
        "<p class='note'><b>Truth set:</b> <a href='https://github.com/broadinstitute/str-truth-set-v2'>"
        "<code>str-truth-set-v2</code></a>. Per-allele true repeat counts "
        "(and repeat purity) are derived from haplotype-resolved long-read genome assemblies of the "
        "same samples (HG002, CHM1&ndash;CHM13), giving the true repeat number at each tandem-repeat "
        "locus. Negative-control loci are filtered out before training; truth repeat purity is not "
        "used as a training filter, but is available below as an opt-in <b>Repeat Purity Filter</b> "
        "stratification pill.</p>",
        "<p class='note'>Held-out accuracy is measured by 5-fold cross validation: "
        "each fold trains on ~19 chromosomes and tests on the held-out ones.</p>",
        "<h2>ExpansionHunter output fields used for model training</h2>",
        "<p class='note'>The raw per-allele <code>AlleleQualityMetrics.Alleles[]</code> fields read from "
        "each ExpansionHunter output JSON.</p>",
        _eh_output_glossary(),
        "<h2>Feature definitions (derived from ExpansionHunter output fields)</h2>",
        "<p class='note'>The model features, with their <code>full_nonspanning</code> importance rank "
        "(<code>#1</code> = most important there). These are derived (and two engineered: "
        "<code>ci_asymmetry</code>, <code>ci_over_eh</code>) from the ExpansionHunter output fields "
        "above.</p>",
        _feature_glossary(results),
        "<h2>Mean absolute error: raw EH allele size vs allele size after LCF correction</h2>",
        _img_toggle(mae_png, mae_homopolymer_png, "mae"),
        "<p class='note'>The length-correction factor "
        "<code>LCF = (ExpansionHunter called allele size)/(True allele size)</code> is "
        "applied only where <code>pOk &lt; 0.5</code>; calls with <code>pOk &ge; 0.5</code> keep the raw "
        "EH allele size. "
        "MAE is computed over held-out alleles, and is measured in repeat units. The pOk &lt; 0.5 threshold "
        "concentrates the correction on the "
        "<code>full_nonspanning</code> allele size bucket, where flanking/IRR sizing makes raw EH most error-prone.</p>",
        "<h3>Confusion matrix (argmax prediction, row-normalized)</h3>",
        _img_toggle(confusion_png, confusion_homopolymer_png, "cm"),
        "<p class='note'>Rows are the <b>true</b> direction, columns the <b>argmax</b>-predicted "
        "direction; each cell shows the allele count and its row-percent, and the diagonal is "
        "per-class <b>recall</b>. Row-normalizing (rather than showing raw counts) keeps the rare "
        "TOO_LONG / TOO_SHORT rows readable next to the dominant OK row, and exposes the asymmetry "
        "in <i>where</i> the predictor misroutes calls. This is the hard-decision view of the same "
        "probabilities shown calibrated above.</p>",
        ("<h3>ROC curves: detecting TOO_LONG / TOO_SHORT calls</h3>" if roc_png else ""),
        (_img_toggle(roc_png, roc_homopolymer_png, "roc") if roc_png else ""),
        ("<p class='note'>One-vs-rest ROC for the two error directions the pOk &lt; 0.5 threshold must "
         "catch: true "
         "positive rate vs false positive rate as the probability threshold sweeps, allele size buckets "
         "overlaid. The legend <code>AUC</code> matches <code>TOO_LONG AUC</code> / "
         "<code>TOO_SHORT AUC</code> in the table; the dashed diagonal is chance. ROC is "
         "threshold-independent but, because OK calls dominate, reads optimistically &mdash; see the "
         "PR-ROC curves below for the rare-class precision trade-off.</p>" if roc_png else ""),
        ("<h3>PR-ROC curves: detecting TOO_LONG / TOO_SHORT calls</h3>" if pr_png else ""),
        (_img_toggle(pr_png, pr_homopolymer_png, "pr") if pr_png else ""),
        ("<p class='note'>One-vs-rest precision&ndash;recall for the two error directions the pOk &lt; 0.5 "
         "threshold must catch. Because OK calls dominate, ROC-AUC looks optimistic; PR-ROC shows the real "
         "trade-off when the positive class is rare. <code>AP</code> is the average precision (area "
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
        "<code>#</code> ranks are set by its own homopolymer full_nonspanning order).</p>",
        "<h2>Add-one-feature ablation (LCF prediction)</h2>",
        _img_toggle(ablation_png, ablation_homopolymer_png, "abl"),
        "<p class='note'>The LCF prediction (the regressor predicting the length-correction factor "
        "<code>LCF = exp(t)</code>, so the corrected size is <code>eh/LCF</code>) is re-fit using only "
        "its top-1 most-important feature, then top-2, ... up to all features (x-axis; added in each "
        "allele size bucket's own importance order). The y-axis is the <b>held-out MAE</b> "
        "<code>mean|true &minus; eh/LCF|</code> (repeat units, fold-0 test chromosomes, within-pool CV). "
        "<b>x = 0 is raw EH</b> (no correction); <b>x &ge; 1</b> apply the LCF fit on that many features "
        "(the y-axis is broken so the large raw-EH baseline and the corrected detail are both readable). "
        "Ungated (the correction is applied to every allele, unlike the "
        "gated MAE bar chart above which compares raw EH vs the pOk&lt;0.5-gated correction).</p>",
    ]
    parts.append(dataset_sections_html)
    parts += ["</body></html>"]
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("wrote %s" % out_html)


HELDOUT_DATASET = "heldout_hprc"  # gen_datasets key regenerated by default (see main())


def _maybe_regenerate_heldout_samples(args):
    """Regenerates the held-out HPRC eval + stacked artifacts before the dataset sections are built.

    Runs by default; skipped when ``--skip-heldout-samples`` is set, in ``--render-text-only`` mode (a
    pure re-render that regenerates nothing), or when no model was given (the apply needs one).
    Downloads nothing -- ``gen_datasets.generate`` is a no-op when the per-sample parquets haven't been
    built locally (build them with ``heldout.py --build-only`` / ``RUN_HELDOUT_SAMPLES=1 ./train_model.sh``).
    """
    if args.skip_heldout_samples or args.render_text_only:
        return
    if not args.model:
        print("held-out HPRC: no --model given, skipping regeneration (pass --skip-heldout-samples to "
              "silence)", flush=True)
        return
    import gen_datasets
    print("\n==== regenerate held-out HPRC artifacts (default; --skip-heldout-samples to skip) ====",
          flush=True)
    if gen_datasets.generate(HELDOUT_DATASET, args.model, args.out_dir) == 0:
        print("  held-out HPRC: no per-sample parquets under data_eval_43/real_43/ -- skipping "
              "(build them with `heldout.py --build-only` or `RUN_HELDOUT_SAMPLES=1 ./train_model.sh`)",
              flush=True)


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
                        help="skip regenerating the held-out HPRC eval/stacked artifacts (by default "
                             "they are regenerated from any locally-built held-out parquets, needs --model)")
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
                df, branch = _load_genotyping_regime_df(r["genotyping_regime"], args.data_dir)
                print("=== ablation %s (%d rows) ===" % (r["genotyping_regime"], len(df)), flush=True)
                r["ablation"] = _ablation_curve(df, branch, folds[0], [f for f, _, _ in r["importance"]])
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

    def _png(name, plot_fn):
        """Path to ``out_dir/name``, regenerating it via ``plot_fn(path)`` unless ``--render-text-only``
        (which reuses the cached PNG and never calls ``plot_fn``). Returns None when the PNG is absent,
        so its report section is skipped rather than embedding a missing file."""
        path = os.path.join(args.out_dir, name)
        if not args.render_text_only:
            plot_fn(path)
        return path if os.path.exists(path) else None

    mae_png = _png("mae_raw_vs_gated.png", lambda p: plot_mae(results, p))
    importance_png = _png("feature_importance.png", lambda p: plot_importance_panel(results, p))
    ablation_png = _png("ablation.png", lambda p: plot_ablation(results, p))

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
    homo_json = os.path.join(args.out_dir, "results_homopolymer.json")
    if args.render_text_only or os.path.exists(homo_json):
        homo_results = None
        if not args.render_text_only:
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
                                       lambda p: plot_mae(homo_results, p))
            ablation_homopolymer_png = _png("ablation_homopolymer.png",
                                            lambda p: plot_ablation(homo_results, p))
            importance_homopolymer_png = _png("feature_importance_homopolymer.png",
                                              lambda p: plot_importance_panel(homo_results, p,
                                                                              loci_label="homopolymer loci"))
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

    # By default, regenerate the held-out HPRC eval/stacked artifacts from any locally-built held-out
    # parquets so the section always reflects the current model (no download; --skip-heldout-samples
    # opts out, --render-text-only never regenerates data). The other datasets' artifacts are still
    # produced only by an explicit gen_datasets.py run.
    _maybe_regenerate_heldout_samples(args)

    # Per-dataset external-validation sections (Dataset / Homopolymers / LCF pills). Rendered from the
    # eval_<key>.json + eval_<key>_violin.npz + stacked_<key>.json artifacts (gen_datasets.py); absent
    # datasets are simply skipped.
    dataset_sections_html = build_dataset_sections(args.out_dir, regenerate=not args.render_text_only)

    render_html(results, mae_png, importance_png, ablation_png,
                args.model or "(model not specified)",
                os.path.join(args.out_dir, "model_report.html"),
                mae_homopolymer_png=mae_homopolymer_png,
                ablation_homopolymer_png=ablation_homopolymer_png,
                importance_homopolymer_png=importance_homopolymer_png,
                pr_png=pr_png, roc_png=roc_png, confusion_png=confusion_png,
                prob_violins_png=prob_violins_png,
                pr_homopolymer_png=pr_homopolymer_png,
                roc_homopolymer_png=roc_homopolymer_png,
                confusion_homopolymer_png=confusion_homopolymer_png,
                prob_violins_homopolymer_png=prob_violins_homopolymer_png,
                dataset_sections_html=dataset_sections_html)


if __name__ == "__main__":
    main()
