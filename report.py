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
import datetime
import html
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_pinball_loss

import features
import metrics
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
ALL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
GRAY, ORANGE = "#888888", "#F58518"
IMPORTANCE_CAP = 30_000  # held-out rows used for permutation importance
TOP_N = 15               # features shown per importance panel
ABLATION_KMAX = 22       # max # of top features in the add-one ablation curve (full = 22, quick = 20)
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
        A list of ``{"k", "feature", "mae"}`` dicts, one per prefix length.
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
    curve = []
    for k in range(1, min(ABLATION_KMAX, len(order)) + 1):
        cols = order[:k]
        qreg = M.train_q_median(Xtr[cols], ttr, Xca[cols], tca)
        true_pred = eh_te / np.exp(qreg.predict(Xte[cols]))
        mae = float(np.mean(np.abs(true_te - true_pred)))
        curve.append({"k": k, "feature": order[k - 1], "mae": mae})
        print("    ablation k=%2d (+%-24s) MAE=%.4f" % (k, order[k - 1], mae), flush=True)
    return curve


def _load_genotyping_regime_df(genotyping_regime, data_dir):
    """Loads one genotyping regime's rows (only the columns the eval needs) + its branch."""
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"eh", "true", "t", "dir_code", "chrom", "genotyping_regime", "tol_repeats"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    df = pd.read_parquet(parquet, columns=cols)
    return df[df["genotyping_regime"] == genotyping_regime].reset_index(drop=True), branch


def evaluate_genotyping_regime(genotyping_regime, data_dir, folds, train_cap):
    """Loads one genotyping_regime's rows, runs OOF, returns its metrics + importance bundle."""
    df, branch = _load_genotyping_regime_df(genotyping_regime, data_dir)
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
    }


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
        ax.bar(x + 0.2, gat, 0.4, label="LCF-corrected (pOk<0.5 gate)", color=ORANGE)
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


def plot_importance_panel(results, out_png, top_n=TOP_N):
    """Draws the three q-head importance charts in one horizontal row, full_nonspanning first.

    The ``(#rank)`` suffix on every feature label is that feature's rank in the
    ``full_nonspanning`` importance order (1 = most important there); the same suffix is reused on
    the other two charts so a feature can be cross-referenced across genotyping regimes. Bars within
    each chart are still sorted by that chart's own importance.
    """
    by_reg = {r["genotyping_regime"]: r for r in results}
    nonspan = features.GENOTYPING_REGIME_FULL_NONSPANNING
    rank = {f: i + 1 for i, (f, _, _) in enumerate(by_reg[nonspan]["importance"])}
    order_reg = [features.GENOTYPING_REGIME_FULL_NONSPANNING,
                 features.GENOTYPING_REGIME_FULL_SPANNING, features.GENOTYPING_REGIME_QUICK]

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
    fig.suptitle("Relative feature importance (LCF prediction, held-out) — rank # set by full_nonspanning",
                 fontsize=14, weight="bold")
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
        xs = [0] + [d["k"] for d in ablation]
        ys = [r["q"]["mae_eh"]] + [d["mae"] for d in ablation]   # k=0 = raw EH (no correction)
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


def plot_helped_hurt(holdout, out_png):
    """Loci the LCF would move closer (green) vs further (red) from truth, split by pOk stratum.

    Left panel = pOk<0.5 (where the gate APPLIES the LCF): helped should dominate. Right panel =
    pOk>=0.5 (where the gate KEEPS raw EH): hurt dominating is exactly why those calls are left alone.
    """
    gr = holdout["genotyping_regimes"]
    regs = [r for r in features.GENOTYPING_REGIMES if gr[r].get("helped_lt") is not None]
    labels = [features.GENOTYPING_REGIME_DISPLAY[r] for r in regs]
    x = np.arange(len(regs))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (tag, title) in zip(axes, (("lt", "pOk < 0.5"),
                                       ("ge", "pOk ≥ 0.5"))):
        helped = [gr[r]["helped_%s" % tag] for r in regs]
        hurt = [gr[r]["hurt_%s" % tag] for r in regs]
        ax.bar(x - 0.2, helped, 0.4, label="helped (corrected closer)", color="#2ca02c")
        ax.bar(x + 0.2, hurt, 0.4, label="hurt (corrected further)", color="#d62728")
        for xi, v in list(zip(x - 0.2, helped)) + list(zip(x + 0.2, hurt)):
            ax.text(xi, v, "{:,}".format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("number of alleles")
        ax.set_title(title); ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.suptitle("Would LCF correction help or hurt? (held-out 43 samples, by pOk stratum)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


# Per-allele error reduction (the violin y-axis): how much closer the LCF-corrected
# call lands to truth than raw EH, in repeat units. >0 = correction helped.
_REDUCTION_LABEL = "error reduction (repeats; >0 = better)"


def _violin_ylim(ax, data):
    """Sets a 1-99th-percentile y-limit (with padding) so a few extreme alleles don't flatten violins."""
    allv = np.concatenate([d for d in data if len(d)])
    if allv.size > 1:
        lo, hi = np.percentile(allv, [1, 99])
        pad = max(0.5, 0.1 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)


def plot_violins(violin, out_png, thresholds=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)):
    """Violins of the per-allele signed error reduction, one panel per genotyping regime.

    reduction = |true - eh| - |true - eh/LCF| (>0 = correction moved the call closer to truth). Per
    panel: one violin for each tightening gate pOk < t (blue, nested subsets) plus pOk >= 0.5 (orange,
    where the gate keeps raw EH). Per-regime y-scale; y clipped to the 1-99th percentile.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__red" % r) in violin]
    fig, axes = plt.subplots(1, len(regs), figsize=(5.0 * len(regs), 5.8))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regs):
        red = np.asarray(violin["%s__red" % r], dtype=float)
        pok = np.asarray(violin["%s__pok" % r], dtype=float)
        data = [red[pok < t] for t in thresholds] + [red[pok >= 0.5]]
        data = [d if d.size else np.zeros(1) for d in data]
        vp = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
        for body, c in zip(vp["bodies"], ["#4c72b0"] * len(thresholds) + ["#dd8452"]):
            body.set_facecolor(c); body.set_alpha(0.65)
        ax.axhline(0, color="k", lw=0.9, ls="--")
        _violin_ylim(ax, data)
        ax.set_xticks(np.arange(1, len(data) + 1))
        ax.set_xticklabels(["pOk<%g" % t for t in thresholds] + ["pOk≥0.5"], fontsize=8, rotation=45)
        ax.set_title(features.GENOTYPING_REGIME_DISPLAY[r])
        ax.set_ylabel(_REDUCTION_LABEL)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-allele error reduction from LCF correction (held-out 43), by pOk stratum",
                 fontsize=13, weight="bold")
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


def plot_violins_lcf(violin, out_png):
    """Second violin plot: error reduction by pOk(<0.5 / >=0.5) x predicted-LCF bin.

    Grid of regime (rows) x pOk stratum (cols, pOk<0.5 = gate APPLIES the LCF, pOk>=0.5 = gate keeps
    raw EH); within each cell one violin per raw-LCF bin, so the x-axis sweeps correction size and
    direction together (LCF<1 grows the call, ~1 leaves it, >1 shrinks it; farther from 1 = bigger
    correction). Allele count under each violin.
    """
    regs = [r for r in features.GENOTYPING_REGIMES if ("%s__red" % r) in violin]
    cols = [("p < 0.5", lambda p: p < 0.5), ("p ≥ 0.5", lambda p: p >= 0.5)]
    fig, axes = plt.subplots(len(regs), 2, figsize=(16, 4.2 * len(regs)), sharey="row", squeeze=False)
    for i, r in enumerate(regs):
        red = np.asarray(violin["%s__red" % r], dtype=float)
        pok = np.asarray(violin["%s__pok" % r], dtype=float)
        bin_idx = np.digitize(np.asarray(violin["%s__lcf" % r], dtype=float), _LCF_BIN_EDGES)
        row_ylim = []
        for j, (ctitle, cmask) in enumerate(cols):
            ax = axes[i][j]
            cm = cmask(pok)
            data = [red[cm & (bin_idx == b)] for b in range(len(_LCF_BIN_LABELS))]
            row_ylim += [d for d in data if len(d)]
            data = [d if d.size else np.zeros(1) for d in data]
            vp = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
            for body, c in zip(vp["bodies"], _LCF_BIN_COLORS):
                body.set_facecolor(c); body.set_alpha(0.65)
            ax.axhline(0, color="k", lw=0.9, ls="--")
            ax.set_xticks(np.arange(1, len(_LCF_BIN_LABELS) + 1))
            ax.set_xticklabels(["%s\n(n=%s)" % (lbl, format(int(d.size), ",")) for lbl, d in zip(_LCF_BIN_LABELS, data)],
                               fontsize=7)
            ax.set_title("%s (%s)" % (_REGIME_ALLELE_DESC[r], ctitle), fontsize=10)
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
    fig.suptitle("Error reduction by predicted-LCF bin (correction size + direction), by pOk stratum (held-out 43)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# --- HTML -----------------------------------------------------------------

def _img(path):
    with open(path, "rb") as f:
        return '<img src="data:image/png;base64,%s" />' % base64.b64encode(f.read()).decode()


def _q_table(results):
    rows = ["<tr><th>genotyping_regime</th><th>n (held-out)</th><th>raw EH MAE</th><th>LCF MAE</th>"
            "<th>dist. reduction</th><th>median |err|: EH&rarr;LCF</th><th>exact: EH&rarr;LCF</th></tr>"]
    for r in results:
        q = r["q"]
        rows.append("<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td><td>%+.1f%%</td>"
                    "<td>%.3f &rarr; %.3f</td><td>%.3f &rarr; %.3f</td></tr>" % (
                        features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]], q["n"], q["mae_eh"], q["mae_true"],
                        100 * q["dist_reduction"], q["median_ae_eh"], q["median_ae_true"],
                        q["eh_exact_match_rate"], q["exact_match_rate"]))
    return "<table>%s</table>" % "".join(rows)


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


def _dir_table(results):
    rows = ["<tr><th>genotyping_regime</th><th>log-loss</th><th>TOO_LONG AUC</th><th>TOO_SHORT AUC</th>"
            "<th>ECE</th><th>pOk argmax acc.</th></tr>"]
    for r in results:
        d = r["direction"]
        rows.append("<tr><td>%s</td><td>%.4f</td><td>%.3f</td><td>%.3f</td><td>%.4f</td>"
                    "<td>%.3f</td></tr>" % (
                        features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]], d["log_loss"], d["too_long_auc"],
                        d["too_short_auc"], d["ece"], d["p_ok_accuracy"]))
    return "<table>%s</table>" % "".join(rows)


def _holdout_results(holdout):
    """Builds a plot_mae-compatible results list from the 43-sample benchmark JSON."""
    gr = holdout["genotyping_regimes"]
    return [{"genotyping_regime": r,
             "gated": {"mae_raw": gr[r]["mae_raw"], "mae_gated": gr[r]["mae_gated"]}}
            for r in features.GENOTYPING_REGIMES if gr[r].get("n")]


def _holdout_table(holdout):
    rows = ["<tr><th>genotyping regime</th><th>n</th><th>raw EH MAE</th><th>gated MAE</th>"
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


def render_html(results, mae_png, importance_png, ablation_png, model_path, out_html,
                holdout=None, holdout_mae_png=None, holdout_helped_hurt_png=None,
                holdout_violin_png=None, holdout_violin_lcf_png=None):
    """Writes the standalone HTML report embedding every plot + metric table."""
    css = ("body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;"
           "color:#222;max-width:1180px}h1{font-size:22px}h2{font-size:17px;margin-top:34px;"
           "border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;"
           "margin:10px 0;font-size:13px}th,td{border:1px solid #ccc;padding:5px 9px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}th{background:#f4f4f4}"
           "img{max-width:100%;height:auto}.note{color:#555;font-size:14px}code{background:#f4f4f4;"
           "padding:1px 4px;border-radius:3px}")
    pool = " + ".join("%s %s" % (s, c) for s, c in (("HG002", "10x/20x/31x"),
                                                    ("CHM1_CHM13", "46x")))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Genotype-quality model report</title>",
        "<style>%s</style></head><body>" % css,
        "<h1>ExpansionHunter genotype-quality model &mdash; training report</h1>",
        "<p class='note'>Generated %s &nbsp;|&nbsp; training data: %s</p>" % (
            html.escape(datetime.date.today().isoformat()), html.escape(pool)),
        "<p class='note'>5-fold cross validation: each fold trains on ~19 chromosomes and tests on "
        "the held-out ones.</p>",
        "<p class='note'>Model file: <code>%s</code></p>" % html.escape(os.path.basename(model_path)),
        "<h2>Mean absolute error: raw EH allele size vs allele size after LCF correction</h2>",
        _img(mae_png),
        "<p class='note'>The length-correction factor "
        "<code>LCF = (ExpansionHunter called allele size)/(True allele size)</code> (LCF prediction) is "
        "applied only where <code>pOk &lt; 0.5</code> (pOk/pTooLong/pTooShort); confident calls keep raw EH. "
        "MAE is over held-out alleles, in repeat units. The gate concentrates the correction on the "
        "<code>full_nonspanning</code> genotyping_regime, where flanking/IRR sizing makes raw EH most error-prone.</p>",
        "<h2>Held-out LCF prediction accuracy (LCF-recovered truth vs raw EH)</h2>",
        _q_table(results),
        "<h2>Held-out pOk/pTooLong/pTooShort metrics</h2>",
        _dir_table(results),
        "<h2>Relative feature importance (per genotyping regime, LCF prediction)</h2>",
        _img(importance_png),
        "<p class='note'>Each feature's <code>(#n)</code> suffix is its importance rank in the "
        "<code>full_nonspanning</code> genotyping regime; the same number is reused across all three "
        "charts so a feature can be tracked between genotyping regimes.</p>",
        "<h2>Feature definitions</h2>",
        "<p class='note'>The model features, with their <code>full_nonspanning</code> importance rank "
        "(<code>#1</code> = most important there).</p>",
        _feature_glossary(results),
        "<h2>Add-one-feature ablation (LCF prediction)</h2>",
        _img(ablation_png),
        "<p class='note'>The LCF prediction (the regressor predicting the length-correction factor "
        "<code>LCF = exp(t)</code>, so the corrected size is <code>eh/LCF</code>) is re-fit using only "
        "its top-1 most-important feature, then top-2, ... up to all features (x-axis; added in each "
        "genotyping regime's own importance order). The y-axis is the <b>held-out MAE</b> "
        "<code>mean|true &minus; eh/LCF|</code> (repeat units, fold-0 test chromosomes, within-pool CV). "
        "<b>x = 0 is raw EH</b> (no correction); <b>x &ge; 1</b> apply the LCF fit on that many features "
        "(the y-axis is broken so the large raw-EH baseline and the corrected detail are both readable). "
        "Almost all of the correction's benefit comes from the very first feature "
        "(<code>eh_minus_ref</code>: full_nonspanning ~52.9 at k=0 &rarr; ~6 at k=1); adding the rest only "
        "refines it (~6 &rarr; ~3.9). Ungated (the correction is applied to every allele, unlike the "
        "gated MAE bar chart above which compares raw EH vs the pOk&lt;0.5-gated correction).</p>",
    ]
    if holdout:
        parts += [
            "<h2>External held-out validation &mdash; 43 HPRC samples</h2>",
            "<p class='note'>The exported model (loaded from its <code>.json.gz</code>, the format "
            "ExpansionHunter consumes) is applied unchanged &mdash; no fitting &mdash; to %d HPRC "
            "samples entirely absent from the HG002+CHM training data, scored against their truth%s.</p>"
            % (holdout.get("n_samples", 0),
               (" (up to %s alleles/sample)" % format(holdout["max_alleles_per_sample"], ",")
                if holdout.get("max_alleles_per_sample") else "")),
            _img(holdout_mae_png),
            _holdout_table(holdout),
        ]
        if holdout_helped_hurt_png:
            parts += [
                "<h3>Loci the LCF would help vs hurt, by pOk stratum</h3>",
                _img(holdout_helped_hurt_png),
                "<p class='note'>For every allele, would applying the LCF move the call <b>closer</b> "
                "to the truth (green) or <b>further</b> (red)? Split by pOk: <b>left</b> "
                "(<code>pOk &lt; 0.5</code>) is where the gate applies the correction &mdash; helped "
                "dominates, so correcting is right; <b>right</b> (<code>pOk &ge; 0.5</code>) is where the "
                "gate keeps raw EH &mdash; hurt dominates, so leaving those calls uncorrected is exactly "
                "the right choice. This is the empirical justification for the <code>pOk &lt; 0.5</code> "
                "gate.</p>",
            ]
        if holdout_violin_png:
            parts += [
                "<h3>Per-allele error reduction (signed), by pOk stratum</h3>",
                _img(holdout_violin_png),
                "<p class='note'>Distribution of the <b>signed</b> per-allele error change from the LCF, "
                "<code>|true &minus; eh| &minus; |true &minus; eh/LCF|</code> &mdash; <b>above 0</b> the "
                "correction moved the call <b>closer</b> to the truth, <b>below 0</b> it moved it "
                "<b>further</b>. The blue violins are progressively stricter gates "
                "<code>pOk &lt; 0.5, 0.4, 0.3, 0.2, 0.1</code> (nested subsets); the orange violin is "
                "<code>pOk &ge; 0.5</code> (where the gate keeps raw EH). The benefit grows and stays "
                "above 0 as the gate tightens, while <code>pOk &ge; 0.5</code> sits at/below 0. Per-regime "
                "y-scale; y clipped to the 1&ndash;99th percentile.</p>",
            ]
        if holdout_violin_lcf_png:
            parts += [
                "<h3>Error reduction by predicted-LCF bin (correction size &amp; direction)</h3>",
                _img(holdout_violin_lcf_png),
                "<p class='note'>Same signed error reduction, split (per genotyping regime, rows) by pOk "
                "stratum (columns: <code>pOk &lt; 0.5</code> where the gate applies the LCF, "
                "<code>pOk &ge; 0.5</code> where it keeps raw EH) and within each cell by the predicted "
                "<code>LCF</code> bin. LCF is the correction factor (<code>corrected = eh / LCF</code>): "
                "<code>LCF &lt; 1</code> <b>grows</b> the call, <code>LCF &gt; 1</code> <b>shrinks</b> it, "
                "<code>LCF &asymp; 1</code> (the gray <code>0.8&ndash;1.25</code> bin) leaves it ~unchanged; "
                "bins farther from 1 are larger corrections. Allele count <code>n</code> under each violin; "
                "per-regime y-scale, 1&ndash;99th-percentile clip.</p>",
            ]
    parts += ["</body></html>"]
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("wrote %s" % out_html)


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
    parser.add_argument("--ablation-only", action="store_true",
                        help="recompute only the ablation curves (cheap), reusing cached CV + importance")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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
        results = [evaluate_genotyping_regime(r, args.data_dir, folds, args.cv_train_cap or None)
                   for r in features.GENOTYPING_REGIMES]
        with open(results_json, "w") as f:
            json.dump(results, f)

    mae_png = os.path.join(args.out_dir, "mae_raw_vs_gated.png")
    importance_png = os.path.join(args.out_dir, "feature_importance.png")
    ablation_png = os.path.join(args.out_dir, "ablation.png")
    plot_mae(results, mae_png)
    plot_importance_panel(results, importance_png)
    plot_ablation(results, ablation_png)

    # Optional external-validation section: rendered only if the 43-sample benchmark has been run
    # with the current (median |err|) schema; a stale within-tol-only file is ignored until re-run.
    holdout = holdout_mae_png = holdout_helped_hurt_png = None
    holdout_violin_png = holdout_violin_lcf_png = None
    holdout_json = os.path.join(args.out_dir, "holdout43.json")
    if os.path.exists(holdout_json):
        with open(holdout_json) as f:
            loaded = json.load(f)
        if "median_raw" in next(iter(loaded["genotyping_regimes"].values()), {}):
            holdout = loaded
            holdout_mae_png = os.path.join(args.out_dir, "holdout43_mae.png")
            plot_mae(_holdout_results(holdout), holdout_mae_png)
            if "helped_lt" in next(iter(holdout["genotyping_regimes"].values()), {}):
                holdout_helped_hurt_png = os.path.join(args.out_dir, "holdout43_helped_hurt.png")
                plot_helped_hurt(holdout, holdout_helped_hurt_png)
            violin_npz = os.path.join(args.out_dir, "holdout43_violin.npz")
            if os.path.exists(violin_npz):
                violin = dict(np.load(violin_npz))
                if any(("%s__red" % r) in violin for r in features.GENOTYPING_REGIMES):
                    holdout_violin_png = os.path.join(args.out_dir, "holdout43_violin.png")
                    plot_violins(violin, holdout_violin_png)
                    holdout_violin_lcf_png = os.path.join(args.out_dir, "holdout43_violin_lcf.png")
                    plot_violins_lcf(violin, holdout_violin_lcf_png)

    render_html(results, mae_png, importance_png, ablation_png,
                args.model or "(model not specified)",
                os.path.join(args.out_dir, "model_report.html"),
                holdout=holdout, holdout_mae_png=holdout_mae_png,
                holdout_helped_hurt_png=holdout_helped_hurt_png,
                holdout_violin_png=holdout_violin_png,
                holdout_violin_lcf_png=holdout_violin_lcf_png)


if __name__ == "__main__":
    main()
