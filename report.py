"""5-fold chromosome-clean evaluation + the HTML training report.

The deployable model (``train.py``) is fit on all real data, so it has no held-out
set of its own. This module measures held-out accuracy honestly with 5-fold
chromosome-clean cross-validation: the 24 chromosomes are partitioned into 5
disjoint test groups, each genotyping_regime's heads are trained out-of-fold on the other
chromosomes (early-stopped on a held-out calib chromosome subset), and the pooled
out-of-fold predictions feed the report. Splitting by chromosome group -- never by
row -- ensures no locus leaks between train and test.

The report (a single standalone ``.html`` with embedded plots) shows:
  - the raw-EH vs gated-LCF MAE chart (apply the LCF only where ``P_OK < 0.5``),
    per genotyping_regime, on a broken linear axis;
  - per-genotyping_regime held-out accuracy + direction-head metrics;
  - per-genotyping_regime q-head permutation feature importance (relative).

Coding rules: no type hints, Google docstrings, ``print()``. Determinism: ``SEED``.
"""

import argparse
import base64
import datetime
import html
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


def evaluate_genotyping_regime(genotyping_regime, data_dir, folds, train_cap):
    """Loads one genotyping_regime's rows, runs OOF, returns its metrics + importance bundle."""
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"eh", "true", "t", "dir_code", "chrom", "genotyping_regime", "tol_repeats"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    df = pd.read_parquet(parquet, columns=cols)
    df = df[df["genotyping_regime"] == genotyping_regime].reset_index(drop=True)
    print("=== %s (branch %s): %d rows ===" % (genotyping_regime, branch, len(df)), flush=True)

    oof, ranked = collect_oof(df, branch, folds, train_cap)
    return {
        "genotyping_regime": genotyping_regime,
        "n_rows": len(df),
        "q": metrics.q_metrics(oof["eh"], oof["true"], oof["true_pred"], oof["t"], oof["t_pred"],
                               oof["tol_repeats"]),
        "direction": metrics.direction_metrics(
            oof["dir_code"], np.column_stack([oof["p_ok"], oof["p_long"], oof["p_short"]])),
        "gated": metrics.gated_mae(oof["eh"], oof["true"], oof["true_pred"], oof["p_ok"]),
        "importance": ranked,
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
        ax.bar(x + 0.2, gat, 0.4, label="LCF-corrected (P_OK<0.5 gate)", color=ORANGE)
        ax.grid(axis="y", alpha=0.3)

    title = "Mean absolute error: raw EH vs gated LCF correction"
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


def plot_importance(ranked, genotyping_regime, out_png):
    """Draws relative q-head permutation importance (normalized to the max) as horizontal bars."""
    top = ranked[:15]
    peak = max((m for _, m, _ in top), default=1.0) or 1.0
    pos = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.32 * len(top) + 1.0)))
    ax.barh(pos, [m / peak for _, m, _ in top], xerr=[s / peak for _, _, s in top],
            color="#4c72b0", ecolor="gray", capsize=3)
    ax.set_yticks(pos); ax.set_yticklabels([f for f, _, _ in top])
    ax.set_xlabel("relative permutation importance (mean pinball-loss drop)")
    ax.set_title("%s -- q-head feature importance (held-out)" % features.GENOTYPING_REGIME_DISPLAY[genotyping_regime])
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


# --- HTML -----------------------------------------------------------------

def _img(path):
    with open(path, "rb") as f:
        return '<img src="data:image/png;base64,%s" />' % base64.b64encode(f.read()).decode()


def _q_table(results):
    rows = ["<tr><th>genotyping_regime</th><th>n (held-out)</th><th>raw EH MAE</th><th>LCF MAE</th>"
            "<th>dist. reduction</th><th>within-tol: EH&rarr;LCF</th><th>exact: EH&rarr;LCF</th></tr>"]
    for r in results:
        q = r["q"]
        rows.append("<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td><td>%+.1f%%</td>"
                    "<td>%.3f &rarr; %.3f</td><td>%.3f &rarr; %.3f</td></tr>" % (
                        features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]], q["n"], q["mae_eh"], q["mae_true"],
                        100 * q["dist_reduction"], q["eh_within_tol_rate"], q["within_tol_rate"],
                        q["eh_exact_match_rate"], q["exact_match_rate"]))
    return "<table>%s</table>" % "".join(rows)


def _dir_table(results):
    rows = ["<tr><th>genotyping_regime</th><th>log-loss</th><th>TOO_LONG AUC</th><th>TOO_SHORT AUC</th>"
            "<th>ECE</th><th>P_OK argmax acc.</th></tr>"]
    for r in results:
        d = r["direction"]
        rows.append("<tr><td>%s</td><td>%.4f</td><td>%.3f</td><td>%.3f</td><td>%.4f</td>"
                    "<td>%.3f</td></tr>" % (
                        features.GENOTYPING_REGIME_DISPLAY[r["genotyping_regime"]], d["log_loss"], d["too_long_auc"],
                        d["too_short_auc"], d["ece"], d["p_ok_accuracy"]))
    return "<table>%s</table>" % "".join(rows)


def render_html(results, mae_png, importance_pngs, model_path, out_html):
    """Writes the standalone HTML report embedding every plot + metric table."""
    css = ("body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;"
           "color:#222;max-width:980px}h1{font-size:22px}h2{font-size:17px;margin-top:34px;"
           "border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;"
           "margin:10px 0;font-size:13px}th,td{border:1px solid #ccc;padding:5px 9px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}th{background:#f4f4f4}"
           "img{max-width:100%;height:auto}.note{color:#555;font-size:12px}"
           ".imp{display:flex;flex-wrap:wrap;gap:10px}.imp img{max-width:48%}code{background:#f4f4f4;"
           "padding:1px 4px;border-radius:3px}")
    pool = " + ".join("%s %s" % (s, c) for s, c in (("HG002", "10x/20x/31x"),
                                                    ("CHM1_CHM13", "46x")))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Genotype-quality model report</title>",
        "<style>%s</style></head><body>" % css,
        "<h1>ExpansionHunter genotype-quality model &mdash; training report</h1>",
        "<p class='note'>Generated %s &nbsp;|&nbsp; training pool: %s (real data only) &nbsp;|&nbsp; "
        "held-out accuracy from 5-fold chromosome-clean CV.</p>" % (
            html.escape(datetime.date.today().isoformat()), html.escape(pool)),
        "<p class='note'>Model file: <code>%s</code></p>" % html.escape(os.path.basename(model_path)),
        "<h2>Mean absolute error: raw EH vs gated LCF correction</h2>",
        _img(mae_png),
        "<p class='note'>The length-correction factor <code>LCF = eh/true</code> (q-median head) is "
        "applied only where <code>P_OK &lt; 0.5</code> (direction head); confident calls keep raw EH. "
        "MAE is over held-out alleles, in repeat units. The gate concentrates the correction on the "
        "<code>full_nonspanning</code> genotyping_regime, where flanking/IRR sizing makes raw EH most error-prone.</p>",
        "<h2>Held-out q-head accuracy (LCF-recovered truth vs raw EH)</h2>",
        _q_table(results),
        "<h2>Held-out direction-head metrics</h2>",
        _dir_table(results),
        "<h2>Relative feature importance (per genotyping_regime, q-head)</h2>",
        "<div class='imp'>%s</div>" % "".join(_img(p) for p in importance_pngs if p),
        "</body></html>",
    ]
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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    folds = make_folds(n_folds=args.folds)
    results = [evaluate_genotyping_regime(r, args.data_dir, folds, args.cv_train_cap or None)
               for r in features.GENOTYPING_REGIMES]

    mae_png = os.path.join(args.out_dir, "mae_raw_vs_gated.png")
    plot_mae(results, mae_png)
    importance_pngs = []
    for r in results:
        if r["importance"]:
            p = os.path.join(args.out_dir, "importance_%s.png" % r["genotyping_regime"])
            plot_importance(r["importance"], r["genotyping_regime"], p)
            importance_pngs.append(p)

    render_html(results, mae_png, importance_pngs, args.model or "(model not specified)",
                os.path.join(args.out_dir, "model_report.html"))


if __name__ == "__main__":
    main()
