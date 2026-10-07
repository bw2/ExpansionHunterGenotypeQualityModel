"""Learning curves for choosing the exported model's size (``train.ITERATIONS_BY_REGIME``).

With whole people held out, the held-out loss keeps improving to any iteration ceiling, so early
stopping never stops and the model size has to be chosen from curves instead. This script fits one
regime with ``train.fit_heads`` once, to ``--direction-iterations`` / ``--q-median-iterations``
(default 4,000 / 8,000), keeping 10 validation people and chromosomes 1, 4, 6 and 12 out of all
fitting, then scores the gated MAE at checkpoints along the boosting path (direction ``d`` paired with
q-median ``2d``; the isotonic calibration is refit at each checkpoint, as ``train_direction`` would at
that length) on three cells:

- ``train``: rows the direction head was actually fit on (``train.direction_training_rows``),
  weighted back to the pool's representative mix (the fit rows are a quota sample),
- ``val_known``: validation people on the fitted chromosomes (new people at loci the training people
  cover, which is about 99% of a new person's variant calls on this catalog),
- ``val_new``: validation people on chromosomes 1, 4, 6, 12 (loci no one was trained on).

``--plot`` renders every result in ``--results`` as an HTML page of curves, and ``--pick`` prints the
smallest checkpoint whose ``val_known`` gated MAE is within ``--tolerance`` of the longest fit's.
``ITERATIONS_BY_REGIME`` was set on 2026-10-06 with ``--tolerance 0.02``, from an earlier version of
this script that also held 5 early-stopping people out of training; quick was then cut by hand from
that rule's 500 to 150 for ExpansionHunter runtime, so ``--pick`` does not reproduce it.

Usage:
    python3 model_size_curves.py --regime full_nonspanning --results curves.jsonl
    python3 model_size_curves.py --results curves.jsonl --plot curves.html --pick --tolerance 0.02

Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import base64
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import dataset
import features
import model as M
import train

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
UNSEEN_CHROMS = ("1", "12", "4", "6")
N_VALIDATION_PEOPLE = 10
TRAIN_CAP = 1_000_000
CALIB_CAP = 200_000
CELL_CAP = 100_000
CHECKPOINTS = (100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000)
STRATA = (("nonhomo", "non-homopolymer"), ("homo", "homopolymer"))
# Measured on the 2026-10-05 export: 18,009,180 gzipped bytes for 31,100 trees of up to 31 leaves.
GZ_BYTES_PER_TREE = 18_009_180 / 31_100


def load_regime(data_dir, regime):
    """Returns the regime's whole pool (as train.py reads it) plus ``true``, ``eh`` and ``representative``."""
    src = "quick" if regime == features.GENOTYPING_REGIME_QUICK else "full"
    path = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"t", "dir_code", "chrom", "genotyping_regime", "true", "eh", "motif_size", "individual",
               "representative"})
    cols = [c for c in pq.ParquetFile(path).schema.names if c in need]
    return pq.read_table(path, columns=cols, filters=[("genotyping_regime", "=", regime)]).to_pandas().reset_index(drop=True)


def split(df):
    """Returns ``(validation, unseen, calib)`` boolean masks over ``df``'s rows (seeded)."""
    individual = df["individual"].to_numpy().astype(object)
    chrom = df["chrom"].astype(str).to_numpy().astype(object)
    candidates = sorted(set(individual.tolist()) - set(train.ALWAYS_TRAIN_INDIVIDUALS))
    validation = np.isin(individual, np.random.default_rng(SEED + 100).choice(
        candidates, N_VALIDATION_PEOPLE, replace=False).tolist())
    calib = np.zeros(len(df), dtype=bool)
    calib[~validation] = train._calib_mask(individual[~validation], chrom[~validation], SEED)
    return validation, np.isin(chrom, UNSEEN_CHROMS), calib


def _calibrators(raw, y):
    cals = {}
    for i in range(M.N_CLASSES):
        if int((y == i).sum()) in (0, y.size):
            cals[i] = None
            continue
        cals[i] = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(
            raw[:, i], (y == i).astype(float))
    return cals


def _calibrated(cals, raw):
    cols = [raw[:, i] if cals[i] is None else cals[i].predict(raw[:, i]) for i in range(M.N_CLASSES)]
    cal = np.clip(np.column_stack(cols), 0.0, 1.0)
    s = cal.sum(axis=1)
    out = np.full_like(cal, 1.0 / M.N_CLASSES)
    out[s > 0] = cal[s > 0] / s[s > 0, None]
    return out


def _staged(generator, keep):
    return {i: np.asarray(v).copy() for i, v in enumerate(generator, start=1) if i in keep}


def curves_for_regime(data_dir, regime, direction_iterations, q_median_iterations):
    """Fits one regime once and returns its curves (see the module docstring) as a JSON-able dict."""
    t0 = time.time()
    df = load_regime(data_dir, regime)
    branch = features.GENOTYPING_REGIME_BRANCH[regime]
    validation, unseen, calib = split(df)
    rep = df["representative"].to_numpy(bool)
    individual = df["individual"].to_numpy().astype(object)
    chrom = df["chrom"].astype(str).to_numpy().astype(object)
    always = np.isin(individual, train.ALWAYS_TRAIN_INDIVIDUALS)
    train_pool = np.where(~validation & ~calib & ~unseen)[0]
    calib_rows = train._cap(np.where(calib & ~unseen & rep)[0], CALIB_CAP, SEED + 2)
    qreg, dmodel, _, Xca = train.fit_heads(
        df, train_pool, calib_rows, np.where(always, individual + "|" + chrom, individual), branch, TRAIN_CAP,
        (direction_iterations, q_median_iterations), "%s curves" % regime)
    checkpoints = [k for k in CHECKPOINTS if k <= direction_iterations and 2 * k <= q_median_iterations]
    rng = np.random.default_rng(SEED + 7)
    def sample(idx):
        return np.sort(rng.choice(idx, CELL_CAP, replace=False)) if idx.size > CELL_CAP else idx
    # The training cell is drawn from the rows the direction head was actually fit on (most of the
    # pool is never fit: the head trains on a TRAIN_CAP quota sample of it). That quota sample
    # over-represents rare, error-prone calls, so each row is weighted by 1 / (its quota cell's inclusion
    # rate among the pool's representative rows), which puts the training cell on the same
    # representative scale as the validation cells.
    fit_rows = train.direction_training_rows(df, train_pool, TRAIN_CAP)
    fit_rep = fit_rows[rep[fit_rows]]
    motif, eh_all = df["motif_size"].to_numpy(float), df["eh"].to_numpy(float)
    pool_cells = dataset.quota_cells(motif, eh_all, train_pool[rep[train_pool]])
    fit_cells = dataset.quota_cells(motif, eh_all, fit_rep)
    inclusion = {c: (fit_cells == c).sum() / (pool_cells == c).sum() for c in np.unique(fit_cells)}
    cells = {"train": sample(fit_rep),
             "val_known": sample(np.where(validation & ~unseen & rep)[0]),
             "val_new": sample(np.where(validation & unseen & rep)[0])}
    train_weight = {"train": 1.0 / np.vectorize(inclusion.get)(dataset.quota_cells(motif, eh_all, cells["train"]))}
    clf = dmodel["clf"]
    raw_calib = _staged(clf.staged_predict_proba(Xca), set(checkpoints))
    y_calib = df["dir_code"].to_numpy(int)[calib_rows]
    cals = {k: _calibrators(raw_calib[k], y_calib) for k in checkpoints}
    gated, log_loss, uncorrected = {}, {}, {}
    for cell, idx in cells.items():
        sub = df.iloc[idx]
        X, _ = features.build_matrix(sub, branch)
        y = sub["dir_code"].to_numpy(int)
        eh, true = sub["eh"].to_numpy(float), sub["true"].to_numpy(float)
        homo = sub["motif_size"].to_numpy() == 1
        w = train_weight.get(cell, np.ones(len(idx)))
        raw = _staged(clf.staged_predict_proba(X), set(checkpoints))
        t = _staged(qreg.staged_predict(X), {2 * k for k in checkpoints})
        for s, _ in STRATA:
            m = homo if s == "homo" else ~homo
            uncorrected.setdefault(cell, {})[s] = float(np.average(np.abs(eh - true)[m], weights=w[m]))
        for k in checkpoints:
            proba = _calibrated(cals[k], raw[k])
            p_ok = M.round_like_emitted(proba[:, 0])
            lcf = M.round_like_emitted(np.exp(t[2 * k]))
            err = np.abs(np.where(p_ok < 0.5, np.round(eh / lcf), eh) - true)
            p_true = np.clip(proba[np.arange(len(y)), y], 1e-6, 1.0)
            log_loss.setdefault(cell, []).append([k, float(np.average(-np.log(p_true), weights=w))])
            for s, _ in STRATA:
                m = homo if s == "homo" else ~homo
                gated.setdefault(cell, {}).setdefault(s, []).append([k, float(np.average(err[m], weights=w[m]))])
    return {"regime": regime, "direction_iterations": direction_iterations,
            "q_median_iterations": q_median_iterations, "regularization": {k: v for k, v in M._GBM_KWARGS.items()},
            "seconds": round(time.time() - t0), "cells_n": {c: int(len(i)) for c, i in cells.items()},
            "gated_mae": gated, "log_loss": log_loss, "uncorrected_mae": uncorrected}


def smallest_within(result, tolerance):
    """Returns the smallest checkpoint whose ``val_known`` gated MAE, for both strata, is within
    ``tolerance`` (a fraction) of the longest checkpoint's."""
    curve = result["gated_mae"]["val_known"]
    best = {s: curve[s][-1][1] for s, _ in STRATA}
    for i, (k, _) in enumerate(curve["nonhomo"]):
        if all(curve[s][i][1] <= best[s] * (1 + tolerance) for s, _ in STRATA):
            return k
    return curve["nonhomo"][-1][0]


def _figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"train": ("rows the direction head was fit on", "#7f7f7f"), "val_known": ("new people, known loci", "#1f77b4"),
              "val_new": ("new people, unseen loci", "#d62728")}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (s, label) in zip(axes[:2], STRATA):
        for cell, (cell_label, color) in styles.items():
            ks, ys = zip(*result["gated_mae"][cell][s])
            ax.plot(ks, ys, "-o", ms=3, color=color, label=cell_label)
            ax.axhline(result["uncorrected_mae"][cell][s], color=color, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("direction iterations d (q-median: 2d)")
        ax.set_ylabel("gated MAE (repeats)")
        ax.set_title("%s, %s" % (result["regime"], label))
        ax.grid(alpha=0.3)
        ax.secondary_xaxis("top", functions=(lambda d: 5 * d * GZ_BYTES_PER_TREE / 1e6,
                                             lambda mb: mb * 1e6 / GZ_BYTES_PER_TREE / 5)).set_xlabel("~MB gzipped")
    for cell, (cell_label, color) in styles.items():
        ks, ys = zip(*result["log_loss"][cell])
        axes[2].plot(ks, ys, "-o", ms=3, color=color, label=cell_label)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("direction iterations d")
    axes[2].set_ylabel("direction log loss")
    axes[2].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot(results, out_html, tolerance):
    """Writes an HTML page with one curve figure per result and the size each would pick."""
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Model Size Curves</title><style>"
             "body{font-family:Helvetica,Arial,sans-serif;margin:24px;background:#fff;color:#222}"
             "img{max-width:100%}</style></head><body><h1>Accuracy vs model size</h1>"
             "<p>Gated MAE at checkpoints of one fixed-length fit per regime (see model_size_curves.py). "
             "Dotted lines: uncorrected ExpansionHunter. Top axis: approximate gzipped size of that "
             "regime's part of the model.</p>"]
    for r in results:
        k = smallest_within(r, tolerance)
        parts.append("<h2>%s</h2><p>Smallest checkpoint within %.0f%% of the longest fit on new people at "
                     "known loci: <b>%d</b> direction / %d q-median iterations.</p>"
                     % (r["regime"], 100 * tolerance, k, 2 * k))
        parts.append("<img src='data:image/png;base64,%s'>" % _figure(r))
    parts.append("</body></html>")
    with open(out_html, "w") as f:
        f.write("".join(parts))
    print("wrote %s" % out_html, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--regime", choices=features.GENOTYPING_REGIMES, help="fit this regime and append its curves")
    parser.add_argument("--direction-iterations", type=int, default=4000)
    parser.add_argument("--q-median-iterations", type=int, default=8000)
    parser.add_argument("--results", required=True, help="JSON-lines file the curves are appended to / read from")
    parser.add_argument("--plot", help="write an HTML page of every result in --results")
    parser.add_argument("--pick", action="store_true", help="print the size each regime's curve picks")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.regime:
        result = curves_for_regime(args.data_dir, args.regime, args.direction_iterations, args.q_median_iterations)
        with open(args.results, "a") as f:
            f.write(json.dumps(result) + "\n")
        print("appended %s curves to %s (%ds)" % (args.regime, args.results, result["seconds"]), flush=True)
    results = [json.loads(line) for line in open(args.results)] if (args.plot or args.pick) else []
    if args.pick:
        for r in results:
            k = smallest_within(r, args.tolerance)
            print("%s: %d direction / %d q-median iterations" % (r["regime"], k, 2 * k))
    if args.plot:
        plot(results, args.plot, args.tolerance)


if __name__ == "__main__":
    main()
