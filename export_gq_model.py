"""Export the trained genotype-quality model to the EH C++ schema (.json.gz).

Fits the per-regime experts (q-median LCF head + 3-class direction head w/ per-class isotonic
calibrators) on the HG002+CHM pool with the SIM rows EXCLUDED (clean real-data-only model),
serializes each sklearn estimator into the schema that ehunter/genotype_quality/
GenotypeQualityModel.cpp parses (see GENOTYPE_QUALITY_CPP_INTEGRATION_PLAN.md), round-trip
verifies the serialized trees/softmax/isotonic reproduce sklearn's predictions, and writes the
gzipped JSON. Also persists the fitted models (results/models/lcf_clean_<regime>.joblib) for the
held-out benchmark.

Serialization facts (validated against sklearn 1.6.1):
- HistGradientBoosting leaf values already include the learning-rate shrinkage, so a prediction is
  baseline + sum_trees(eval) with no extra factor (regressor matches .predict, classifier
  softmax(raw) matches .predict_proba to ~1e-16).
- node fields: is_leaf / value / feature_idx / num_threshold / missing_go_to_left / left / right.
- direction classifier classes_ == [0,1,2] == [OK, TOO_LONG, TOO_SHORT]; _baseline_prediction and
  _predictors[i] are in that class order.

Coding rules: print(), Google docstrings, no type hints.
"""

import argparse
import gzip
import json
import os

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import features
import model_direction as MD
import model_q as MQ

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
REGIMES = ["fast", "full_spanning", "full_nonspanning"]
MODELS_DIR = os.path.join(HERE, "results", "models")
DEFAULT_OUT = os.path.join(
    HERE, "model",
    "genotype_quality_model_from_HG002_and_CHM1_CHM13.20260624.json.gz")


def _calib_mask(chrom, seed):
    """Boolean mask holding out ~10% of chromosomes (seeded) for early-stopping calib."""
    ch = np.array(sorted(set(chrom[pd.notna(chrom)].tolist())))
    n = max(1, int(round(0.1 * len(ch))))
    cc = set(np.random.default_rng(seed).choice(ch, n, replace=False).tolist())
    return np.isin(chrom.astype(object), list(cc))


def _cap(idx, cap, seed):
    """Seeded subsample of an index array to ``cap`` rows."""
    if cap and idx.size > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def fit_regime(regime, train_cap):
    """Fits q-median + direction for one regime on the SIM-EXCLUDED HG002+CHM pool.

    Returns (q_regressor, direction_model_dict, branch, feature_names, X_check) where X_check is a
    small held-out (calib) feature matrix used only for round-trip verification.
    """
    branch = "fast" if regime == "fast" else "full"
    src = "fast" if regime == "fast" else "full"
    parquet = os.path.join(HERE, "data", "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"t", "dir_code", "chrom", "regime", "sample"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    df = pd.read_parquet(parquet, columns=cols)
    df = df[(df["regime"] == regime) & (df["sample"] != "sim")]
    print("  [%s] pool rows (sim excluded)=%d" % (regime, len(df)), flush=True)

    calib_mask = _calib_mask(df["chrom"].to_numpy(), SEED)
    tr = _cap(np.where(~calib_mask)[0], train_cap, SEED)
    ca = _cap(np.where(calib_mask)[0], max(1, train_cap // 5), SEED + 1)
    Xtr, names = features.build_matrix(df.iloc[tr], branch)
    Xca, _ = features.build_matrix(df.iloc[ca], branch)
    ttr = df["t"].to_numpy(float)[tr]
    tca = df["t"].to_numpy(float)[ca]
    ytr = df["dir_code"].to_numpy(int)[tr]
    yca = df["dir_code"].to_numpy(int)[ca]
    del df
    print("  [%s] fit q(median)+direction on %d train / %d calib ..." % (regime, tr.size, ca.size),
          flush=True)
    qreg = MQ.train_q(Xtr, ttr, Xca, tca, quantiles=[0.5])["models"][0.5]
    dmodel = MD.train_direction(Xtr, ytr, Xca, yca)
    return qreg, dmodel, branch, list(names), Xca[:5000]


def ser_tree(pred):
    """Serializes one sklearn TreePredictor to the schema's flat node array."""
    out = []
    for nd in pred.nodes:
        if nd["is_leaf"]:
            out.append({"leaf": True, "value": float(nd["value"])})
        else:
            out.append({"feature": int(nd["feature_idx"]), "threshold": float(nd["num_threshold"]),
                        "missing_left": bool(nd["missing_go_to_left"]),
                        "left": int(nd["left"]), "right": int(nd["right"])})
    return {"nodes": out}


def ser_qhead(qreg):
    """Serializes the q-median regressor to a QHEAD (predicts t; LCF = exp(t))."""
    return {"baseline": float(np.ravel(qreg._baseline_prediction)[0]),
            "trees": [ser_tree(p[0]) for p in qreg._predictors]}


def ser_iso(cal):
    """Serializes one isotonic calibrator (or passthrough => empty knots)."""
    if not hasattr(cal, "X_thresholds_"):  # _PassthroughCalibrator
        return {"x": [], "y": [], "increasing": True}
    return {"x": [float(v) for v in cal.X_thresholds_],
            "y": [float(v) for v in cal.y_thresholds_], "increasing": True}


def ser_dirhead(dmodel):
    """Serializes the direction classifier + isotonic calibrators to a DIRHEAD."""
    clf = dmodel["clf"]
    cals = dmodel["calibrators"]
    if list(clf.classes_) != [0, 1, 2]:
        raise RuntimeError("direction classes_ != [0,1,2] (a class was absent): %s" % clf.classes_)
    bl = np.ravel(clf._baseline_prediction)
    if bl.size != 3:
        raise RuntimeError("direction baseline size %d != 3" % bl.size)
    trees = []
    for it in clf._predictors:
        if len(it) != 3:
            raise RuntimeError("direction iter has %d trees != 3" % len(it))
        trees.append([ser_tree(it[0]), ser_tree(it[1]), ser_tree(it[2])])
    return {"baseline": [float(bl[0]), float(bl[1]), float(bl[2])],
            "trees": trees,
            "calibrators": [ser_iso(cals[0]), ser_iso(cals[1]), ser_iso(cals[2])]}


# ---- round-trip inference reimplemented from the SERIALIZED dict (mirrors the C++) ----

def _tree_eval(nodes, x):
    n = 0
    while not nodes[n].get("leaf", False):
        nd = nodes[n]
        xv = x[nd["feature"]]
        if np.isnan(xv):
            n = nd["left"] if nd["missing_left"] else nd["right"]
        else:
            n = nd["left"] if xv <= nd["threshold"] else nd["right"]
    return nodes[n]["value"]


def _iso_apply(iso, v):
    x, y = iso["x"], iso["y"]
    if not x:
        return v
    if v <= x[0]:
        return y[0]
    if v >= x[-1]:
        return y[-1]
    hi = int(np.searchsorted(x, v, side="right"))
    lo = hi - 1
    span = x[hi] - x[lo]
    if span <= 0:
        return y[lo]
    return y[lo] + (v - x[lo]) / span * (y[hi] - y[lo])


def _qhead_lcf(q, X):
    base = q["baseline"]
    return np.exp(np.array([base + sum(_tree_eval(t["nodes"], X[r]) for t in q["trees"])
                            for r in range(len(X))]))


def _dirhead_proba(d, X):
    bl = d["baseline"]
    out = np.zeros((len(X), 3))
    for r in range(len(X)):
        raw = np.array([bl[c] + sum(_tree_eval(tr[c]["nodes"], X[r]) for tr in d["trees"])
                        for c in range(3)])
        e = np.exp(raw - raw.max())
        p = e / e.sum()
        cal = np.clip([_iso_apply(d["calibrators"][c], p[c]) for c in range(3)], 0, 1)
        s = cal.sum()
        out[r] = cal / s if s > 0 else np.full(3, 1 / 3)
    return out


def verify(regime, qreg, dmodel, qj, dj, Xc):
    """Asserts the serialized dicts reproduce sklearn predictions on Xc."""
    lcf_skl = np.exp(qreg.predict(Xc))
    proba_skl = MD.predict_proba(dmodel, Xc)
    Xnp = np.asarray(Xc, dtype=float)  # serialized reimpl indexes features positionally
    lcf_ser = _qhead_lcf(qj, Xnp)
    proba_ser = _dirhead_proba(dj, Xnp)
    dl = float(np.max(np.abs(lcf_skl - lcf_ser)))
    dp = float(np.max(np.abs(proba_skl - proba_ser)))
    print("  [%s] verify n=%d  max|LCF diff|=%.2e  max|proba diff|=%.2e" % (regime, len(Xc), dl, dp),
          flush=True)
    if dl > 1e-6 or dp > 1e-6:
        raise RuntimeError("round-trip mismatch (%s): LCF %.2e proba %.2e" % (regime, dl, dp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cap", type=int, default=1_000_000)
    ap.add_argument("--out", default=os.path.normpath(DEFAULT_OUT))
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    regimes_json = {}
    feat_names = {}
    for regime in REGIMES:
        print("==== %s ====" % regime, flush=True)
        qreg, dmodel, branch, names, Xc = fit_regime(regime, args.train_cap)
        feat_names[branch] = names
        qj = ser_qhead(qreg)
        dj = ser_dirhead(dmodel)
        verify(regime, qreg, dmodel, qj, dj, Xc)
        regimes_json[regime] = {"q_median": qj, "direction": dj}
        joblib.dump({"q": {0.5: qreg}, "dir": dmodel, "branch": branch, "regime": regime},
                    os.path.join(MODELS_DIR, "lcf_clean_%s.joblib" % regime))
        print("  [%s] %d q-trees, %d dir-iters" % (regime, len(qj["trees"]), len(dj["trees"])), flush=True)

    model = {"format_version": 1,
             "feature_names": {"fast": feat_names["fast"], "full": feat_names["full"]},
             "regimes": regimes_json}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    blob = json.dumps(model, separators=(",", ":")).encode()
    with gzip.open(args.out, "wb", compresslevel=9) as f:
        f.write(blob)
    # confirm it reloads
    with gzip.open(args.out, "rb") as f:
        json.loads(f.read())
    print("\nwrote %s (%d features fast / %d full, json %d bytes, gz %d bytes)"
          % (args.out, len(feat_names["fast"]), len(feat_names["full"]), len(blob),
             os.path.getsize(args.out)), flush=True)


if __name__ == "__main__":
    main()
