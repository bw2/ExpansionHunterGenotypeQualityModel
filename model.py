"""The two genotype-quality model heads, their serialization, and round-trip checks.

Each genotyping_regime expert is two scikit-learn ``HistGradientBoosting`` heads:

- **q-median** (``train_q_median``) -- a quantile regressor on
  ``t = log(eh) - log(true)``; the length-correction factor is ``LCF = exp(t)``.
- **direction** (``train_direction``) -- a 3-class classifier
  ``[pOk, pTooLong, pTooShort]`` plus one per-class isotonic calibrator.

Both heads use fixed, principled capacity/regularization (NOT tuned on test) and a
MANUAL ``warm_start`` early-stopping loop scored on an externally supplied,
chromosome-clean calibration set -- never sklearn's internal ``early_stopping``,
whose random validation split would leak loci across the chromosome-clean folds.
Every estimator is built with ``random_state=SEED`` for determinism.

``serialize_genotyping_regime`` emits each head in the C++ schema (``GenotypeQualityModel.cpp``),
and ``verify_genotyping_regime`` re-implements inference from the serialized dict to assert it
reproduces sklearn's predictions before the model is written.

Serialization facts (validated against sklearn 1.6.1): HistGradientBoosting leaf
values already include the learning-rate shrinkage, so a prediction is
``baseline + sum_trees(eval)`` with no extra factor; the regressor matches
``.predict`` and ``softmax(raw)`` matches ``.predict_proba`` to ~1e-16.

Pure apart from fitting the estimators returned; no module-level mutable state.
"""

import gzip
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

import features

SEED = 20260616
N_CLASSES = 3

# Fixed capacity/regularization shared by both heads.
_GBM_KWARGS = dict(
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=300,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=False,
    random_state=SEED,
)

# Early-stop monitor: grow max_iter by this step, stop after this many steps with
# no calib improvement, never exceed this many boosting iterations.
_STEP = 25
_PATIENCE = 3
_MAX_TOTAL_ITER = 500
_IMPROVE_TOL = 1e-6


# --- losses (used by the early-stop monitor) ------------------------------

def _pinball_loss(y_true, y_pred, quantile):
    """Mean pinball (quantile) loss at ``quantile``."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def _log_loss(y_true, proba):
    """Mean multinomial cross-entropy on probabilities clipped to avoid ``log(0)``."""
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    return float(np.mean(-np.log(proba[np.arange(len(y_true)), y_true])))


def _fit_early_stop(make_estimator, fit, score, X_calib):
    """Runs the shared warm-start early-stopping loop and returns the best estimator.

    Builds one ``warm_start=True`` estimator, grows ``max_iter`` by ``_STEP`` and
    refits (appending iterations) until the calib score has not improved for
    ``_PATIENCE`` steps, then refits a FRESH estimator fixed at the best iteration.
    Boosting is deterministic given ``random_state`` + data, so the fresh fit
    reproduces exactly the trees the warm-start held at that iteration.

    Args:
        make_estimator: Zero-arg callable returning a fresh unfitted estimator.
        fit: ``fit(estimator)`` -- fits it on the (closed-over) training data.
        score: ``score(estimator) -> calib_loss`` lower-is-better.
        X_calib: Unused here; kept so callers document the calib dependency.

    Returns:
        The fitted estimator at the best observed iteration.
    """
    est = make_estimator()
    est.set_params(warm_start=True)
    best_loss, best_n_iter, stalled, n_iter = np.inf, _STEP, 0, 0
    while n_iter < _MAX_TOTAL_ITER:
        n_iter = min(n_iter + _STEP, _MAX_TOTAL_ITER)
        est.set_params(max_iter=n_iter)
        fit(est)
        loss = score(est)
        if loss < best_loss - _IMPROVE_TOL:
            best_loss, best_n_iter, stalled = loss, n_iter, 0
        else:
            stalled += 1
            if stalled >= _PATIENCE:
                break
    best = make_estimator()
    best.set_params(warm_start=False, max_iter=best_n_iter)
    fit(best)
    return best


# --- q-median head --------------------------------------------------------

def train_q_median(X, t, X_calib, t_calib):
    """Fits the early-stopped median (0.5-quantile) regressor of ``t``.

    Returns:
        A fitted ``HistGradientBoostingRegressor`` (``loss='quantile'``, q=0.5).
    """
    def make():
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **_GBM_KWARGS)
    return _fit_early_stop(
        make,
        fit=lambda e: e.fit(X, t),
        score=lambda e: _pinball_loss(t_calib, e.predict(X_calib), 0.5),
        X_calib=X_calib)


def predict_lcf(qreg, X):
    """Returns the per-row length-correction factor ``LCF = exp(t_median)``."""
    return np.exp(qreg.predict(X))


# --- direction head -------------------------------------------------------

def _proba_in_class_order(clf, X):
    """Returns ``clf.predict_proba`` scattered into fixed ``[pOk, pTooLong, pTooShort]``.

    Columns for classes absent from the training labels are left all-zero, so the
    output is always shape ``(n, 3)`` in canonical order.
    """
    raw = clf.predict_proba(X)
    full = np.zeros((raw.shape[0], N_CLASSES))
    for col, cls in enumerate(clf.classes_):
        full[:, int(cls)] = raw[:, col]
    return full


def train_direction(X, y, X_calib, y_calib):
    """Fits the early-stopped 3-class classifier plus per-class isotonic calibrators.

    The classifier is fit with early stopping monitored on the calib set; then one
    ``IsotonicRegression`` per class maps its raw one-vs-rest probability to the
    empirical outcome on the calib set. A class absent from (or the only class in)
    the calib split gets a pass-through calibrator so the simplex renormalization in
    ``predict_proba`` does not zero it out.

    Returns:
        ``{"clf": clf, "calibrators": {idx: IsotonicRegression or None}}`` (None ==
        pass-through).
    """
    def make():
        return HistGradientBoostingClassifier(loss="log_loss", **_GBM_KWARGS)
    clf = _fit_early_stop(
        make,
        fit=lambda e: e.fit(X, y),
        score=lambda e: _log_loss(y_calib, _proba_in_class_order(e, X_calib)),
        X_calib=X_calib)

    raw_calib = _proba_in_class_order(clf, X_calib)
    y_calib = np.asarray(y_calib, dtype=int)
    calibrators = {}
    for idx in range(N_CLASSES):
        n_pos = int((y_calib == idx).sum())
        if n_pos == 0 or n_pos == y_calib.size:
            calibrators[idx] = None  # pass-through
            continue
        cal = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        cal.fit(raw_calib[:, idx], (y_calib == idx).astype(float))
        calibrators[idx] = cal
    return {"clf": clf, "calibrators": calibrators}


def predict_proba(model, X):
    """Returns calibrated, simplex-renormalized class probabilities, shape ``(n, 3)``.

    Applies each isotonic calibrator (pass-through where ``None``), clips to
    ``[0, 1]``, and renormalizes rows to sum to 1 (all-zero rows fall back to 1/3).
    Calibration is assessed AFTER this renormalization, so report code must score on
    these returned columns.
    """
    raw = _proba_in_class_order(model["clf"], X)
    cals = model["calibrators"]
    cols = [raw[:, i] if cals[i] is None else cals[i].predict(raw[:, i]) for i in range(N_CLASSES)]
    cal = np.clip(np.column_stack(cols), 0.0, 1.0)
    row_sums = cal.sum(axis=1)
    out = np.full_like(cal, 1.0 / N_CLASSES)
    nz = row_sums > 0
    out[nz] = cal[nz] / row_sums[nz, None]
    return out


# --- serialization to the C++ schema --------------------------------------

# sklearn HistGradientBoosting marks "missing-only" split nodes with a +/-inf num_threshold (every
# non-missing value routes one way, missing the other). Standard JSON has no Infinity, so serialize
# those as a large finite sentinel: every real feature value is far below it, so the `x <= threshold`
# routing is byte-for-byte identical, but the JSON is valid for any strict parser (incl. the C++ one).
_FINITE_CLAMP = 3.0e38  # < float32 max (3.4e38), so the C++ reader keeps it finite even as a 32-bit float


def _finite_threshold(v):
    """Clamps a non-finite split threshold to +/- ``_FINITE_CLAMP``; finite values pass through.

    NaN is intentionally left as-is so ``json.dumps(allow_nan=False)`` rejects it loudly rather than a
    bad threshold slipping through silently.
    """
    v = float(v)
    if v == np.inf:
        return _FINITE_CLAMP
    if v == -np.inf:
        return -_FINITE_CLAMP
    return v


def _ser_tree(pred):
    """Serializes one sklearn TreePredictor to the schema's flat node array."""
    nodes = []
    for nd in pred.nodes:
        if nd["is_leaf"]:
            nodes.append({"leaf": True, "value": float(nd["value"])})
        else:
            nodes.append({"feature": int(nd["feature_idx"]),
                          "threshold": _finite_threshold(nd["num_threshold"]),
                          "missing_left": bool(nd["missing_go_to_left"]),
                          "left": int(nd["left"]), "right": int(nd["right"])})
    return {"nodes": nodes}


def _ser_qhead(qreg):
    """Serializes the q-median regressor (predicts ``t``; ``LCF = exp(t)``)."""
    return {"baseline": float(np.ravel(qreg._baseline_prediction)[0]),
            "trees": [_ser_tree(p[0]) for p in qreg._predictors]}


def _ser_iso(cal):
    """Serializes one isotonic calibrator (``None`` pass-through => empty knots)."""
    if cal is None:
        return {"x": [], "y": [], "increasing": True}
    return {"x": [float(v) for v in cal.X_thresholds_],
            "y": [float(v) for v in cal.y_thresholds_], "increasing": True}


def _ser_dirhead(dmodel):
    """Serializes the direction classifier + isotonic calibrators."""
    clf = dmodel["clf"]
    if list(clf.classes_) != [0, 1, 2]:
        raise RuntimeError("direction classes_ != [0,1,2] (a class was absent): %s" % clf.classes_)
    bl = np.ravel(clf._baseline_prediction)
    if bl.size != N_CLASSES:
        raise RuntimeError("direction baseline size %d != 3" % bl.size)
    trees = []
    for it in clf._predictors:
        if len(it) != N_CLASSES:
            raise RuntimeError("direction iter has %d trees != 3" % len(it))
        trees.append([_ser_tree(it[0]), _ser_tree(it[1]), _ser_tree(it[2])])
    cals = dmodel["calibrators"]
    return {"classes": list(features.DIR_CLASS_NAMES),
            "baseline": [float(bl[0]), float(bl[1]), float(bl[2])],
            "trees": trees,
            "calibrators": [_ser_iso(cals[0]), _ser_iso(cals[1]), _ser_iso(cals[2])]}


def serialize_genotyping_regime(qreg, dmodel):
    """Returns the ``{"q_median", "direction"}`` JSON dict for one genotyping_regime expert."""
    return {"q_median": _ser_qhead(qreg), "direction": _ser_dirhead(dmodel)}


# --- round-trip inference from the SERIALIZED dict (mirrors the C++) -------

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
    return y[lo] if span <= 0 else y[lo] + (v - x[lo]) / span * (y[hi] - y[lo])


def _ser_lcf(qj, X):
    base = qj["baseline"]
    return np.exp(np.array([base + sum(_tree_eval(t["nodes"], X[r]) for t in qj["trees"])
                            for r in range(len(X))]))


def _ser_proba(dj, X):
    bl = dj["baseline"]
    out = np.zeros((len(X), N_CLASSES))
    for r in range(len(X)):
        raw = np.array([bl[c] + sum(_tree_eval(tr[c]["nodes"], X[r]) for tr in dj["trees"])
                        for c in range(N_CLASSES)])
        e = np.exp(raw - raw.max())
        p = e / e.sum()
        cal = np.clip([_iso_apply(dj["calibrators"][c], p[c]) for c in range(N_CLASSES)], 0, 1)
        s = cal.sum()
        out[r] = cal / s if s > 0 else np.full(N_CLASSES, 1.0 / N_CLASSES)
    return out


def verify_genotyping_regime(genotyping_regime, qreg, dmodel, genotyping_regime_json, X_check, tol=1e-6):
    """Asserts the serialized dict reproduces sklearn predictions on ``X_check``.

    Raises:
        RuntimeError: If max abs LCF or proba difference exceeds ``tol``.
    """
    Xnp = np.asarray(X_check, dtype=float)  # serialized reimpl indexes features positionally
    dl = float(np.max(np.abs(predict_lcf(qreg, X_check) - _ser_lcf(genotyping_regime_json["q_median"], Xnp))))
    dp = float(np.max(np.abs(predict_proba(dmodel, X_check) - _ser_proba(genotyping_regime_json["direction"], Xnp))))
    print("  [%s] round-trip verify n=%d  max|LCF diff|=%.2e  max|proba diff|=%.2e"
          % (genotyping_regime, len(X_check), dl, dp), flush=True)
    if dl > tol or dp > tol:
        raise RuntimeError("round-trip mismatch (%s): LCF %.2e proba %.2e" % (genotyping_regime, dl, dp))


# ---- vectorized inference from the SERIALIZED model (the format ExpansionHunter loads) ----
# These read the exact ``.json[.gz]`` the C++ consumes and evaluate it with no sklearn objects, so
# any consumer (e.g. the held-out benchmark) APPLIES the deployed model rather than re-fitting it.
# Trees are compiled to flat arrays once per genotyping regime, then evaluated vectorized over rows
# (one descent per tree); the logic mirrors ``_ser_lcf`` / ``_ser_proba`` and ``GenotypeQualityModel.cpp``.

def load(path):
    """Loads a serialized model dict from a ``.json`` / ``.json.gz`` file (gzip auto-detected)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return json.loads(f.read())


def _compile_tree(tree):
    """Compiles one serialized tree's node list to flat numpy arrays for vectorized descent."""
    nodes = tree["nodes"]
    n = len(nodes)
    is_leaf = np.zeros(n, bool)
    value = np.zeros(n, float)
    feat = np.zeros(n, np.intp)
    thr = np.zeros(n, float)
    missing_left = np.zeros(n, bool)
    left = np.zeros(n, np.intp)
    right = np.zeros(n, np.intp)
    for i, nd in enumerate(nodes):
        if nd.get("leaf", False):
            is_leaf[i] = True
            value[i] = nd["value"]
        else:
            feat[i] = nd["feature"]
            thr[i] = nd["threshold"]
            missing_left[i] = nd["missing_left"]
            left[i] = nd["left"]
            right[i] = nd["right"]
    return (is_leaf, value, feat, thr, missing_left, left, right)


def _eval_tree(arrs, X):
    """Vectorized leaf-value lookup for every row of ``X`` through one compiled tree."""
    is_leaf, value, feat, thr, missing_left, left, right = arrs
    node = np.zeros(X.shape[0], np.intp)
    while True:
        idx = np.nonzero(~is_leaf[node])[0]
        if idx.size == 0:
            break
        nd = node[idx]
        xv = X[idx, feat[nd]]
        go_left = np.where(np.isnan(xv), missing_left[nd], xv <= thr[nd])
        node[idx] = np.where(go_left, left[nd], right[nd])
    return value[node]


def compile_genotyping_regime(genotyping_regime_json):
    """Compiles one genotyping regime's serialized heads for fast repeated vectorized inference."""
    q = genotyping_regime_json["q_median"]
    d = genotyping_regime_json["direction"]
    return {
        "q_baseline": float(q["baseline"]),
        "q_trees": [_compile_tree(t) for t in q["trees"]],
        "d_baseline": np.asarray(d["baseline"], float),
        "d_trees": [[_compile_tree(tr) for tr in triple] for triple in d["trees"]],
        "calibrators": d["calibrators"],
    }


def predict_lcf_json(comp, X):
    """Returns ``LCF = exp(t)`` for ``X`` from a compiled genotyping regime (see ``predict_lcf``)."""
    X = np.asarray(X, dtype=float)
    t = np.full(X.shape[0], comp["q_baseline"])
    for arrs in comp["q_trees"]:
        t += _eval_tree(arrs, X)
    return np.exp(t)


def _iso_apply_vec(iso, v):
    """Vectorized isotonic apply (linear interp, clipped to the end knots; passthrough if no knots)."""
    return v if not iso["x"] else np.interp(v, iso["x"], iso["y"])


def predict_proba_json(comp, X):
    """Returns calibrated ``(n, 3)`` probabilities from a compiled genotyping regime (see ``predict_proba``)."""
    X = np.asarray(X, dtype=float)
    raw = np.tile(comp["d_baseline"], (X.shape[0], 1))
    for triple in comp["d_trees"]:
        for c in range(N_CLASSES):
            raw[:, c] += _eval_tree(triple[c], X)
    raw -= raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    p = e / e.sum(axis=1, keepdims=True)
    cal = np.clip(np.column_stack(
        [_iso_apply_vec(comp["calibrators"][c], p[:, c]) for c in range(N_CLASSES)]), 0.0, 1.0)
    row_sums = cal.sum(axis=1)
    out = np.full_like(cal, 1.0 / N_CLASSES)
    nz = row_sums > 0
    out[nz] = cal[nz] / row_sums[nz, None]
    return out
