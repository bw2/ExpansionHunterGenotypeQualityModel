"""Three-class direction classifier for the EH genotype-quality ``direction`` head.

Predicts whether an EH allele call is correct (``OK``), an over-call (``TOO_LONG``)
or an under-call (``TOO_SHORT``) relative to the truth (see ``SPEC.md`` /
``GENOTYPE_QUALITY_METRICS_PLAN.md``). The label is ``dir_code`` written by
``build_dataset`` with the fixed coding ``OK=0, TOO_LONG=1, TOO_SHORT=2``; probability
vectors are always returned in that column order: ``[P_OK, P_TOO_LONG, P_TOO_SHORT]``.

Engine: a multinomial ``HistGradientBoostingClassifier`` (``loss='log_loss'``).
sklearn's classifier does NOT accept ``monotonic_cst`` for >2 classes, so none is
passed. Calibration: one per-class one-vs-rest ``IsotonicRegression`` fit on the
calibration set, applied per column and then RENORMALIZED back to the simplex.

Determinism (SPEC sec "Determinism"): every estimator is built with
``random_state=SEED``. Early stopping is a MANUAL ``warm_start`` monitor loop
scored on the externally supplied calibration set -- never sklearn's internal
``early_stopping=True``, whose random ``validation_fraction`` split would leak
loci/coverages across the chromosome-clean folds.

The functions here are pure apart from fitting the estimators they return; no
module-level mutable state.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression


SEED = 20260616

# Fixed class coding (matches build_dataset ``dir_code``); also the output column
# order of every probability vector.
OK, TOO_LONG, TOO_SHORT = 0, 1, 2
N_CLASSES = 3

# Fixed, principled capacity/regularization (NOT tuned on test); SPEC sec model_direction.
_LEARNING_RATE = 0.05
_MAX_LEAF_NODES = 31
_MIN_SAMPLES_LEAF = 300
_L2_REGULARIZATION = 1.0
_MAX_BINS = 255

# Calib-loss improvement smaller than this counts as "no improvement".
_IMPROVE_TOL = 1e-6


class _PassthroughCalibrator:
    """No-op calibrator for a class with no positive (or no negative) calib examples.

    Fitting ``IsotonicRegression`` on an all-equal one-vs-rest target collapses it to a
    constant 0 (or 1); after the simplex renormalization in ``predict_proba`` that zeros
    out (or saturates) the class for EVERY row -- even rows the raw classifier assigns it
    real mass. A chrom-clean fold with few calibration chroms can easily lack a class, so
    instead of destroying it we pass the raw classifier probability through unchanged.
    """

    @staticmethod
    def predict(x):
        return np.asarray(x, dtype=float)

# Probability clip for the log of a multinomial loss (avoids log(0)).
_EPS = 1e-15


def multiclass_log_loss(y_true, proba, sample_weight=None):
    """Returns the (optionally weighted) mean multinomial cross-entropy log loss.

    For row ``i`` with integer label ``y_true[i]`` the per-sample loss is
    ``-log(proba[i, y_true[i]])``; this returns the mean over all rows, or the
    ``sample_weight``-weighted mean when weights are given. Probabilities are
    clipped to ``[_EPS, 1]`` before the log so a confident wrong call does not
    produce a non-finite loss. Matches ``sklearn.metrics.log_loss`` for 3 classes
    (with ``sample_weight``).

    Args:
        y_true: Array-like of integer class labels in ``{0, 1, 2}``.
        proba: Array of shape ``(n, 3)`` of class probabilities in
            ``[P_OK, P_TOO_LONG, P_TOO_SHORT]`` order.
        sample_weight: Optional per-sample weights; ``None`` => unweighted mean.

    Returns:
        The mean log loss as a float.
    """
    proba = np.clip(np.asarray(proba, dtype=float), _EPS, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    per_sample = -np.log(proba[np.arange(len(y_true)), y_true])
    if sample_weight is None:
        return float(np.mean(per_sample))
    sample_weight = np.asarray(sample_weight, dtype=float)
    return float(np.sum(sample_weight * per_sample) / np.sum(sample_weight))


def make_direction_classifier():
    """Builds an unfitted multinomial ``HistGradientBoostingClassifier``.

    All capacity/regularization knobs are the fixed SPEC defaults and
    ``random_state=SEED`` is set for determinism. ``early_stopping`` is disabled
    so the only stopping signal is the external calib monitor in
    ``fit_with_early_stop``. No ``monotonic_cst`` is passed -- sklearn rejects it
    for multiclass classification.

    Returns:
        An unfitted ``HistGradientBoostingClassifier``.
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=_LEARNING_RATE,
        max_leaf_nodes=_MAX_LEAF_NODES,
        min_samples_leaf=_MIN_SAMPLES_LEAF,
        l2_regularization=_L2_REGULARIZATION,
        max_bins=_MAX_BINS,
        early_stopping=False,
        random_state=SEED,
    )


def _proba_in_class_order(clf, X):
    """Returns ``clf.predict_proba`` reindexed to fixed ``[P_OK, P_TOO_LONG, P_TOO_SHORT]``.

    ``predict_proba`` returns columns in ``clf.classes_`` order, which omits any
    class absent from the training labels. This scatters each column into its
    ``dir_code`` position, leaving missing classes as all-zero columns, so the
    output is always shape ``(n, 3)`` in the canonical order.

    Args:
        clf: A fitted ``HistGradientBoostingClassifier``.
        X: Feature matrix.

    Returns:
        Array of shape ``(n, 3)``.
    """
    raw = clf.predict_proba(X)
    full = np.zeros((raw.shape[0], N_CLASSES))
    for col, cls in enumerate(clf.classes_):
        full[:, int(cls)] = raw[:, col]
    return full


def fit_with_early_stop(X, y, X_calib, y_calib, max_total_iter=500, step=25, patience=3,
                        sample_weight=None, sample_weight_calib=None):
    """Fits the classifier with a manual ``warm_start`` early-stopping loop.

    Builds a single ``warm_start=True`` classifier and repeatedly grows
    ``max_iter`` by ``step``, refitting (which appends ``step`` more boosting
    iterations) and scoring multinomial log loss on the calib set after each
    step. Training stops when the calib loss has not improved for ``patience``
    consecutive steps, or when ``max_total_iter`` is reached.

    Best-iteration recovery: rather than keep the (over-trained) warm-start
    classifier, this refits a FRESH classifier with ``max_iter`` fixed to the
    best observed iteration. Boosting is deterministic given ``random_state``,
    the binning and the data, so the fresh fit reproduces exactly the trees the
    warm-start classifier held at that iteration.

    Args:
        X: Training feature matrix.
        y: Training labels (``dir_code`` in ``{0, 1, 2}``).
        X_calib: Calibration feature matrix (held-out chroms; never test).
        y_calib: Calibration labels.
        max_total_iter: Hard cap on boosting iterations.
        step: Iterations added per monitor step.
        patience: Stop after this many steps without calib improvement.
        sample_weight: Optional per-row training weights (e.g. equal-per-bin
            weights from ``size_bins.bin_weights``); also weights the monitored
            train loss.
        sample_weight_calib: Optional per-row calib weights; weights the calib
            log-loss the early-stopping decision uses, matching the weighted fit.

    Returns:
        A ``(fitted_clf_at_best_iter, history)`` tuple, where ``history`` is a
        list of ``{"n_iter", "train_loss", "calib_loss"}`` dicts (one per monitor
        step) for ``diagnostics.py``.
    """
    clf = make_direction_classifier()
    clf.set_params(warm_start=True)

    history = []
    best_loss = np.inf
    best_n_iter = step
    steps_without_improve = 0
    n_iter = 0

    while n_iter < max_total_iter:
        n_iter = min(n_iter + step, max_total_iter)
        clf.set_params(max_iter=n_iter)
        clf.fit(X, y, sample_weight=sample_weight)
        train_loss = multiclass_log_loss(y, _proba_in_class_order(clf, X), sample_weight)
        calib_loss = multiclass_log_loss(
            y_calib, _proba_in_class_order(clf, X_calib), sample_weight_calib)
        history.append({"n_iter": n_iter, "train_loss": train_loss, "calib_loss": calib_loss})
        if calib_loss < best_loss - _IMPROVE_TOL:
            best_loss = calib_loss
            best_n_iter = n_iter
            steps_without_improve = 0
        else:
            steps_without_improve += 1
            if steps_without_improve >= patience:
                break

    best_clf = make_direction_classifier()
    best_clf.set_params(warm_start=False, max_iter=best_n_iter)
    best_clf.fit(X, y, sample_weight=sample_weight)
    return best_clf, history


def train_direction(X, y, X_calib, y_calib, early_stop_kwargs=None,
                    sample_weight=None, sample_weight_calib=None):
    """Fits the early-stopped classifier plus per-class isotonic calibrators.

    The classifier is fit on ``(X, y)`` with early stopping monitored on the
    calib set. Then, on the SAME calib set, one ``IsotonicRegression`` is fit per
    class mapping the classifier's raw ``P(class)`` to the one-vs-rest binary
    outcome ``y_calib == class``; this de-biases each class column before the
    simplex renormalization done in ``predict_proba``. When calib weights are
    given the isotonic fits are weighted too, so calibration targets the same
    bin-equalized population as the classifier.

    Args:
        X: Training feature matrix.
        y: Training labels (``dir_code`` in ``{0, 1, 2}``).
        X_calib: Calibration feature matrix (held-out chroms; never test).
        y_calib: Calibration labels.
        early_stop_kwargs: Optional dict of overrides forwarded to
            ``fit_with_early_stop`` (e.g. ``max_total_iter``/``step`` for tests).
        sample_weight: Optional per-row training weights; ``None`` => unweighted.
        sample_weight_calib: Optional per-row calib weights (early stop + isotonic).

    Returns:
        ``{"clf": clf, "calibrators": {class_idx: IsotonicRegression or _PassthroughCalibrator},
        "history": history}`` (passthrough used for any class absent from the calib split).
    """
    clf, history = fit_with_early_stop(
        X, y, X_calib, y_calib, sample_weight=sample_weight,
        sample_weight_calib=sample_weight_calib, **(early_stop_kwargs or {}))
    raw_calib = _proba_in_class_order(clf, X_calib)
    y_calib = np.asarray(y_calib, dtype=int)
    calibrators = {}
    for idx in range(N_CLASSES):
        n_pos = int((y_calib == idx).sum())
        if n_pos == 0 or n_pos == y_calib.size:
            # Class absent from (or the only class in) the calib split: isotonic would
            # collapse to a constant, so pass the raw probability through unchanged.
            calibrators[idx] = _PassthroughCalibrator
            continue
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        calibrator.fit(raw_calib[:, idx], (y_calib == idx).astype(float),
                       sample_weight=sample_weight_calib)
        calibrators[idx] = calibrator
    return {"clf": clf, "calibrators": calibrators, "history": history}


def predict_proba(model, X):
    """Returns calibrated, simplex-renormalized class probabilities.

    Applies the classifier, maps each raw class column through its isotonic
    calibrator, clips to ``[0, 1]``, then renormalizes each row to sum to 1.
    Rows whose calibrated columns are all zero fall back to a uniform ``1/3``
    vector. NOTE: calibration is assessed AFTER this renormalization (the
    renormalization can perturb a per-class isotonic fit), so ``evaluate.py``
    must score reliability on these returned columns.

    Args:
        model: A dict from ``train_direction`` (``"clf"`` + ``"calibrators"``).
        X: Feature matrix.

    Returns:
        Array of shape ``(n, 3)`` in ``[P_OK, P_TOO_LONG, P_TOO_SHORT]`` order, rows
        summing to 1.
    """
    raw = _proba_in_class_order(model["clf"], X)
    calibrators = model["calibrators"]
    cal = np.clip(
        np.column_stack([calibrators[idx].predict(raw[:, idx]) for idx in range(N_CLASSES)]),
        0.0, 1.0)
    row_sums = cal.sum(axis=1)
    out = np.full_like(cal, 1.0 / N_CLASSES)
    nonzero = row_sums > 0
    out[nonzero] = cal[nonzero] / row_sums[nonzero, None]
    return out
