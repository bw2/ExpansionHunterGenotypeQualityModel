"""Quantile gradient-boosting regression for the EH genotype-quality ``q`` head.

Models the per-allele log-ratio ``t = log(eh) - log(true)`` with a set of
quantile regressors (see ``GENOTYPE_QUALITY_METRICS_PLAN.md`` / ``SPEC.md``).
One ``HistGradientBoostingRegressor(loss='quantile', quantile=a)`` is fit per
target quantile ``a in QUANTILES``; together they give a predictive distribution
over ``t`` from which a point estimate (``exp(median)``), a recovered truth
(``eh / exp(median)``) and 80% / 90% predictive intervals are derived.

Determinism (SPEC sec "Determinism"): every estimator is built with
``random_state=SEED``. Early stopping is done with a MANUAL ``warm_start``
monitor loop scored on an externally supplied calibration set -- never sklearn's
internal ``early_stopping=True``, whose random ``validation_fraction`` split
would leak loci/coverages across the chromosome-clean folds.

The functions here are pure apart from fitting the estimators they return; no
module-level mutable state.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


SEED = 20260616
QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]

# 80% interval uses the 0.1/0.9 pair, 90% uses 0.05/0.95.
_INTERVAL_QUANTILES = {0.8: (0.1, 0.9), 0.9: (0.05, 0.95)}

# Fixed, principled capacity/regularization (NOT tuned on test); SPEC sec model_q.
_LEARNING_RATE = 0.05
_MAX_LEAF_NODES = 31
_MIN_SAMPLES_LEAF = 300
_L2_REGULARIZATION = 1.0
_MAX_BINS = 255

# Calib-loss improvement smaller than this counts as "no improvement".
_IMPROVE_TOL = 1e-6


def pinball_loss(y_true, y_pred, quantile, sample_weight=None):
    """Returns the (optionally weighted) mean pinball (quantile) loss.

    For target quantile ``q`` the per-sample loss is ``q * d`` when the
    prediction is below the truth (``d = y_true - y_pred >= 0``) and
    ``(1 - q) * (-d)`` when it is above; this returns the mean over all samples,
    or the ``sample_weight``-weighted mean when weights are given. Matches
    ``sklearn.metrics.mean_pinball_loss`` (with ``sample_weight``).

    Args:
        y_true: Array-like of observed targets.
        y_pred: Array-like of predicted quantile values.
        quantile: The target quantile in ``(0, 1)``.
        sample_weight: Optional per-sample weights; ``None`` => unweighted mean.

    Returns:
        The mean pinball loss as a float.
    """
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    per_sample = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    if sample_weight is None:
        return float(np.mean(per_sample))
    sample_weight = np.asarray(sample_weight, dtype=float)
    return float(np.sum(sample_weight * per_sample) / np.sum(sample_weight))


def pinball_for_quantile(estimator, y_true, y_pred, sample_weight=None):
    """Default calib scorer: pinball loss matching ``estimator.quantile``.

    Reading the quantile off the estimator keeps ``fit_with_early_stop`` scoring
    each quantile head with its own matching loss without the caller having to
    bind it (SPEC: "the pinball/quantile loss matching the estimator's quantile").

    Args:
        estimator: A fitted ``HistGradientBoostingRegressor`` with ``loss='quantile'``.
        y_true: Array-like of observed targets.
        y_pred: Array-like of predictions from ``estimator``.
        sample_weight: Optional per-sample weights passed through to ``pinball_loss``.

    Returns:
        The mean pinball loss at the estimator's quantile as a float.
    """
    return pinball_loss(y_true, y_pred, estimator.quantile, sample_weight)


def make_quantile_estimator(quantile):
    """Builds an unfitted quantile ``HistGradientBoostingRegressor``.

    All capacity/regularization knobs are the fixed SPEC defaults and
    ``random_state=SEED`` is set for determinism. ``early_stopping`` is disabled
    so the only stopping signal is the external calib monitor in
    ``fit_with_early_stop``.

    Args:
        quantile: The target quantile in ``(0, 1)``.

    Returns:
        An unfitted ``HistGradientBoostingRegressor``.
    """
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=_LEARNING_RATE,
        max_leaf_nodes=_MAX_LEAF_NODES,
        min_samples_leaf=_MIN_SAMPLES_LEAF,
        l2_regularization=_L2_REGULARIZATION,
        max_bins=_MAX_BINS,
        early_stopping=False,
        random_state=SEED,
    )


def fit_with_early_stop(make_estimator, X, y, X_calib, y_calib,
                        max_total_iter=500, step=25, patience=3,
                        scorer=pinball_for_quantile,
                        sample_weight=None, sample_weight_calib=None):
    """Fits one estimator with a manual ``warm_start`` early-stopping loop.

    Builds a single ``warm_start=True`` estimator and repeatedly grows
    ``max_iter`` by ``step``, refitting (which appends ``step`` more boosting
    iterations) and scoring the calib set after each step. Training stops when
    the calib loss has not improved for ``patience`` consecutive steps, or when
    ``max_total_iter`` is reached.

    Best-iteration recovery: rather than keep the (over-trained) warm-start
    estimator, this refits a FRESH estimator with ``max_iter`` fixed to the best
    observed iteration. Because boosting is deterministic given ``random_state``,
    the binning, and the data, a fresh fit at ``best_n_iter`` reproduces exactly
    the trees the warm-start estimator held at that iteration -- so this is the
    model "at best iter", obtained deterministically and without rewinding.

    Args:
        make_estimator: Zero-arg callable returning a fresh unfitted estimator.
        X: Training feature matrix.
        y: Training targets.
        X_calib: Calibration feature matrix (held-out chroms; never test).
        y_calib: Calibration targets.
        max_total_iter: Hard cap on boosting iterations.
        step: Iterations added per monitor step.
        patience: Stop after this many steps without calib improvement.
        scorer: ``scorer(estimator, y_true, y_pred, sample_weight) -> float`` calib loss.
        sample_weight: Optional per-row training weights (e.g. equal-per-bin
            weights from ``size_bins.bin_weights``); also used to weight the
            monitored train loss.
        sample_weight_calib: Optional per-row calib weights; weights the calib
            loss the early-stopping decision is made on, so stopping reflects the
            same weighted objective as the fit.

    Returns:
        A ``(fitted_estimator_at_best_iter, history)`` tuple, where ``history``
        is a list of ``{"n_iter", "train_loss", "calib_loss"}`` dicts (one per
        monitor step) for ``diagnostics.py``.
    """
    estimator = make_estimator()
    estimator.set_params(warm_start=True)

    history = []
    best_loss = np.inf
    best_n_iter = step
    steps_without_improve = 0
    n_iter = 0

    while n_iter < max_total_iter:
        n_iter = min(n_iter + step, max_total_iter)
        estimator.set_params(max_iter=n_iter)
        estimator.fit(X, y, sample_weight=sample_weight)
        train_loss = scorer(estimator, y, estimator.predict(X), sample_weight)
        calib_loss = scorer(estimator, y_calib, estimator.predict(X_calib), sample_weight_calib)
        history.append({"n_iter": n_iter, "train_loss": train_loss,
                        "calib_loss": calib_loss})
        if calib_loss < best_loss - _IMPROVE_TOL:
            best_loss = calib_loss
            best_n_iter = n_iter
            steps_without_improve = 0
        else:
            steps_without_improve += 1
            if steps_without_improve >= patience:
                break

    best_estimator = make_estimator()
    best_estimator.set_params(warm_start=False, max_iter=best_n_iter)
    best_estimator.fit(X, y, sample_weight=sample_weight)
    return best_estimator, history


def train_q(X, t, X_calib, t_calib, quantiles=QUANTILES, early_stop_kwargs=None,
            sample_weight=None, sample_weight_calib=None):
    """Fits one early-stopped quantile regressor per quantile.

    Args:
        X: Training feature matrix.
        t: Training targets ``t = log(eh) - log(true)``.
        X_calib: Calibration feature matrix (held-out chroms).
        t_calib: Calibration targets.
        quantiles: Quantiles to fit; defaults to ``QUANTILES``.
        early_stop_kwargs: Optional dict of overrides forwarded to
            ``fit_with_early_stop`` (e.g. ``max_total_iter``/``step`` for tests).
        sample_weight: Optional per-row training weights (shared across quantile
            heads); ``None`` => unweighted.
        sample_weight_calib: Optional per-row calib weights for the early-stop
            decision.

    Returns:
        ``{"models": {q: estimator}, "histories": {q: history}}`` keyed by quantile.
    """
    early_stop_kwargs = early_stop_kwargs or {}
    models = {}
    histories = {}
    for a in quantiles:
        model, history = fit_with_early_stop(
            lambda a=a: make_quantile_estimator(a),
            X, t, X_calib, t_calib,
            sample_weight=sample_weight, sample_weight_calib=sample_weight_calib,
            **early_stop_kwargs)
        models[a] = model
        histories[a] = history
    return {"models": models, "histories": histories}


def predict_q(models, X):
    """Predicts every quantile, repairing quantile crossing per row.

    Quantile heads are fit independently and can cross; this sorts the predicted
    values within each row ascending and reassigns them to the ascending
    quantile order, which is the standard monotone rearrangement and leaves the
    marginal of each quantile unchanged.

    Args:
        models: ``{quantile: fitted_estimator}`` (e.g. ``train_q(...)["models"]``).
        X: Feature matrix.

    Returns:
        ``{quantile: yhat_array}`` with non-crossing predictions.
    """
    quantiles = sorted(models.keys())
    preds = np.column_stack([models[a].predict(X) for a in quantiles])
    preds = np.sort(preds, axis=1)
    return {a: preds[:, i] for i, a in enumerate(quantiles)}


def point_q(models, X):
    """Returns the point estimate of ``q = eh / true``: ``exp(median yhat)``.

    Args:
        models: ``{quantile: fitted_estimator}`` (must include ``0.5``).
        X: Feature matrix.

    Returns:
        Array of point ``q`` estimates.
    """
    predictions = predict_q(models, X)
    if 0.5 not in predictions:
        raise KeyError("point_q requires a 0.5 (median) quantile model")
    return np.exp(predictions[0.5])


def recover_true(models, X, eh):
    """Recovers the truth estimate ``true_pred = eh / point_q``.

    Args:
        models: ``{quantile: fitted_estimator}`` (must include ``0.5``).
        X: Feature matrix.
        eh: Array-like of EH allele calls (the label source).

    Returns:
        Array of recovered truth estimates.
    """
    return np.asarray(eh, dtype=float) / point_q(models, X)


def predictive_interval(models, X, level):
    """Returns the ``(lo, hi)`` predictive interval for ``t`` at a coverage level.

    Args:
        models: ``{quantile: fitted_estimator}``.
        X: Feature matrix.
        level: Coverage level; ``0.8`` -> (0.1, 0.9) pair, ``0.9`` -> (0.05, 0.95).

    Returns:
        A ``(lo_array, hi_array)`` tuple of ``t``-space bounds.
    """
    if level not in _INTERVAL_QUANTILES:
        raise ValueError(f"level must be one of {sorted(_INTERVAL_QUANTILES)}; got {level}")
    lo_q, hi_q = _INTERVAL_QUANTILES[level]
    predictions = predict_q(models, X)
    return predictions[lo_q], predictions[hi_q]
