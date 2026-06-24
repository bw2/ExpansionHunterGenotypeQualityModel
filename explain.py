"""Model-agnostic feature explanations for the EH genotype-quality models.

HistGBM has no reliable native feature importance, so all explanation here is
model-agnostic and computed on HELD-OUT rows (SPEC sec ``explain.py``):

- ``permutation_importance_plot`` -- ``sklearn.inspection.permutation_importance``
  on held-out ``X`` / ``y``, sorted most->least, drawn as a bar chart with std
  whiskers.
- ``partial_dependence_plots`` -- PDP + ICE (``PartialDependenceDisplay`` with
  ``kind='both'`` where the estimator supports it) for a chosen set of top
  features.
- ``pinball_scorer`` / ``ProbaEstimator`` -- shared scoring helpers so callers
  score permutation importance by the head's TRUE objective (quantile pinball
  loss for the q head; calibrated multinomial log loss for the direction head)
  instead of the estimator's default ``score`` (R^2 / accuracy).

The fitted estimator and the held-out matrices are passed IN; this module never
imports the model/feature modules, so it is independently testable. Determinism:
``permutation_importance`` is seeded with ``SEED``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import make_scorer, mean_pinball_loss

from plot_io import _save_svg_png


SEED = 20260616


def pinball_scorer(alpha=0.5):
    """Returns a ``make_scorer`` for the quantile pinball loss at ``alpha``.

    ``permutation_importance`` ranks features by the drop in this score when a column is shuffled.
    A bare regressor's default score is R^2, not the pipeline's quantile loss, so pass this scorer
    to score the q (quantile) head by the SAME objective it is trained on. ``greater_is_better=False``
    negates the loss, so an important feature still yields a POSITIVE permutation importance (the
    loss rises -> the negated score drops).
    """
    return make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)


class ProbaEstimator:
    """Minimal ``predict_proba`` adapter so a non-sklearn probability model can be permutation-scored.

    Wraps a ``proba_fn(X) -> (n, n_classes)`` callable (e.g. the isotonic-CALIBRATED,
    simplex-renormalized direction probabilities) into an estimator that
    ``permutation_importance`` can score with ``scoring="neg_log_loss"`` -- reproducing the
    deployed calibrated-log-loss objective instead of the raw classifier's default accuracy. Stays
    model-agnostic (no model/feature-module import) so this module remains independently testable.
    ``fit`` is a no-op present only to satisfy sklearn's "estimator must implement fit" check;
    ``permutation_importance`` only scores, never refits.
    """
    _estimator_type = "classifier"

    def __init__(self, proba_fn, n_classes):
        self._proba_fn = proba_fn
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self._proba_fn(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def permutation_importance_plot(estimator, X, y, feature_names, out_prefix,
                                scoring=None, n_repeats=10, seed=SEED):
    """Permutation importance on held-out data, drawn as a sorted bar chart.

    Runs ``sklearn.inspection.permutation_importance`` (which repeatedly shuffles
    one column at a time and measures the score drop) on the supplied held-out
    ``X`` / ``y``, then plots the per-feature mean importance most->least with the
    repeat-to-repeat std as error bars.

    Args:
        estimator: A fitted estimator with a ``predict`` (and ``score``) method.
        X: Held-out feature matrix (held-out chroms; never the training rows).
        y: Held-out targets.
        feature_names: Column names of ``X`` (same order as the columns).
        out_prefix: Path prefix for the saved ``.svg`` / ``.png`` figures.
        scoring: Optional scoring passed to ``permutation_importance`` (``None``
            uses the estimator's default ``score``).
        n_repeats: Number of shuffles per feature.
        seed: RNG seed for the shuffles (determinism).

    Returns:
        A list of ``(feature, importance_mean, importance_std)`` tuples sorted by
        importance mean, most important first.
    """
    result = permutation_importance(estimator, X, y, scoring=scoring,
                                    n_repeats=n_repeats, random_state=seed)
    order = np.argsort(result.importances_mean)[::-1]
    ranked = [(feature_names[i], float(result.importances_mean[i]),
               float(result.importances_std[i])) for i in order]

    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.32 * len(ranked) + 1.0)))
    positions = np.arange(len(ranked))[::-1]  # most important at the top
    ax.barh(positions, [r[1] for r in ranked], xerr=[r[2] for r in ranked],
            color="#4c72b0", ecolor="gray", capsize=3)
    ax.set_yticks(positions)
    ax.set_yticklabels([r[0] for r in ranked])
    ax.set_xlabel("permutation importance (mean score drop)")
    ax.set_title("permutation importance (held-out)")
    ax.grid(True, axis="x", alpha=0.3)
    _save_svg_png(fig, out_prefix)
    return ranked


def partial_dependence_plots(estimator, X, feature_names, top_features, out_prefix):
    """Draws PDP + ICE for the given top features into one figure.

    Maps each requested feature (by name, or by integer column index) to its
    column in ``X`` and calls ``PartialDependenceDisplay.from_estimator`` with
    ``kind='both'`` (partial-dependence average plus ICE lines). Estimators that
    do not support ICE for the requested target fall back to ``kind='average'``.

    Args:
        estimator: A fitted estimator supported by ``PartialDependenceDisplay``.
        X: Feature matrix to evaluate dependence over (held-out rows).
        feature_names: Column names of ``X`` (same order as the columns).
        top_features: Features to plot, given as names or integer column indices.
        out_prefix: Path prefix for the saved ``.svg`` / ``.png`` figures.

    Returns:
        The list of feature names actually plotted (requested names not found in
        ``feature_names`` are skipped with a printed warning).
    """
    indices = []
    for feat in top_features:
        if isinstance(feat, (int, np.integer)):
            indices.append(int(feat))
        elif feat in feature_names:
            indices.append(feature_names.index(feat))
        else:
            print(f"partial_dependence_plots: skipping unknown feature {feat!r}")
    if not indices:
        raise ValueError("no valid features to plot for partial dependence")

    plotted = [feature_names[i] for i in indices]
    n_cols = min(3, len(indices))
    fig, axes = plt.subplots(int(np.ceil(len(indices) / n_cols)), n_cols,
                             figsize=(4.2 * n_cols, 3.6 * int(np.ceil(len(indices) / n_cols))),
                             squeeze=False)
    try:
        PartialDependenceDisplay.from_estimator(
            estimator, X, features=indices, feature_names=feature_names,
            kind="both", ax=axes.ravel()[:len(indices)])
    except (ValueError, TypeError):
        # ICE ('both') is unsupported for this estimator/target -> PDP only.
        for ax in axes.ravel():
            ax.clear()
        PartialDependenceDisplay.from_estimator(
            estimator, X, features=indices, feature_names=feature_names,
            kind="average", ax=axes.ravel()[:len(indices)])
    for ax in axes.ravel()[len(indices):]:
        ax.set_visible(False)
    fig.suptitle("partial dependence (PDP + ICE)")
    _save_svg_png(fig, out_prefix)
    return plotted
