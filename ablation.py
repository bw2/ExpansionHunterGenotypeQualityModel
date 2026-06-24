"""Feature-set ablation for the EH genotype-quality models.

Selects a small, defensible feature set without ever touching the test rows
(SPEC sec ``ablation.py``). Everything here ranks and scores on the 2 CALIB
chroms of ONE fixed fold; the chosen minimal set is scored once on test by
``run_cv.py`` -- not here.

- ``add_one_curve`` -- order features by permutation importance, then add them
  one at a time (most important first) and watch the calib score; the smallest
  prefix within 1% of the peak is the minimal set.
- ``grouped_ablation`` -- drop a whole feature *family* and measure the calib
  score lost, giving each family's marginal contribution.

The ranking and scoring callables are passed IN, so this module imports neither
the model nor the feature modules and is independently testable. It introduces no
randomness of its own (any randomness lives in the supplied ``rank_fn`` /
``score_fn``), so it is deterministic given deterministic callables.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_io import _save_svg_png


SEED = 20260616

# A prefix counts as "as good as peak" if within this fraction of the peak score.
_WITHIN_FRACTION_OF_PEAK = 0.01


def add_one_curve(rank_fn, score_fn, feature_names, out_prefix):
    """Importance-ranked add-one curve; returns the minimal near-peak feature set.

    ``rank_fn()`` returns the features ordered most->least important (by
    permutation importance on the CALIB chroms). For ``k = 1..n`` the top-``k``
    features are scored with ``score_fn`` (trained on train chroms, scored on the
    SAME calib chroms -- never test). The minimal set is the smallest ``k`` whose
    calib score is within 1% of the best score reached along the curve.

    Args:
        rank_fn: Callable ``rank_fn() -> [feature, ...]`` ordered most->least
            important. Names not in ``feature_names`` are ignored.
        score_fn: Callable ``score_fn(feature_subset) -> float`` (higher is
            better), trained on train chroms and scored on calib chroms.
        feature_names: The full candidate feature list.
        out_prefix: Path prefix for the saved ``.svg`` / ``.png`` figures.

    Returns:
        A ``(k, feature_subset)`` tuple: the minimal number of top features
        within 1% of the peak calib score, and that prefix of the ranking.
    """
    ranked = [f for f in rank_fn() if f in feature_names]
    scores = [float(score_fn(ranked[:k])) for k in range(1, len(ranked) + 1)]
    peak = max(scores)
    band = peak - _WITHIN_FRACTION_OF_PEAK * abs(peak)
    best_k = next(k for k, s in enumerate(scores, start=1) if s >= band)
    subset = ranked[:best_k]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(range(1, len(scores) + 1), scores, "-o", color="#1f77b4", markersize=4)
    ax.axhline(band, linestyle=":", color="gray", linewidth=1, label="within 1% of peak")
    thresh_feat = ranked[best_k - 1]  # the k-th ranked feature = last one added to reach the minimal set
    ax.axvline(best_k, linestyle="--", color="#d62728", linewidth=1,
               label=f"minimal k={best_k}: {thresh_feat}")
    # name that threshold feature on the plot, written vertically along the dashed line
    ax.text(best_k, 0.02, " " + thresh_feat, rotation=90, va="bottom", ha="left",
            color="#d62728", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("number of top-ranked features (added one at a time)")
    ax.set_ylabel("calib score")
    ax.set_title("add-one feature curve (calib)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # feature counts are integers, never 2.5
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_svg_png(fig, out_prefix)
    # persist the curve so it can be re-plotted without re-fitting (out_prefix is a path prefix)
    with open(out_prefix + "_data.json", "w") as f:
        json.dump({"ks": list(range(1, len(scores) + 1)), "scores": scores, "ranked": ranked,
                   "minimal_k": best_k, "band": band, "peak": peak}, f, indent=1)
    return best_k, subset


def grouped_ablation(score_fn, feature_families, all_features, out_prefix):
    """Per-family marginal contribution by dropping each family in turn.

    Scores the full feature set once, then for each family scores again with that
    family's features removed; the family's marginal contribution is the calib
    score lost (``full_score - score_without_family``). Scored on calib chroms
    only.

    Args:
        score_fn: Callable ``score_fn(feature_subset) -> float`` (higher is
            better), trained on train chroms and scored on calib chroms.
        feature_families: Dict ``family_name -> [feature, ...]`` (from
            ``features.py``; passed in, not imported here).
        all_features: The full feature list scored as the baseline.
        out_prefix: Path prefix for the saved ``.svg`` / ``.png`` figures.

    Returns:
        A dict ``family_name -> marginal_contribution`` (one entry per family).
    """
    full_score = float(score_fn(list(all_features)))
    marginals = {}
    for family, feats in feature_families.items():
        drop = set(feats)
        marginals[family] = full_score - float(
            score_fn([f for f in all_features if f not in drop]))

    families = list(marginals.keys())
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.5 * len(families) + 1.0)))
    positions = np.arange(len(families))[::-1]
    ax.barh(positions, [marginals[fam] for fam in families], color="#55a868")
    ax.set_yticks(positions)
    ax.set_yticklabels(families)
    ax.axvline(0.0, color="gray", linewidth=1)
    ax.set_xlabel("marginal contribution (full_score - score_without_family)")
    ax.set_title("grouped family ablation (calib)")
    ax.grid(True, axis="x", alpha=0.3)
    _save_svg_png(fig, out_prefix)
    return marginals
