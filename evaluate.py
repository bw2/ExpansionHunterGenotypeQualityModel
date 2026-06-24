"""Held-out evaluation metrics for the genotype-quality ``q`` and direction models.

This module scores model predictions that were produced on held-out chromosomes
(see ``SPEC.md`` sec 8 / "evaluate.py"). It never trains or imports the model
modules -- every function operates purely on arrays / DataFrames the caller
passes in, so the same code scores any fold, cross-split, or branch.

Two heads are evaluated:

- q head (``model_q``): regresses the per-allele log-ratio
  ``t = log(eh) - log(true)`` with five quantiles. ``evaluate_q`` reports point
  accuracy on ``t`` and on the recovered truth ``true_pred = eh / exp(t_median)``,
  the exact-match rate against the raw-EH baseline, and the empirical coverage of
  the fitted 80% / 90% predictive intervals. There is deliberately NO PIT
  histogram: five independently fit quantiles do not define a proper CDF
  (SPEC sec 6 caveat), so a PIT would be meaningless.

- direction head (``model_direction``): a 3-class ``[P_OK, P_TOO_LONG, P_TOO_SHORT]``
  classifier (``OK=0, TOO_LONG=1, TOO_SHORT=2``). ``evaluate_direction`` reports
  multiclass log-loss, TOO_LONG/TOO_SHORT one-vs-rest reliability + ECE (on the
  renormalized proba), one-vs-rest ROC-AUC / PR-AUC, and the 3x3 confusion
  matrix. Branch-specific coarse ranking baselines come from
  ``direction_baseline`` (full: ``eh_q`` + ``ci_width``; fast: ``ci_width`` only).

Both heads stratify by ``source`` (always real-only / sim-only / pooled),
``motif_size`` (bins ``1,2,3,4,5,6+``), ``coverage``, and ``sample``.

Pure functions, no module-level mutable state.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


# Quantiles the q model fits; the 80%/90% intervals use the (0.1,0.9)/(0.05,0.95)
# pairs (matches ``model_q.QUANTILES``).
QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]

# Default mapping {quantile: DataFrame column} used by ``stratified_q``.
DEFAULT_QUANTILE_COLS = {0.05: "q05", 0.1: "q10", 0.5: "q50", 0.9: "q90", 0.95: "q95"}

# Fixed direction class coding / proba column order (matches model_direction).
_DIR_LABELS = [0, 1, 2]  # OK, TOO_LONG, TOO_SHORT

# Scalar metric keys returned by ``evaluate_q`` (used to shape the empty result).
_Q_METRIC_KEYS = (
    "mae_t", "rmse_t", "mae_true", "rmse_true",
    "exact_match_rate", "eh_exact_match_rate",
    "within_tol_rate", "eh_within_tol_rate",
    "coverage_80", "coverage_90",
    "mae_eh", "dist_reduction", "mean_frac_closer",
)


# --- q head ---------------------------------------------------------------

def evaluate_q(t_true, t_pred_median, eh, true, quantile_preds, tol=None):
    """Scores the q head on one set of held-out predictions.

    Truth can be fractional (a sim allele size or a melted real allele), so every
    "exact match" is taken against ``round(true)`` rather than ``true`` itself.

    Args:
        t_true: Observed log-ratio ``t = log(eh) - log(true)`` per allele.
        t_pred_median: Predicted median of ``t`` (the 0.5 quantile head).
        eh: EH allele calls (the label source).
        true: Truth allele sizes (may be fractional).
        quantile_preds: Dict ``{0.05, 0.1, 0.5, 0.9, 0.95: array}`` of predicted
            ``t`` quantiles, aligned to ``t_true``.
        tol: Optional per-allele size-dependent tolerance in repeats
            (``size_tolerance``). When given, adds ``within_tol_rate`` (model) and
            ``eh_within_tol_rate`` (raw-EH baseline) = fraction within +/-tol; both
            are NaN when ``tol`` is omitted.

    Returns:
        Dict with ``n`` and: ``mae_t`` / ``rmse_t`` (on log-ratio ``t``);
        ``mae_true`` / ``rmse_true`` on the recovered truth
        ``true_pred = eh / exp(t_pred_median)``; ``exact_match_rate`` =
        ``mean(round(true_pred) == round(true))``; ``eh_exact_match_rate`` =
        ``mean(round(eh) == round(true))`` (raw-EH baseline); ``coverage_80`` =
        ``mean(q10 <= t_true <= q90)`` and ``coverage_90`` =
        ``mean(q05 <= t_true <= q95)``. An empty input yields ``n=0`` and NaN
        metrics (so stratified groups never crash).
    """
    t_true = np.asarray(t_true, dtype=float)
    t_pred_median = np.asarray(t_pred_median, dtype=float)
    eh = np.asarray(eh, dtype=float)
    true = np.asarray(true, dtype=float)
    if t_true.size == 0:
        return dict({"n": 0}, **{k: float("nan") for k in _Q_METRIC_KEYS})

    resid = t_true - t_pred_median
    true_pred = eh / np.exp(t_pred_median)
    # Distance-to-truth before applying q (raw EH) vs after (q-recovered), in repeat units.
    # "% closer" = how much q shrinks the call-to-truth distance (the user's metric): pooled as
    # 1 - MAE_after/MAE_before, and per-allele as mean((before-after)/before) over alleles EH
    # didn't already nail (before>0). q HELPS when these are > 0; q hurts if negative.
    dist_before = np.abs(true - eh)
    dist_after = np.abs(true - true_pred)
    mae_eh = float(np.mean(dist_before))
    mae_true_v = float(np.mean(dist_after))
    moved = dist_before > 0
    if tol is None:
        within_model = within_eh = float("nan")
    else:
        tol = np.asarray(tol, dtype=float)
        within_model = float(np.mean(np.abs(np.round(true_pred) - np.round(true)) <= tol))
        within_eh = float(np.mean(np.abs(np.round(eh) - np.round(true)) <= tol))
    return {
        "n": int(t_true.size),
        "mae_t": float(np.mean(np.abs(resid))),
        "rmse_t": float(np.sqrt(np.mean(resid ** 2))),
        "mae_true": mae_true_v,
        "rmse_true": float(np.sqrt(np.mean((true_pred - true) ** 2))),
        "mae_eh": mae_eh,
        "dist_reduction": float(1.0 - mae_true_v / mae_eh) if mae_eh > 0 else float("nan"),
        "mean_frac_closer": float(np.mean(
            (dist_before[moved] - dist_after[moved]) / dist_before[moved])) if moved.any()
            else float("nan"),
        "exact_match_rate": float(np.mean(np.round(true_pred) == np.round(true))),
        "eh_exact_match_rate": float(np.mean(np.round(eh) == np.round(true))),
        "within_tol_rate": within_model,
        "eh_within_tol_rate": within_eh,
        "coverage_80": float(np.mean(
            (t_true >= np.asarray(quantile_preds[0.1], dtype=float)) &
            (t_true <= np.asarray(quantile_preds[0.9], dtype=float)))),
        "coverage_90": float(np.mean(
            (t_true >= np.asarray(quantile_preds[0.05], dtype=float)) &
            (t_true <= np.asarray(quantile_preds[0.95], dtype=float)))),
    }


def stratified_q(df_eval, t_true_col="t_true", t_pred_col=None, eh_col="eh",
                 true_col="true", quantile_cols=None, tol_col="tol_repeats", **stratum_cols):
    """Runs ``evaluate_q`` per stratum group.

    ``df_eval`` carries the per-row q-eval inputs (``t_true``, the median and the
    five quantile predictions, ``eh``, ``true``) plus the stratum columns
    (``source``, ``motif_size``, ``coverage``, ``sample``).

    Args:
        df_eval: DataFrame with the columns named below.
        t_true_col: Column holding observed ``t``.
        t_pred_col: Column holding the predicted median ``t``; defaults to the
            0.5-quantile column from ``quantile_cols`` (they are the same value).
        eh_col: Column holding the EH call.
        true_col: Column holding the (possibly fractional) truth.
        quantile_cols: ``{quantile: column}`` map; defaults to
            ``DEFAULT_QUANTILE_COLS``.
        **stratum_cols: Optional overrides for the stratum column names
            (``source_col``, ``motif_col``, ``coverage_col``, ``sample_col``).

    Returns:
        Nested dict ``{stratum_type: {stratum_value: q_metrics}}`` always
        including ``source`` with ``real`` / ``sim`` / ``pooled`` slices.
    """
    quantile_cols = quantile_cols or DEFAULT_QUANTILE_COLS
    median_col = t_pred_col or quantile_cols[0.5]

    def metric_fn(sub):
        return evaluate_q(
            sub[t_true_col].to_numpy(dtype=float),
            sub[median_col].to_numpy(dtype=float),
            sub[eh_col].to_numpy(dtype=float),
            sub[true_col].to_numpy(dtype=float),
            {a: sub[col].to_numpy(dtype=float) for a, col in quantile_cols.items()},
            tol=sub[tol_col].to_numpy(dtype=float) if tol_col in sub else None,
        )

    return _stratify(df_eval, metric_fn, **stratum_cols)


# --- direction head -------------------------------------------------------

def evaluate_direction(y_true, proba, branch, baseline_cols=None):
    """Scores the 3-class direction head on one set of held-out predictions.

    Probabilities are renormalized to the simplex first (defensive: model output
    already sums to 1), so the TOO_LONG/TOO_SHORT one-vs-rest reliability and ECE are
    computed on coherent probabilities. ``branch`` is recorded for reporting
    context; baseline construction is branch-specific and lives in
    ``direction_baseline``.

    Args:
        y_true: Integer ``dir_code`` labels in ``{0, 1, 2}`` (OK/TOO_LONG/TOO_SHORT).
        proba: Array ``(n, 3)`` in ``[P_OK, P_TOO_LONG, P_TOO_SHORT]`` order.
        branch: ``"full"`` or ``"fast"`` (carried through for reporting).
        baseline_cols: Optional ``{signal_name: score_array}`` of coarse ranking
            baseline scores (already oriented so higher => more likely miscall);
            their TOO_LONG/TOO_SHORT AUCs are attached under ``"baselines"``.

    Returns:
        Dict with ``n``, ``branch``, ``log_loss``, ``too_long_auc`` / ``too_short_auc``
        (one-vs-rest ROC-AUC), ``too_long_ap`` / ``too_short_ap`` (PR-AUC), ``confusion``
        (3x3 list, predicted = argmax), ``reliability`` (``over`` / ``under``
        each with bin centers + empirical freq + per-bin pred mean + count +
        ECE), ``ece`` (top-level ``over`` / ``under`` mirror) and ``baselines``.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError("proba must have shape (n, 3) for [P_OK, P_TOO_LONG, P_TOO_SHORT]")
    if y_true.size == 0:
        return {
            "n": 0, "branch": branch, "log_loss": float("nan"),
            "too_long_auc": float("nan"), "too_short_auc": float("nan"),
            "too_long_ap": float("nan"), "too_short_ap": float("nan"),
            "confusion": [[0, 0, 0] for _ in range(3)],
            "reliability": {"too_long": None, "too_short": None},
            "ece": {"too_long": float("nan"), "too_short": float("nan")},
            "baselines": _baseline_aucs(y_true, baseline_cols),
        }

    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / 3.0),
                      where=row_sums > 0)
    over_rel = _reliability(y_true == 1, proba[:, 1])
    under_rel = _reliability(y_true == 2, proba[:, 2])
    return {
        "n": int(y_true.size),
        "branch": branch,
        "log_loss": float(log_loss(y_true, proba, labels=_DIR_LABELS)),
        "too_long_auc": _safe_auc(y_true == 1, proba[:, 1], roc_auc_score),
        "too_short_auc": _safe_auc(y_true == 2, proba[:, 2], roc_auc_score),
        "too_long_ap": _safe_auc(y_true == 1, proba[:, 1], average_precision_score),
        "too_short_ap": _safe_auc(y_true == 2, proba[:, 2], average_precision_score),
        "confusion": confusion_matrix(
            y_true, np.argmax(proba, axis=1), labels=_DIR_LABELS).tolist(),
        "reliability": {"too_long": over_rel, "too_short": under_rel},
        "ece": {"too_long": over_rel["ece"], "too_short": under_rel["ece"]},
        "baselines": _baseline_aucs(y_true, baseline_cols),
    }


def direction_baseline(df, branch, y_col="dir_code", ci_col="ci_width",
                       eh_q_col="eh_q"):
    """Builds branch-specific coarse ranking baselines and returns their AUCs.

    This is a CALIBRATION-FREE ranking baseline, not a probabilistic classifier:
    each signal is just oriented so that a larger value flags a more likely
    miscall, and we report how well that ranking separates TOO_LONG (and TOO_SHORT) from
    the rest by one-vs-rest ROC-AUC.

    - full branch: ``ci_width`` (wider CI => more uncertain) and ``-eh_q``
      (lower EH quality => more likely miscall).
    - fast branch: ``ci_width`` ONLY -- ``eh_q`` is absent on the fast path, so it
      is never used even if a stray column is present.

    A signal whose column is missing or all-NaN is skipped (never an error).

    Args:
        df: DataFrame with ``y_col`` plus the signal columns below.
        branch: ``"full"`` or ``"fast"``.
        y_col: Column holding integer ``dir_code`` labels.
        ci_col: CI-width column name.
        eh_q_col: EH-quality column name (full branch only).

    Returns:
        Dict ``{signal_name: {"too_long_auc": float, "too_short_auc": float}}`` for each
        available signal (``"ci_width"`` and, on full, ``"neg_eh_q"``).
    """
    scores = {}
    if ci_col in df.columns and not df[ci_col].isna().all():
        scores["ci_width"] = df[ci_col].to_numpy(dtype=float)
    if branch == "full" and eh_q_col in df.columns and not df[eh_q_col].isna().all():
        scores["neg_eh_q"] = -df[eh_q_col].to_numpy(dtype=float)
    return _baseline_aucs(df[y_col].to_numpy(), scores)


def stratified_direction(df_eval, branch, y_col="dir_code",
                         proba_cols=("p_ok", "p_too_long", "p_too_short"),
                         with_baseline=True, **stratum_cols):
    """Runs ``evaluate_direction`` per stratum group.

    Args:
        df_eval: DataFrame with ``y_col``, the three proba columns, the stratum
            columns, and (for baselines) ``ci_width`` / ``eh_q``.
        branch: ``"full"`` or ``"fast"``.
        y_col: Column holding integer ``dir_code`` labels.
        proba_cols: The ``[P_OK, P_TOO_LONG, P_TOO_SHORT]`` column names, in order.
        with_baseline: If true, attach ``direction_baseline`` per stratum.
        **stratum_cols: Optional overrides for the stratum column names.

    Returns:
        Nested dict ``{stratum_type: {stratum_value: direction_metrics}}`` always
        including ``source`` with ``real`` / ``sim`` / ``pooled`` slices.
    """
    def metric_fn(sub):
        result = evaluate_direction(
            sub[y_col].to_numpy(),
            sub[list(proba_cols)].to_numpy(dtype=float),
            branch)
        if with_baseline:
            result["baselines"] = direction_baseline(sub, branch, y_col=y_col)
        return result

    return _stratify(df_eval, metric_fn, **stratum_cols)


# --- shared helpers -------------------------------------------------------

def _safe_auc(y_binary, score, metric):
    """Returns ``metric(y_binary, score)`` or NaN when it is undefined.

    Drops rows with a NaN score, then returns NaN if fewer than two classes
    remain (ROC-AUC / PR-AUC need both a positive and a negative).

    Args:
        y_binary: Boolean / 0-1 one-vs-rest labels.
        score: Real-valued ranking scores (NaNs allowed; dropped).
        metric: ``roc_auc_score`` or ``average_precision_score``.

    Returns:
        The metric as a float, or ``float("nan")``.
    """
    y_binary = np.asarray(y_binary)
    score = np.asarray(score, dtype=float)
    keep = ~np.isnan(score)
    if not np.any(keep):
        return float("nan")
    y_binary = y_binary[keep]
    if np.unique(y_binary).size < 2:
        return float("nan")
    return float(metric(y_binary, score[keep]))


def _reliability(y_binary, p, n_bins=10):
    """Returns a one-vs-rest reliability curve and its ECE.

    Bins the predicted probabilities into ``n_bins`` equal-width bins over
    ``[0, 1]`` and, per bin, reports the mean predicted probability and the
    empirical frequency of the positive class. ECE is the count-weighted mean
    absolute gap between the two over non-empty bins.

    Args:
        y_binary: Boolean / 0-1 one-vs-rest labels.
        p: Predicted positive-class probabilities in ``[0, 1]``.
        n_bins: Number of equal-width probability bins.

    Returns:
        Dict with ``bin_centers``, ``bin_pred_mean``, ``bin_freq`` (NaN for empty
        bins), ``bin_count``, and ``ece``.
    """
    y_binary = np.asarray(y_binary, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    pred_means, freqs, counts = [], [], []
    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        count = int(np.count_nonzero(in_bin))
        counts.append(count)
        if count == 0:
            pred_means.append(float("nan"))
            freqs.append(float("nan"))
            continue
        pred_means.append(float(np.mean(p[in_bin])))
        freqs.append(float(np.mean(y_binary[in_bin])))
        ece += (count / p.size) * abs(pred_means[-1] - freqs[-1])
    return {
        "bin_centers": ((edges[:-1] + edges[1:]) / 2.0).tolist(),
        "bin_pred_mean": pred_means,
        "bin_freq": freqs,
        "bin_count": counts,
        "ece": float(ece),
    }


def _baseline_aucs(y_true, scores_by_signal):
    """Returns TOO_LONG/TOO_SHORT one-vs-rest ROC-AUCs for each baseline signal.

    Args:
        y_true: Integer ``dir_code`` labels.
        scores_by_signal: ``{name: score_array}`` (may be None / empty).

    Returns:
        Dict ``{name: {"too_long_auc": float, "too_short_auc": float}}``.
    """
    y_true = np.asarray(y_true)
    return {
        name: {
            "too_long_auc": _safe_auc(y_true == 1, score, roc_auc_score),
            "too_short_auc": _safe_auc(y_true == 2, score, roc_auc_score),
        }
        for name, score in (scores_by_signal or {}).items()
    }


def _motif_bin_masks(df, motif_col):
    """Yields ``(label, boolean_mask)`` for motif bins ``1,2,3,4,5,6+``."""
    motif = df[motif_col]
    masks = [(str(k), motif == k) for k in (1, 2, 3, 4, 5)]
    masks.append(("6+", motif >= 6))
    return masks


def _cov_key(cov):
    """Returns a compact string key for a (usually whole-number) coverage value."""
    return f"{float(cov):g}"


def macro_over_bins(bin_metrics, keys, min_n=1):
    """Equal-weight-per-bin (macro) average of scalar metrics across size bins.

    This is the reporting counterpart of the equal-per-bin TRAINING weights: each
    occupied size bin contributes equally to the headline regardless of how many
    alleles it holds, so large expansions/contractions are no longer drowned out
    by the small-event bulk. NaN per-bin metrics (e.g. an AUC in a bin with only
    one direction class present) are skipped.

    Args:
        bin_metrics: ``{bin_key: metrics_dict}`` (the ``"size_bin"`` slice of a
            ``stratified_*`` result).
        keys: Scalar metric keys to macro-average.
        min_n: Minimum bin allele count to include a bin.

    Returns:
        Dict ``{key: macro_mean}`` plus ``n_bins`` (bins that contributed).
    """
    usable = [m for m in bin_metrics.values() if m.get("n", 0) >= min_n]
    out = {}
    for k in keys:
        vals = [m[k] for m in usable if k in m and not np.isnan(float(m[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["n_bins"] = len(usable)
    return out


def _stratify(df, metric_fn, source_col="source", motif_col="motif_size",
              coverage_col="coverage", sample_col="sample", bin_col="size_bin"):
    """Applies ``metric_fn`` over source / motif / coverage / sample / size-bin strata.

    ``source`` always carries ``pooled`` / ``real`` / ``sim`` (per SPEC); the
    other stratum types are included only when their column is present. The
    ``size_bin`` stratum (keyed by the integer ``size_bins`` index, stringified)
    feeds the equal-per-bin macro headline via ``macro_over_bins``. Empty slices
    are passed through to ``metric_fn`` (which returns an ``n=0`` result).

    Args:
        df: DataFrame carrying the metric inputs and stratum columns.
        metric_fn: ``metric_fn(sub_df) -> metrics dict``.
        source_col, motif_col, coverage_col, sample_col, bin_col: Stratum column names.

    Returns:
        Nested dict ``{stratum_type: {stratum_value: metrics}}``.
    """
    out = {"source": {"pooled": metric_fn(df)}}
    if source_col in df.columns:
        out["source"]["real"] = metric_fn(df[df[source_col] == "real"])
        out["source"]["sim"] = metric_fn(df[df[source_col] == "sim"])
    if motif_col in df.columns:
        out["motif_size"] = {label: metric_fn(df[mask])
                             for label, mask in _motif_bin_masks(df, motif_col)}
    if coverage_col in df.columns:
        # Stratify coverage only over the REAL rows: real rows carry the NOMINAL run coverage
        # (10/20/31/46), while sim rows keep the extractor's MEASURED fractional Coverage, so
        # including sim would explode the report into dozens of one-allele fractional buckets
        # (e.g. 9.8x, 10.3x). features.py documents this real-vs-sim coverage inconsistency.
        cov_rows = df[df[source_col] == "real"] if source_col in df.columns else df
        out["coverage"] = {_cov_key(cov): metric_fn(cov_rows[cov_rows[coverage_col] == cov])
                           for cov in sorted(cov_rows[coverage_col].dropna().unique())}
    if sample_col in df.columns:
        out["sample"] = {str(s): metric_fn(df[df[sample_col] == s])
                         for s in sorted(df[sample_col].dropna().unique())}
    if bin_col in df.columns:
        out["size_bin"] = {str(int(b)): metric_fn(df[df[bin_col] == b])
                           for b in sorted(df[bin_col].dropna().unique())}
    return out
