"""Pure held-out evaluation metrics for the q and direction heads.

Every function takes plain arrays of out-of-fold predictions + truth and returns a
flat dict, so the same code scores any genotyping_regime or fold. Distances are in repeat
units. Truth may be fractional, so every "exact match" is taken against
``round(true)``.

- ``q_metrics``        -- point accuracy of the LCF-recovered truth
  ``true_pred = eh / LCF`` vs the raw-EH baseline.
- ``direction_metrics``-- 3-class ``[P_OK, P_TOO_LONG, P_TOO_SHORT]`` log-loss,
  one-vs-rest ROC-AUCs, calibration error, P_OK accuracy, and the confusion matrix.
- ``gated_mae``        -- the deployment metric: raw-EH MAE vs the MAE after
  applying the LCF only where ``P_OK < 0.5`` (the recommended gate).

Pure functions, no module-level mutable state.
"""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score

_DIR_LABELS = [0, 1, 2]  # OK, TOO_LONG, TOO_SHORT


def _safe_auc(y_binary, score, metric=roc_auc_score):
    """Returns ``metric(y_binary, score)`` or NaN when only one class is present."""
    y_binary = np.asarray(y_binary)
    if np.unique(y_binary).size < 2:
        return float("nan")
    return float(metric(y_binary, np.asarray(score, dtype=float)))


def _ece(y_binary, p, n_bins=10):
    """Expected calibration error of a one-vs-rest probability against its outcome."""
    y_binary = np.asarray(y_binary, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        c = int(in_bin.sum())
        if c:
            ece += (c / p.size) * abs(p[in_bin].mean() - y_binary[in_bin].mean())
    return float(ece)


def q_metrics(eh, true, true_pred, t_true, t_pred, tol):
    """Scores the q head on one set of out-of-fold predictions.

    Args:
        eh: Raw EH allele calls.
        true: Truth allele sizes (may be fractional).
        true_pred: LCF-recovered truth ``eh / LCF``.
        t_true: Observed log-ratio ``t = log(eh) - log(true)``.
        t_pred: Predicted median of ``t``.
        tol: Per-allele size-dependent tolerance in repeats.

    Returns:
        Dict of scalar metrics (``n``, ``mae_t``, ``mae_true``, ``mae_eh``,
        ``dist_reduction``, ``exact_match_rate``, ``eh_exact_match_rate``,
        ``within_tol_rate``, ``eh_within_tol_rate``).
    """
    eh, true, true_pred = (np.asarray(a, dtype=float) for a in (eh, true, true_pred))
    t_true, t_pred, tol = (np.asarray(a, dtype=float) for a in (t_true, t_pred, tol))
    if eh.size == 0:
        return {"n": 0}
    mae_eh = float(np.mean(np.abs(true - eh)))
    mae_true = float(np.mean(np.abs(true - true_pred)))
    return {
        "n": int(eh.size),
        "mae_t": float(np.mean(np.abs(t_true - t_pred))),
        "mae_true": mae_true,
        "mae_eh": mae_eh,
        "dist_reduction": float(1.0 - mae_true / mae_eh) if mae_eh > 0 else float("nan"),
        "exact_match_rate": float(np.mean(np.round(true_pred) == np.round(true))),
        "eh_exact_match_rate": float(np.mean(np.round(eh) == np.round(true))),
        "within_tol_rate": float(np.mean(np.abs(np.round(true_pred) - np.round(true)) <= tol)),
        "eh_within_tol_rate": float(np.mean(np.abs(np.round(eh) - np.round(true)) <= tol)),
    }


def direction_metrics(dir_code, proba):
    """Scores the 3-class direction head on one set of out-of-fold probabilities.

    Args:
        dir_code: Integer labels in ``{0, 1, 2}`` (OK / TOO_LONG / TOO_SHORT).
        proba: Array ``(n, 3)`` in ``[P_OK, P_TOO_LONG, P_TOO_SHORT]`` order.

    Returns:
        Dict with ``n``, ``log_loss``, ``too_long_auc`` / ``too_short_auc``,
        ``too_long_ap`` / ``too_short_ap``, ``ece``, ``p_ok_accuracy`` (argmax ==
        label) and ``confusion`` (3x3, predicted = argmax).
    """
    y = np.asarray(dir_code, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if y.size == 0:
        return {"n": 0}
    return {
        "n": int(y.size),
        "log_loss": float(log_loss(y, proba, labels=_DIR_LABELS)),
        "too_long_auc": _safe_auc(y == 1, proba[:, 1]),
        "too_short_auc": _safe_auc(y == 2, proba[:, 2]),
        "too_long_ap": _safe_auc(y == 1, proba[:, 1], average_precision_score),
        "too_short_ap": _safe_auc(y == 2, proba[:, 2], average_precision_score),
        "ece": float(np.mean([_ece(y == 1, proba[:, 1]), _ece(y == 2, proba[:, 2])])),
        "p_ok_accuracy": float(np.mean(np.argmax(proba, axis=1) == y)),
        "confusion": confusion_matrix(y, np.argmax(proba, axis=1), labels=_DIR_LABELS).tolist(),
    }


def gated_mae(eh, true, true_pred, p_ok, gate=0.5):
    """Returns raw-EH MAE vs gated-LCF-corrected MAE (apply LCF only where P_OK < gate).

    Args:
        eh: Raw EH allele calls.
        true: Truth allele sizes.
        true_pred: LCF-recovered truth ``eh / LCF``.
        p_ok: Direction-head ``P_OK`` per allele.
        gate: Apply the correction only where ``p_ok < gate``.

    Returns:
        Dict with ``n``, ``mae_raw`` (mean ``|true - eh|``) and ``mae_gated`` (mean
        ``|true - corrected|``, ``corrected = eh/LCF`` where ``p_ok < gate`` else ``eh``).
    """
    eh, true, true_pred, p_ok = (np.asarray(a, dtype=float) for a in (eh, true, true_pred, p_ok))
    if eh.size == 0:
        return {"n": 0, "mae_raw": float("nan"), "mae_gated": float("nan")}
    corrected = np.where(p_ok < gate, true_pred, eh)
    return {
        "n": int(eh.size),
        "mae_raw": float(np.mean(np.abs(true - eh))),
        "mae_gated": float(np.mean(np.abs(true - corrected))),
    }
