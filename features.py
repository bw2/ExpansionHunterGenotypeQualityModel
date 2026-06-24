"""Model feature contract, size-dependent tolerance, genotyping_regime routing, and labels.

Two disjoint feature contracts (the JSON carries ``feature_names`` so the C++
consumer assembles features in the exact same order):

- ``quick`` -- the optimized-streaming ``processLocusFast`` (QuickGenotype) rows.
- ``full``  -- everything else; ``QUICK_FEATURES`` plus the two flank-normalized depths.

``ci_asymmetry`` and ``ci_over_eh`` are engineered here (next to the feature list)
from the raw CI columns. ``build_matrix`` returns a float DataFrame with exactly
the branch's feature columns, NaN preserved for ``HistGradientBoosting`` to handle.

The label helpers turn a joined ``(eh, true)`` frame into the training targets:
``t = log(eh) - log(true)`` (the q head's target; ``LCF = exp(t)``) and the
size-tolerance direction code (the direction head's target). What counts as a
"correct" call is not a fixed +/-1 band -- it widens with the true allele size in
bp, because read length limits how precisely a large repeat can be sized.

Pure functions, no module-level mutable state, no randomness.
"""

import numpy as np
import pandas as pd

# Per-branch model feature lists. Identifiers, labels, and leakage columns are
# excluded. ``coverage`` is deliberately NOT a feature (inconsistent real/sim
# semantics); the consistently-measured ``depth`` is used instead.
QUICK_FEATURES = [
    "motif_size",
    "num_repeats_in_reference", "ref_size_bp",
    "eh", "eh_minus_ref", "allele_rank",
    "ci_width", "ci_asymmetry", "ci_over_eh",
    "spanning_total", "hq_unamb_total",
    "spanning_at_called", "spanning_above_called", "flanking_above_called",
    "support_frac",
    "depth", "hq_unambiguous_reads", "strand_bias_phred",
    "mean_inserted_bases", "mean_deleted_bases",
]

# Full branch adds the two full-only flank-normalized depth columns.
FULL_FEATURES = QUICK_FEATURES + ["left_flank_norm_depth", "right_flank_norm_depth"]

# Engineered columns and the raw inputs they derive from (NOT present in the parquet
# as features; rebuilt by ``add_engineered``).
ENGINEERED_FEATURES = ("ci_asymmetry", "ci_over_eh")
ENGINEERED_RAW_INPUTS = ("ci_start", "ci_end", "ci_width", "eh")

# Direction class coding -- also the output column order of every probability vector.
OK, TOO_LONG, TOO_SHORT = 0, 1, 2
DIR_CLASS_NAMES = ["OK", "TOO_LONG", "TOO_SHORT"]

# Genotyping branches (feature contracts) and genotyping_regimes (one model-set each).
BRANCH_QUICK = "quick"
BRANCH_FULL = "full"
GENOTYPING_REGIME_QUICK = "quick"
GENOTYPING_REGIME_FULL_SPANNING = "full_spanning"
GENOTYPING_REGIME_FULL_NONSPANNING = "full_nonspanning"
GENOTYPING_REGIMES = (GENOTYPING_REGIME_QUICK, GENOTYPING_REGIME_FULL_SPANNING, GENOTYPING_REGIME_FULL_NONSPANNING)

# Branch (feature contract) each genotyping_regime reads.
GENOTYPING_REGIME_BRANCH = {GENOTYPING_REGIME_QUICK: BRANCH_QUICK, GENOTYPING_REGIME_FULL_SPANNING: BRANCH_FULL,
                 GENOTYPING_REGIME_FULL_NONSPANNING: BRANCH_FULL}

# Display labels for the report's genotyping_regime axis (the genotyping_regime names are already the
# display names; kept as an explicit map so the report never hardcodes them).
GENOTYPING_REGIME_DISPLAY = {GENOTYPING_REGIME_QUICK: "quick", GENOTYPING_REGIME_FULL_SPANNING: "full_spanning",
                  GENOTYPING_REGIME_FULL_NONSPANNING: "full_nonspanning"}


def add_engineered(df):
    """Returns a copy of ``df`` with the engineered CI columns added.

    ``ci_asymmetry = ((ci_end - eh) - (eh - ci_start)) / (ci_width + 1)`` and
    ``ci_over_eh = ci_width / (eh + 1)``, each set to 0 wherever its raw inputs are
    missing (legitimate negatives/zeros are kept).
    """
    df = df.copy()
    eh = pd.to_numeric(df["eh"], errors="coerce")
    ci_start = pd.to_numeric(df["ci_start"], errors="coerce")
    ci_end = pd.to_numeric(df["ci_end"], errors="coerce")
    ci_width = pd.to_numeric(df["ci_width"], errors="coerce")
    ci_missing = ci_start.isna() | ci_end.isna() | ci_width.isna()
    df["ci_asymmetry"] = (((ci_end - eh) - (eh - ci_start)) / (ci_width + 1)).where(~ci_missing, 0.0)
    df["ci_over_eh"] = (ci_width / (eh + 1)).where(~(ci_width.isna() | eh.isna()), 0.0)
    return df


def feature_names(branch):
    """Returns the ordered feature list for ``branch`` (``"full"`` or ``"quick"``)."""
    if branch not in (BRANCH_FULL, BRANCH_QUICK):
        raise ValueError("branch must be 'full' or 'quick'; got %r" % branch)
    return list(FULL_FEATURES if branch == BRANCH_FULL else QUICK_FEATURES)


def build_matrix(df, branch):
    """Builds the float feature matrix for one branch.

    Adds the engineered columns, selects the branch's feature list (in order),
    casts to float, and preserves NaN for the gradient booster to handle natively.

    Returns:
        ``(X, names)`` where ``X`` is a float DataFrame whose columns are exactly
        ``names`` (``FULL_FEATURES`` or ``QUICK_FEATURES``).
    """
    names = feature_names(branch)
    df = add_engineered(df)
    missing = [c for c in names if c not in df.columns]
    assert not missing, "missing feature columns for %s branch: %s" % (branch, missing)
    return df[names].astype(float), names


# --- size-dependent tolerance + genotyping_regime + labels ---------------------------

def tol_repeats(bp):
    """Returns the per-allele OK tolerance (in repeats) for true allele sizes ``bp``.

    Tiers keyed on true allele size in bp (``motif_size * round(true)``):
    ``<50 -> 0`` (exact), ``[50,120] -> 1``, ``(120,270) -> 2``, ``[270,600) -> 4``,
    ``>=600 -> 8``.
    """
    bp = np.asarray(bp, dtype=float)
    tol = np.full(bp.shape, 8, dtype=int)   # >= 600
    tol = np.where(bp < 600, 4, tol)        # [270, 600)
    tol = np.where(bp < 270, 2, tol)        # (120, 270)
    tol = np.where(bp <= 120, 1, tol)       # [50, 120]
    tol = np.where(bp < 50, 0, tol)         # < 50
    return tol


def direction_codes(eh, true, tol):
    """Returns ``{OK=0, TOO_LONG=1, TOO_SHORT=2}`` per allele.

    ``dr = round(eh) - round(true)``; OK if ``|dr| <= tol``, TOO_LONG if ``dr > tol``,
    TOO_SHORT if ``dr < -tol``.
    """
    dr = np.round(np.asarray(eh, dtype=float)) - np.round(np.asarray(true, dtype=float))
    tol = np.asarray(tol, dtype=float)
    return np.where(dr > tol, TOO_LONG, np.where(dr < -tol, TOO_SHORT, OK)).astype(int)


def genotyping_regime_of(genotyping_branch, spanning_at_called):
    """Maps each allele to its genotyping_regime.

    ``quick`` branch -> ``quick``. Full-branch rows split on spanning support at the
    called size: ``>= 1`` -> ``full_spanning`` else ``full_nonspanning`` (NaN -> 0).
    """
    branch = np.asarray(genotyping_branch, dtype=object)
    span = np.nan_to_num(np.asarray(spanning_at_called, dtype=float), nan=0.0)
    out = np.where(span >= 1, GENOTYPING_REGIME_FULL_SPANNING, GENOTYPING_REGIME_FULL_NONSPANNING).astype(object)
    out[branch == BRANCH_QUICK] = GENOTYPING_REGIME_QUICK
    return out


def add_labels(df):
    """Adds the q/direction targets and routing columns to a joined frame in place.

    Expects numeric-coercible ``eh``, ``true``, ``motif_size``, ``genotyping_branch``
    and ``spanning_at_called``. Adds ``q``, ``t``, ``allele_bp``, ``tol_repeats``,
    ``dir_code``, ``direction`` and ``genotyping_regime``. Returns the same ``df``.
    """
    eh = pd.to_numeric(df["eh"], errors="coerce").to_numpy(dtype=float)
    true = pd.to_numeric(df["true"], errors="coerce").to_numpy(dtype=float)
    motif = pd.to_numeric(df["motif_size"], errors="coerce").to_numpy(dtype=float)
    df["q"] = eh / true
    df["t"] = np.log(eh) - np.log(true)
    df["allele_bp"] = (motif * np.round(true)).astype("float32")
    tol = tol_repeats(df["allele_bp"].to_numpy())
    df["tol_repeats"] = tol.astype("int16")
    df["dir_code"] = direction_codes(eh, true, tol)
    df["direction"] = pd.Series(df["dir_code"], index=df.index).map(
        dict(enumerate(DIR_CLASS_NAMES)))
    df["genotyping_regime"] = genotyping_regime_of(
        df["genotyping_branch"].to_numpy(),
        pd.to_numeric(df["spanning_at_called"], errors="coerce").to_numpy())
    return df
