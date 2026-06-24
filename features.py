"""Per-branch model feature lists and the feature-matrix builder.

The genotype-quality models read two disjoint contracts (SPEC sec "features.py"):
the ``fast`` branch (optimized-streaming ``processLocusFast`` rows) and the
``full`` branch (everything else, which additionally carries the flank-normalized
depths). ``FAST_FEATURES`` / ``FULL_FEATURES`` enumerate the model inputs for each;
``FULL_FEATURES`` is ``FAST_FEATURES`` plus the two full-only flank-normalized depth
columns. (EH's own ``QD`` / ``eh_q`` quality scores were dropped as model features --
permutation importance found them uninformative -- but ``eh_q`` is still extracted as
the direction baseline in ``evaluate``.)

Two engineered columns are derived here from the raw row (rather than in the
extractor) so the definition lives next to the feature list:
``ci_asymmetry`` and ``ci_over_eh``. ``build_matrix`` returns a float DataFrame
(NaN preserved -- ``HistGradientBoosting`` handles missing values natively) with
exactly the branch's feature columns. ``FEATURE_FAMILIES`` groups the features
for the grouped ablation in ``ablation.py``.

Determinism: pure functions, no module-level mutable state, no randomness.
"""

import pandas as pd


# Per-branch model feature lists (SPEC sec "features.py"). Identifiers, labels,
# and leakage columns are deliberately excluded. ``coverage`` is deliberately NOT
# a model feature: real rows carry the NOMINAL run coverage while sim rows keep the
# extractor's per-locus MEASURED Coverage, so it has inconsistent semantics across
# the real/sim domains (the consistently-measured ``depth`` is used instead). It is
# still kept as a parquet column for coverage stratification in the evaluator.
FAST_FEATURES = [
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

# Full branch adds the two full-only flank-normalized depth columns. (EH's QD / eh_q are
# extracted but NOT model features -- permutation importance found them uninformative.)
FULL_FEATURES = FAST_FEATURES + [
    "left_flank_norm_depth", "right_flank_norm_depth",
]

# Feature groups for grouped ablation (SPEC sec "ablation.py"). These list the
# full set of members per family; ``feature_families`` filters to a branch.
FEATURE_FAMILIES = {
    "tier2_aqm": [
        "depth", "hq_unambiguous_reads", "strand_bias_phred",
        "mean_inserted_bases", "mean_deleted_bases",
        "left_flank_norm_depth", "right_flank_norm_depth",
    ],
    "ci": ["ci_width", "ci_asymmetry", "ci_over_eh"],
    "read_fractions": [
        # ``frac_spanning`` is extracted by eh_json_features but is NOT a model feature in either
        # branch (absent from FAST_FEATURES / FULL_FEATURES), so it is intentionally not listed here.
        "spanning_total", "spanning_at_called", "spanning_above_called",
        "flanking_above_called", "support_frac",
    ],
}

# Columns NOT present in the source parquet -- ``add_engineered`` derives them at runtime from the
# raw CI inputs below. Any consumer that reads feature names straight from the parquet must strip
# these and rebuild them via ``add_engineered`` (build_matrix does this).
ENGINEERED_FEATURES = ("ci_asymmetry", "ci_over_eh")
ENGINEERED_RAW_INPUTS = ("ci_start", "ci_end", "ci_width", "eh")


def add_engineered(df):
    """Returns a copy of ``df`` with the engineered CI columns added.

    Computes (vectorized):

    - ``ci_asymmetry = ((ci_end - eh) - (eh - ci_start)) / (ci_width + 1)``,
      set to 0 wherever any of ``ci_start`` / ``ci_end`` / ``ci_width`` is
      missing.
    - ``ci_over_eh = ci_width / (eh + 1)``, set to 0 wherever ``ci_width`` or
      ``eh`` is missing.

    Where the inputs are present the computed value is kept (including legitimate
    negatives / zeros); only NaN that stems from missing inputs is filled with 0.

    Args:
        df: DataFrame with raw ``eh``, ``ci_start``, ``ci_end``, ``ci_width``
            columns.

    Returns:
        A new DataFrame equal to ``df`` plus ``ci_asymmetry`` and ``ci_over_eh``.
    """
    df = df.copy()
    eh = pd.to_numeric(df["eh"], errors="coerce")
    ci_start = pd.to_numeric(df["ci_start"], errors="coerce")
    ci_end = pd.to_numeric(df["ci_end"], errors="coerce")
    ci_width = pd.to_numeric(df["ci_width"], errors="coerce")

    ci_missing = ci_start.isna() | ci_end.isna() | ci_width.isna()
    df["ci_asymmetry"] = (((ci_end - eh) - (eh - ci_start)) / (ci_width + 1)
                          ).where(~ci_missing, 0.0)

    df["ci_over_eh"] = (ci_width / (eh + 1)
                        ).where(~(ci_width.isna() | eh.isna()), 0.0)
    return df


def build_matrix(df, branch):
    """Builds the float feature matrix for one branch.

    Adds the engineered columns, selects the branch's feature list, casts every
    column (including bools) to float, and preserves NaN for the gradient
    booster to handle natively.

    Args:
        df: DataFrame of raw per-allele rows for this branch.
        branch: ``"full"`` or ``"fast"``.

    Returns:
        A ``(X, feature_names)`` tuple where ``X`` is a float DataFrame whose
        columns are exactly ``feature_names`` (``FULL_FEATURES`` or
        ``FAST_FEATURES``), in that order.

    Raises:
        ValueError: If ``branch`` is not ``"full"`` or ``"fast"``.
        AssertionError: If any required feature column is absent from ``df``.
    """
    if branch not in ("full", "fast"):
        raise ValueError(f"branch must be 'full' or 'fast'; got {branch!r}")
    df = add_engineered(df)
    feature_names = FULL_FEATURES if branch == "full" else FAST_FEATURES
    missing = [c for c in feature_names if c not in df.columns]
    assert not missing, f"missing feature columns for {branch} branch: {missing}"
    return df[feature_names].astype(float), list(feature_names)


def feature_families(branch):
    """Returns ``FEATURE_FAMILIES`` filtered to a branch's feature set.

    Members not present in the branch (e.g. the flank-normalized depths on the fast
    branch) are dropped, and any family left empty is omitted -- so the result is
    exactly the families ablation can score on this branch.

    Args:
        branch: ``"full"`` or ``"fast"``.

    Returns:
        A dict ``{family: [members in branch]}`` with empty families removed.
    """
    if branch not in ("full", "fast"):
        raise ValueError(f"branch must be 'full' or 'fast'; got {branch!r}")
    allowed = set(FULL_FEATURES if branch == "full" else FAST_FEATURES)
    return {family: members for family, members in (
                (family, [m for m in members if m in allowed])
                for family, members in FEATURE_FAMILIES.items())
            if members}
