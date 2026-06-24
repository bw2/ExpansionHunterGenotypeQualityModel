"""Allele-size (expansion/contraction) bins + equal-per-bin sample weights.

The truth set is dominated by small events: ~85% of alleles have
``|true - ref| <= 2`` repeats, while large expansions/contractions -- the
clinically interesting regime -- are 100s of alleles vs >1M in the core bins
(see ``REPORT.md`` "size imbalance"). Left uncorrected, both the model losses
and the headline metrics are effectively the small-event bins only.

This module defines the size bins (the SIGNED bins of the truth-vs-reference
plot in ``REPORT.md``) and a CAPPED inverse-frequency weighting that gives each
occupied bin equal total weight, except that a per-allele weight is capped so a
tiny tail bin cannot make a single allele dominate the objective.

``delta = round(true) - round(num_repeats_in_reference)`` is the per-allele
event size in repeat units (the x-axis of that plot). Pure functions only.
"""

import numpy as np


# The 27 signed bins of the truth-vs-reference figure, in order. Each interior
# bin is an inclusive integer range ``[lo, hi]``; the two ends are open.
BIN_LABELS = [
    "<=-31", "-30:-26", "-25:-21", "-20:-19", "-18:-17", "-16:-15", "-14:-13",
    "-12:-11", "-10:-9", "-8:-7", "-6:-5", "-4:-3", "-2:-1", "0", "1:2", "3:4",
    "5:6", "7:8", "9:10", "11:12", "13:14", "15:16", "17:18", "19:20", "21:25",
    "26:30", ">=31",
]

# Inclusive upper bound of every bin except the last (open) one. ``assign_bins``
# uses these as the searchsorted edges; len == len(BIN_LABELS) - 1 == 26.
_UPPER_EDGES = np.array(
    [-31, -26, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8,
     10, 12, 14, 16, 18, 20, 25, 30], dtype=float)

N_BINS = len(BIN_LABELS)


def deltas(true, ref_repeats):
    """Returns the per-allele event size ``round(true) - round(ref_repeats)``.

    Args:
        true: Array-like of truth allele sizes (repeat units; may be fractional).
        ref_repeats: Array-like of reference repeat counts (``num_repeats_in_reference``).

    Returns:
        Integer-valued float ndarray of the rounded size difference per allele.
    """
    return np.round(np.asarray(true, dtype=float)) - np.round(np.asarray(ref_repeats, dtype=float))


def assign_bins(true, ref_repeats):
    """Maps each allele to its size-bin index in ``[0, N_BINS)``.

    The index follows ``BIN_LABELS`` order: 0 is the most extreme contraction
    (``<=-31``), 13 is the no-change bin (``0``), 26 is the most extreme
    expansion (``>=31``). Rows whose delta is NaN (NaN truth OR NaN
    ``num_repeats_in_reference`` -- e.g. a malformed reference region) are routed
    to the sentinel index ``-1`` ("unbinned") rather than silently landing in the
    extreme-expansion bin 26: searchsorted sorts NaN above every edge, which would
    otherwise pollute (and ~50x up-weight) the large-expansion stratum. Callers
    that iterate ``range(N_BINS)`` or ``BIN_LABELS`` naturally skip ``-1``.

    Args:
        true: Array-like of truth allele sizes.
        ref_repeats: Array-like of reference repeat counts.

    Returns:
        Integer ndarray of bin indices in ``[0, N_BINS)``, or ``-1`` for a NaN delta.
    """
    d = deltas(true, ref_repeats)
    idx = np.searchsorted(_UPPER_EDGES, d, side="left").astype(int)
    idx[np.isnan(d)] = -1
    return idx


def bin_weights(bins, cap=50.0):
    """Returns capped inverse-frequency weights giving each bin ~equal total weight.

    Each occupied bin is assigned the same total weight ``n / n_occupied_bins``,
    so a rare bin's few alleles are up-weighted to match a common bin. Weights
    are kept at mean 1 (the loss scale, and thus the early-stopping /
    regularization behaviour, matches the unweighted fit). ``cap`` is a HARD
    per-allele ceiling in mean-1 units ("an allele counts like at most ``cap``
    average alleles"): the overflow shaved off the capped (tiny-bin) alleles is
    redistributed proportionally over the uncapped alleles so the mean stays 1
    (water-filling). With the default ``cap=50`` only the few-hundred-allele tail
    bins hit the ceiling; everything down to a few thousand alleles is fully
    equalized.

    Args:
        bins: Integer array-like of bin indices (e.g. from ``assign_bins``).
        cap: Hard max per-allele weight in mean-1 units; ``0`` / ``None`` disables
            the cap (pure inverse-frequency, still mean-1 normalized).

    Returns:
        Float ndarray of per-allele weights (same length as ``bins``, mean ~1).
    """
    bins = np.asarray(bins, dtype=int)
    n = bins.size
    if n == 0:
        return np.empty(0, dtype=float)
    uniq, counts = np.unique(bins, return_counts=True)
    count_of = dict(zip(uniq.tolist(), counts.tolist()))
    w = np.array([(n / uniq.size) / count_of[b] for b in bins], dtype=float)  # mean(w) == 1
    if not cap:
        return w
    cap = float(cap)
    # Water-filling: clamp to cap, spread the shaved-off mass over the still-uncapped
    # alleles (proportional, preserving their inverse-frequency shape), repeat until stable.
    for _ in range(100):
        over = w > cap
        if not over.any():
            break
        deficit = float(w[over].sum() - cap * over.sum())  # mass removed by clamping
        w[over] = cap
        free = ~over
        if deficit <= 0 or not free.any():
            break
        w[free] += deficit * (w[free] / w[free].sum())
    return w
