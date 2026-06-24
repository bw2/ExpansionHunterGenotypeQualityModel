"""Size-dependent accuracy tolerance + genotyping-regime routing.

What counts as a "correct" EH allele call is not a fixed +/-1 repeat band: it
depends on the allele's physical size, because read-length limits how precisely
a repeat can be sized. Small alleles (well within a read) must be exact; large
expansions (sized indirectly from flanking / in-repeat reads) are only
resolvable to within several repeats. The tolerance table below (in repeats,
keyed on the TRUE allele size in bp = ``motif_size * round(true)``) encodes that:

    true bp        tolerance (repeats)
    < 50           0   (exact)
    50 - 120       +/-1
    120 - 270      +/-2
    270 - 600      +/-4
    >= 600         +/-8

Boundary rule: ``<50 -> 0``, ``[50,120] -> 1``, ``(120,270) -> 2``,
``[270,600) -> 4``, ``>=600 -> 8``. Same table for spanning and non-spanning
alleles (the per-regime models absorb the regime difference via features).

Regimes (one model-set each): ``fast`` (processLocusFast), and within the full
genotyper ``full_spanning`` (>=1 read spans the repeat at the called size,
``spanning_at_called >= 1``) vs ``full_nonspanning`` (flanking / in-repeat only).

Pure functions only; no module-level mutable state.
"""

import numpy as np

# Direction class coding shared with model_direction / build_dataset.
OK, TOO_LONG, TOO_SHORT = 0, 1, 2

REGIME_FAST = "fast"
REGIME_FULL_SPANNING = "full_spanning"
REGIME_FULL_NONSPANNING = "full_nonspanning"
REGIMES = (REGIME_FAST, REGIME_FULL_SPANNING, REGIME_FULL_NONSPANNING)


def tol_repeats(bp):
    """Returns the per-allele tolerance (repeats) for the given true allele sizes in bp.

    Args:
        bp: Array-like of true allele sizes in bp (``motif_size * round(true)``).

    Returns:
        Integer ndarray of tolerances in repeats (one of 0, 1, 2, 4, 8).
    """
    bp = np.asarray(bp, dtype=float)
    tol = np.full(bp.shape, 8, dtype=int)   # >= 600
    tol = np.where(bp < 600, 4, tol)        # [270, 600)
    tol = np.where(bp < 270, 2, tol)        # (120, 270)
    tol = np.where(bp <= 120, 1, tol)       # [50, 120]
    tol = np.where(bp < 50, 0, tol)         # < 50
    return tol


def direction_codes(eh, true, tol):
    """Returns size-tolerance direction codes ``{OK=0, TOO_LONG=1, TOO_SHORT=2}`` per allele.

    ``dr = round(eh) - round(true)``; OK if ``|dr| <= tol``, TOO_LONG if ``dr > tol``
    (EH over-called), TOO_SHORT if ``dr < -tol`` (EH under-called).

    Args:
        eh: Array-like of EH allele calls.
        true: Array-like of truth allele sizes.
        tol: Array-like (or scalar) of per-allele tolerances in repeats.

    Returns:
        Integer ndarray of direction codes.
    """
    dr = np.round(np.asarray(eh, dtype=float)) - np.round(np.asarray(true, dtype=float))
    tol = np.asarray(tol, dtype=float)
    return np.where(dr > tol, TOO_LONG, np.where(dr < -tol, TOO_SHORT, OK)).astype(int)


def regime_of(genotyping_branch, spanning_at_called):
    """Maps each allele to its regime label.

    ``fast`` branch rows -> ``fast``. Full-branch rows split by spanning support
    at the called size: ``>= 1`` spanning read -> ``full_spanning`` else
    ``full_nonspanning`` (NaN spanning count treated as 0 -> non-spanning).

    Args:
        genotyping_branch: Array-like of ``"full"`` / ``"fast"`` per allele.
        spanning_at_called: Array-like spanning-read count at the called size.

    Returns:
        Object ndarray of regime labels (one of ``REGIMES``).
    """
    branch = np.asarray(genotyping_branch, dtype=object)
    span = np.nan_to_num(np.asarray(spanning_at_called, dtype=float), nan=0.0)
    out = np.where(span >= 1, REGIME_FULL_SPANNING, REGIME_FULL_NONSPANNING).astype(object)
    out[branch == REGIME_FAST] = REGIME_FAST
    return out
