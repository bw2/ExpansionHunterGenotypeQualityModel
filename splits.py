"""Chromosome-group cross-validation folds and cross-domain splits.

All evaluation in the genotype-quality project splits data by *group* -- never by
random row -- so that no locus, allele, or chromosome leaks between train and
test (SPEC sec "Module interfaces", ``splits.py``). Three kinds of split live
here:

- ``make_cv_folds`` -- Monte-Carlo 10-fold CV over the 24 chromosomes. Each fold
  independently draws 5 test chroms from the 24; from the remaining 19 it draws 2
  calibration chroms and keeps the other 17 for training. Reproducible for a
  given seed; persisted to / loaded from JSON.
- ``cross_sample_split`` -- HG002 -> CHM cross-sample generalization (held-out
  sample). ``cross_coverage_splits`` -- leave-one-HG002-coverage-out, which is a
  coverage-ROBUSTNESS check on the SAME (seen) loci, not a held-out-locus split
  (see its docstring).
- ``cross_domain_splits`` -- real<->sim transfer splits.

Determinism (SPEC sec "Determinism"): the only randomness is the chrom draw in
``make_cv_folds``, driven by ``numpy.random.default_rng(SEED)``. No module-level
mutable state.
"""

import json

import numpy as np
import pandas as pd


SEED = 20260616
ALL_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


def make_cv_folds(chroms=ALL_CHROMS, n_folds=10, n_test=5, n_calib=2, seed=SEED):
    """Builds Monte-Carlo chromosome CV folds.

    Each of the ``n_folds`` folds is an independent random draw: ``n_test``
    chroms are sampled (without replacement) from ``chroms`` for the test set,
    ``n_calib`` more from the remainder for calibration, and the rest become the
    training set. Folds are independent of one another (a chrom may be in the
    test set of several folds) -- this is Monte-Carlo CV, not a partition of the
    chroms into disjoint folds. A single seeded RNG drives all folds, so the
    whole fold list is reproducible for a given ``seed``.

    Args:
        chroms: Sequence of chromosome names to split; defaults to ``ALL_CHROMS``.
        n_folds: Number of folds to generate.
        n_test: Number of test chroms drawn per fold.
        n_calib: Number of calibration chroms drawn (from the non-test chroms).
        seed: RNG seed for the chrom draws.

    Returns:
        A list of ``n_folds`` dicts, each ``{"test": [...], "calib": [...],
        "train": [...]}`` whose three lists partition ``chroms`` (sizes
        ``n_test`` / ``n_calib`` / ``len(chroms) - n_test - n_calib``).
    """
    rng = np.random.default_rng(seed)
    chroms = list(chroms)
    folds = []
    for _ in range(n_folds):
        test = [str(c) for c in rng.choice(chroms, size=n_test, replace=False)]
        remaining = [c for c in chroms if c not in test]
        calib = [str(c) for c in rng.choice(remaining, size=n_calib, replace=False)]
        train = [c for c in remaining if c not in calib]
        folds.append({"test": test, "calib": calib, "train": train})
    return folds


def save_folds(folds, path):
    """Writes ``folds`` to ``path`` as indented JSON.

    Args:
        folds: Fold list as returned by ``make_cv_folds``.
        path: Destination JSON path.
    """
    with open(path, "w") as f:
        json.dump(folds, f, indent=2)


def load_folds(path):
    """Loads a fold list previously written by ``save_folds``.

    Args:
        path: Source JSON path.

    Returns:
        The fold list (same structure as ``make_cv_folds`` returns).
    """
    with open(path) as f:
        return json.load(f)


def fold_masks(df, fold):
    """Returns ``(train, calib, test)`` boolean row masks for one fold.

    Membership is decided by ``df["chrom"]`` (compared as strings) against the
    fold's ``train`` / ``calib`` / ``test`` chrom lists.

    Args:
        df: DataFrame with a ``chrom`` column.
        fold: One fold dict from ``make_cv_folds``.

    Returns:
        A ``(train_mask, calib_mask, test_mask)`` tuple of boolean numpy arrays.
    """
    chrom = df["chrom"].astype(str)
    return (chrom.isin(set(fold["train"])).to_numpy(),
            chrom.isin(set(fold["calib"])).to_numpy(),
            chrom.isin(set(fold["test"])).to_numpy())


def cross_sample_split(df):
    """Returns the HG002 -> CHM cross-sample masks.

    Train on every HG002 row (``sample`` starting with ``"HG002"``), test on
    every CHM row (``sample`` starting with ``"CHM"``).

    Args:
        df: DataFrame with a ``sample`` column.

    Returns:
        A ``(train_mask, test_mask)`` tuple of boolean numpy arrays.
    """
    sample = df["sample"].astype(str)
    return (sample.str.startswith("HG002").to_numpy(),
            sample.str.startswith("CHM").to_numpy())


def cross_coverage_splits(df):
    """Returns leave-one-coverage-out splits within HG002 (10x / 20x / 31x).

    For each held-out coverage the test mask is the HG002 rows at that coverage
    and the train mask is the HG002 rows at the other two coverages. CHM (46x) is
    never included on either side.

    CAVEAT (not a held-out-locus split): HG002 is the same sample sequenced at all
    three coverages, so every test locus is ALSO in the training set at the other
    two coverages -- only the coverage differs. This therefore measures coverage
    ROBUSTNESS on SEEN loci, not generalization to unseen loci; a model could
    memorize a per-locus eh->true mapping and still score well here. Report it as
    coverage robustness, not as a leak-free generalization number.

    Args:
        df: DataFrame with ``sample`` and ``coverage`` columns.

    Returns:
        A list of ``(held_out_coverage, train_mask, test_mask)`` tuples, one per
        coverage in ``(10, 20, 31)``.
    """
    is_hg002 = df["sample"].astype(str).str.startswith("HG002").to_numpy()
    coverage = pd.to_numeric(df["coverage"], errors="coerce").to_numpy()
    return [(held, is_hg002 & (coverage != held), is_hg002 & (coverage == held))
            for held in (10, 20, 31)]


def cross_domain_splits(df):
    """Returns the two real<->sim transfer splits.

    One split trains on ``source == "real"`` and tests on ``source == "sim"``;
    the other reverses the two.

    Args:
        df: DataFrame with a ``source`` column.

    Returns:
        A list of two ``(name, train_mask, test_mask)`` tuples.
    """
    source = df["source"].astype(str).to_numpy()
    return [("train_real_test_sim", source == "real", source == "sim"),
            ("train_sim_test_real", source == "sim", source == "real")]
