"""Fit the deployable per-genotyping_regime experts and export the model ``.json.gz``.

Fits the three genotyping_regime experts (``quick`` / ``full_spanning`` / ``full_nonspanning``),
each a q-median LCF head + a 3-class direction head with per-class isotonic
calibration, on the full real-data pool, serializes them into the schema the EH C++
parses (``GenotypeQualityModel.cpp``, ``format_version == 2``), round-trip-verifies
that the serialized trees/softmax/isotonic reproduce sklearn's predictions, and
writes the gzipped JSON.

Each genotyping_regime reads its branch parquet (``quick`` reads ``data/parquet/quick.parquet``;
the two full genotyping_regimes read ``data/parquet/full.parquet``) and is restricted to its
own ``genotyping_regime`` rows. The isotonic calibration is fit on 5 held-out people plus HG002's and
CHM1_CHM13's rows on one autosome (``_calib_mask``); every chromosome is trained on. Each head is fit
to a fixed number of iterations per regime (``ITERATIONS_BY_REGIME``, chosen from learning curves), as
early stopping on held-out people never stops. The training rows are capped by a seeded quota sample
spread over motif type and called allele size (``dataset.quota_sample``), and the q-median head is fit
only on rows the gate fires on (``fit_heads``).
Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import datetime
import gzip
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import dataset
import features
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
FORMAT_VERSION = 2


def _reject_nonfinite(tok):
    """``json.loads`` ``parse_constant`` hook: fail if the model JSON carries Infinity/NaN tokens."""
    raise ValueError("model JSON contains non-finite token %r (invalid standard JSON)" % tok)


def _default_out(training_population="variant_loci", length_head_training_population=None,
                 hom_ref_locus_direction_training_rows=None, quota_by_motif_size_category=False,
                 full_nonspanning_balance_too_long_and_too_short_by_called_size=False):
    """Returns the default dated output path (today's date).

    A non-default training population, q-median head population, number of hom-ref-locus direction rows, quota by
    motif-size category or full_nonspanning too-long / too-short balancing goes into the name, before the date (which
    ``compare_models._model_sort_key`` reads), so such a model never overwrites today's default one.
    """
    today = datetime.date.today().strftime("%Y%m%d")
    choices = "" if training_population == "variant_loci" else "_" + training_population
    if length_head_training_population not in (None, training_population):
        choices += "_length_head_%s" % length_head_training_population
    if hom_ref_locus_direction_training_rows is not None:
        choices += "_variant_locus_direction_rows_plus_%d_hom_ref_locus_rows" % hom_ref_locus_direction_training_rows
    if quota_by_motif_size_category:
        choices += "_quota_by_motif_size_category"
    if full_nonspanning_balance_too_long_and_too_short_by_called_size:
        choices += "_full_nonspanning_too_long_too_short_balanced_by_called_size"
    return os.path.join(HERE, "model",
                        "genotype_quality_model_from_HG002_and_CHM1_CHM13%s.%s.json.gz" % (choices, today))


# People who are never held out whole: HG002 is the only source at low coverage (10x/20x) and, with
# CHM1_CHM13, has the most accurate truth. Each gives one autosome to the calibration set
# (_calib_mask) and trains on all its other chromosomes.
ALWAYS_TRAIN_INDIVIDUALS = ("HG002", "CHM1_CHM13")
N_CALIB_INDIVIDUALS = 5

# Boosting iterations of the exported model, (direction head, q-median head) per regime. Chosen on
# 2026-10-06 from learning curves (model_size_curves.py, --pick --tolerance 0.02; 10 validation people
# and chromosomes 1/4/6/12 held out of all fitting): with people rather than chromosomes held out, the held-out loss keeps
# improving to any ceiling, because more trees keep learning catalog loci that new people share (98.8-
# 99.9% of a new person's variant calls are at loci some training person is variant at), so early
# stopping never stops. The curves' checkpoints pair d direction iterations with 2d q-median
# iterations (d = 100 ... 4,000); each regime's size is the smallest checkpoint whose gated MAE on new
# people at known loci is within 2% of the (4,000, 8,000) fit's, for non-homopolymers and homopolymers
# alike. On unseen loci the gated MAE was flat by then (no sign of harmful overfitting). Stronger
# regularization (min_samples_leaf 3,000) was worse at every size. The quick regime was then cut by
# hand from that rule's 500 to 150 for runtime (so --pick does not reproduce it): quick calls are ~93%
# of the alleles ExpansionHunter scores (HG002 chr22, every catalog locus), so at 500 its 2,500 trees
# per allele made up ~69% of all tree evaluations. Over a 1-tree model, optimized-streaming user CPU
# time (1 thread) rose 6.3% at 500 and 4.5% at 150. The quick curve is nearly flat: 150 is not a checkpoint, but
# it lies between 100 (gated MAE 1.9% non-homopolymer / 4.7% homopolymer above the 4,000-iteration fit
# on new people at known loci) and 250 (1.2% / 2.9%).
ITERATIONS_BY_REGIME = {
    features.GENOTYPING_REGIME_QUICK: (150, 300),
    features.GENOTYPING_REGIME_FULL_SPANNING: (3000, 6000),
    features.GENOTYPING_REGIME_FULL_NONSPANNING: (3000, 6000),
}


def _calibration_people(individual, chrom, seed):
    """Returns ``(held, autosome)``: the ``N_CALIB_INDIVIDUALS`` people ``_calib_mask`` holds out whole (seeded) and
    the autosome of the ``ALWAYS_TRAIN_INDIVIDUALS`` it holds out. Spiked samples (``dataset.is_spiked_sample``) are
    never drawn as calibration people and don't change which people are, so a table with them calibrates on the same
    people; an always-trained person's spiked rows on that autosome are held out with the person's (``_calib_mask``).

    Raises:
        RuntimeError: if a held-out person has spiked samples in the table: their reads would then also train the
            model. Spike other people.
    """
    individual = np.asarray(individual, dtype=object)
    chrom = np.asarray(chrom, dtype=object)
    rng = np.random.default_rng(seed)
    candidates = sorted(i for i in set(individual.tolist()) - set(ALWAYS_TRAIN_INDIVIDUALS)
                        if not dataset.is_spiked_sample(i))
    held = rng.choice(candidates, N_CALIB_INDIVIDUALS, replace=False).tolist()
    spiked_people = {dataset.person_of_sample(i) for i in set(individual.tolist()) if dataset.is_spiked_sample(i)}
    if spiked_people & set(held):
        raise RuntimeError("calibration people %s have spiked samples in the table, so their reads would also train "
                           "the model; spike other people" % sorted(spiked_people & set(held)))
    always = np.isin(individual, ALWAYS_TRAIN_INDIVIDUALS)
    autosomes = sorted(c for c in set(chrom[always].tolist()) if str(c).isdigit())
    return held, rng.choice(autosomes).item()


def _calib_mask(individual, chrom, seed, people=None):
    """Returns the isotonic-calibration rows: ``N_CALIB_INDIVIDUALS`` whole people (seeded) plus the
    ``ALWAYS_TRAIN_INDIVIDUALS`` rows on one autosome.

    Holding out people rather than chromosomes lets the trees train on every chromosome, so all loci
    of the catalog are learned. That matches how the model is used: the same catalog loci, genotyped in
    new people. (With chromosomes held out, the 2026-10-05 model's gain on the held-out samples fell by
    up to two thirds on its 4 unseen chromosomes.) The single chromosome of HG002 and CHM1_CHM13 rows
    is there because they are the only 10x, 20x and 46x sources (the other people are 33-41x), so
    without them the calibration would never see low or high coverage; those loci are still learned
    from everyone else.

    Args:
        individual: Per-row person (``dataset.individual_of_part``).
        chrom: Per-row chromosome (no ``chr`` prefix).
        seed: RNG seed.
        people: ``_calibration_people`` of another table of the same people, so both hold out the same ones;
            default: drawn from these rows.
    """
    individual = np.asarray(individual, dtype=object)
    chrom = np.asarray(chrom, dtype=object)
    held, calib_chrom = people or _calibration_people(individual, chrom, seed)
    # A spiked sample holds its person's reads, so its rows on that chromosome are held out with the person's.
    always = np.isin(_persons_of_samples(individual), ALWAYS_TRAIN_INDIVIDUALS)
    return np.isin(individual, held) | (always & (chrom == calib_chrom))


def _persons_of_samples(individual):
    """The person whose reads each row's sample holds (``dataset.person_of_sample``), as an object array."""
    individual = np.asarray(individual, dtype=object)
    return pd.Series(individual).map({i: dataset.person_of_sample(i) for i in set(individual.tolist())}).to_numpy(
        object)


def _cap(idx, cap, seed):
    """Seeded subsample of an index array down to ``cap`` rows."""
    if cap and idx.size > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


# Boosting iterations of the two direction heads fit only to decide which training rows the gate would
# correct (see _out_of_fold_gate_fires). Measured on 2026-10-05 (full_spanning, held-back training
# chromosomes, otherwise identical recipe): gating heads limited to 500 iterations gave a gated MAE of
# 8.715 non-homopolymer / 0.842 homopolymer, against 8.738 / 0.850 with gating heads at 3000, so the
# cheaper size cost nothing measurable. The 2026-10-06 learning curves that set ITERATIONS_BY_REGIME
# used gating heads of exactly this size.
GATING_DIRECTION_ITERATIONS = 500
_PREDICT_CHUNK_ROWS = 500_000


def _predict_p_ok_in_chunks(dmodel, df, idx, branch):
    """Returns the deployed-rounding pOk for rows ``idx`` of ``df``, building the matrix in chunks."""
    return np.concatenate([
        M.round_like_emitted(M.predict_proba(
            dmodel, features.build_matrix(df.iloc[idx[i:i + _PREDICT_CHUNK_ROWS]], branch)[0])[:, features.OK])
        for i in range(0, idx.size, _PREDICT_CHUNK_ROWS)])


# Called-size bins (lower edges in bp; the last bin is open-ended) in which
# ``too_long_too_short_balancing_weights`` gives the too-long and too-short rows equal weight. They start at 150 bp,
# the training samples' read length: past it ExpansionHunter sizes alleles poorly, and in the training genomes the
# longer the call, the more often it is too long (held-out full genotyper without spanning reads, 2026-10-29: 0.9
# too-long rows per too-short row at 150-200 bp, 1.6 at 250-300 bp, 4.9 at 450 bp and above), while the expansions of
# patients are mostly called too short. 50 bp wide up to 450 bp, then one bin, since few calls are longer.
TOO_LONG_TOO_SHORT_BALANCING_CALLED_BP_BIN_EDGES = (150, 200, 250, 300, 350, 400, 450)


def too_long_too_short_balancing_weights(called_bp, dir_code, bin_edges=TOO_LONG_TOO_SHORT_BALANCING_CALLED_BP_BIN_EDGES):
    """Returns per-row weights that give the too-long and too-short rows of each called-size bin equal total weight.

    In each bin (``bin_edges``: lower edges in bp, the last bin open-ended), the TOO_LONG and TOO_SHORT rows' weights
    are scaled so that each class carries half of their combined row count. OK rows, rows below the first edge or
    without a called size, and bins that lack either class keep weight 1. So each bin's OK weight and total weight
    are unchanged, and the called size alone no longer tells a too-long call from a too-short one there. The heads
    trained and calibrated with these weights are shared by calls of every size, so the predictions of calls below the
    first edge can move too (``fit_heads``).

    Args:
        called_bp: Per-row called allele size in bp (``motif_size * round(eh)``).
        dir_code: Per-row direction class code (``features.OK``, ``TOO_LONG`` or ``TOO_SHORT``).
        bin_edges: Increasing lower edges of the bins, in bp.
    """
    called_bp, dir_code = np.asarray(called_bp, dtype=float), np.asarray(dir_code, dtype=int)
    bin_index = np.where(np.isfinite(called_bp), np.searchsorted(bin_edges, called_bp, side="right") - 1, -1)
    weights = np.ones(dir_code.size)
    for b in range(len(bin_edges)):
        too_long = (bin_index == b) & (dir_code == features.TOO_LONG)
        too_short = (bin_index == b) & (dir_code == features.TOO_SHORT)
        n_too_long, n_too_short = int(too_long.sum()), int(too_short.sum())
        if n_too_long and n_too_short:
            weights[too_long] = (n_too_long + n_too_short) / (2 * n_too_long)
            weights[too_short] = (n_too_long + n_too_short) / (2 * n_too_short)
    return weights


def called_bp_of(df):
    """Returns each row's called allele size in bp, ``motif_size * round(eh)``."""
    return df["motif_size"].to_numpy(float) * np.round(df["eh"].to_numpy(float))


def direction_head_sample_weights(df, rows, balance_too_long_and_too_short_by_called_size=False):
    """Returns the sample weights of a direction head fit on ``df``'s ``rows``, or None (unweighted).

    With ``balance_too_long_and_too_short_by_called_size``, the too-long and too-short rows of each called-size bin
    get equal weight (``too_long_too_short_balancing_weights``).
    """
    if balance_too_long_and_too_short_by_called_size:
        return too_long_too_short_balancing_weights(called_bp_of(df.iloc[rows]), df["dir_code"].to_numpy(int)[rows])
    return None


def _out_of_fold_gate_fires(df, train_pool, fold_groups, branch, train_cap, X_calib, y_calib, scored=None,
                            hom_ref_locus_rows=None, quota_by_motif_size_category=False,
                            balance_too_long_and_too_short_by_called_size=False,
                            too_long_too_short_calibration_weight=None):
    """Returns, for each row of the scored pool, whether the gate (pOk < 0.5) fires on it out of fold.

    The pool's groups (``fold_groups``: people for the exported model, with HG002 and CHM1_CHM13 split
    by chromosome; chromosomes for the report's cross-validation) are split into two seeded halves; a
    direction head fit on one half's rows of ``train_pool`` predicts the scored rows of the other half, so no row is
    scored by a model that saw its group. Each half's head is sampled like the exported one
    (``direction_training_rows``, with ``hom_ref_locus_rows`` and ``quota_by_motif_size_category``).
    ``balance_too_long_and_too_short_by_called_size`` and ``too_long_too_short_calibration_weight`` are the head's
    row weighting and calibration weighting, as in ``fit_heads``.
    ``scored`` is ``(rows, pool, fold_groups)`` of another table of the same people whose pool is
    judged instead (``fit_heads``' ``length_head_table``); default ``df``'s own ``train_pool``.
    """
    score_df, score_pool, score_groups = scored or (df, train_pool, fold_groups)
    groups, score_groups = np.asarray(fold_groups, dtype=object), np.asarray(score_groups, dtype=object)
    pool_groups = np.array(sorted(set(groups[train_pool].tolist())))
    half = set(np.random.default_rng(SEED + 4).permutation(pool_groups)[:len(pool_groups) // 2].tolist())
    in_half = np.isin(groups[train_pool], list(half))
    scored_in_half = np.isin(score_groups[score_pool], list(half))
    fires = np.zeros(score_pool.size, dtype=bool)
    for fit_mask, score_mask in ((in_half, ~scored_in_half), (~in_half, scored_in_half)):
        fit_rows = direction_training_rows(df, train_pool[fit_mask], train_cap, hom_ref_locus_rows, SEED + 5,
                                           quota_by_motif_size_category)
        Xfit, _ = features.build_matrix(df.iloc[fit_rows], branch)
        weights = direction_head_sample_weights(df, fit_rows, balance_too_long_and_too_short_by_called_size)
        dmodel = M.train_direction(Xfit, df["dir_code"].to_numpy(int)[fit_rows], X_calib, y_calib,
                                   n_iter=GATING_DIRECTION_ITERATIONS, sample_weight=weights,
                                   too_long_too_short_calibration_weight=too_long_too_short_calibration_weight)
        fires[score_mask] = _predict_p_ok_in_chunks(dmodel, score_df, score_pool[score_mask], branch) < 0.5
    return fires


def direction_training_rows(df, train_pool, train_cap, hom_ref_locus_rows=None, seed=SEED,
                            quota_by_motif_size_category=False):
    """Returns the rows a direction head is fit on: a seeded quota sample of ``train_pool`` up to ``train_cap``.

    With ``hom_ref_locus_rows``, the pool's variant-locus rows and its other rows (loci where the truth equals the
    reference, ``variant_locus`` False) are sampled apart: the variant-locus rows up to ``train_cap``, exactly as a
    pool of only those rows would be, plus the other rows up to ``hom_ref_locus_rows``. With
    ``quota_by_motif_size_category``, the quota cells split motifs into the nine motif-size categories instead of
    homopolymer or not (``dataset.quota_sample``).
    """
    motif, eh = df["motif_size"].to_numpy(float), df["eh"].to_numpy(float)
    if hom_ref_locus_rows is None:
        return dataset.quota_sample(motif, eh, train_pool, train_cap, seed, quota_by_motif_size_category)
    variant_locus = df["variant_locus"].to_numpy(bool)[train_pool]
    return np.sort(np.concatenate([
        dataset.quota_sample(motif, eh, train_pool[variant_locus], train_cap, seed, quota_by_motif_size_category),
        dataset.quota_sample(motif, eh, train_pool[~variant_locus], hom_ref_locus_rows, seed + 7,
                             quota_by_motif_size_category)]))


def fit_heads(df, train_pool, calib_rows, fold_groups, branch, train_cap, iterations, label, length_head_table=None,
              hom_ref_locus_direction_training_rows=None, quota_by_motif_size_category=False,
              balance_too_long_and_too_short_by_called_size=False):
    """Fits one genotyping regime's direction + q-median heads with the shipped recipe.

    Shared by ``fit_genotyping_regime`` (the exported model) and ``report.collect_oof`` (its
    cross-validation), so the report's CV evaluates the recipe that ships. The direction head trains on
    a quota sample of ``train_pool`` (``dataset.quota_sample``) for a fixed number of iterations
    (``ITERATIONS_BY_REGIME``) and is calibrated on ``calib_rows``. The q-median head trains only on pool rows the gate fires on (judged out of
    fold), because ExpansionHunter's LCF is applied only where pOk < 0.5: a median fit on every row is
    dominated by calls that are already right and under-corrects the ones the gate actually corrects.
    The 2026-10-05 experiments on held-back training chromosomes cut the full_spanning gated MAE from
    9.85 to 8.74 (non-homopolymer) and from 1.09 to 0.85 (homopolymer).

    Args:
        df: The regime's rows, with a default RangeIndex (``df.iloc`` positions = row indices).
        train_pool: Row indices training may draw from (no calib or test rows).
        calib_rows: Row indices of the isotonic-calibration set.
        fold_groups: Per-row group labels for the out-of-fold gating split (``_out_of_fold_gate_fires``):
            the person for the exported model, the chromosome for the report's chromosome CV.
        branch: Feature contract (``features.BRANCH_QUICK`` / ``features.BRANCH_FULL``).
        train_cap: Max rows per head fit (None = no cap).
        iterations: ``(direction iterations, q-median iterations)``, e.g. ``ITERATIONS_BY_REGIME[regime]``.
        label: Prefix for progress lines.
        length_head_table: Optional ``(rows, pool, fold_groups)`` of another table of the same people (with ``t``)
            that the q-median head draws its rows from instead of ``df``'s ``train_pool``: the rows of its pool where
            the gate fires, judged out of fold by direction heads fit on ``df``. Decision D3 (b) of
            ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md: the direction head learns from all loci, the length
            correction from the variant-locus table it is measured on.
        hom_ref_locus_direction_training_rows: Optional number of ``train_pool`` rows at loci where the truth equals
            the reference (``variant_locus`` False) that the direction heads (exported and gating) train on, sampled
            apart from the variant-locus rows, which keep their own ``train_cap`` (``direction_training_rows``).
            Default: one quota sample of the whole pool.
        quota_by_motif_size_category: Quota-sample every head's training rows (exported and gating direction heads,
            q-median head) with one cell per motif-size category x called size bin (``dataset.quota_sample``), so
            each motif size gets its own quota instead of homopolymers taking up to half. Rare categories can fall
            short of it where the branch table's per-source cap kept few of their rows (``dataset.quota_sample``).
            Default: homopolymer or not x called size bin.
        balance_too_long_and_too_short_by_called_size: Give the direction heads' (exported and gating) too-long and
            too-short training rows equal weight in each called-size bin from 150 bp up
            (``too_long_too_short_balancing_weights``), and weight the TOO_LONG and TOO_SHORT calibrators' calibration
            rows the same way. Because the calibration is weighted too, this changes the reported pTooLong and
            pTooShort of calls in those bins: they no longer follow how often such calls are too long or too short in
            the training genomes (mostly too long), but treat both directions as equally likely at that size. pOk's
            calibrator stays unweighted, but the reported pOk and the pOk < 0.5 gate (here and in ExpansionHunter, and
            so the q-median head's training rows) can still move, because the calibrated values are divided by their
            sum. The trees and calibrators are shared by calls of every size in the regime, so calls below 150 bp can
            move as well. How much is measured on held-out people, not guaranteed here. Default: unweighted.

    Returns:
        ``(qreg, dmodel, feature_names, X_calib)``.
    """
    motif = df["motif_size"].to_numpy(float)
    y = df["dir_code"].to_numpy(int)
    tr = direction_training_rows(df, train_pool, train_cap, hom_ref_locus_direction_training_rows,
                                 quota_by_motif_size_category=quota_by_motif_size_category)
    ca = calib_rows
    Xca, _ = features.build_matrix(df.iloc[ca], branch)
    print("  [%s] fit direction (%d iterations) on %d train (%.0f%% homopolymer) / %d calib ..."
          % (label, iterations[0], tr.size, 100 * (motif[tr] == 1).mean(), ca.size), flush=True)
    weights = direction_head_sample_weights(df, tr, balance_too_long_and_too_short_by_called_size)
    calibration_weights = None
    if balance_too_long_and_too_short_by_called_size:
        calibration_weights = too_long_too_short_balancing_weights(called_bp_of(df.iloc[ca]), y[ca])
        for name, rows, row_weights in (("training", tr, weights), ("calibration", ca, calibration_weights)):
            called_bp = called_bp_of(df.iloc[rows])
            balanced = called_bp >= TOO_LONG_TOO_SHORT_BALANCING_CALLED_BP_BIN_EDGES[0]
            print("  [%s] %s rows called %d bp or longer: %d; too-long rows weighted %s, too-short rows %s"
                  % (label, name, TOO_LONG_TOO_SHORT_BALANCING_CALLED_BP_BIN_EDGES[0], int(balanced.sum()),
                     sorted(set(np.round(row_weights[balanced & (y[rows] == features.TOO_LONG)], 2).tolist())),
                     sorted(set(np.round(row_weights[balanced & (y[rows] == features.TOO_SHORT)], 2).tolist()))),
                  flush=True)
    Xtr, names = features.build_matrix(df.iloc[tr], branch)
    dmodel = M.train_direction(Xtr, y[tr], Xca, y[ca], n_iter=iterations[0], sample_weight=weights,
                               too_long_too_short_calibration_weight=calibration_weights)
    del Xtr

    length_df, length_pool, length_groups = length_head_table or (df, train_pool, fold_groups)
    fires = _out_of_fold_gate_fires(df, train_pool, fold_groups, branch, train_cap, Xca, y[ca],
                                    scored=(length_df, length_pool, length_groups),
                                    hom_ref_locus_rows=hom_ref_locus_direction_training_rows,
                                    quota_by_motif_size_category=quota_by_motif_size_category,
                                    balance_too_long_and_too_short_by_called_size=(
                                        balance_too_long_and_too_short_by_called_size),
                                    too_long_too_short_calibration_weight=calibration_weights)
    qtr = dataset.quota_sample(length_df["motif_size"].to_numpy(float), length_df["eh"].to_numpy(float),
                               length_pool[fires], train_cap, SEED + 6, quota_by_motif_size_category)
    print("  [%s] fit q-median (%d iterations) on %d train rows the gate fires on (%d of %d pool rows fire) ..."
          % (label, iterations[1], qtr.size, int(fires.sum()), length_pool.size), flush=True)
    Xq, _ = features.build_matrix(length_df.iloc[qtr], branch)
    qreg = M.train_q_median(Xq, length_df["t"].to_numpy(float)[qtr], n_iter=iterations[1])
    return qreg, dmodel, names, Xca


def _fold_groups(individual, chrom):
    """Returns the out-of-fold gating groups (``_out_of_fold_gate_fires``): people, except that HG002 (the only
    10x/20x source) and CHM1_CHM13 (46x) are split by chromosome, so each half's head sees low and high coverage. A
    spiked sample joins its person's group (``dataset.person_of_sample``), as it holds that person's reads."""
    person = _persons_of_samples(individual)
    always = np.isin(person, ALWAYS_TRAIN_INDIVIDUALS)
    return np.where(always, person + "|" + np.asarray(chrom, dtype=object), person)


def _length_head_table(parquet, genotyping_regime, calibration_people):
    """Returns ``fit_heads``' ``length_head_table`` from another branch parquet of the same people: its rows of
    ``genotyping_regime``, the pool without the rows of ``calibration_people`` (``_calibration_people`` of the
    direction head's table, so both tables hold out the same people), and the rows' ``_fold_groups``."""
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"t", "chrom", "genotyping_regime", "individual"})
    df = pd.read_parquet(parquet, columns=[c for c in pq.ParquetFile(parquet).schema.names if c in need])
    df = df[df["genotyping_regime"] == genotyping_regime].reset_index(drop=True)
    individual, chrom = df["individual"].to_numpy().astype(object), df["chrom"].to_numpy().astype(object)
    calib = _calib_mask(individual, chrom, SEED, calibration_people)
    return df, np.where(~calib)[0], _fold_groups(individual, chrom)


def _regime_rows(parquet, columns, genotyping_regime):
    """Returns the ``genotyping_regime`` rows of ``parquet`` (``columns`` only, default RangeIndex)."""
    df = pd.read_parquet(parquet, columns=columns)
    return df[df["genotyping_regime"] == genotyping_regime].reset_index(drop=True)


def fit_genotyping_regime(genotyping_regime, data_dir, train_cap, parquet_subdir="parquet",
                          length_head_parquet_subdir=None, hom_ref_locus_direction_training_rows=None,
                          quota_by_motif_size_category=False, balance_too_long_and_too_short_by_called_size=False):
    """Fits the q-median + direction heads for one genotyping_regime on its full real-data pool.

    Holds out the calibration rows, mostly whole people (``_calib_mask``), then fits with ``fit_heads``
    on every chromosome at the regime's ``ITERATIONS_BY_REGIME``. Reads
    ``data_dir/<parquet_subdir>/<branch>.parquet`` (``dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION``) and
    trains the direction head on its per-allele ``dir_code`` label. With ``length_head_parquet_subdir`` (another
    population's subdir, e.g. "parquet" while ``parquet_subdir`` is "parquet_all_loci"), the q-median head draws its
    rows from that table instead (``fit_heads``' ``length_head_table``), with the same calibration people held out.

    With ``hom_ref_locus_direction_training_rows`` (an all-loci ``parquet_subdir`` and the variant-locus table as
    ``length_head_parquet_subdir``), the direction heads draw their variant-locus rows from the variant-locus table,
    sampled as the variant-locus recipe samples them (for the exported head, the same rows as a model trained on that
    table, since both tables hold out the same people), plus a quota sample of this many rows of the all-loci table
    at loci where the truth equals the reference (``fit_heads``). On the all-loci table alone the training cap leaves
    about half as many expansion rows, and the direction head underfits them. Calibration still uses the all-loci
    table's representative rows. With ``quota_by_motif_size_category``, every head's training rows are spread evenly
    over the nine motif-size categories (``fit_heads``). With ``balance_too_long_and_too_short_by_called_size``, the
    direction heads and their TOO_LONG and TOO_SHORT calibrators give too-long and too-short rows equal weight in each
    called-size bin from 150 bp up (``fit_heads``).

    Returns:
        ``(qreg, dmodel, branch, feature_names, X_check)`` where ``X_check`` is a
        small calib slice used only for round-trip verification.
    """
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, parquet_subdir, "%s.parquet" % src)
    variant_locus_inputs = set() if hom_ref_locus_direction_training_rows is None else {"variant_locus"}
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS) | variant_locus_inputs
            | {"t", "dir_code", "chrom", "genotyping_regime", "individual", "representative"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    missing_label_or_variant_locus_columns = sorted(({"dir_code"} | variant_locus_inputs) - set(cols))
    if missing_label_or_variant_locus_columns:
        raise SystemExit("ERROR: %s has no %s column(s); re-assemble it with %s" % (
            parquet, ", ".join(missing_label_or_variant_locus_columns),
            dataset.command_that_builds_branch_parquets(data_dir, parquet_subdir)))
    df = _regime_rows(parquet, cols, genotyping_regime)
    print("  [%s] pool rows=%d" % (genotyping_regime, len(df)), flush=True)

    missing = [c for c in ("individual", "representative") if c not in df.columns]
    if missing:
        raise SystemExit("ERROR: %s has no %s column(s); re-assemble it with dataset.py" % (parquet, missing))
    individual, chrom = df["individual"].to_numpy().astype(object), df["chrom"].to_numpy().astype(object)
    calibration_people = _calibration_people(individual, chrom, SEED)
    calib = _calib_mask(individual, chrom, SEED, calibration_people)
    print("  [%s] calibration rows: %s" % (genotyping_regime, df[calib].groupby("individual")["chrom"].agg(
        lambda c: "all" if c.nunique() > 1 else "chr%s" % c.iloc[0]).to_dict()), flush=True)
    # Calibration estimates frequencies, so it uses only the representative rows; the training-only
    # quota top-up rows (dataset._cap_rows_per_genotyping_regime) join training only.
    representative = df["representative"].to_numpy(bool)
    held_cap = max(1, train_cap // 5) if train_cap else None
    train_pool, calib_rows = np.where(~calib)[0], _cap(np.where(calib & representative)[0], held_cap, SEED + 2)
    length_head_table = None
    if hom_ref_locus_direction_training_rows is not None:
        # One frame: the variant-locus table's rows first (at the same positions as in that table, so its pool and
        # quota sample are the variant-locus recipe's), then the all-loci table's rows at loci where the truth equals
        # the reference outside the calibration people, then its calibration rows. The variant-locus table's pool is
        # also the q-median head's (D3 (b)).
        length_parquet = os.path.join(data_dir, length_head_parquet_subdir, "%s.parquet" % src)
        hom_ref_locus_pool = np.where(~calib & ~df["variant_locus"].to_numpy(bool))[0]
        all_loci_parts = [df.iloc[hom_ref_locus_pool], df.iloc[calib_rows]]
        del df
        variant_locus_df = _regime_rows(length_parquet, cols, genotyping_regime)
        n_variant_locus = len(variant_locus_df)
        variant_locus_pool = np.where(~_calib_mask(variant_locus_df["individual"].to_numpy().astype(object),
                                                   variant_locus_df["chrom"].to_numpy().astype(object), SEED,
                                                   calibration_people))[0]
        df = pd.concat([variant_locus_df] + all_loci_parts, ignore_index=True)
        del variant_locus_df, all_loci_parts
        train_pool = np.concatenate([variant_locus_pool, n_variant_locus + np.arange(hom_ref_locus_pool.size)])
        calib_rows = n_variant_locus + hom_ref_locus_pool.size + np.arange(calib_rows.size)
        fold_groups = _fold_groups(df["individual"].to_numpy().astype(object), df["chrom"].to_numpy().astype(object))
        length_head_table = (df, variant_locus_pool, fold_groups)
        print("  [%s] direction pool: %d variant-locus rows from %s + %d rows at loci where the truth equals the "
              "reference; q-median head rows from the former" % (
                  genotyping_regime, variant_locus_pool.size, length_parquet, hom_ref_locus_pool.size), flush=True)
    else:
        fold_groups = _fold_groups(individual, chrom)
        if length_head_parquet_subdir not in (None, parquet_subdir):
            length_parquet = os.path.join(data_dir, length_head_parquet_subdir, "%s.parquet" % src)
            length_head_table = _length_head_table(length_parquet, genotyping_regime, calibration_people)
            print("  [%s] q-median head rows from %s: pool rows=%d" % (
                genotyping_regime, length_parquet, length_head_table[1].size), flush=True)
    qreg, dmodel, names, Xca = fit_heads(
        df, train_pool, calib_rows, fold_groups, branch, train_cap,
        ITERATIONS_BY_REGIME[genotyping_regime], genotyping_regime, length_head_table=length_head_table,
        hom_ref_locus_direction_training_rows=hom_ref_locus_direction_training_rows,
        quota_by_motif_size_category=quota_by_motif_size_category,
        balance_too_long_and_too_short_by_called_size=balance_too_long_and_too_short_by_called_size)
    # A seeded sample spans the calibration people and chromosomes; the first rows would be one
    # person's first loci and exercise few tree paths.
    return qreg, dmodel, branch, names, Xca.sample(min(5000, len(Xca)), random_state=SEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--out", help="default: model/genotype_quality_model_from_HG002_and_CHM1_CHM13.<today>.json.gz, "
                                      "with any non-default --training-population / "
                                      "--length-head-training-population / --hom-ref-locus-direction-training-rows / "
                                      "--quota-by-motif-size-category / "
                                      "--full-nonspanning-balance-too-long-and-too-short-by-called-size before the "
                                      "date")
    parser.add_argument("--train-cap", type=int, default=1_000_000,
                        help="max train rows per genotyping_regime fit (0 = no cap)")
    parser.add_argument("--training-population", choices=sorted(dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION),
                        default="variant_loci",
                        help="which assembled branch parquets to train on (dataset.py --training-population)")
    parser.add_argument("--length-head-training-population",
                        choices=sorted(dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION),
                        help="which assembled branch parquets the q-median (length-correction) head draws its rows "
                             "from (default: --training-population); e.g. variant_loci with --training-population "
                             "all_loci trains the direction head on all loci and the length correction on the "
                             "variant-locus rows it is measured on (decision D3 (b) of "
                             "ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md)")
    parser.add_argument("--hom-ref-locus-direction-training-rows", type=int,
                        help="with --training-population all_loci and --length-head-training-population variant_loci: "
                             "train the direction head on the variant-locus recipe's own training sample (up to "
                             "--train-cap rows of the variant-locus table) plus this many rows of the all-loci table at "
                             "loci where the truth equals the reference, such as 500000 (default: a quota sample of the "
                             "all-loci table, which keeps about half as many expansion rows); calibration still uses "
                             "the all-loci table")
    parser.add_argument("--quota-by-motif-size-category", action="store_true",
                        help="quota-sample the training rows with one cell per motif-size category (%s bp) x called "
                             "size bin, so each motif size gets its own quota; rare categories can fall short of it "
                             "where the branch tables' per-source cap kept few of their rows (default: homopolymer "
                             "or not x called size bin, where homopolymers take up to about 40 percent of the rows)"
                             % ", ".join(features.MOTIF_SIZE_CATEGORY_LABELS))
    parser.add_argument("--full-nonspanning-balance-too-long-and-too-short-by-called-size", action="store_true",
                        help="in the full_nonspanning regime (full-genotyper alleles without spanning reads, which "
                             "includes every allele longer than a read), give the direction head's too-long and "
                             "too-short rows equal weight within each called-size bin starting at %s bp, in training "
                             "and in the pTooLong and pTooShort calibration, so a long call's size alone no longer "
                             "makes it likely too long; pOk's own calibrator stays unweighted, but the reported pOk and "
                             "the pOk < 0.5 gate can still move (the probabilities are normalized to sum to 1), and so "
                             "can calls shorter than the first edge, which share the regime's trees and calibrators "
                             "(default: unweighted)"
                             % ", ".join(str(edge) for edge in TOO_LONG_TOO_SHORT_BALANCING_CALLED_BP_BIN_EDGES))
    args = parser.parse_args()
    if args.hom_ref_locus_direction_training_rows is not None:
        if args.hom_ref_locus_direction_training_rows <= 0:
            parser.error("--hom-ref-locus-direction-training-rows must be a positive number of rows, such as 500000")
        if args.training_population != "all_loci" or args.length_head_training_population != "variant_loci":
            parser.error("--hom-ref-locus-direction-training-rows needs --training-population all_loci and "
                         "--length-head-training-population variant_loci (the variant-locus table supplies the "
                         "variant-locus rows)")
    args.out = args.out or _default_out(args.training_population, args.length_head_training_population,
                                        args.hom_ref_locus_direction_training_rows, args.quota_by_motif_size_category,
                                        args.full_nonspanning_balance_too_long_and_too_short_by_called_size)
    parquet_subdir = dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION[args.training_population]
    length_head_parquet_subdir = (None if args.length_head_training_population is None
                                  else dataset.BRANCH_PARQUET_SUBDIR_BY_TRAINING_POPULATION[
                                      args.length_head_training_population])

    # Refuse to train on parquets that are missing or older than their upstream JSON/TSV sources.
    for subdir in sorted({parquet_subdir, length_head_parquet_subdir or parquet_subdir}):
        dataset.assert_parquets_up_to_date(args.data_dir, parquet_subdir=subdir)

    genotyping_regimes_json = {}
    feat_names = {}
    for genotyping_regime in features.GENOTYPING_REGIMES:
        print("==== %s ====" % genotyping_regime, flush=True)
        qreg, dmodel, branch, names, X_check = fit_genotyping_regime(
            genotyping_regime, args.data_dir, args.train_cap or None, parquet_subdir=parquet_subdir,
            length_head_parquet_subdir=length_head_parquet_subdir,
            hom_ref_locus_direction_training_rows=args.hom_ref_locus_direction_training_rows,
            quota_by_motif_size_category=args.quota_by_motif_size_category,
            balance_too_long_and_too_short_by_called_size=(
                args.full_nonspanning_balance_too_long_and_too_short_by_called_size
                and genotyping_regime == features.GENOTYPING_REGIME_FULL_NONSPANNING))
        feat_names[branch] = names
        genotyping_regimes_json[genotyping_regime] = M.serialize_genotyping_regime(qreg, dmodel)
        M.verify_genotyping_regime(genotyping_regime, qreg, dmodel, genotyping_regimes_json[genotyping_regime], X_check)
        print("  [%s] %d q-trees, %d dir-iters"
              % (genotyping_regime, len(genotyping_regimes_json[genotyping_regime]["q_median"]["trees"]),
                 len(genotyping_regimes_json[genotyping_regime]["direction"]["trees"])), flush=True)

    model = {"format_version": FORMAT_VERSION,
             "feature_names": {"quick": feat_names["quick"], "full": feat_names["full"]},
             "genotyping_regimes": genotyping_regimes_json}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # allow_nan=False => raise rather than emit non-standard Infinity/NaN tokens (the serializer
    # clamps sklearn's +/-inf split thresholds; this guards anything else slipping through).
    blob = json.dumps(model, separators=(",", ":"), allow_nan=False).encode()
    # gzip.open would stamp the current time and the output filename into the gzip header, so two
    # runs that fit byte-identical trees would still produce different archives. Writing through
    # GzipFile with mtime=0 and an empty filename keeps the whole artifact a function of SEED + data.
    with open(args.out, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as f:
            f.write(blob)
    with gzip.open(args.out, "rb") as f:
        json.loads(f.read(), parse_constant=_reject_nonfinite)  # confirm it reloads + is finite
    print("\nwrote %s (%d quick / %d full features, json %d bytes, gz %d bytes)"
          % (args.out, len(feat_names["quick"]), len(feat_names["full"]), len(blob),
             os.path.getsize(args.out)), flush=True)


if __name__ == "__main__":
    main()
