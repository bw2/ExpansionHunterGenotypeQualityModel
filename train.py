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


def _default_out():
    """Returns the default dated output path (today's date)."""
    today = datetime.date.today().strftime("%Y%m%d")
    return os.path.join(HERE, "model",
                        "genotype_quality_model_from_HG002_and_CHM1_CHM13.%s.json.gz" % today)


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


def _calib_mask(individual, chrom, seed):
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
    """
    individual = np.asarray(individual, dtype=object)
    chrom = np.asarray(chrom, dtype=object)
    rng = np.random.default_rng(seed)
    candidates = sorted(set(individual.tolist()) - set(ALWAYS_TRAIN_INDIVIDUALS))
    held = rng.choice(candidates, N_CALIB_INDIVIDUALS, replace=False).tolist()
    always = np.isin(individual, ALWAYS_TRAIN_INDIVIDUALS)
    autosomes = sorted(c for c in set(chrom[always].tolist()) if str(c).isdigit())
    calib_chrom = rng.choice(autosomes).item()
    return np.isin(individual, held) | (always & (chrom == calib_chrom))


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
        M.round_like_emitted(M.predict_proba(dmodel, features.build_matrix(df.iloc[idx[i:i + _PREDICT_CHUNK_ROWS]],
                                                                          branch)[0])[:, 0])
        for i in range(0, idx.size, _PREDICT_CHUNK_ROWS)])


def _out_of_fold_gate_fires(df, train_pool, fold_groups, branch, train_cap, X_calib, y_calib):
    """Returns, for each row of ``train_pool``, whether the gate (pOk < 0.5) fires on it out of fold.

    The pool's groups (``fold_groups``: people for the exported model, with HG002 and CHM1_CHM13 split
    by chromosome; chromosomes for the report's cross-validation) are split into two seeded halves; a
    direction head fit on one half predicts the other, so no row is scored by a model that saw it.
    """
    groups = np.asarray(fold_groups, dtype=object)
    pool_groups = np.array(sorted(set(groups[train_pool].tolist())))
    half = set(np.random.default_rng(SEED + 4).permutation(pool_groups)[:len(pool_groups) // 2].tolist())
    in_half = np.isin(groups[train_pool], list(half))
    motif, eh = df["motif_size"].to_numpy(float), df["eh"].to_numpy(float)
    fires = np.zeros(train_pool.size, dtype=bool)
    for fit_mask in (in_half, ~in_half):
        fit_rows = dataset.quota_sample(motif, eh, train_pool[fit_mask], train_cap, SEED + 5)
        Xfit, _ = features.build_matrix(df.iloc[fit_rows], branch)
        dmodel = M.train_direction(Xfit, df["dir_code"].to_numpy(int)[fit_rows], X_calib, y_calib,
                                   n_iter=GATING_DIRECTION_ITERATIONS)
        fires[~fit_mask] = _predict_p_ok_in_chunks(dmodel, df, train_pool[~fit_mask], branch) < 0.5
    return fires


def direction_training_rows(df, train_pool, train_cap):
    """Returns the rows ``fit_heads`` fits the direction head on: a seeded quota sample of ``train_pool``."""
    return dataset.quota_sample(df["motif_size"].to_numpy(float), df["eh"].to_numpy(float), train_pool,
                                train_cap, SEED)


def fit_heads(df, train_pool, calib_rows, fold_groups, branch, train_cap, iterations, label):
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

    Returns:
        ``(qreg, dmodel, feature_names, X_calib)``.
    """
    motif, eh = df["motif_size"].to_numpy(float), df["eh"].to_numpy(float)
    t, y = df["t"].to_numpy(float), df["dir_code"].to_numpy(int)
    tr = direction_training_rows(df, train_pool, train_cap)
    ca = calib_rows
    Xca, _ = features.build_matrix(df.iloc[ca], branch)
    print("  [%s] fit direction (%d iterations) on %d train (%.0f%% homopolymer) / %d calib ..."
          % (label, iterations[0], tr.size, 100 * (motif[tr] == 1).mean(), ca.size), flush=True)
    Xtr, names = features.build_matrix(df.iloc[tr], branch)
    dmodel = M.train_direction(Xtr, y[tr], Xca, y[ca], n_iter=iterations[0])
    del Xtr

    fires = _out_of_fold_gate_fires(df, train_pool, fold_groups, branch, train_cap, Xca, y[ca])
    qtr = dataset.quota_sample(motif, eh, train_pool[fires], train_cap, SEED + 6)
    print("  [%s] fit q-median (%d iterations) on %d train rows the gate fires on (%d of %d pool rows fire) ..."
          % (label, iterations[1], qtr.size, int(fires.sum()), train_pool.size), flush=True)
    Xq, _ = features.build_matrix(df.iloc[qtr], branch)
    qreg = M.train_q_median(Xq, t[qtr], n_iter=iterations[1])
    return qreg, dmodel, names, Xca


def fit_genotyping_regime(genotyping_regime, data_dir, train_cap):
    """Fits the q-median + direction heads for one genotyping_regime on its full real-data pool.

    Holds out the calibration rows, mostly whole people (``_calib_mask``), then fits with ``fit_heads``
    on every chromosome at the regime's ``ITERATIONS_BY_REGIME``.

    Returns:
        ``(qreg, dmodel, branch, feature_names, X_check)`` where ``X_check`` is a
        small calib slice used only for round-trip verification.
    """
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"t", "dir_code", "chrom", "genotyping_regime", "individual", "representative"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    df = pd.read_parquet(parquet, columns=cols)
    df = df[df["genotyping_regime"] == genotyping_regime]
    print("  [%s] pool rows=%d" % (genotyping_regime, len(df)), flush=True)

    df = df.reset_index(drop=True)
    missing = [c for c in ("individual", "representative") if c not in df.columns]
    if missing:
        raise SystemExit("ERROR: %s has no %s column(s); re-assemble it with dataset.py" % (parquet, missing))
    individual, chrom = df["individual"].to_numpy().astype(object), df["chrom"].to_numpy().astype(object)
    calib = _calib_mask(individual, chrom, SEED)
    print("  [%s] calibration rows: %s" % (genotyping_regime, df[calib].groupby("individual")["chrom"].agg(
        lambda c: "all" if c.nunique() > 1 else "chr%s" % c.iloc[0]).to_dict()), flush=True)
    # Calibration estimates frequencies, so it uses only the representative rows; the training-only
    # quota top-up rows (dataset._cap_rows_per_genotyping_regime) join training only.
    representative = df["representative"].to_numpy(bool)
    held_cap = max(1, train_cap // 5) if train_cap else None
    # Out-of-fold gating splits people into halves, but HG002 (the only 10x/20x source) and CHM1_CHM13
    # (46x) are split by chromosome, so each half's head sees low and high coverage.
    always = np.isin(individual, ALWAYS_TRAIN_INDIVIDUALS)
    fold_groups = np.where(always, individual + "|" + chrom, individual)
    qreg, dmodel, names, Xca = fit_heads(
        df, np.where(~calib)[0], _cap(np.where(calib & representative)[0], held_cap, SEED + 2), fold_groups,
        branch, train_cap, ITERATIONS_BY_REGIME[genotyping_regime], genotyping_regime)
    # A seeded sample spans the calibration people and chromosomes; the first rows would be one
    # person's first loci and exercise few tree paths.
    return qreg, dmodel, branch, names, Xca.sample(min(5000, len(Xca)), random_state=SEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--out", default=_default_out())
    parser.add_argument("--train-cap", type=int, default=1_000_000,
                        help="max train rows per genotyping_regime fit (0 = no cap)")
    args = parser.parse_args()

    # Refuse to train on parquets that are missing or older than their upstream JSON/TSV sources.
    dataset.assert_parquets_up_to_date(args.data_dir)

    genotyping_regimes_json = {}
    feat_names = {}
    for genotyping_regime in features.GENOTYPING_REGIMES:
        print("==== %s ====" % genotyping_regime, flush=True)
        qreg, dmodel, branch, names, X_check = fit_genotyping_regime(genotyping_regime, args.data_dir,
                                                          args.train_cap or None)
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
