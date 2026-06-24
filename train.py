"""Fit the deployable per-genotyping_regime experts and export the model ``.json.gz``.

Fits the three genotyping_regime experts (``quick`` / ``full_spanning`` / ``full_nonspanning``),
each a q-median LCF head + a 3-class direction head with per-class isotonic
calibration, on the full real-data pool, serializes them into the schema the EH C++
parses (``GenotypeQualityModel.cpp``, ``format_version == 2``), round-trip-verifies
that the serialized trees/softmax/isotonic reproduce sklearn's predictions, and
writes the gzipped JSON.

Each genotyping_regime reads its branch parquet (``quick`` reads ``data/parquet/quick.parquet``;
the two full genotyping_regimes read ``data/parquet/full.parquet``) and is restricted to its
own ``genotyping_regime`` rows. ~10% of chromosomes are held out (seeded) for the early-stop
calibration set, and the training rows are capped (a seeded subsample) -- HistGBM on
~1M rows is statistically equivalent to the full pool at ``min_samples_leaf=300`` but
far faster. Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import datetime
import gzip
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import features
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260616
FORMAT_VERSION = 2


def _default_out():
    """Returns the default dated output path (today's date)."""
    today = datetime.date.today().strftime("%Y%m%d")
    return os.path.join(HERE, "model",
                        "genotype_quality_model_from_HG002_and_CHM1_CHM13.%s.json.gz" % today)


def _calib_mask(chrom, seed):
    """Boolean mask holding out ~10% of chromosomes (seeded) for the early-stop calib set."""
    ch = np.array(sorted(set(chrom[pd.notna(chrom)].tolist())))
    n = max(1, int(round(0.1 * len(ch))))
    held = set(np.random.default_rng(seed).choice(ch, n, replace=False).tolist())
    return np.isin(chrom.astype(object), list(held))


def _cap(idx, cap, seed):
    """Seeded subsample of an index array down to ``cap`` rows."""
    if cap and idx.size > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def fit_genotyping_regime(genotyping_regime, data_dir, train_cap):
    """Fits the q-median + direction heads for one genotyping_regime on its full real-data pool.

    Returns:
        ``(qreg, dmodel, branch, feature_names, X_check)`` where ``X_check`` is a
        small calib slice used only for round-trip verification.
    """
    branch = features.GENOTYPING_REGIME_BRANCH[genotyping_regime]
    src = "quick" if genotyping_regime == features.GENOTYPING_REGIME_QUICK else "full"
    parquet = os.path.join(data_dir, "parquet", "%s.parquet" % src)
    need = (set(features.FULL_FEATURES) | set(features.ENGINEERED_RAW_INPUTS)
            | {"t", "dir_code", "chrom", "genotyping_regime"})
    cols = [c for c in pq.ParquetFile(parquet).schema.names if c in need]
    df = pd.read_parquet(parquet, columns=cols)
    df = df[df["genotyping_regime"] == genotyping_regime]
    print("  [%s] pool rows=%d" % (genotyping_regime, len(df)), flush=True)

    calib = _calib_mask(df["chrom"].to_numpy(), SEED)
    tr = _cap(np.where(~calib)[0], train_cap, SEED)
    ca = _cap(np.where(calib)[0], max(1, train_cap // 5) if train_cap else None, SEED + 1)
    Xtr, names = features.build_matrix(df.iloc[tr], branch)
    Xca, _ = features.build_matrix(df.iloc[ca], branch)
    ttr, tca = df["t"].to_numpy(float)[tr], df["t"].to_numpy(float)[ca]
    ytr, yca = df["dir_code"].to_numpy(int)[tr], df["dir_code"].to_numpy(int)[ca]
    print("  [%s] fit q-median + direction on %d train / %d calib ..." % (genotyping_regime, tr.size, ca.size),
          flush=True)
    qreg = M.train_q_median(Xtr, ttr, Xca, tca)
    dmodel = M.train_direction(Xtr, ytr, Xca, yca)
    return qreg, dmodel, branch, names, Xca.iloc[:5000]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--out", default=_default_out())
    parser.add_argument("--train-cap", type=int, default=1_000_000,
                        help="max train rows per genotyping_regime fit (0 = no cap)")
    args = parser.parse_args()

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
    blob = json.dumps(model, separators=(",", ":")).encode()
    with gzip.open(args.out, "wb", compresslevel=9) as f:
        f.write(blob)
    with gzip.open(args.out, "rb") as f:
        json.loads(f.read())  # confirm it reloads
    print("\nwrote %s (%d quick / %d full features, json %d bytes, gz %d bytes)"
          % (args.out, len(feat_names["quick"]), len(feat_names["full"]), len(blob),
             os.path.getsize(args.out)), flush=True)


if __name__ == "__main__":
    main()
