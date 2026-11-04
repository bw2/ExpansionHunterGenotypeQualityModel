"""Compares the simulated expanded alleles with real confirmed expansions, as the models see them.

Two checks of whether the simulated alleles look like real ones to a genotype quality model:

1. Each model's pOk, pTooLong and pTooShort for the simulated under-calls (true allele longer than the call) and for the
   real confirmed expansions, by called size. If the models treat them alike, rows of simulated alleles teach them about
   the real ones; if not, the simulation misses something the models use.
2. The model inputs of each real expansion, as a percentile among the simulated under-calls of a similar called size
   (within a factor of 1.5) and the same genotyping regime, and how well a classifier tells the two apart
   (cross-validated ROC-AUC; near 0.5 means the inputs don't give the simulation away).

The real expansions are confirmed pathogenic expansions in other people's genomes: the long allele at each finding's
locus in that person's ExpansionHunter JSON. ``--known-findings-tsv`` lists the findings, with columns run, finding
(rows with "expected expanded" are used), sample_id and locus_id (the disease locus id of ``disease_loci.tsv``), and
``--real-expansion-jsons-dir`` holds each sample's ``<sample_id>.json.gz``, genotyped with ``disease_loci.catalog.json``
like the spiked samples (``spike_in_expansions.genotype_on_hail_batch``), so the real and simulated alleles share the
catalog, the ExpansionHunter build and its settings.

Usage:
    python3 simulated_expansions/compare_simulated_and_real_expansions.py \\
        --tables-dir simulated_expansions/work_spiked_samples/tables \\
        --manifest-tsv simulated_expansions/work_spiked_samples/spiked/manifest.tsv --known-findings-tsv TSV \\
        --real-expansion-jsons-dir DIR --run optimized_streaming_on_minicram --model LABEL=MODEL_JSON_GZ ... OUT_TSV
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import eh_json  # noqa: E402
import features  # noqa: E402
import model as M  # noqa: E402

DISEASE_LOCI_TSV = os.path.join(HERE, "disease_loci.tsv")
SIZE_FACTOR = 1.5
COMPARED_INPUTS = ("hq_unamb_total", "inrepeat_total", "flanking_total", "spanning_total", "coverage", "depth",
                   "read_repeat_purity", "strand_bias_phred", "ci_width", "left_flank_norm_depth",
                   "right_flank_norm_depth", "mean_inserted_bases", "mean_deleted_bases")
CALLED_BP_BINS = (0, 150, 300, 600, 1200, np.inf)


def simulated_alleles(tables_dir, manifest_tsv):
    """The rows of the simulated alleles in the spiked samples' parquets: at each spiked locus, the row whose truth
    is a simulated allele (``copies`` in the manifest, or a patient's shorter ``shorter_simulated_allele_copies``) and
    that has its own quality metrics, as the rows training and the real expansions' rows do (not the second copy of a
    homozygous call, which a patient's two alleles often get, the row with its own metrics then taking the shorter
    allele's truth). Of a patient's two, only the longer call's row, as ``real_expansions`` takes one call per
    expansion."""
    manifest = pd.read_csv(manifest_tsv, sep="\t")
    rows = []
    for path in sorted(glob.glob(os.path.join(tables_dir, "*.parquet"))):
        label = os.path.basename(path)[:-len(".parquet")]
        df = pd.read_parquet(path)
        # reindex: a manifest without shorter_simulated_allele_copies gets an empty column, which never equals a
        # truth, so its rows are carriers only.
        spiked = manifest[manifest["label"] == label].reindex(
            columns=["locus_id", "copies", "shorter_simulated_allele_copies", "gene"])
        df = df.merge(spiked, on="locus_id", how="inner", validate="many_to_one")
        df = df[((df["true"] == df["copies"]) | (df["true"] == df["shorter_simulated_allele_copies"]))
                & df["has_own_quality_metrics"].astype(bool)]
        rows.append(df.sort_values("eh").drop_duplicates("locus_id", keep="last").assign(label=label))
    return pd.concat(rows, ignore_index=True)


def real_expansions(known_findings_tsv, jsons_dir, run, disease_loci_tsv=DISEASE_LOCI_TSV):
    """The long allele's row at each "expected expanded" finding of ``run``, from its sample's EH JSON, at the
    catalog locus ``disease_loci_tsv`` gives the finding's disease locus."""
    catalog_locus_of = pd.read_csv(disease_loci_tsv, sep="\t").set_index("disease_locus_id")["locus_id"]
    findings = pd.read_csv(known_findings_tsv, sep="\t")
    findings = findings[(findings["run"] == run) & (findings["finding"] == "expected expanded")]
    rows = []
    for finding in findings.to_dict("records"):
        path = os.path.join(jsons_dir, "%s.json.gz" % finding["sample_id"])
        locus_rows = [r for r in eh_json.extract_rows(path) if r["locus_id"] == catalog_locus_of[finding["locus_id"]]
                      and r["has_own_quality_metrics"]]
        if locus_rows:
            long_allele = max(locus_rows, key=lambda r: r["eh"])
            rows.append({**long_allele, "gene": finding["locus_id"], "sample": finding["sample_id"]})
    return pd.DataFrame(rows)


def scores(df, model_path):
    """``(pOk, pTooLong, pTooShort)`` arrays of ``model_path`` for rows with model inputs."""
    proba = M.predict_proba_by_genotyping_regime(df, M.load(model_path), model_path)
    return proba[:, features.OK], proba[:, features.TOO_LONG], proba[:, features.TOO_SHORT]


def with_called_bp(df):
    df = df.copy()
    df["called_bp"] = df["eh"] * df["motif_size"]
    df["regime"] = features.genotyping_regime_of(df["genotyping_branch"].to_numpy(object),
                                                 df["spanning_at_called"].to_numpy(float))
    df["size_bin"] = pd.cut(df["called_bp"], CALLED_BP_BINS, right=False).astype(str)
    return df


def input_percentiles(real, simulated):
    """For each real expansion and input, its percentile among the simulated under-calls of the same regime with a
    called size within a factor of ``SIZE_FACTOR`` (NaN with fewer than 20 of them)."""
    out = []
    for r in real.to_dict("records"):
        similar = simulated[(simulated["regime"] == r["regime"])
                            & simulated["called_bp"].between(r["called_bp"] / SIZE_FACTOR, r["called_bp"] * SIZE_FACTOR)]
        row = {"sample": r["sample"], "gene": r["gene"], "called_bp": r["called_bp"], "n_similar_simulated": len(similar)}
        for name in COMPARED_INPUTS:
            values = similar[name].dropna()
            row[name] = (100.0 * (values < r[name]).mean() if len(values) >= 20 and pd.notna(r[name]) else np.nan)
        out.append(row)
    return pd.DataFrame(out)


def classifier_auc(real, simulated, seed=0, groups=None):
    """Cross-validated ROC-AUC of telling real expansions from simulated under-calls by ``COMPARED_INPUTS``.

    Args:
        groups: Optional group of each row (``real``'s, then ``simulated``'s): rows of one group stay in one fold, e.g.
            an allele's real row and its simulated twin, which would otherwise teach the classifier the test row.
    """
    X = pd.concat([real[list(COMPARED_INPUTS)], simulated[list(COMPARED_INPUTS)]], ignore_index=True).to_numpy(float)
    y = np.r_[np.ones(len(real)), np.zeros(len(simulated))]
    n_splits = min(5, int(y.sum()))
    folds = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if groups is None
             else list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups)))
    predicted = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=seed), X, y, cv=folds,
                                  method="predict_proba")[:, 1]
    return roc_auc_score(y, predicted)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", required=True,
                        help="the spiked samples' per-source parquets, such as make_spiked_samples.sh's "
                             "work_spiked_samples/tables")
    parser.add_argument("--manifest-tsv", required=True,
                        help="their manifest, such as work_spiked_samples/spiked/manifest.tsv")
    parser.add_argument("--known-findings-tsv", required=True)
    parser.add_argument("--real-expansion-jsons-dir", required=True)
    parser.add_argument("--run", default="optimized_streaming_on_minicram")
    parser.add_argument("--model", action="append", required=True, help="LABEL=MODEL_JSON_GZ, one or more")
    parser.add_argument("out_tsv")
    args = parser.parse_args()

    simulated = with_called_bp(simulated_alleles(args.tables_dir, args.manifest_tsv))
    real = with_called_bp(real_expansions(args.known_findings_tsv, args.real_expansion_jsons_dir, args.run))
    under_calls = simulated[simulated["true"] > simulated["eh"]]
    print("simulated alleles: %d; called shorter than the truth: %d; real confirmed expansions: %d"
          % (len(simulated), len(under_calls), len(real)))
    print("\nsimulated alleles by called size: share under-called, mean true/called size")
    by_bin = simulated.groupby("size_bin")
    print(pd.DataFrame({"alleles": by_bin.size(), "under-called": by_bin.apply(lambda d: (d["true"] > d["eh"]).mean()),
                        "true / called": by_bin.apply(lambda d: (d["true"] / d["eh"]).mean())}).to_string())

    rows = []
    for spec in args.model:
        label, path = spec.split("=", 1)
        for name, df in (("simulated under-calls", under_calls), ("real expansions", real)):
            p_ok, p_too_long, p_too_short = scores(df, path)
            frame = df.assign(pOk=p_ok, pTooLong=p_too_long, pTooShort=p_too_short)
            for size_bin, group in frame.groupby("size_bin"):
                rows.append({"model": label, "alleles": name, "called size": size_bin, "n": len(group),
                             "median pOk": group["pOk"].median(), "pTooLong >= 0.5": (group["pTooLong"] >= 0.5).mean(),
                             "pTooShort >= 0.5": (group["pTooShort"] >= 0.5).mean()})
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False, float_format=lambda v: "%.2f" % v))
    summary.to_csv(args.out_tsv, sep="\t", index=False)

    percentiles = input_percentiles(real, under_calls)
    print("\nreal expansions' inputs as percentiles among simulated under-calls of a similar called size:")
    print(percentiles.to_string(index=False, float_format=lambda v: "%.0f" % v))
    percentiles.to_csv(os.path.splitext(args.out_tsv)[0] + ".input_percentiles.tsv", sep="\t", index=False)
    full = under_calls[under_calls["regime"] != features.GENOTYPING_REGIME_QUICK]
    print("\nROC-AUC telling real expansions from simulated full-genotyper under-calls by their inputs: %.2f"
          % classifier_auc(real[real["regime"] != features.GENOTYPING_REGIME_QUICK], full))


if __name__ == "__main__":
    main()
