"""Cross-population held-out benchmark: apply the EXPORTED model to 43 HPRC samples.

The deployed model is loaded straight from its ``.json[.gz]`` -- the exact format ExpansionHunter
consumes -- and applied (NO fitting, no re-training) to 43 HPRC short-read samples that are entirely
absent from the HG002+CHM training pool, scored against their truth. This is the realistic "train on
some samples, apply to new samples" test. A single optimized-streaming source per sample (the same
1.6M-locus catalog, ``EHv5-bw2-optimized``) supplies all three genotyping regimes via routing: its
``QuickGenotype`` rows are the ``quick`` regime and its full-genotyper-fallback rows split into
``full_spanning`` / ``full_nonspanning``.

corrected call = ``eh / LCF`` (q-median head); the gate applies it only where ``pOk < 0.5``
(direction head), else keeps raw EH. Metrics are accumulated as running sums so the tens of millions
of ``quick`` rows never sit in RAM at once. Writes ``report/holdout43.json`` for the report section.

Coding rules: no type hints, Google docstrings, ``print()``, ``gcloud`` (macOS).
"""

import argparse
import datetime
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

import dataset
import eh_json
import features
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
GCS_ROOT = "gs://str-truth-set-v2/tool_results"
VARIANT = "EHv5-bw2-optimized"
CATALOG = "combined_catalog_43_samples_1.6M_loci_maxdepth150"

# The 43 held-out HPRC short-read samples (absent from the HG002+CHM training pool).
SAMPLES = [
    "HG00438", "HG00514", "HG00621", "HG00673", "HG00733", "HG00735", "HG00741", "HG01071",
    "HG01106", "HG01109", "HG01175", "HG01243", "HG01258", "HG01358", "HG01361", "HG01891",
    "HG01928", "HG01952", "HG01978", "HG02055", "HG02080", "HG02145", "HG02148", "HG02257",
    "HG02572", "HG02622", "HG02630", "HG02717", "HG02723", "HG02818", "HG02886", "HG03098",
    "HG03125", "HG03453", "HG03486", "HG03492", "HG03516", "HG03540", "HG03579", "NA12878",
    "NA18906", "NA19240", "NA20129",
]


def _discover_cov(sample):
    """Returns the coverage label (e.g. ``"32x"``) auto-discovered from the GCS path."""
    out = subprocess.run(["gsutil", "ls", "%s/%s/illumina/%s/" % (GCS_ROOT, sample, VARIANT)],
                         capture_output=True, text=True).stdout
    for line in out.split():
        m = re.search(r"/(\d+)x_coverage/", line)
        if m:
            return "%sx" % m.group(1)
    raise RuntimeError("no Nx_coverage dir for %s" % sample)


def _catalog_base(sample, cov):
    return "%s/%s/illumina/%s/%s_coverage/%s/" % (GCS_ROOT, sample, VARIANT, cov, CATALOG)


def build_sample(sample, data_dir, force):
    """Downloads + joins one held-out sample and writes its per-sample parquet."""
    out_path = os.path.join(data_dir, "real_43", "%s.parquet" % sample)
    if os.path.exists(out_path) and not force:
        return out_path
    cov = _discover_cov(sample)
    base = _catalog_base(sample, cov)
    listing = subprocess.run(["gsutil", "ls", base + "json/"],
                             capture_output=True, text=True, check=True).stdout.split()
    json_remote = sorted(p for p in listing if p.endswith(".json") or p.endswith(".json.gz"))
    tsv_remote = ("%s%s.tandem_repeat_genotypes.for_comparison.with_%s_vs_Truth_columns."
                  "alleles.tsv.gz" % (base, sample, VARIANT))
    print("=== %s (%s): %d json file(s) ===" % (sample, cov, len(json_remote)), flush=True)

    dl_dir = os.path.join(data_dir, "real_43", "_downloads", sample)
    json_local = dataset._download(json_remote, dl_dir)
    tsv_local = dataset._download([tsv_remote], dl_dir)[0]

    rows = []
    for path in json_local:
        rows.extend(eh_json.extract_rows(path, sample_id=sample))
    merged = dataset._join_truth(pd.DataFrame(rows), dataset._load_truth_tsv(tsv_local, VARIANT))
    merged = merged.drop(columns=["sample_id"])
    for c in merged.select_dtypes("float64").columns:
        merged[c] = merged[c].astype("float32")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print("    %d rows (%d matched truth)" % (len(merged), int(merged["true"].notna().sum())), flush=True)
    return out_path


EPS = 1e-9              # distance-change tolerance for the helped/hurt comparison
VIOLIN_PER_SAMPLE = 6000  # per-sample cap on the (reduction, pOk) pairs kept for the violins


def _new_acc():
    # sum_* feed the MAE; err_* hold the per-allele |error| (float32) for an exact pooled median;
    # helped_*/hurt_*/n_* count, per pOk stratum, how many alleles the LCF would move closer to
    # (helped) / further from (hurt) the truth than raw EH -- evaluated on EVERY allele, so the
    # pOk>=0.5 stratum shows what the gate avoids by keeping raw EH there. For the violins,
    # red/pok/lcf_all are a capped per-allele sample of (signed error reduction, pOk, predicted LCF)
    # over the FULL pOk range, so any later stratification (pOk threshold, LCF bin) is derived from these.
    return dict(n=0, sum_db=0.0, sum_da=0.0, ex_eh=0, ex_gated=0, pok_correct=0, err_raw=[], err_gated=[],
                n_lt=0, helped_lt=0, hurt_lt=0, n_ge=0, helped_ge=0, hurt_ge=0,
                red_all=[], pok_all=[], lcf_all=[])


def _accumulate(acc, sub, comp, branch):
    """Folds one sample's rows (one genotyping regime) into the running accumulator (gated correction)."""
    X, _ = features.build_matrix(sub, branch)
    eh = sub["eh"].to_numpy(float)
    true = sub["true"].to_numpy(float)
    lcf = M.predict_lcf_json(comp, X)
    true_pred = eh / lcf
    proba = M.predict_proba_json(comp, X)
    p_ok = proba[:, 0]
    corrected = np.where(p_ok < 0.5, true_pred, eh)        # gate: correct only low-confidence calls
    d_raw = np.abs(true - eh)
    d_gated = np.abs(true - corrected)
    acc["n"] += int(eh.size)
    acc["sum_db"] += float(d_raw.sum())
    acc["sum_da"] += float(d_gated.sum())
    acc["err_raw"].append(d_raw.astype(np.float32))
    acc["err_gated"].append(d_gated.astype(np.float32))
    acc["ex_eh"] += int((np.round(eh) == np.round(true)).sum())
    acc["ex_gated"] += int((np.round(corrected) == np.round(true)).sum())
    acc["pok_correct"] += int((np.argmax(proba, axis=1) == sub["dir_code"].to_numpy(int)).sum())
    # Would the LCF-corrected call (eh/LCF) be closer (helped) or further (hurt) than raw EH?
    # Computed on every allele, then split by pOk stratum: pOk<0.5 is where the gate APPLIES the
    # correction; pOk>=0.5 is where it KEEPS raw EH (so its hurt count is the regret avoided).
    d_corr = np.abs(true - true_pred)
    lt = p_ok < 0.5
    for tag, mask in (("lt", lt), ("ge", ~lt)):
        db, da = d_raw[mask], d_corr[mask]
        acc["n_%s" % tag] += int(mask.sum())
        acc["helped_%s" % tag] += int((da < db - EPS).sum())
        acc["hurt_%s" % tag] += int((da > db + EPS).sum())
    # Violin samples (signed error reduction = |raw err| - |corrected err|, >0 = corrected closer):
    # keep (reduction, pOk, LCF) over the full pOk range so any later stratification is
    # derived from these. Strided subsample => bounded, deterministic spread.
    red = d_raw - d_corr
    idx = slice(None) if red.size <= VIOLIN_PER_SAMPLE else slice(None, None, red.size // VIOLIN_PER_SAMPLE)
    acc["red_all"].append(red[idx][:VIOLIN_PER_SAMPLE].astype(np.float32))
    acc["pok_all"].append(p_ok[idx][:VIOLIN_PER_SAMPLE].astype(np.float32))
    acc["lcf_all"].append(lcf[idx][:VIOLIN_PER_SAMPLE].astype(np.float32))


def _finalize(acc, n_samples):
    n = acc["n"]
    if n == 0:
        return {"n": 0}
    mae_raw, mae_gated = acc["sum_db"] / n, acc["sum_da"] / n
    return {
        "n": n, "n_samples": n_samples,
        "mae_raw": mae_raw, "mae_gated": mae_gated,
        "median_raw": float(np.median(np.concatenate(acc["err_raw"]))),
        "median_gated": float(np.median(np.concatenate(acc["err_gated"]))),
        "dist_reduction": (1.0 - mae_gated / mae_raw) if mae_raw > 0 else float("nan"),
        "exact_eh": acc["ex_eh"] / n, "exact_gated": acc["ex_gated"] / n,
        "p_ok_accuracy": acc["pok_correct"] / n,
        "n_pok_lt": acc["n_lt"], "helped_lt": acc["helped_lt"], "hurt_lt": acc["hurt_lt"],
        "n_pok_ge": acc["n_ge"], "helped_ge": acc["helped_ge"], "hurt_ge": acc["hurt_ge"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data_eval_43"))
    parser.add_argument("--model", default=os.path.join(
        HERE, "model", "genotype_quality_model_from_HG002_and_CHM1_CHM13.%s.json.gz"
        % datetime.date.today().strftime("%Y%m%d")),
        help="exported model .json[.gz] to apply (the format ExpansionHunter loads)")
    parser.add_argument("--out", default=os.path.join(HERE, "report", "holdout43.json"))
    parser.add_argument("--build-only", action="store_true", help="only download + build parquets")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--samples", nargs="+", default=None, help="subset (default: all 43)")
    parser.add_argument("--max-alleles-per-sample", type=int, default=2_000_000,
                        help="seeded per-sample allele cap for the JSON-model apply (0 = all alleles)")
    args = parser.parse_args()

    wanted = args.samples or SAMPLES
    print("==== build %d held-out sample parquet(s) ====" % len(wanted), flush=True)
    for s in wanted:
        build_sample(s, args.data_dir, args.force)
    if args.build_only:
        return

    print("\n==== load + compile the exported model: %s ====" % os.path.basename(args.model), flush=True)
    model_json = M.load(args.model)["genotyping_regimes"]
    compiled = {r: (M.compile_genotyping_regime(model_json[r]), features.GENOTYPING_REGIME_BRANCH[r])
                for r in features.GENOTYPING_REGIMES}

    acc = {r: _new_acc() for r in features.GENOTYPING_REGIMES}
    cap = args.max_alleles_per_sample or None
    paths = sorted(p for p in (os.path.join(args.data_dir, "real_43", "%s.parquet" % s)
                               for s in wanted) if os.path.exists(p))
    print("\n==== predict + gate on %d held-out samples (no fitting, cap %s alleles/sample) ===="
          % (len(paths), cap or "none"), flush=True)
    for p in paths:
        df, _ = dataset.label_and_filter(pd.read_parquet(p))
        if cap and len(df) > cap:
            df = df.sample(cap, random_state=20260616).reset_index(drop=True)
        for regime in features.GENOTYPING_REGIMES:
            sub = df[df["genotyping_regime"] == regime]
            if not sub.empty:
                _accumulate(acc[regime], sub, *compiled[regime])
        print("  %s done" % os.path.basename(p), flush=True)

    out = {"n_samples": len(paths), "max_alleles_per_sample": cap,
           "genotyping_regimes": {r: _finalize(acc[r], len(paths)) for r in features.GENOTYPING_REGIMES}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    # Violin samples per genotyping regime: full-range per-allele (reduction, pOk, LCF);
    # all pOk-threshold + LCF-bin stratifications are derived from these.
    def _cat(key, r):
        return np.concatenate(acc[r][key]) if acc[r][key] else np.zeros(0, np.float32)
    violin = {}
    for r in features.GENOTYPING_REGIMES:
        for short, key in (("red", "red_all"), ("pok", "pok_all"), ("lcf", "lcf_all")):
            violin["%s__%s" % (r, short)] = _cat(key, r)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_violin.npz", **violin)
    print("\nwrote %s" % args.out, flush=True)
    for r in features.GENOTYPING_REGIMES:
        m = out["genotyping_regimes"][r]
        if m.get("n"):
            print("  %-18s n=%-9d raw MAE %.3f -> gated %.3f (distred %+.1f%%)  median|err| %.3f->%.3f"
                  % (r, m["n"], m["mae_raw"], m["mae_gated"], 100 * m["dist_reduction"],
                     m["median_raw"], m["median_gated"]), flush=True)


if __name__ == "__main__":
    main()
