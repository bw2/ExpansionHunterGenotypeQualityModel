"""Cross-population held-out benchmark: apply the EXPORTED model to the remaining 30 HPRC samples.

The deployed model is loaded straight from its ``.json[.gz]`` -- the exact format ExpansionHunter
consumes -- and applied (NO fitting, no re-training) to the remaining 30 HPRC short-read samples (of
the original 43-sample panel) that are entirely absent from the HG002+CHM training pool -- the other
13 were promoted into training, see ``dataset.PROMOTED_HELDOUT_SAMPLES`` -- scored against their
truth. This is the realistic "train on
some samples, apply to new samples" test. A single optimized-streaming source per sample (the same
1.6M-locus catalog, ``EHv5-bw2-optimized``) supplies all three genotyping regimes via routing: its
``QuickGenotype`` rows are the ``quick`` regime and its full-genotyper-fallback rows split into
``full_spanning`` / ``full_nonspanning``.

corrected call = ``eh / LCF`` (q-median head); the gate applies it only where ``pOk < 0.5``
(direction head), else keeps raw EH. The MAE is a running sum, but the exact pooled median retains
every kept allele's ``|error|`` in RAM (bounded by ``--max-alleles-per-sample``), so peak memory grows
with the total kept alleles. ``main()`` writes a standalone ``report/heldout.json`` benchmark dump.
The report's held-out-43 section is NOT fed from that file -- it is produced by
``gen_datasets.py --dataset heldout43``, which reuses ``run_eval`` here to emit the
``report/eval_heldout43.json`` / ``report/stacked_heldout43.json`` artifacts ``report.py`` consumes.

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
CATALOG = "combined_catalog_43_samples_1.6M_loci"

# The remaining 30 held-out HPRC short-read samples (absent from the training pool). 13 of the
# original 43 (see ``dataset.PROMOTED_HELDOUT_SAMPLES``) were promoted into training for
# ancestry/sex diversity at large allele sizes, so they are excluded here to avoid double-counting
# them in the external validation set.
SAMPLES = [
    "HG00438", "HG00514", "HG00673", "HG00733", "HG00735", "HG00741", "HG01071",
    "HG01109", "HG01175", "HG01243", "HG01358", "HG01361", "HG01891",
    "HG01952", "HG01978", "HG02145", "HG02148", "HG02257",
    "HG02572", "HG02630", "HG02717", "HG02723", "HG02818", "HG02886", "HG03098",
    "HG03486", "HG03516", "HG03540", "HG03579",
    "NA19240",
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
    cov = _discover_cov(sample)
    base = _catalog_base(sample, cov)
    listing = subprocess.run(["gsutil", "ls", base + "json/"],
                             capture_output=True, text=True, check=True).stdout.split()
    json_remote = sorted(p for p in listing if p.endswith(".json") or p.endswith(".json.gz"))

    dl_dir = os.path.join(data_dir, "real_43", "_downloads", sample)
    genotypes_tsv_remote = dataset._truth_genotypes_tsv_remote(sample)
    # Checked even when the parquet cache below is about to be reused -- see dataset.build_combo.
    dataset._check_freshness(sample, [
        ("json shard %d" % i, r, os.path.join(dl_dir, os.path.basename(r)))
        for i, r in enumerate(json_remote)
    ] + [
        ("truth-genotypes TSV", genotypes_tsv_remote,
         os.path.join(dl_dir, os.path.basename(genotypes_tsv_remote))),
    ])

    if os.path.exists(out_path) and not force:
        return out_path
    print("=== %s (%s): %d json file(s) ===" % (sample, cov, len(json_remote)), flush=True)

    json_local = dataset._download(json_remote, dl_dir)
    genotypes_tsv_local = dataset._download([genotypes_tsv_remote], dl_dir)[0]

    rows = []
    for path in json_local:
        rows.extend(eh_json.extract_rows(path, sample_id=sample))
    merged = dataset._join_truth(
        pd.DataFrame(rows), dataset._load_truth_from_genotypes_tsv(genotypes_tsv_local), sample)
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
    # pOk>=0.5 stratum shows what the gate avoids by keeping raw EH there. Homopolymer (1 bp motif)
    # loci are EXCLUDED from every scalar metric and from the red/pok/lcf violin sample (they are
    # summarized on their own in the by-motif-size violins); only the mred/mpok/motif sample keeps
    # them. red/pok/lcf_all = capped per-allele (signed error reduction, pOk, predicted LCF) over the
    # full pOk range (non-homopolymer); mred/mpok/motif_all = the same reduction + pOk + motif size in
    # bp over ALL loci, for the by-motif-size violins.
    # h_* are the homopolymer-only (1 bp motif) counterparts of n/sum_db/sum_da/err, for the separate
    # homopolymer MAE bar chart (everything else above is non-homopolymer).
    # pdiff_all / h_pdiff_all = per-allele direction-head lean (pTooLong - pTooShort), strided +
    # per-sample capped exactly like red/pok/lcf, for the by-lean error-reduction violins.
    return dict(n=0, sum_db=0.0, sum_da=0.0, ex_eh=0, ex_gated=0, pok_correct=0, err_raw=[], err_gated=[],
                n_lt=0, helped_lt=0, hurt_lt=0, n_ge=0, helped_ge=0, hurt_ge=0,
                red_all=[], pok_all=[], lcf_all=[], pdiff_all=[], mred_all=[], mpok_all=[], motif_all=[],
                h_n=0, h_sum_db=0.0, h_sum_da=0.0, h_err_raw=[], h_err_gated=[],
                h_n_lt=0, h_helped_lt=0, h_hurt_lt=0, h_n_ge=0, h_helped_ge=0, h_hurt_ge=0,
                h_red_all=[], h_pok_all=[], h_lcf_all=[], h_pdiff_all=[])


def _strided(a, idx):
    return a[idx][:VIOLIN_PER_SAMPLE].astype(np.float32)


def _accumulate(acc, sub, comp, branch):
    """Folds one sample's rows (one genotyping regime) into the running accumulator (gated correction).

    Homopolymer (1 bp motif) loci are dropped from every scalar metric and the red/pok/lcf violin
    sample; only the by-motif-size sample (mred/mpok/motif) retains them.
    """
    X, _ = features.build_matrix(sub, branch)
    eh = sub["eh"].to_numpy(float)
    true = sub["true"].to_numpy(float)
    motif = sub["motif_size"].to_numpy(float)
    lcf = M.predict_lcf_json(comp, X)
    true_pred = eh / lcf
    proba = M.predict_proba_json(comp, X)
    p_ok = proba[:, 0]
    corrected = np.where(p_ok < 0.5, true_pred, eh)        # gate: correct only low-confidence calls
    d_raw = np.abs(true - eh)
    d_gated = np.abs(true - corrected)
    d_corr = np.abs(true - true_pred)                      # ungated correction (helped/hurt + reduction)
    red = d_raw - d_corr                                   # signed error reduction (>0 = corrected closer)

    # By-motif-size violin sample: keep ALL loci (incl. homopolymers). Strided => bounded, deterministic.
    midx = slice(None) if red.size <= VIOLIN_PER_SAMPLE else slice(None, None, red.size // VIOLIN_PER_SAMPLE)
    acc["mred_all"].append(_strided(red, midx))
    acc["mpok_all"].append(_strided(p_ok, midx))
    acc["motif_all"].append(_strided(motif, midx))

    # Homopolymer-only (1 bp motif) MAE accumulation (for the separate homopolymer bar chart).
    homo = motif == 1
    if homo.any():
        acc["h_n"] += int(homo.sum())
        acc["h_sum_db"] += float(d_raw[homo].sum())
        acc["h_sum_da"] += float(d_gated[homo].sum())
        acc["h_err_raw"].append(d_raw[homo].astype(np.float32))
        acc["h_err_gated"].append(d_gated[homo].astype(np.float32))
        # Homopolymer helped/hurt by pOk stratum (ungated d_corr vs raw EH, mirrors the non-homo block).
        h_lt = p_ok[homo] < 0.5
        h_db, h_da = d_raw[homo], d_corr[homo]
        for tag, mask in (("lt", h_lt), ("ge", ~h_lt)):
            acc["h_n_%s" % tag] += int(mask.sum())
            acc["h_helped_%s" % tag] += int((h_da[mask] < h_db[mask] - EPS).sum())
            acc["h_hurt_%s" % tag] += int((h_da[mask] > h_db[mask] + EPS).sum())
        # Homopolymer violin sample: (signed reduction, pOk, predicted LCF), strided + per-sample capped.
        hred, hpok, hlcf = red[homo], p_ok[homo], lcf[homo]
        hidx = slice(None) if hred.size <= VIOLIN_PER_SAMPLE else slice(None, None, hred.size // VIOLIN_PER_SAMPLE)
        acc["h_red_all"].append(_strided(hred, hidx))
        acc["h_pok_all"].append(_strided(hpok, hidx))
        acc["h_lcf_all"].append(_strided(hlcf, hidx))
        acc["h_pdiff_all"].append(_strided(proba[homo, 1] - proba[homo, 2], hidx))

    # Everything else EXCLUDES homopolymer (1 bp motif) loci.
    keep = motif != 1
    if not keep.any():
        return
    eh, true, corrected = eh[keep], true[keep], corrected[keep]
    d_raw, d_gated, d_corr, red = d_raw[keep], d_gated[keep], d_corr[keep], red[keep]
    proba, p_ok, lcf = proba[keep], p_ok[keep], lcf[keep]
    dir_code = sub["dir_code"].to_numpy(int)[keep]
    acc["n"] += int(eh.size)
    acc["sum_db"] += float(d_raw.sum())
    acc["sum_da"] += float(d_gated.sum())
    acc["err_raw"].append(d_raw.astype(np.float32))
    acc["err_gated"].append(d_gated.astype(np.float32))
    acc["ex_eh"] += int((np.round(eh) == np.round(true)).sum())
    acc["ex_gated"] += int((np.round(corrected) == np.round(true)).sum())
    acc["pok_correct"] += int((np.argmax(proba, axis=1) == dir_code).sum())
    # Would the LCF-corrected call (eh/LCF) be closer (helped) or further (hurt) than raw EH?
    # Computed on every (non-homopolymer) allele, then split by pOk stratum: pOk<0.5 is where the gate
    # APPLIES the correction; pOk>=0.5 is where it KEEPS raw EH (so its hurt count is the regret avoided).
    lt = p_ok < 0.5
    for tag, mask in (("lt", lt), ("ge", ~lt)):
        db, da = d_raw[mask], d_corr[mask]
        acc["n_%s" % tag] += int(mask.sum())
        acc["helped_%s" % tag] += int((da < db - EPS).sum())
        acc["hurt_%s" % tag] += int((da > db + EPS).sum())
    # Violin samples (non-homopolymer): (reduction, pOk, LCF) over the full pOk range.
    idx = slice(None) if red.size <= VIOLIN_PER_SAMPLE else slice(None, None, red.size // VIOLIN_PER_SAMPLE)
    acc["red_all"].append(_strided(red, idx))
    acc["pok_all"].append(_strided(p_ok, idx))
    acc["lcf_all"].append(_strided(lcf, idx))
    acc["pdiff_all"].append(_strided(proba[:, 1] - proba[:, 2], idx))


def _homopolymer_summary(acc):
    """Homopolymer-only (1 bp motif) MAE summary for the separate bar chart."""
    hn = acc["h_n"]
    if hn == 0:
        return {"n": 0}
    return {"n": hn,
            "mae_raw": acc["h_sum_db"] / hn, "mae_gated": acc["h_sum_da"] / hn,
            "median_raw": float(np.median(np.concatenate(acc["h_err_raw"]))),
            "median_gated": float(np.median(np.concatenate(acc["h_err_gated"]))),
            "n_pok_lt": acc["h_n_lt"], "helped_lt": acc["h_helped_lt"], "hurt_lt": acc["h_hurt_lt"],
            "n_pok_ge": acc["h_n_ge"], "helped_ge": acc["h_helped_ge"], "hurt_ge": acc["h_hurt_ge"]}


def _finalize(acc, n_samples):
    n = acc["n"]
    if n == 0:
        return {"n": 0, "homopolymer": _homopolymer_summary(acc)}
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
        "homopolymer": _homopolymer_summary(acc),
    }


def run_eval(paths, model_path, out_json, max_alleles):
    """Applies the exported model to ``paths`` (per-allele parquets) and writes ``out_json`` + its
    ``*_violin.npz``; returns the metrics dict.

    Shared by the 43-sample benchmark and the per-dataset (HG002 genome / HG002 exome) evaluations so
    every dataset produces identical artifacts (scalar metrics JSON + the violin/pdiff/lcf npz). No
    fitting -- the deployed model is loaded from its ``.json[.gz]`` and applied with the ``pOk<0.5``
    gate.
    """
    print("\n==== load + compile the exported model: %s ====" % os.path.basename(model_path), flush=True)
    model_json = M.load(model_path)["genotyping_regimes"]
    compiled = {r: (M.compile_genotyping_regime(model_json[r]), features.GENOTYPING_REGIME_BRANCH[r])
                for r in features.GENOTYPING_REGIMES}

    acc = {r: _new_acc() for r in features.GENOTYPING_REGIMES}
    cap = max_alleles or None
    print("\n==== predict + gate on %d parquet(s) (no fitting, cap %s alleles/sample) ===="
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
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    # Violin samples per genotyping regime: red/pok/lcf/pdiff are non-homopolymer; mred/mpok/motif keep
    # ALL loci (the by-motif-size violins); h* are the homopolymer-only counterparts.
    def _cat(key, r):
        return np.concatenate(acc[r][key]) if acc[r][key] else np.zeros(0, np.float32)
    violin = {}
    for r in features.GENOTYPING_REGIMES:
        for short, key in (("red", "red_all"), ("pok", "pok_all"), ("lcf", "lcf_all"),
                           ("pdiff", "pdiff_all"),
                           ("mred", "mred_all"), ("mpok", "mpok_all"), ("motif", "motif_all"),
                           ("hred", "h_red_all"), ("hpok", "h_pok_all"), ("hlcf", "h_lcf_all"),
                           ("hpdiff", "h_pdiff_all")):
            violin["%s__%s" % (r, short)] = _cat(key, r)
    np.savez_compressed(os.path.splitext(out_json)[0] + "_violin.npz", **violin)
    print("\nwrote %s" % out_json, flush=True)
    for r in features.GENOTYPING_REGIMES:
        m = out["genotyping_regimes"][r]
        if m.get("n"):
            print("  %-18s n=%-9d raw MAE %.3f -> gated %.3f (distred %+.1f%%)  median|err| %.3f->%.3f"
                  % (r, m["n"], m["mae_raw"], m["mae_gated"], 100 * m["dist_reduction"],
                     m["median_raw"], m["median_gated"]), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data_eval_43"))
    parser.add_argument("--model", default=os.path.join(
        HERE, "model", "genotype_quality_model_from_HG002_and_CHM1_CHM13.%s.json.gz"
        % datetime.date.today().strftime("%Y%m%d")),
        help="exported model .json[.gz] to apply (the format ExpansionHunter loads)")
    parser.add_argument("--out", default=os.path.join(HERE, "report", "heldout.json"))
    parser.add_argument("--build-only", action="store_true", help="only download + build parquets")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="subset (default: all %d)" % len(SAMPLES))
    parser.add_argument("--max-alleles-per-sample", type=int, default=2_000_000,
                        help="seeded per-sample allele cap for the JSON-model apply (0 = all alleles)")
    args = parser.parse_args()

    wanted = args.samples or SAMPLES
    print("==== build %d held-out sample parquet(s) ====" % len(wanted), flush=True)
    for s in wanted:
        build_sample(s, args.data_dir, args.force)
    if args.build_only:
        return

    paths = sorted(p for p in (os.path.join(args.data_dir, "real_43", "%s.parquet" % s)
                               for s in wanted) if os.path.exists(p))
    run_eval(paths, args.model, args.out, args.max_alleles_per_sample)


if __name__ == "__main__":
    main()
