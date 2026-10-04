"""Cross-population held-out benchmark: apply the EXPORTED model to the 81 held-out samples.

The deployed model is loaded straight from its ``.json[.gz]`` -- the exact format ExpansionHunter
consumes -- and applied (NO fitting, no re-training) to the 81 held-out short-read samples in
``SAMPLES`` that are entirely absent from the training pool (the samples that train alongside
HG002+CHM1_CHM13 are listed in ``dataset.PROMOTED_HELDOUT_SAMPLES``), scored against their truth.
This is the realistic "train on some samples, apply to new samples" test. A single optimized-streaming source per sample (the
TRExplorer v2.1 catalog, ``dataset.EH_RESULTS_ROOT``) supplies all three genotyping regimes via routing: its
``QuickGenotype`` rows are the ``quick`` regime and its full-genotyper-fallback rows split into
``full_spanning`` / ``full_nonspanning``.

corrected call = ``eh / LCF`` (q-median head); the gate applies it only where ``pOk < 0.5``
(direction head), else keeps raw EH. The MAE is a running sum, but the exact pooled median retains
every kept allele's ``|error|`` in RAM (bounded by ``--max-alleles-per-sample``), so peak memory grows
with the total kept alleles. ``main()`` writes a standalone ``report/heldout.json`` benchmark dump.
The report's held-out HPRC section is NOT fed from that file -- it is produced by
``gen_datasets.py --dataset heldout_hprc``, which reuses ``run_eval`` here to emit the
``report/eval_heldout_hprc.json`` / ``report/stacked_heldout_hprc.json`` artifacts ``report.py`` consumes.

Coding rules: no type hints, Google docstrings, ``print()``, ``gcloud`` (macOS).
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

import dataset
import features
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))

# The 81 held-out short-read samples (absent from the training pool): the 138 1kGP samples with a
# DipCall high-confidence BED, a truth-genotypes TSV and a Broad short-read CRAM, minus the 49 of
# them that train alongside HG002+CHM1_CHM13 (``dataset.PROMOTED_HELDOUT_SAMPLES``; see there for how
# the split was chosen on 2026-09-25 and updated on 2026-09-28 and 2026-10-01), and minus the 8 whose DipCall truth
# lost almost all of chrX/chrY (listed in str-truth-set-v2's
# filter_vcfs_v2/samples_excluded_from_downstream_analyses.tsv). Includes the 30 samples held out
# before 2026-09-25 and the ten pOk fast-path diagnosis samples.
SAMPLES = [
    "HG00423", "HG00438", "HG00514", "HG00544", "HG00558", "HG00597", "HG00609", "HG00639",
    "HG00642", "HG00673", "HG00733", "HG00735", "HG00738", "HG00741", "HG01071", "HG01099",
    "HG01109", "HG01114", "HG01175", "HG01243", "HG01252", "HG01255", "HG01261", "HG01358",
    "HG01361", "HG01496", "HG01884", "HG01891", "HG01940", "HG01952", "HG01969", "HG01975",
    "HG01978", "HG01981", "HG01993", "HG02004", "HG02015", "HG02027", "HG02056", "HG02074",
    "HG02132", "HG02145", "HG02148", "HG02257", "HG02280", "HG02293", "HG02300", "HG02514",
    "HG02523", "HG02572", "HG02587", "HG02602", "HG02630", "HG02668", "HG02698", "HG02717",
    "HG02723", "HG02735", "HG02738", "HG02809", "HG02818", "HG02886", "HG02984", "HG03017",
    "HG03050", "HG03098", "HG03486", "HG03516", "HG03540", "HG03579", "HG03654", "HG03683",
    "HG03704", "HG03834", "HG03942", "HG04157", "HG04160", "HG04184", "HG04187", "HG04199",
    "NA19240",
]


def build_sample(sample, data_dir, force):
    """Downloads + joins one held-out sample and writes its per-sample parquet.

    ``sample`` is both the sample label of its ``dataset.EH_RESULTS_ROOT`` folder and its truth sample id.
    """
    out_path = os.path.join(data_dir, "real_43", "%s.parquet" % sample)
    json_remote = dataset._list_json_inputs(sample)

    dl_dir = os.path.join(data_dir, "real_43", "_downloads", sample)
    json_locals = [os.path.join(dl_dir, os.path.basename(r)) for r in json_remote]
    truth_sources = dataset._truth_sources(sample, dl_dir)
    # Checked even when the parquet cache below is about to be reused -- see dataset.build_combo.
    # Deletes any bucket-newer / content-changed local copy so _download re-fetches it below.
    json_sources = [("json shard %d" % i, r, l) for i, (r, l) in enumerate(zip(json_remote, json_locals))]
    dataset._check_freshness(sample, json_sources + truth_sources)

    if dataset._parquet_reusable(out_path, json_sources + truth_sources, force):
        return out_path
    print("=== %s: %d json file(s) ===" % (sample, len(json_remote)), flush=True)

    cloud_versions = dataset._cloud_versions(json_sources + truth_sources)
    json_local = dataset._download(json_remote, dl_dir)
    genotypes_tsv_local, high_confidence_bed_local = dataset._download([r for _, r, _ in truth_sources], dl_dir)
    # _check_freshness could only judge shards that already existed locally -- see dataset.build_combo.
    dataset.assert_eh_build_matches(
        sample, [("json shard %d" % i, p) for i, p in enumerate(json_local)])

    merged = dataset.extract_rows_and_join_truth(
        json_local, genotypes_tsv_local, high_confidence_bed_local, sample, sample).drop(columns=["sample_id"])
    # Feature columns stay float64 all the way to fit(). Downcasting here used to halve the parquet
    # and frame size, but it permanently quantized every value: sklearn's HistGradientBoosting upcasts
    # back to float64 internally (X_DTYPE) and bins to uint8, so the downcast bought nothing at fit
    # time while forcing the C++ scorer to reproduce it exactly. See features.build_matrix.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    dataset._record_sources_and_remove_downloads(out_path, json_sources + truth_sources, cloud_versions)
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


def _accumulate(acc, sub, comp, branch, names):
    """Folds one sample's rows (one genotyping regime) into the running accumulator (gated correction).

    ``names`` is the feature list the model being applied declares (see ``run_eval``); the compiled
    trees index it positionally.

    Homopolymer (1 bp motif) loci are dropped from every scalar metric and the red/pok/lcf violin
    sample; only the by-motif-size sample (mred/mpok/motif) retains them.
    """
    X, _ = features.build_matrix(sub, branch, names)
    eh = sub["eh"].to_numpy(float)
    true = sub["true"].to_numpy(float)
    motif = sub["motif_size"].to_numpy(float)
    # Rounded to the 3 decimals ExpansionHunter emits: this benchmark exists to describe what the
    # deployed binary does, and 3 decimals is all its JSON carries (model.round_like_emitted).
    lcf = M.round_like_emitted(M.predict_lcf_json(comp, X))
    true_pred = eh / lcf
    proba = M.round_like_emitted(M.predict_proba_json(comp, X))
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


def assert_parquets_carry_contract(paths):
    """Raises if any per-allele parquet no longer matches the current extractor/feature contract.

    Delegates to ``dataset.parquet_contract_complaint``, the SAME check the training path applies in
    ``_parquet_reusable`` / ``assert_parquets_up_to_date``, so the eval path cannot drift into
    accepting a parquet the training path would rebuild. Checking only column NAMES here used to let
    two conditions through: float32 feature columns (which then die inside ``build_matrix`` with the
    bare AssertionError this guard exists to prevent) and a missing ``has_own_quality_metrics``
    column, which does not fail at all -- ``dataset.label_and_filter`` falls back to keeping every
    row, so the benchmark silently scores the homozygous rank-1 alleles inference never produces.
    """
    complaints = {p: c for p, c in ((p, dataset.parquet_contract_complaint(p)) for p in paths) if c}
    if complaints:
        raise RuntimeError(
            "%d parquet(s) no longer match the current feature contract: %s."
            % (len(complaints), _rebuild_instructions(complaints)))


def assert_parquets_supply_features(paths, names, requested_by):
    """Raises if any per-allele parquet is missing one of ``names``.

    ``names`` is whatever list is about to be turned into a feature matrix -- the current
    ``features.py`` contract, or an older MODEL's declared list when that model is being applied (see
    ``run_eval``). ``requested_by`` names the source of the list, so the error says which side is
    asking for the column that is missing.
    """
    stale = features.missing_feature_columns(paths, names)
    if stale:
        raise RuntimeError(
            "%d parquet(s) are missing feature column(s) required by %s: %s."
            % (len(stale), requested_by,
               _rebuild_instructions({p: "missing %s" % m for p, m in stale.items()})))


def _rebuild_instructions(complaints):
    """Renders ``{path: complaint}`` as ``<path> <complaint> -- rebuild with: <command>`` lines.

    The rebuild command is chosen PER PATH (``dataset.rebuild_command_for``): ``heldout.py`` cannot
    rebuild a training-combo parquet under ``data/real_quick/`` and ``dataset.py`` cannot rebuild a
    held-out sample's, so naming one command for every path sends the reader to a no-op.
    """
    return "; ".join("%s %s -- rebuild with: %s" % (p, c, dataset.rebuild_command_for(p))
                     for p, c in sorted(complaints.items()))


def run_eval(paths, model_path, out_json, max_alleles):
    """Applies the exported model to ``paths`` (per-allele parquets) and writes ``out_json`` + its
    ``*_violin.npz``; returns the metrics dict.

    Shared by the 43-sample benchmark and the per-dataset (HG002 genome / HG002 exome) evaluations so
    every dataset produces identical artifacts (scalar metrics JSON + the violin/pdiff/lcf npz). No
    fitting -- the deployed model is loaded from its ``.json[.gz]`` and applied with the ``pOk<0.5``
    gate.
    """
    print("\n==== load + compile the exported model: %s ====" % os.path.basename(model_path), flush=True)
    model = M.load(model_path)
    # Build each regime's matrix from the MODEL's own declared feature list, not this checkout's:
    # the compiled trees index features positionally, so that list -- in that order -- is the only
    # correct matrix for this model. It also means a model exported under an older contract can be
    # applied here (what compare_models needs) instead of being refused.
    declared = M.feature_names_of(model, model_path)
    # Two different questions, both required: does each parquet still match the CURRENT extractor
    # contract (so the rows and dtypes are the ones training would produce), and can it supply the
    # columns THIS model declares? compare_models reaches run_eval without rebuilding anything, so
    # this is the only place either is checked for it.
    assert_parquets_carry_contract(paths)
    assert_parquets_supply_features(paths, sorted(set().union(*declared.values())), model_path)
    model_json = model["genotyping_regimes"]
    compiled = {r: (M.compile_genotyping_regime(model_json[r]),
                    features.GENOTYPING_REGIME_BRANCH[r],
                    declared[features.GENOTYPING_REGIME_BRANCH[r]])
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
