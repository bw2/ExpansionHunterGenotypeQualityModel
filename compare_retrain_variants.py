"""Scores several models once each on the held-out samples and compares them in pairs (step 5 of
ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md).

``compare_models.py`` compares two models and scores both every time, so comparing seven models in eleven pairs would
score each model several times (about 20 minutes per model on the 81 held-out samples). This script scores each model
once, with the same two evaluations compare_models runs:

- the variant-locus view, ``heldout.run_eval``: gated MAE per genotyping regime and motif stratum, at the per-sample
  allele cap the report uses;
- the all-loci view, ``heldout.all_loci_early_warning``: on the representative fast-path loci of the all-loci sample,
  how well low pOk flags misses (reference-length calls whose truth is more than 2 repeats from the reference).

Each model's results are saved in ``--out-dir`` (``<label>.results.json``) with the model's content fingerprint, the
allele cap, the samples and a hash of the scoring modules' source (``SCORING_MODULES``, hashed when this script
imported them), so a rerun skips the models already scored. Saved results older than any held-out parquet, or
scored by other scoring code, are scored again, so a rebuilt parquet or changed scoring code never pairs old results
with new ones; results in one ``--out-dir`` belong to one ``--data-dir``. Then
each ``--pair`` is printed the way
compare_models prints a comparison (``compare_models.paired_gated_mae_comparison``), and everything is written to one
HTML page, ``<out-dir>/retrain_variants.<YYYY-MM-DD>_model.html``, dated by the newest model file.

Usage:
    python3 compare_retrain_variants.py --out-dir report/retrain_variants \\
        --model deployed=model/genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.20261017.json.gz \\
        --model baseline=model/genotype_quality_model_from_HG002_and_CHM1_CHM13.20261024.json.gz \\
        --pair baseline:deployed

Without ``--pair``, every model is compared with the first one.

Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import datetime
import hashlib
import html
import json
import os
import re

import compare_models
import dataset
import features
import gen_datasets
import heldout
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))

# Modules whose code decides a model's held-out numbers (row selection, labels, inference, the gate, the metrics):
# saved results scored by other versions of them are stale.
SCORING_MODULES = (heldout, M, dataset, features)


def scoring_code_hash(modules):
    """Returns the SHA-256 of the source files of ``modules``, in order."""
    digest = hashlib.sha256()
    for module in modules:
        with open(module.__file__, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


# The scoring code this process runs, hashed when it was imported and saved with each model's results. Results are
# reused only by a run of the same code, so results scored before a scoring module was edited mid-run are never
# paired with results of the edited code (their file time would make them look newer than the edit).
SCORING_CODE_HASH = scoring_code_hash(SCORING_MODULES)


def report_file_name(model_paths):
    """Returns ``retrain_variants.<YYYY-MM-DD>_model.html``, dated by the newest ``.<YYYYMMDD>.json.gz`` model file,
    as model reports are named (``report.report_html_name``); without any dated file, ``retrain_variants.html``."""
    dates = [m.group(1) for m in (re.search(r"\.(\d{8})\.json(?:\.gz)?$", os.path.basename(p)) for p in model_paths)
             if m]
    if not dates:
        return "retrain_variants.html"
    newest = max(dates)
    return "retrain_variants.%s-%s-%s_model.html" % (newest[:4], newest[4:6], newest[6:])


def parse_labeled_models(values):
    """Returns ``{label: path}`` in the order given, from ``LABEL=PATH`` strings.

    Raises:
        ValueError: On a value without ``=``, a repeated label, or a path that does not exist.
    """
    models = {}
    for value in values:
        label, sep, path = value.partition("=")
        if not sep or not label or not path:
            raise ValueError("--model takes LABEL=PATH; got %r" % value)
        if label in models:
            raise ValueError("--model label %r is given twice" % label)
        if not os.path.exists(path):
            raise ValueError("--model %s: no such file %s" % (label, path))
        models[label] = path
    return models


def parse_pairs(values, labels):
    """Returns ``[(new label, previous label), ...]`` from ``NEW:PREV`` strings; by default every model vs the first.

    Raises:
        ValueError: On a value without ``:`` or a label that is not one of ``labels``.
    """
    if not values:
        return [(label, labels[0]) for label in labels[1:]]
    pairs = []
    for value in values:
        new, sep, prev = value.partition(":")
        if not sep or new not in labels or prev not in labels or new == prev:
            raise ValueError("--pair takes NEW:PREV with two different --model labels (%s); got %r"
                             % (", ".join(labels), value))
        pairs.append((new, prev))
    return pairs


def _file_stem_of(label):
    """The file-name stem for a model label: runs of characters other than letters and digits become ``_``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _early_warning_from_json(early_warning):
    """Restores the float flag-rate keys of a saved ``heldout.all_loci_early_warning`` result (JSON keys are strings)."""
    if early_warning is None:
        return None
    return dict(early_warning, recall={subset: {float(rate): value for rate, value in by_rate.items()}
                                       for subset, by_rate in early_warning["recall"].items()})


def scored_results(label, model_path, paths, out_dir, max_alleles):
    """Returns a model's saved results, scoring it first unless ``out_dir`` already holds them for this model file.

    Saved results are reused only when they are newer than every parquet in ``paths`` (a rebuilt parquet makes them
    stale, as ``dataset.assert_parquets_up_to_date`` treats the branch tables), were scored by this process's scoring
    code (``SCORING_CODE_HASH``; results saved before the hash was recorded must instead be newer than every source
    file of ``SCORING_MODULES``), and the model's content fingerprint, the allele cap and the samples all match;
    otherwise the model is scored again and the file replaced. The file is written
    only once both evaluations are done, so an interrupted run never leaves results that look complete.

    Returns:
        ``{"model_path", "fingerprint", "max_alleles_per_sample", "samples", "variant_locus", "early_warning"}``:
        ``variant_locus`` is the ``heldout.run_eval`` result and ``early_warning`` the
        ``heldout.all_loci_early_warning`` one (None when no parquet has hom-ref loci).
    """
    stem = _file_stem_of(label)
    results_path = os.path.join(out_dir, "%s.results.json" % stem)
    samples = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    fingerprint = M.fingerprint(model_path)
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        results_time = os.path.getmtime(results_path)
        if "scoring_code_hash" in saved:
            other_scoring_code = saved["scoring_code_hash"] != SCORING_CODE_HASH
        else:
            other_scoring_code = results_time <= max(os.path.getmtime(m.__file__) for m in SCORING_MODULES)
        if other_scoring_code or results_time <= max((os.path.getmtime(p) for p in paths), default=0):
            print("  %s: %s is older than a held-out parquet or was scored by other scoring code; scoring again"
                  % (label, results_path), flush=True)
        elif (saved["fingerprint"], saved["max_alleles_per_sample"], saved["samples"]) == (fingerprint, max_alleles,
                                                                                           samples):
            print("  %s: reusing %s" % (label, results_path), flush=True)
            return dict(saved, early_warning=_early_warning_from_json(saved["early_warning"]))
        else:
            print("  %s: %s is for another model file, cap or sample set; scoring again" % (label, results_path),
                  flush=True)
    print("\n==== scoring %s: %s ====" % (label, os.path.basename(model_path)), flush=True)
    variant_locus = heldout.run_eval(paths, model_path, os.path.join(out_dir, "%s.variant_locus.json" % stem),
                                     max_alleles)
    results = {"model_path": os.path.abspath(model_path), "fingerprint": fingerprint,
               "max_alleles_per_sample": max_alleles, "samples": samples, "scoring_code_hash": SCORING_CODE_HASH,
               "variant_locus": variant_locus,
               "early_warning": heldout.all_loci_early_warning(paths, model_path)}
    tmp = results_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, results_path)
    # Reloaded so a fresh run hands back the same types (string-keyed JSON, restored rates) as a reused one.
    with open(results_path) as f:
        saved = json.load(f)
    return dict(saved, early_warning=_early_warning_from_json(saved["early_warning"]))


def _cell(value, fmt):
    """Formats a number for the HTML tables; NaN or None becomes an empty cell."""
    return "" if value is None or value != value else fmt % value


def _table(header, rows):
    """An HTML table from a header list and rows of already-escaped cell strings."""
    return "<table><tr>%s</tr>%s</table>" % (
        "".join("<th>%s</th>" % h for h in header),
        "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in row) for row in rows))


def report_html(results_by_label, pairs, comparisons, category_comparisons=None):
    """Returns the HTML page: the models, gated MAE per stratum and per motif-size category for all of them, each
    pair, and the early warning.

    Args:
        results_by_label: ``{label: scored_results(...)}`` in display order.
        pairs: ``[(new label, previous label), ...]``.
        comparisons: ``{(new, prev): compare_models.paired_gated_mae_comparison rows}``.
        category_comparisons: ``{(new, prev): compare_models.paired_gated_mae_by_motif_size_category rows}``
            (default: none shown).
    """
    labels = list(results_by_label)
    esc = html.escape
    first = results_by_label[labels[0]]
    parts = ["<h1>Retrain variants on the held-out samples</h1>",
             "<p>Written %s. %d held-out samples, at most %s alleles per sample (the variant-locus view), and the "
             "representative fast-path loci of the all-loci sample (the all-loci view). Each model is applied as "
             "saved, with its own "
             "pOk &lt; 0.5 gate for the length correction.</p>"
             % (datetime.date.today().isoformat(), len(first["samples"]),
                first["max_alleles_per_sample"] or "all")]

    parts.append("<h2>Models</h2>")
    parts.append(_table(["label", "file", "content fingerprint"],
                        [[esc(label), esc(os.path.basename(r["model_path"])), esc(r["fingerprint"].split("@")[-1])]
                         for label, r in results_by_label.items()]))

    parts.append("<h2>Variant-locus view: gated MAE (repeats)</h2>"
                 "<p>Alleles at loci whose truth differs from the reference, one row per scored allele. Lower is "
                 "better; the lowest in each row is in bold. Raw MAE is the error of the uncorrected calls, the same "
                 "for every model.</p>")
    mae_rows = []
    for rk in features.GENOTYPING_REGIMES:
        for stratum, key, _, _ in compare_models.MOTIF_STRATA:
            by_label = {}
            for label in labels:
                regime = results_by_label[label]["variant_locus"]["genotyping_regimes"][rk]
                by_label[label] = regime if key is None else regime[key]
            if not by_label[labels[0]].get("n"):
                continue
            best = min(by_label[label]["mae_gated"] for label in labels)
            mae_rows.append([esc(rk), esc(stratum), "{:,}".format(by_label[labels[0]]["n"]),
                             _cell(by_label[labels[0]]["mae_raw"], "%.4f")]
                            + [("<b>%s</b>" if by_label[label]["mae_gated"] == best else "%s")
                               % _cell(by_label[label]["mae_gated"], "%.4f") for label in labels])
    parts.append(_table(["regime", "motif stratum", "alleles", "raw MAE"] + [esc(label) for label in labels],
                        mae_rows))

    parts.append("<h2>Variant-locus view: gated MAE by motif size category (repeats)</h2>"
                 "<p>The same alleles split by motif size in bp (homopolymers are 1), pooled over alleles. Lower is "
                 "better; the lowest in each row is in bold.</p>")
    by_category = {label: {rk: r["variant_locus"]["genotyping_regimes"][rk].get("by_motif_size_category")
                           for rk in features.GENOTYPING_REGIMES} for label, r in results_by_label.items()}
    if any(by_category[label][rk] is None for label in labels for rk in features.GENOTYPING_REGIMES):
        parts.append("<p>Not available: a model's results were scored before the categories were reported.</p>")
    else:
        category_rows = []
        for rk in features.GENOTYPING_REGIMES:
            for category in features.MOTIF_SIZE_CATEGORY_LABELS:
                first_metrics = by_category[labels[0]][rk].get(category)
                if not first_metrics:
                    continue
                gated = [by_category[label][rk][category]["mae_gated"] for label in labels]
                category_rows.append([esc(rk), esc(category), "{:,}".format(first_metrics["n"]),
                                      _cell(first_metrics["mae_raw"], "%.4f")]
                                     + [("<b>%s</b>" if g == min(gated) else "%s") % _cell(g, "%.4f") for g in gated])
        parts.append(_table(["regime", "motif size (bp)", "alleles", "raw MAE"] + [esc(label) for label in labels],
                            category_rows))

    parts.append("<h2>Pairs</h2><p>Per stratum, the per-sample paired difference in gated MAE (new minus previous) "
                 "with a bootstrap 95%% interval over samples, every sample weighted equally, so the gated MAE shown "
                 "here is the mean of the per-sample values (the table above pools alleles). A stratum is won only "
                 "when the interval excludes 0; one with fewer than %d samples cannot be won. The intervals are not "
                 "adjusted for the number of strata, so a single win or loss can be chance.</p>"
                 % compare_models.MIN_SAMPLES_FOR_VERDICT)
    for new, prev in pairs:
        rows = comparisons[(new, prev)]
        verdict, wins = compare_models.verdict_of(rows)
        parts.append("<h3>%s vs %s: %s</h3><p>Strata won: %s %d, %s %d, no clear difference %d, too few samples %d."
                     "</p>" % (esc(new), esc(prev), esc(verdict), esc(new), wins["new"], esc(prev), wins["prev"],
                               wins["tie"], wins["too few samples"]))
        parts.append(_table(
            ["regime", "motif stratum", "%s gated MAE, mean over samples" % esc(prev),
             "%s gated MAE, mean over samples" % esc(new), "mean paired difference", "95% interval",
             "samples where %s is better" % esc(new), "winner"],
            [[esc(row["regime"]), esc(row["stratum"]), _cell(row["prev_mean_over_samples"], "%.4f"),
              _cell(row["new_mean_over_samples"], "%.4f"), _cell(row["deltas"].mean(), "%+.4f"),
              "[%+.4f, %+.4f]" % tuple(row["interval"]),
              "%d of %d" % (int((row["deltas"] < 0).sum()), len(row["deltas"])),
              esc({"new": new, "prev": prev}.get(row["winner"], row["winner"]))] for row in rows]))
        category_rows = (category_comparisons or {}).get((new, prev))
        if category_rows:
            parts.append("<details><summary>%s vs %s by motif size category (context, not counted in the verdict)"
                         "</summary>%s</details>" % (esc(new), esc(prev), _table(
                             ["regime", "motif size (bp)", "%s gated MAE, mean over samples" % esc(prev),
                              "%s gated MAE, mean over samples" % esc(new), "mean paired difference", "95% interval",
                              "winner"],
                             [[esc(row["regime"]), esc(row["category"]), _cell(row["prev_mean_over_samples"], "%.4f"),
                               _cell(row["new_mean_over_samples"], "%.4f"), _cell(row["deltas"].mean(), "%+.4f"),
                               "[%+.4f, %+.4f]" % tuple(row["interval"]),
                               esc({"new": new, "prev": prev}.get(row["winner"], row["winner"]))]
                              for row in category_rows])))

    parts.append("<h2>All-loci view: early warning on the fast path</h2>"
                 "<p>Representative fast-path loci of the all-loci sample. A miss is a reference-length call whose "
                 "truth is more than 2 repeats from the reference. ROC-AUC is of 1 - pOk for the misses among "
                 "reference-length calls, averaged over samples. Recall is the share of all misses that fall among the "
                 "1%, 2%, 5% or 10% of reference-length calls with the lowest pOk, pooled over samples. Higher is "
                 "better for both.</p>")
    warnings = {label: r["early_warning"] for label, r in results_by_label.items()}
    if any(w is None for w in warnings.values()):
        parts.append("<p>Not available: no held-out parquet was built with --keep-hom-ref-loci.</p>")
    else:
        rows = [["ROC-AUC of 1 - pOk for misses"] + [_cell(warnings[label]["auc_miss_mean"], "%.3f") for label in labels],
                ["share of fast-path calls below pOk 0.5"]
                + [_cell(warnings[label]["share_below_half_mean"], "%.3f") for label in labels]]
        for subset_name in warnings[labels[0]]["recall"]:
            for rate in heldout.EARLY_WARNING_FLAG_RATES:
                rows.append(["misses in the lowest %d%% of pOk (%s)" % (round(100 * rate), esc(subset_name))]
                            + [_cell(warnings[label]["recall"][subset_name][rate], "%.3f") for label in labels])
        parts.append(_table(["measure"] + [esc(label) for label in labels], rows))

    style = ("body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;max-width:1200px;color:#222}"
             "table{border-collapse:collapse;margin:8px 0 20px}"
             "th,td{border:1px solid #ccc;padding:3px 8px;font-size:13px;text-align:right}"
             "th{background:#f2f2f2}td:first-child,td:nth-child(2){text-align:left}"
             "h2{margin-top:32px}h3{margin-bottom:4px}")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Retrain variants</title><style>%s</style>"
            "</head><body>%s</body></html>" % (style, "\n".join(parts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="LABEL=PATH of a model to score; repeatable")
    parser.add_argument("--pair", action="append",
                        help="NEW:PREV labels to compare; repeatable; default: every model vs the first one")
    parser.add_argument("--out-dir", required=True, help="where each model's results and the HTML page go")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data_eval_43"),
                        help="dir holding the held-out real_43/<sample>.parquet files")
    parser.add_argument("--max-alleles-per-sample", type=int, default=gen_datasets.EVAL_CAP_DEFAULT,
                        help="seeded per-sample allele cap for the variant-locus view (0 = all alleles), as in "
                             "compare_models.py")
    args = parser.parse_args()
    try:
        models = parse_labeled_models(args.model)
        pairs = parse_pairs(args.pair, list(models))
    except ValueError as e:
        parser.error(str(e))

    paths = sorted(p for p in (os.path.join(args.data_dir, "real_43", "%s.parquet" % s) for s in heldout.SAMPLES)
                   if os.path.exists(p))
    if not paths:
        raise SystemExit("ERROR: no held-out parquets under %s/real_43/" % args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print("%d held-out samples, %d models, %d pairs" % (len(paths), len(models), len(pairs)), flush=True)
    results_by_label = {label: scored_results(label, path, paths, args.out_dir, args.max_alleles_per_sample)
                        for label, path in models.items()}

    comparisons, category_comparisons = {}, {}
    for new, prev in pairs:
        print("\n==================== %s vs %s ====================" % (new, prev))
        samples = results_by_label[prev]["samples"]
        prev_results, new_results = results_by_label[prev]["variant_locus"], results_by_label[new]["variant_locus"]
        comparisons[(new, prev)] = compare_models.paired_gated_mae_comparison(prev_results, new_results, samples)
        compare_models.print_paired_comparison(comparisons[(new, prev)], samples)
        category_comparisons[(new, prev)] = compare_models.paired_gated_mae_by_motif_size_category(
            prev_results, new_results, samples)
        compare_models.print_motif_size_category_comparison(category_comparisons[(new, prev)])
        compare_models.print_early_warning_comparison(results_by_label[prev]["early_warning"],
                                                      results_by_label[new]["early_warning"])
    out_html = os.path.join(args.out_dir, report_file_name(models.values()))
    with open(out_html, "w") as f:
        f.write(report_html(results_by_label, pairs, comparisons, category_comparisons))
    print("\nwrote %s" % out_html)


if __name__ == "__main__":
    main()
