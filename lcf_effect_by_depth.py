"""Does the length correction factor (LCF) hurt calls supported by few reads? Step 0 of
ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md.

The June REViewer deep-dive found 4 of 20 failures at HG002 10x where, with 3-8 reads, the corrected call
overshot the truth or moved away from it. Coverage has been a model input since 2026-10-01, but nothing makes the
correction smaller when reads are few. This script applies the deployed 2026-10-17 model, unchanged, to tables built
from the same ExpansionHunter build as its training data, and asks, for the calls the gate corrects (pOk < 0.5),
whether the corrected call ``round(eh / LCF)`` is closer to the truth than the raw call. Calls are binned by
per-allele ``depth`` and, separately, by ``spanning_total``, within each genotyping regime and motif stratum
(homopolymer or not).

Data sets:

- the 81 held-out samples (``heldout.SAMPLES``, 33-41x, never trained on), from ``data_eval_43/real_43.eh_b1fbc23/``;
- HG002 at 10x, 20x and 31x, chromosome 7 only (``HG002_CALIBRATION_CHROMOSOME``), from
  ``data/real_quick.eh_b1fbc23/``: the same loci and truth at three depths. The model never trained on HG002's rows
  there, but it learned those loci from the other training people.

Per bin: calls, the share the correction moves closer to and further from the truth, the mean absolute error (MAE,
in repeats) before and after, and the distance reduction (1 - MAE after / MAE before) with a 95% bootstrap interval
that resamples samples in the held-out set and loci in each HG002 set.

Outputs, in ``--out-dir`` and named after the model like the model report (``report.report_html_name``), e.g.
``lcf_effect_by_depth.2026-10-17_plus50_model.html`` and ``lcf_effect_by_depth.2026-10-17_plus50_model.gated_rows.parquet``
(one row per corrected call). Applying the model takes
about 2 hours; ``--reuse-gated-rows`` renders the report from the saved rows instead.

Coding rules: no type hints, Google docstrings, ``print()``.
"""

import argparse
import base64
import html
import io
import json
import os
import shutil

import numpy as np
import pandas as pd

import dataset
import features
import heldout
import model as M
import report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    HERE, "model", "genotype_quality_model_from_HG002_and_CHM1_CHM13_plus50.20261017.json.gz")

# The ExpansionHunter build of the 2026-10-17 model's training JSONs. Its tables are consistent with the model;
# tables from a later build would add the input changes since then (85bd281, 5210d1f, f37a139) to its errors.
TRAINING_DATA_EH_BUILD = "b1fbc23"
# Where those tables were kept when data_eval_43/real_43/ and data/real_quick/ were rebuilt from EH e90307c on
# 2026-10-23.
DEFAULT_HELDOUT_TABLES_DIR = os.path.join(HERE, "data_eval_43", "real_43.eh_b1fbc23")
DEFAULT_HG002_TABLES_DIR = os.path.join(HERE, "data", dataset.SOURCE_SUBDIR + ".eh_b1fbc23")

# The chromosome whose HG002 rows the 2026-10-17 model's trees never trained on: train._calib_mask gives HG002's and
# CHM1_CHM13's rows on one autosome to the isotonic calibration and keeps them out of training. Checked on
# 2026-10-23 by running _calib_mask on the 2026-10-08 data/parquet tables the model was trained on: chromosome 7,
# in all three regimes. The calibrators did see these rows, so pOk was fit to them; the LCF never was. The trees did
# train on chromosome 7 rows of the other training people, so these are new reads at loci the model has learned.
HG002_CALIBRATION_CHROMOSOME = "7"
HG002_COVERAGES = ("10x", "20x", "31x")
HG002_DATA_SETS = tuple("HG002 %s, chr%s" % (coverage, HG002_CALIBRATION_CHROMOSOME) for coverage in HG002_COVERAGES)
HELDOUT_DATA_SET = "81 held-out samples"
# Chart colors: the held-out set in orange, HG002's three depths as a light-to-dark blue ramp (both checked with the
# dataviz skill's palette validator, as categorical colors and as an ordinal ramp).
DATA_SET_COLORS = dict(zip((HELDOUT_DATA_SET,) + HG002_DATA_SETS, ("#eb6834", "#86b6ef", "#2a78d6", "#104281")))

# ExpansionHunter users apply the LCF only where pOk is below this (the gate the reports evaluate).
GATE_P_OK = 0.5

# Lower edges of the bins; each bin runs up to the next edge, the last one has no upper end. depth is fractional,
# spanning_total a read count.
DEPTH_BIN_EDGES = (0, 2, 3, 4, 6, 8, 12, 20, 40)
SPANNING_TOTAL_BIN_EDGES = (0, 1, 2, 3, 4, 6, 10, 20, 40)
BINNINGS = (("depth", DEPTH_BIN_EDGES, False), ("spanning_total", SPANNING_TOTAL_BIN_EDGES, True))
MOTIF_STRATA = ("non-homopolymer", "homopolymer")

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261023
_BOOTSTRAP_CHUNK = 100  # resamples drawn at once, so a bin with many loci never needs a huge index matrix

# Every bin is in the tables. A bin with fewer than MIN_CALLS_FOR_VERDICT corrected calls is left out of the verdict,
# and one with fewer than MIN_CALLS_TO_PLOT out of the charts too.
MIN_CALLS_FOR_VERDICT = 100
MIN_CALLS_TO_PLOT = 20
# The charts' distance-reduction axis; a bin below its floor is drawn on the floor with a downward marker.
PLOT_FLOOR_PERCENT = -100


def tables_to_score(heldout_tables_dir, hg002_tables_dir):
    """Returns ``[(data set, sample, parquet path, chromosome or None)]``, the HG002 sets first.

    Args:
        heldout_tables_dir: Folder holding ``<sample>.parquet`` for each held-out sample.
        hg002_tables_dir: Folder holding ``HG002_<coverage>.parquet``.

    Raises:
        SystemExit: If a table is missing.
    """
    tables = [(data_set, "HG002_%s" % coverage, os.path.join(hg002_tables_dir, "HG002_%s.parquet" % coverage),
               HG002_CALIBRATION_CHROMOSOME) for data_set, coverage in zip(HG002_DATA_SETS, HG002_COVERAGES)]
    tables += [(HELDOUT_DATA_SET, sample, os.path.join(heldout_tables_dir, "%s.parquet" % sample), None)
               for sample in heldout.SAMPLES]
    missing = [path for _, _, path, _ in tables if not os.path.exists(path)]
    if missing:
        raise SystemExit("ERROR: %d table(s) missing, e.g. %s" % (len(missing), missing[0]))
    return tables


def assert_tables_come_from_eh_build(paths, eh_build):
    """Raises unless every table's ``.sources.json`` records only JSONs from ExpansionHunter build ``eh_build``.

    Raises:
        SystemExit: Naming each table built from another build (or with no build record).
    """
    wrong = {}
    for path in paths:
        with open(dataset._sources_record_path(path)) as f:
            builds = sorted(set(json.load(f).get("eh_build_by_json", {}).values()), key=str)
        if builds != [eh_build]:
            wrong[path] = builds
    if wrong:
        raise SystemExit("ERROR: %d table(s) are not from ExpansionHunter build %s: %s"
                         % (len(wrong), eh_build, "; ".join("%s: %s" % item for item in sorted(wrong.items()))))


def gated_rows_of_table(path, data_set, sample, chromosome, compiled, declared):
    """Applies the model to one table and returns its calls the gate corrects, one row each.

    Rows are filtered as every evaluation here filters them (``dataset.label_and_filter``), and pOk and the LCF
    are rounded to the 3 decimals ExpansionHunter writes. ``corrected`` is the whole-repeat call
    ``round(eh / LCF)``.

    Args:
        path: Per-allele parquet.
        data_set: Name stored in the ``data_set`` column.
        sample: Name stored in the ``sample`` column.
        chromosome: Keep only this chromosome (no ``chr`` prefix), or None for all.
        compiled: ``{genotyping regime: model.compile_genotyping_regime(...)}``.
        declared: ``model.feature_names_of(...)``, the model's own feature lists.
    """
    df, _ = dataset.label_and_filter(dataset.variant_locus_rows(pd.read_parquet(path)))
    if chromosome is not None:
        df = df[df["chrom"] == chromosome]
    parts = []
    for regime in features.GENOTYPING_REGIMES:
        sub = df[df["genotyping_regime"] == regime]
        if sub.empty:
            continue
        branch = features.GENOTYPING_REGIME_BRANCH[regime]
        X = features.build_matrix(sub, branch, declared[branch])[0].to_numpy(float)
        p_ok = M.round_like_emitted(M.predict_proba_json(compiled[regime], X))[:, features.OK]
        gated = p_ok < GATE_P_OK
        lcf = M.round_like_emitted(M.predict_lcf_json(compiled[regime], X[gated]))
        corrected_calls = sub[gated]
        eh = corrected_calls["eh"].to_numpy(float)
        parts.append(pd.DataFrame({
            # Arrow strings: millions of these across the held-out set would take far more memory as objects.
            "data_set": data_set, "sample": sample,
            "locus_id": pd.array(corrected_calls["locus_id"].to_numpy(), dtype="string[pyarrow]"),
            "genotyping_regime": regime, "motif_size": corrected_calls["motif_size"].to_numpy(float),
            "depth": corrected_calls["depth"].to_numpy(float),
            "spanning_total": corrected_calls["spanning_total"].to_numpy(float),
            "eh": eh, "true": corrected_calls["true"].to_numpy(float), "p_ok": p_ok[gated], "lcf": lcf,
            "corrected": np.round(eh / lcf)}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_gated_rows(model_path, heldout_tables_dir, hg002_tables_dir, eh_build, out_path):
    """Applies the model to every table of ``tables_to_score``, one at a time, and writes the corrected calls.

    Each table's calls are saved as soon as they are computed, in ``<out_path>.parts.<hash of the model file>/``, and
    a rerun skips the tables saved there, so a run stopped by the memory guard loses at most one table. The hash keeps
    a retrained model with the same name from picking them up. The folder is removed once ``out_path`` is written.
    """
    tables = tables_to_score(heldout_tables_dir, hg002_tables_dir)
    paths = [path for _, _, path, _ in tables]
    assert_tables_come_from_eh_build(paths, eh_build)
    serialized = M.load(model_path)
    declared = M.feature_names_of(serialized, model_path)
    # The same two checks heldout.run_eval makes before applying a model to tables.
    heldout.assert_parquets_carry_contract(paths)
    heldout.assert_parquets_supply_features(paths, sorted(set().union(*declared.values())), model_path)
    compiled = {regime: M.compile_genotyping_regime(serialized["genotyping_regimes"][regime])
                for regime in features.GENOTYPING_REGIMES}
    parts_dir = "%s.parts.%s" % (out_path, M.fingerprint(model_path).rsplit("@", 1)[1])
    os.makedirs(parts_dir, exist_ok=True)
    part_paths = [os.path.join(parts_dir, "%s.parquet" % sample) for _, sample, _, _ in tables]
    for i, ((data_set, sample, path, chromosome), part_path) in enumerate(zip(tables, part_paths)):
        if os.path.exists(part_path):
            print("  %d/%d %s: saved by an earlier run" % (i + 1, len(tables), sample), flush=True)
            continue
        part = gated_rows_of_table(path, data_set, sample, chromosome, compiled, declared)
        # Written under another name and then renamed, so a run stopped mid-write leaves no partial part behind.
        part.to_parquet(part_path + ".partial", index=False)
        os.replace(part_path + ".partial", part_path)
        print("  %d/%d %s: %d corrected calls" % (i + 1, len(tables), sample, len(part)), flush=True)
    gated_rows = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
    for column in ("data_set", "sample", "genotyping_regime"):
        gated_rows[column] = gated_rows[column].astype("category")
    gated_rows.to_parquet(out_path, index=False)
    shutil.rmtree(parts_dir)
    print("wrote %s (%d corrected calls)" % (out_path, len(gated_rows)), flush=True)


def bin_labels(edges, whole_numbers):
    """Returns a label per bin of ``edges`` (lower edges): ``"2-3"`` is [2, 3); for whole numbers ``"4-5"``."""
    labels = []
    for low, high in zip(edges, list(edges[1:]) + [None]):
        if high is None:
            labels.append("%g+" % low)
        elif whole_numbers:
            labels.append("%g" % low if high - low == 1 else "%g-%g" % (low, high - 1))
        else:
            labels.append("%g-%g" % (low, high))
    return labels


def bin_indices(values, edges):
    """Returns each value's bin index in ``edges`` (lower edges), or -1 for NaN or a value below the first edge."""
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="right") - 1
    return np.where(np.isnan(values), -1, idx)


def distance_reduction_interval(error_before, error_after, cluster_ids, n_resamples=BOOTSTRAP_RESAMPLES,
                                seed=BOOTSTRAP_SEED):
    """Returns the 95% percentile-bootstrap interval of ``1 - sum(error_after) / sum(error_before)``.

    Resamples whole clusters (samples, or loci) with replacement, so correlated calls move together. Returns
    ``(nan, nan)`` with fewer than 2 clusters.
    """
    codes = pd.factorize(np.asarray(cluster_ids))[0]
    before_by_cluster = np.bincount(codes, weights=np.asarray(error_before, dtype=float))
    after_by_cluster = np.bincount(codes, weights=np.asarray(error_after, dtype=float))
    n_clusters = before_by_cluster.size
    if n_clusters < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reductions = []
    for start in range(0, n_resamples, _BOOTSTRAP_CHUNK):
        picks = rng.integers(0, n_clusters, size=(min(_BOOTSTRAP_CHUNK, n_resamples - start), n_clusters))
        with np.errstate(divide="ignore", invalid="ignore"):
            reductions.append(1.0 - after_by_cluster[picks].sum(axis=1) / before_by_cluster[picks].sum(axis=1))
    low, high = np.nanpercentile(np.concatenate(reductions), [2.5, 97.5])
    return float(low), float(high)


def summarize_calls(error_before, error_after, cluster_ids):
    """Returns the per-bin measures for one group of corrected calls (see the module docstring)."""
    error_before = np.asarray(error_before, dtype=float)
    error_after = np.asarray(error_after, dtype=float)
    mae_before, mae_after = error_before.mean(), error_after.mean()
    low, high = distance_reduction_interval(error_before, error_after, cluster_ids)
    return {"calls": int(error_before.size),
            "share_closer": float((error_after < error_before - heldout.EPS).mean()),
            "share_further": float((error_after > error_before + heldout.EPS).mean()),
            "mae_before": float(mae_before), "mae_after": float(mae_after),
            "distance_reduction": float(1.0 - mae_after / mae_before) if mae_before > 0 else float("nan"),
            "interval_low": low, "interval_high": high}


def summarize(gated_rows):
    """Returns one row of measures per data set, regime, motif stratum, binning and bin (plus an "all" bin).

    The bootstrap resamples samples in a data set of several samples and loci in a data set of one.
    """
    rows = []
    for data_set, calls_of_set in gated_rows.groupby("data_set", observed=True, sort=False):
        cluster_column = "sample" if calls_of_set["sample"].nunique() > 1 else "locus_id"
        for regime in features.GENOTYPING_REGIMES:
            calls_of_regime = calls_of_set[calls_of_set["genotyping_regime"] == regime]
            for stratum in MOTIF_STRATA:
                homopolymer = calls_of_regime["motif_size"].to_numpy(float) == 1
                calls = calls_of_regime[homopolymer if stratum == "homopolymer" else ~homopolymer]
                if calls.empty:
                    continue
                true = calls["true"].to_numpy(float)
                error_before = np.abs(true - calls["eh"].to_numpy(float))
                error_after = np.abs(true - calls["corrected"].to_numpy(float))
                clusters = calls[cluster_column].to_numpy()
                for binning, edges, whole_numbers in BINNINGS:
                    idx = bin_indices(calls[binning].to_numpy(float), edges)
                    labels = bin_labels(edges, whole_numbers)
                    groups = [("all", np.ones(len(calls), bool))]
                    groups += [("missing", idx == -1)] if (idx == -1).any() else []
                    groups += [(label, idx == i) for i, label in enumerate(labels)]
                    for label, in_bin in groups:
                        if in_bin.any():
                            rows.append({"data_set": data_set, "genotyping_regime": regime, "motif_stratum": stratum,
                                         "binning": binning, "bin": label,
                                         **summarize_calls(error_before[in_bin], error_after[in_bin],
                                                           clusters[in_bin])})
    return pd.DataFrame(rows)


def _bins_judged(summary):
    """Returns the bins (not "all") with at least ``MIN_CALLS_FOR_VERDICT`` calls."""
    return summary[(summary["bin"] != "all") & (summary["calls"] >= MIN_CALLS_FOR_VERDICT)]


def harmful_bins(summary):
    """Returns the judged bins where the correction does more harm than good: the distance reduction's interval lies
    below 0, so the corrected calls are further from the truth on average beyond the bootstrap's doubt."""
    bins = _bins_judged(summary)
    return bins[bins["interval_high"] < 0]


def bins_moving_more_calls_further(summary):
    """Returns the judged bins not in ``harmful_bins`` where the correction moves more calls further from the truth
    than closer. A raw comparison of two shares with no interval, and one that can hold while the error falls (many
    calls moved a little the wrong way, a few moved a lot the right way), so it is not evidence of net harm."""
    bins = _bins_judged(summary)
    return bins[~(bins["interval_high"] < 0) & (bins["share_further"] > bins["share_closer"])]


def output_prefix(out_dir, model_path, eh_build):
    """Returns ``<out_dir>/lcf_effect_by_depth.<what report.report_html_name puts after model_report.>``, so the
    files are named after the model the way the model report is. Tables from an ExpansionHunter build other than
    ``TRAINING_DATA_EH_BUILD`` get ``lcf_effect_by_depth_on_eh_<build>_tables`` in place of ``lcf_effect_by_depth``,
    so their saved calls (and the resume folder named after them) never mix with or overwrite that build's."""
    name = ("lcf_effect_by_depth" if eh_build == TRAINING_DATA_EH_BUILD
            else "lcf_effect_by_depth_on_eh_%s_tables" % eh_build)
    return os.path.join(out_dir, name + "." + report.report_html_name(model_path).split(".", 1)[1][:-len(".html")])


def _bins_of(binning):
    """Returns the bin labels of ``binning`` in order, with "missing" first."""
    edges, whole_numbers = {name: (e, w) for name, e, w in BINNINGS}[binning]
    return ["missing"] + bin_labels(edges, whole_numbers)


def _png_data_uri(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def distance_reduction_figure(summary, binning):
    """Returns a figure with a panel per motif stratum (rows) and genotyping regime (columns), showing each data
    set's distance reduction by bin of ``binning`` with its 95% interval. Bins with fewer than
    ``MIN_CALLS_TO_PLOT`` calls are left out."""
    bins = _bins_of(binning)
    data_sets = [d for d in DATA_SET_COLORS if d in set(summary["data_set"])]
    offsets = np.linspace(-0.27, 0.27, len(data_sets))
    fig, axes = plt.subplots(len(MOTIF_STRATA), len(features.GENOTYPING_REGIMES), figsize=(15, 8), sharey=True,
                             squeeze=False)
    plotted = [0.0]  # every drawn value, to fit the shared y-axis
    for row, stratum in enumerate(MOTIF_STRATA):
        for col, regime in enumerate(features.GENOTYPING_REGIMES):
            ax = axes[row, col]
            panel = summary[(summary["binning"] == binning) & (summary["genotyping_regime"] == regime)
                            & (summary["motif_stratum"] == stratum) & (summary["bin"] != "all")
                            & (summary["calls"] >= MIN_CALLS_TO_PLOT)]
            shown_bins = [b for b in bins if b in set(panel["bin"])]
            for offset, data_set in zip(offsets, data_sets):
                points = panel[panel["data_set"] == data_set].set_index("bin").reindex(shown_bins).dropna(
                    subset=["calls"])
                if points.empty:
                    continue
                x = np.array([shown_bins.index(b) for b in points.index]) + offset
                # Points and interval ends below the axis floor are drawn on it.
                y, low, high = (np.maximum(100 * points[c].to_numpy(float), PLOT_FLOOR_PERCENT)
                                for c in ("distance_reduction", "interval_low", "interval_high"))
                below_floor = 100 * points["distance_reduction"].to_numpy(float) < PLOT_FLOOR_PERCENT
                plotted.extend(v for v in np.concatenate([y, low, high]) if not np.isnan(v))
                color = DATA_SET_COLORS[data_set]
                ax.plot(x, y, color=color, linewidth=1.2, alpha=0.6)
                # A percentile interval can miss its own point estimate when the resamples are skewed.
                ax.errorbar(x[~below_floor], y[~below_floor],
                            yerr=[np.maximum(y - low, 0)[~below_floor], np.maximum(high - y, 0)[~below_floor]],
                            fmt="o", color=color,
                            markersize=6, elinewidth=1.2, capsize=0, label=data_set if row == col == 0 else None)
                ax.plot(x[below_floor], y[below_floor], "v", color=color, markersize=7)
            ax.axhline(0, color="#52514e", linewidth=0.9)
            ax.set_xticks(range(len(shown_bins)))
            ax.set_xticklabels(shown_bins, rotation=45, ha="right", fontsize=9)
            ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title("%s, %s" % (regime, stratum), fontsize=11)
            if col == 0:
                ax.set_ylabel("distance reduction (%)")
            if row == len(MOTIF_STRATA) - 1:
                ax.set_xlabel(binning)
    axes[0, 0].set_ylim(min(plotted) - 5, max(plotted) + 5)  # shared by every panel
    fig.legend(loc="upper center", ncol=len(data_sets), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _percent(value):
    return "n/a" if np.isnan(value) else "%+.0f%%" % (100 * value)


def _reduction_cell(row):
    interval = ("" if np.isnan(row["interval_low"])
                else " [%+.0f, %+.0f]" % (100 * row["interval_low"], 100 * row["interval_high"]))
    return _percent(row["distance_reduction"]) + interval


def _measures_table(rows, first_columns):
    """Renders ``rows`` (summary rows) as an HTML table: ``first_columns`` then the measures."""
    head = "".join("<th>%s</th>" % html.escape(c) for c in first_columns) + (
        "<th>calls</th><th>moved closer</th><th>moved further</th><th>MAE before</th><th>MAE after</th>"
        "<th>distance reduction [95% interval]</th>")
    body = []
    for _, row in rows.iterrows():
        body.append("<tr>%s<td>%d</td><td>%.0f%%</td><td>%.0f%%</td><td>%.2f</td><td>%.2f</td><td>%s</td></tr>" % (
            "".join("<td>%s</td>" % html.escape(str(row[c])) for c in first_columns), row["calls"],
            100 * row["share_closer"], 100 * row["share_further"], row["mae_before"], row["mae_after"],
            _reduction_cell(row)))
    return "<table><tr>%s</tr>%s</table>" % (head, "".join(body))


def _ordered(summary, binning):
    """Returns ``summary`` sorted by data set, regime, motif stratum and bin order ("all" first)."""
    order = {"data_set": list(DATA_SET_COLORS), "genotyping_regime": list(features.GENOTYPING_REGIMES),
             "motif_stratum": list(MOTIF_STRATA), "bin": ["all"] + _bins_of(binning)}
    keys = [summary[c].map({v: i for i, v in enumerate(values)}) for c, values in order.items()]
    return summary.assign(_k0=keys[0], _k1=keys[1], _k2=keys[2], _k3=keys[3]).sort_values(
        ["_k0", "_k1", "_k2", "_k3"]).drop(columns=["_k0", "_k1", "_k2", "_k3"])


def render_html(summary, model_path, eh_build, gated_rows_path, out_html):
    """Writes the standalone HTML report of ``summary`` (see ``summarize``); ``eh_build`` is the ExpansionHunter build
    the tables were checked against."""
    css = ("body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;"
           "color:#222;max-width:1180px}h1{font-size:22px}h2{font-size:17px;margin-top:34px;"
           "border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;"
           "margin:10px 0;font-size:13px}th,td{border:1px solid #ccc;padding:5px 9px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}th{background:#f4f4f4}"
           "img{max-width:100%;height:auto}.note{color:#555;font-size:14px}code{background:#f4f4f4;"
           "padding:1px 4px}summary{cursor:pointer;margin:8px 0;font-size:14px}")
    depth_rows = summary[summary["binning"] == "depth"]
    harmful = harmful_bins(summary)
    bin_columns = ["data_set", "genotyping_regime", "motif_stratum", "binning", "bin"]
    if harmful.empty:
        verdict = ("<p>No bin with at least %d corrected calls is made worse by the correction: in every one, the "
                   "distance reduction's interval reaches 0 or above.</p>" % MIN_CALLS_FOR_VERDICT)
    else:
        verdict = ("<p>%d bin(s) with at least %d corrected calls where the correction does more harm than good (the "
                   "distance reduction's interval lies below 0):</p>%s"
                   % (len(harmful), MIN_CALLS_FOR_VERDICT, _measures_table(harmful, bin_columns)))
    more_further = bins_moving_more_calls_further(summary)
    if not more_further.empty:
        verdict += ("<p>%d other bin(s) where the correction moves more calls further from the truth than closer, "
                    "without a distance reduction below 0. That compares two shares with no interval, and can hold "
                    "while the error falls, so it is not evidence of net harm:</p>%s"
                    % (len(more_further), _measures_table(more_further, bin_columns)))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>LCF effect by read depth</title>",
        "<style>%s</style></head><body>" % css,
        "<h1>Does the length correction factor hurt calls with few reads?</h1>",
        "<p class='note'>Model: <code>%s</code>. Corrected calls: <code>%s</code>. Step 0 of "
        "<code>ALL_LOCI_AND_RELATIVE_TOLERANCE_RETRAIN_PLAN.md</code>.</p>"
        % (html.escape(os.path.basename(model_path)), html.escape(os.path.basename(gated_rows_path))),
        "<p class='note'>The length correction factor (LCF) is applied only where the model's pOk is below %g, giving "
        "the corrected call round(eh / LCF). For every such call, this report compares the corrected call's distance "
        "from the truth with the raw call's, in bins of the allele's read depth (<code>depth</code>, "
        "ExpansionHunter's per-allele depth) and of the locus's spanning reads (<code>spanning_total</code>), "
        "separately for each genotyping regime and for homopolymer (1 bp motif) and other loci. The tables come "
        "from ExpansionHunter build %s, %s. Data sets:</p><ul class='note'>"
        "<li><b>%s</b>: never trained on, 33-41x.</li>"
        "<li><b>HG002 at 10x, 20x and 31x, chromosome %s only</b>: the same loci and truth at three depths. The model "
        "trained on HG002's other chromosomes; HG002's chromosome %s rows went to the pOk calibration instead, so its "
        "trees (the LCF's included) never trained on them. They did train on chromosome %s of the other training "
        "people, so these are new reads at loci the model has learned, not unseen loci.</li></ul>"
        % (GATE_P_OK, eh_build,
           "the build of the model's training data" if eh_build == TRAINING_DATA_EH_BUILD
           else "not the build of the 2026-10-17 model's training data (%s)" % TRAINING_DATA_EH_BUILD,
           HELDOUT_DATA_SET, HG002_CALIBRATION_CHROMOSOME, HG002_CALIBRATION_CHROMOSOME, HG002_CALIBRATION_CHROMOSOME),
        "<p class='note'><b>Moved closer / further</b>: share of the corrected calls whose distance from the truth "
        "the correction shrinks / grows (the rest are unchanged). <b>MAE</b>: mean absolute error, in repeats, "
        "before and after the correction. <b>Distance reduction</b>: 1 - MAE after / MAE before, so above 0 means "
        "the correction helps. <b>95%% interval</b>: percentile bootstrap (%d resamples) over samples in the "
        "held-out set and over loci in each HG002 set. Depth bins include their lower edge (2-3 is 2 up to 3); "
        "spanning-read bins are whole numbers.</p>" % BOOTSTRAP_RESAMPLES,
        "<h2>Verdict</h2>", verdict,
        "<h2>All depths together</h2>",
        _measures_table(_ordered(depth_rows[depth_rows["bin"] == "all"], "depth"),
                        ["data_set", "genotyping_regime", "motif_stratum"]),
    ]
    for binning, _, _ in BINNINGS:
        parts.append("<h2>By %s</h2>" % binning)
        parts.append("<p class='note'>Bins with fewer than %d corrected calls are left out of the chart. A downward "
                     "triangle is a bin below %d%%, drawn at the floor.</p>" % (MIN_CALLS_TO_PLOT, PLOT_FLOOR_PERCENT))
        parts.append("<img src='%s'>" % _png_data_uri(distance_reduction_figure(summary, binning)))
        rows = _ordered(summary[summary["binning"] == binning], binning)
        for regime in features.GENOTYPING_REGIMES:
            for stratum in MOTIF_STRATA:
                table_rows = rows[(rows["genotyping_regime"] == regime) & (rows["motif_stratum"] == stratum)]
                parts.append("<details><summary>Table: %s, %s, by %s</summary>%s</details>"
                             % (regime, stratum, binning, _measures_table(table_rows, ["data_set", "bin"])))
    parts.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w") as f:
        f.write("\n".join(parts))
    print("wrote %s" % out_html, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="model .json[.gz] to apply")
    parser.add_argument("--heldout-tables-dir", default=DEFAULT_HELDOUT_TABLES_DIR,
                        help="holds <sample>.parquet for the held-out samples (default: %(default)s)")
    parser.add_argument("--hg002-tables-dir", default=DEFAULT_HG002_TABLES_DIR,
                        help="holds HG002_<coverage>.parquet (default: %(default)s)")
    parser.add_argument("--expected-eh-build", default=TRAINING_DATA_EH_BUILD,
                        help="refuse tables built from JSONs of any other ExpansionHunter build")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "report"))
    parser.add_argument("--reuse-gated-rows", action="store_true",
                        help="render from the corrected calls written by an earlier run instead of applying the model")
    args = parser.parse_args()
    prefix = output_prefix(args.out_dir, args.model, args.expected_eh_build)
    gated_rows_path = prefix + ".gated_rows.parquet"
    if not args.reuse_gated_rows:
        write_gated_rows(args.model, args.heldout_tables_dir, args.hg002_tables_dir, args.expected_eh_build,
                         gated_rows_path)
    elif not os.path.exists(gated_rows_path):
        raise SystemExit("ERROR: --reuse-gated-rows, but %s does not exist" % gated_rows_path)
    render_html(summarize(pd.read_parquet(gated_rows_path)), args.model, args.expected_eh_build, gated_rows_path,
                prefix + ".html")


if __name__ == "__main__":
    main()
