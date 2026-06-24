"""Unify the real + simulated sources into one parquet per genotyping branch.

Reads the per-combo source parquets written by ``data_gcs.py`` (full-branch real),
``data_fast_real.py`` (fast-branch real), and ``data_sim.py`` (both branches), then for each
branch (``full``, ``fast``) computes the labels and applies the row filtering from
GENOTYPE_QUALITY_METRICS_PLAN.md (sec 3-4) and writes
``data/parquet/{full,fast}.parquet``.

Branch assembly (SPEC sec 5.3, avoiding near-duplicate-locus leakage):
  - ``full`` = all full-branch real rows (low-mem-streaming EHv5) + simulated rows from the
    ``low-mem-streaming`` run.
  - ``fast`` = fast-branch real rows (``QuickGenotype``) from the regenerated optimized-streaming
    run + simulated ``optimized-streaming`` rows tagged ``genotyping_branch == "fast"``.
  The optimized-streaming full-genotyper *fallback* rows (real and sim) are dropped from the
  primary parquets: they sit on the same loci as the low-mem-streaming full rows, so pooling them
  would leak a locus across train/test within the full branch.

Labels (per allele): ``q = eh/true``, ``t = log(eh) - log(true)``; ``direction`` / ``dir_code``
0/1/2 via the SIZE-DEPENDENT tolerance band (``size_tolerance``): ``dr = round(eh) - round(true)``,
OK if ``|dr| <= tol``, TOO_LONG if ``> tol``, TOO_SHORT if ``< -tol``, where ``tol`` grows with the true
allele size in bp (small alleles must be exact). Also adds ``allele_bp``, ``tol_repeats`` and
``regime`` (``fast`` / ``full_spanning`` / ``full_nonspanning``).
Filtering: drop EH no-calls / missing or non-positive ``eh``/``true``; keep
``purity >= 0.9`` and non-negative-control loci (sim passes by construction, ``purity = 1.0``);
drop chrM. ``eh`` and all features come from the JSON (already true of the source parquets).

Coding rules: no type hints, Google docstrings, ``print()``. Determinism: no randomness.
"""

import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd

import features
import size_tolerance

VALID_CHROMS = set(str(i) for i in range(1, 23)) | {"X", "Y"}

# Columns kept from each source parquet (everything downstream build_dataset / run_cv needs).
# The high-cardinality string `locus_id` is dropped after `chrom` is derived from it, and floats
# are downcast -- both essential to keep the ~4M-row fast-branch assembly within RAM.
_KEEP = (set(features.FAST_FEATURES) | set(features.FULL_FEATURES)
         | {"ci_start", "ci_end", "ci_width", "eh", "true", "purity", "is_negative_locus",
            "motif_size", "locus_id", "chrom_raw", "source", "sample", "coverage", "data_type",
            "genotyping_branch", "quick_genotype", "analysis_mode",
            # eh_q is not a model feature, but evaluate.direction_baseline uses it as the
            # full-branch "neg_eh_q" baseline, so it must survive into the parquet.
            "eh_q"})
_CATEGORICAL = ["source", "sample", "genotyping_branch", "analysis_mode", "chrom"]


def normalize_chrom(df):
    """Returns a normalized chromosome Series ({1..22, X, Y}) for the rows of ``df``.

    Real rows carry ``locus_id`` like ``"1-591733-591751-A"`` (chrom = first token); simulated
    rows carry ``chrom_raw`` like ``"chr12"``. Any leading ``chr`` is stripped so both map to the
    same space. chrM / unknown contigs become NaN (dropped by the caller).
    """
    if "chrom_raw" in df.columns and df["chrom_raw"].notna().any():
        raw = df["chrom_raw"].where(df["chrom_raw"].notna(),
                                    df["locus_id"].astype(str).str.split("-").str[0])
    else:
        raw = df["locus_id"].astype(str).str.split("-").str[0]
    chrom = raw.astype(str).str.replace(r"^chr", "", regex=True)
    return chrom.where(chrom.isin(VALID_CHROMS))


def add_labels_and_filter(df):
    """Adds q / t / direction labels and applies the plan's row filtering.

    Args:
        df: Assembled per-branch rows (real + sim) with ``eh``, ``true``, ``purity``,
            ``is_negative_locus`` and the source/feature columns.

    Returns:
        A ``(kept_df, drop_counts)`` tuple. ``kept_df`` has ``chrom``, ``q``, ``t``,
        ``direction``, ``dir_code`` added; ``drop_counts`` is a dict of rows removed per reason.
    """
    n0 = len(df)
    df = df.copy()
    if "chrom" not in df.columns:  # assemble_branch derives chrom up front; tests pass raw frames
        df["chrom"] = normalize_chrom(df)
    eh = pd.to_numeric(df["eh"], errors="coerce")
    true = pd.to_numeric(df["true"], errors="coerce")

    drops = {}
    keep = pd.Series(True, index=df.index)

    bad_chrom = df["chrom"].isna()
    drops["chrM_or_unknown_contig"] = int(bad_chrom.sum())
    keep &= ~bad_chrom

    # Negative-control loci are dropped BEFORE the missing-label filter so that a matched
    # negative-control row (which legitimately carries no truth size) is attributed to
    # negative_control_locus rather than miscounted under missing_eh_or_true. fillna(False):
    # an UNMATCHED row has a NaN flag (and NaN true) -- it is not a known negative control, so it
    # must fall through to the missing_eh_or_true filter, not be mislabeled a negative control.
    negative = df["is_negative_locus"].fillna(False).astype(bool)
    drops["negative_control_locus"] = int((negative & keep).sum())
    keep &= ~negative

    no_label = eh.isna() | true.isna()
    drops["missing_eh_or_true"] = int((no_label & keep).sum())
    keep &= ~no_label

    nonpos = (eh <= 0) | (true <= 0)
    drops["nonpositive_eh_or_true"] = int((nonpos & keep).sum())
    keep &= ~nonpos

    # A row with no parseable repeat unit has motif_size NaN, so allele_bp = motif*round(true)
    # would be NaN and size_tolerance.tol_repeats would default it to the WIDEST (tol=8) band,
    # silently labeling gross miscalls as OK. Such loci cannot be sized -- drop them.
    motif = pd.to_numeric(df["motif_size"], errors="coerce")
    bad_motif = motif.isna() | (motif <= 0)
    drops["missing_motif_size"] = int((bad_motif & keep).sum())
    keep &= ~bad_motif

    purity = pd.to_numeric(df["purity"], errors="coerce")
    impure = purity < 0.9  # NaN purity -> False here; guard below
    drops["impure_below_0.9"] = int((impure & keep).sum())
    keep &= ~impure
    no_purity = purity.isna()
    drops["missing_purity"] = int((no_purity & keep).sum())
    keep &= ~no_purity

    df = df[keep].copy()
    eh = eh[keep]
    true = true[keep]
    df["q"] = eh / true
    df["t"] = np.log(eh) - np.log(true)
    add_size_tolerance_labels(df, eh=eh, true=true)
    drops["_kept"] = len(df)
    drops["_input"] = n0
    return df, drops


def add_size_tolerance_labels(df, eh=None, true=None):
    """Adds the size-tolerance direction labels + ``allele_bp`` / ``tol_repeats`` / ``regime``.

    Mutates ``df`` in place and returns it. ``direction`` / ``dir_code`` use the size-dependent
    tolerance band (``size_tolerance``) instead of a fixed +/-1: small alleles must be exact,
    large expansions are graded against a wider band. ``regime`` routes each allele to its
    expert model (fast / full_spanning / full_nonspanning). Shared by ``build_dataset`` (fresh
    build) and ``relabel.py`` (cheap in-place re-derive from the existing parquet).

    Args:
        df: Per-allele frame with ``eh``, ``true``, ``motif_size``, ``genotyping_branch`` and
            ``spanning_at_called``.
        eh: Optional pre-coerced numeric ``eh`` (defaults to ``df["eh"]``).
        true: Optional pre-coerced numeric ``true`` (defaults to ``df["true"]``).

    Returns:
        The same ``df`` with the new columns added.
    """
    eh = pd.to_numeric(df["eh"], errors="coerce") if eh is None else eh
    true = pd.to_numeric(df["true"], errors="coerce") if true is None else true
    motif = pd.to_numeric(df["motif_size"], errors="coerce")
    df["allele_bp"] = (motif * np.round(true)).astype("float32")
    tol = size_tolerance.tol_repeats(df["allele_bp"].to_numpy())
    df["tol_repeats"] = tol.astype("int16")
    df["dir_code"] = size_tolerance.direction_codes(eh.to_numpy(), true.to_numpy(), tol)
    df["direction"] = pd.Series(df["dir_code"], index=df.index).map(
        {0: "OK", 1: "TOO_LONG", 2: "TOO_SHORT"})
    df["regime"] = size_tolerance.regime_of(
        df["genotyping_branch"].to_numpy(),
        pd.to_numeric(df["spanning_at_called"], errors="coerce").to_numpy())
    return df


def _load_sim(data_dir):
    """Loads the simulated rows (sample relabeled to 'sim'), or an empty frame."""
    sim_path = os.path.join(data_dir, "sim", "sim_rows.parquet")
    if not os.path.exists(sim_path):
        return pd.DataFrame()
    sim = pd.read_parquet(sim_path)
    sim["sample"] = "sim"  # real samples are HG002 / CHM1_CHM13; sim is its own group
    return sim


def _prep(d):
    """Trims a source frame to the kept columns, derives ``chrom``, drops the heavy ``locus_id``
    string, downcasts floats and categorizes low-cardinality strings -- keeps the concat lean."""
    d = d[[c for c in sorted(_KEEP) if c in d.columns]].copy()  # sorted: stable column order
    # data_type is only stamped on the newer (element/exome) parquets; the original illumina real
    # parquets and the sim rows predate it -- default by source so CV domain-splitting always works.
    if "data_type" not in d.columns:
        d["data_type"] = d["source"].map({"real": "illumina", "sim": "sim"}) if "source" in d.columns else "illumina"
    d["chrom"] = normalize_chrom(d)
    d = d.drop(columns=[c for c in ("locus_id", "chrom_raw") if c in d.columns])
    for c in d.select_dtypes("float64").columns:
        d[c] = d[c].astype("float32")
    for c in _CATEGORICAL:
        if c in d.columns:
            d[c] = d[c].astype("category")
    return d


def assemble_branch(data_dir, branch):
    """Assembles ONE branch's pre-label rows, holding only that branch's sources in memory.

    ``full`` = full-branch real (low-mem-streaming EHv5) + simulated low-mem-streaming rows.
    ``fast`` = fast-branch (QuickGenotype) real rows + simulated optimized-streaming fast rows.
    Built per-branch (not both at once), each source trimmed/downcast before concat, to bound RAM.
    """
    src = "real_full" if branch == "full" else "real_fast"
    parts = []
    for p in sorted(glob.glob(os.path.join(data_dir, src, "*.parquet"))):
        d = pd.read_parquet(p)
        if branch == "fast":
            d = d[d["genotyping_branch"] == "fast"]
        parts.append(_prep(d))
        del d
        gc.collect()
    sim = _load_sim(data_dir)
    if not sim.empty:
        sub = (sim[sim["analysis_mode"] == "low-mem-streaming"] if branch == "full"
               else sim[(sim["analysis_mode"] == "optimized-streaming")
                        & (sim["genotyping_branch"] == "fast")])
        parts.append(_prep(sub))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def report(branch, kept, drops):
    """Prints a per-branch build summary (counts, drops, direction + source slices)."""
    print("\n==================== %s branch ====================" % branch.upper())
    print("  input rows: %d -> kept %d" % (drops["_input"], drops["_kept"]))
    for k, v in drops.items():
        if not k.startswith("_"):
            print("    dropped %-26s %d" % (k, v))
    if kept.empty:
        print("  (no rows kept)")
        return
    print("  by source:", kept["source"].value_counts().to_dict())
    print("  by direction:", kept["direction"].value_counts().to_dict())
    if "chrom" in kept:
        print("  #chroms:", kept["chrom"].nunique())
    ms = pd.to_numeric(kept["motif_size"], errors="coerce")
    bins = {"1": (ms == 1), "2": (ms == 2), "3": (ms == 3), "4": (ms == 4),
            "5": (ms == 5), "6+": (ms >= 6)}
    print("  motif strata:", {k: int(v.sum()) for k, v in bins.items()})


def main():
    """Builds ``data/parquet/{full,fast}.parquet`` from the available source parquets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--branches", nargs="+", choices=["full", "fast"], default=["full", "fast"],
                        help="which branch parquet(s) to (re)build")
    parser.add_argument("--allow-partial", action="store_true",
                        help="build a branch from only the sources present (e.g. sim-only). By "
                             "default a branch whose REAL source parquets are missing is SKIPPED "
                             "so a stray rerun cannot overwrite an existing multi-million-row "
                             "branch parquet with a tiny sim-only frame")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "parquet")
    os.makedirs(out_dir, exist_ok=True)
    for branch in args.branches:
        src = "real_full" if branch == "full" else "real_fast"
        if not glob.glob(os.path.join(args.data_dir, src, "*.parquet")) and not args.allow_partial:
            print("\n%s branch: NO real source parquets in %s/ -- skipping to avoid clobbering "
                  "%s.parquet with a sim-only frame (pass --allow-partial to build anyway)"
                  % (branch, src, branch))
            continue
        df = assemble_branch(args.data_dir, branch)  # one branch's sources at a time
        if df.empty:
            print("\n%s branch: NO source rows found -- skipping" % branch)
            continue
        kept, drops = add_labels_and_filter(df)
        del df
        report(branch, kept, drops)
        # Write to a temp path then atomically replace, so a crash mid-write (or the sim-only
        # guard above) can never leave a truncated/partial branch parquet in place.
        out_path = os.path.join(out_dir, "%s.parquet" % branch)
        tmp_path = out_path + ".tmp"
        kept.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        print("  wrote %d rows -> %s" % (len(kept), out_path))
        del kept
        gc.collect()


if __name__ == "__main__":
    main()
