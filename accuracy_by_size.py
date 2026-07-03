"""Stacked-bar "tool accuracy by true allele size" plot -- a faithful matplotlib replica of
str-truth-set's ``figures_and_tables/plot_tool_accuracy_by_allele_size.py`` (the figure the
str-truth-set-v2 tool-comparison viewer shows).

Two panels share the x-axis ``True Allele Size - Number of Repeats in Reference`` (= the truth's
``DiffFromRefRepeats``), binned into 27 size bins:
  - left  : stacked allele COUNTS ("Number of Alleles"),
  - right : stacked FRACTIONS ("Fraction of Alleles", to 1.0) with per-bin allele counts on top.

Each allele is colored by how the genotyped call compares to truth (``DiffRepeats = call - truth``):
No Call / Called Hom Ref / Called Het Ref / Wrong Direction, then the signed magnitude bands
(-21 or more ... -2, Same, 2 ... 21 or more). "Same" = within +/-1 repeat (widening for long
alleles), so the green "exactly right" count matches str-truth-set's published numbers.

The category boundaries, colors, x-bins, override precedence, and "exactly right" numerator are
transcribed verbatim from the v1 script (verified byte-for-byte against the example SVG). Pure
functions; no global state. Coding rules: no type hints, Google docstrings, ``print()``.
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Stacked categories in ascending hue order (= legend top->bottom). Verbatim from the v1 script's
# hue_sorter ordering; the colors are the exact hex codes it emits (fixed specials + Blues_r(5)/
# Oranges_r(5) ramps for the 5 negative / 5 positive magnitude bands).
CATEGORIES = ("No Call", "Called Hom Ref", "Called Het Ref", "Wrong Direction",
              "-21 or more", "-8 to -20", "-5 to -7", "-3 to -4", "-2", "Same",
              "2", "3 to 4", "5 to 7", "8 to 20", "21 or more")
CATEGORY_COLORS = {
    "No Call": "#C5C5C5", "Called Hom Ref": "#C9342A", "Called Het Ref": "#734B4B",
    "Wrong Direction": "#99FFEF",
    "-21 or more": "#105BA4", "-8 to -20": "#3787C0", "-5 to -7": "#6CAED6",
    "-3 to -4": "#ABD0E6", "-2": "#D6E6F4", "Same": "#50AA44",
    "2": "#FEDFC0", "3 to 4": "#FDB97D", "5 to 7": "#FD8E3D", "8 to 20": "#E95E0D",
    "21 or more": "#B63C02"}

# x-axis bins on (true - ref) repeats. Upper-inclusive integer edge of bins 0..25; bin 26 is the
# open ">= +31" tail and bin 0 the open "<= -31" tail (bin_size=2 pairing, special wide >=21 bins).
_X_UPPER = np.array([-31, -26, -21, -19, -17, -15, -13, -11, -9, -7, -5, -3, -1, 0,
                     2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30], dtype=float)
X_LABELS = ("-31 or more", "-26 to -30", "-21 to -25", "-19 to -20", "-17 to -18", "-15 to -16",
            "-13 to -14", "-11 to -12", "-9 to -10", "-7 to -8", "-5 to -6", "-3 to -4", "-1 to -2",
            "0", "+1 to +2", "+3 to +4", "+5 to +6", "+7 to +8", "+9 to +10", "+11 to +12",
            "+13 to +14", "+15 to +16", "+17 to +18", "+19 to +20", "+21 to +25", "+26 to +30",
            "+31 or more")


def xbin(true_minus_ref):
    """Maps each (true - ref) value to a size-bin index in ``[0, len(X_LABELS))`` (rounded to int)."""
    return np.clip(np.searchsorted(_X_UPPER, np.round(np.asarray(true_minus_ref, dtype=float)),
                                   side="left"), 0, len(X_LABELS) - 1)


def classify(d, n, drr_truth, drr_tool, is_ref_allele, is_hom_ref):
    """Returns the per-allele category (one of ``CATEGORIES``) for each allele.

    Mirrors the v1 script exactly. ``d = call - truth`` (NaN => No Call); ``n`` = called NumRepeats
    (for the long-allele "Same" tolerance widening); ``drr_truth``/``drr_tool`` = each side's
    DiffFromRef (for Wrong Direction); ``is_ref_allele``/``is_hom_ref`` = the tool's ref-call flags.
    Override precedence (last wins): magnitude band -> Wrong Direction -> Het Ref -> Hom Ref; No Call
    (NaN d) survives because the override conditions require non-NaN tool columns.
    """
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    drr_truth = np.asarray(drr_truth, dtype=float)
    drr_tool = np.asarray(drr_tool, dtype=float)
    is_ref_allele = np.asarray(is_ref_allele, dtype=bool)
    is_hom_ref = np.asarray(is_hom_ref, dtype=bool)

    nocall = np.isnan(d)
    di = np.round(np.where(nocall, 0.0, d))
    a = np.abs(di)
    neg = di < -1
    same = (a <= 1) | ((n > 120) & (a <= 2)) | ((n > 240) & (a <= 3)) | ((n > 360) & (a <= 4))

    lab = np.full(d.shape, "Same", dtype=object)
    for mask, pos_label, neg_label in (
            (a == 2, "2", "-2"),
            ((a >= 3) & (a <= 4), "3 to 4", "-3 to -4"),
            ((a >= 5) & (a <= 7), "5 to 7", "-5 to -7"),
            ((a >= 8) & (a <= 20), "8 to 20", "-8 to -20"),
            (a >= 21, "21 or more", "-21 or more")):
        m = mask & ~same
        lab[m & ~neg] = pos_label
        lab[m & neg] = neg_label

    wrong_dir = ((drr_truth != 0) & (drr_tool != 0) & ~np.isnan(drr_truth) & ~np.isnan(drr_tool)
                 & (np.sign(drr_truth) != np.sign(drr_tool)))
    lab[wrong_dir] = "Wrong Direction"
    lab[(drr_truth != 0) & is_ref_allele] = "Called Het Ref"
    lab[(drr_truth != 0) & is_hom_ref] = "Called Hom Ref"
    lab[nocall] = "No Call"
    return lab


# Column names in the `*.for_comparison.with_<tool>_vs_Truth_columns.alleles.tsv.gz` table.
def _cols(tool):
    return {
        "d": "DiffRepeats: Allele: %s - Truth" % tool,
        "n": "NumRepeats: Allele: %s" % tool,
        "drr_truth": "DiffFromRefRepeats: Allele: Truth",
        "drr_tool": "DiffFromRefRepeats: Allele: %s" % tool,
        "is_ref": "IsRef: Allele: %s" % tool,
        "is_hom_ref": "IsHomRef: %s" % tool,
        "motif": "MotifSize",
        "locus": "LocusId",
        "true_repeats": "NumRepeats: Allele: Truth",
        "purity": "RepeatPurity: Allele: Truth",
    }


def _as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def categorize_tsv(tsv_path, tool):
    """Loads a for-comparison alleles TSV and returns a frame with ``category``, ``xbin``, ``motif``.

    Returns one row per allele with the raw (uncorrected) category, the (true-ref) size-bin index,
    the motif size, ``locus``, and the raw inputs needed to recompute the category after LCF
    correction (``d``, ``n``, ``drr_truth``, ``drr_tool``, ``is_ref``, ``is_hom_ref``,
    ``true_repeats``).
    """
    c = _cols(tool)
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False, dtype={c["locus"]: str},
                     usecols=list(c.values()))
    out = pd.DataFrame({
        "locus": df[c["locus"]].astype(str).str.replace(r"^chr", "", regex=True),
        "motif": pd.to_numeric(df[c["motif"]], errors="coerce"),
        "d": pd.to_numeric(df[c["d"]], errors="coerce"),
        "n": pd.to_numeric(df[c["n"]], errors="coerce"),
        "drr_truth": pd.to_numeric(df[c["drr_truth"]], errors="coerce"),
        "drr_tool": pd.to_numeric(df[c["drr_tool"]], errors="coerce"),
        "is_ref": _as_bool(df[c["is_ref"]]),
        "is_hom_ref": _as_bool(df[c["is_hom_ref"]]),
        "true_repeats": pd.to_numeric(df[c["true_repeats"]], errors="coerce"),
        "purity": pd.to_numeric(df[c["purity"]], errors="coerce"),
    })
    out["category"] = classify(out["d"], out["n"], out["drr_truth"], out["drr_tool"],
                               out["is_ref"], out["is_hom_ref"])
    out["xbin"] = xbin(out["drr_truth"])
    return out


def assign_allele_rank(cat):
    """Adds a per-locus ``allele_rank`` (size-sorted by called repeats then truth) matching the
    parquet's rank convention, so model predictions can be joined back onto the TSV rows."""
    cat = cat.sort_values(["locus", "n"], kind="mergesort")
    cat = cat.assign(allele_rank=cat.groupby("locus", sort=False).cumcount())
    return cat.sort_index()


# LCF-correction variants shown on the report's "LCF correction" pill. Each applies
# corrected = round(eh / LCF) only to the alleles passing its gate; "raw" applies no correction.
# (key, pill label, title note, gate). A gate's "pok" is the pOk upper bound; "nonspanning" restricts
# it to full_nonspanning-regime alleles. Gating is on pOk alone (no LCF-magnitude guard), matching the
# eval-side LCF correction.
CORRECTION_VARIANTS = (
    ("raw", "Raw EH", "", None),
    ("p050", "LCF-corrected (p < 0.5)", " — LCF-corrected (pOk < 0.5)", {"pok": 0.5}),
    ("p025", "LCF-corrected (p < 0.25)", " — LCF-corrected (pOk < 0.25)", {"pok": 0.25}),
    ("p050ns", "LCF-corrected (p < 0.5 and non-spanning)",
     " — LCF-corrected (pOk < 0.5, non-spanning only)", {"pok": 0.5, "nonspanning": True}),
)

# Repeat-purity filter for the accuracy-by-size pill. "off" counts every allele; the filtered variant
# keeps only alleles whose truth repeat purity exceeds the threshold (purity is on a 0-1 scale, so 0.95
# = 95% pure). (key, pill label, min purity or None).
PURITY_VARIANTS = (
    ("off", "Off", None),
    ("p95", "> 0.95 pure", 0.95),
)


def _gate_mask(lcf, pok, non_spanning, gate):
    """Boolean mask of the alleles a correction-variant ``gate`` applies to: a valid predicted LCF and
    pOk below the gate's threshold (the non-spanning gate also requires the full_nonspanning regime)."""
    g = (pok < gate["pok"]) & (lcf > 0) & ~np.isnan(lcf) & ~np.isnan(pok)
    if gate.get("nonspanning"):
        g = g & non_spanning
    return g


def _corrected_category(raw_call, corrected_call, gated, ref, true_round, drr_truth, locus):
    """Recomputes the category for the LCF-corrected scenario of one gate.

    The effective call is the corrected call where the gate applies, else the raw call. Crucially
    ``is_hom_ref`` is recomputed per-locus from those effective calls -- so an allele whose corrected
    call leaves the reference no longer counts as a reference call (and its locus is no longer
    homozygous-reference), instead of being frozen in the raw "Called Hom Ref" band. Returns the
    category for every row; callers keep it only for the gated rows.
    """
    eff = np.where(gated, corrected_call, raw_call)
    is_ref = np.round(eff - ref) == 0
    is_hom_ref = pd.Series(is_ref).groupby(locus).transform("all").to_numpy(dtype=bool)
    return classify(np.round(eff - true_round), eff, drr_truth, np.round(eff - ref), is_ref, is_hom_ref)


def predict_lcf_pok(parquet_paths, model_path, cap=None):
    """Applies the deployed model to the raw per-allele parquet(s); returns one row per called allele
    with ``locus`` (chr-stripped), ``allele_rank``, ``lcf``, ``pok`` and ``non_spanning`` (the
    full_nonspanning genotyping regime) -- the inputs the gated LCF correction needs. No fitting; the
    model is loaded from its serialized ``.json[.gz]``. ``cap`` seed-subsamples each parquet to bound
    the apply (``None`` = all alleles).
    """
    import model as M
    import features
    mj = M.load(model_path)["genotyping_regimes"]
    comp = {r: (M.compile_genotyping_regime(mj[r]), features.GENOTYPING_REGIME_BRANCH[r])
            for r in features.GENOTYPING_REGIMES}
    out = []
    for p in parquet_paths:
        df = pd.read_parquet(p)
        if cap and len(df) > cap:
            df = df.sample(cap, random_state=20260616).reset_index(drop=True)
        df = df.assign(_regime=features.genotyping_regime_of(
            df["genotyping_branch"].to_numpy(),
            pd.to_numeric(df["spanning_at_called"], errors="coerce").to_numpy()))
        for r, (cg, branch) in comp.items():
            sub = df[df["_regime"] == r]
            if sub.empty:
                continue
            X, _ = features.build_matrix(sub, branch)
            out.append(pd.DataFrame({
                "locus": sub["locus_id"].astype(str).str.replace(r"^chr", "", regex=True).to_numpy(),
                "allele_rank": sub["allele_rank"].to_numpy(),
                "lcf": M.predict_lcf_json(cg, X), "pok": M.predict_proba_json(cg, X)[:, 0],
                "non_spanning": r == features.GENOTYPING_REGIME_FULL_NONSPANNING}))
    cols = ["locus", "allele_rank", "lcf", "pok", "non_spanning"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def add_corrected_categories(cat, preds):
    """Returns ``cat`` joined with ``preds`` plus one ``category__<key>`` column per gated entry in
    ``CORRECTION_VARIANTS``: the category after applying corrected = ``round(eh / LCF)`` only to the
    alleles passing that variant's gate; every other allele keeps its raw category. Joins ``preds``
    (from ``predict_lcf_pok``) by ``(locus, allele_rank)``. The corrected call + its category are the
    same for any gated allele across variants -- only the gate (which alleles get it) differs.
    """
    m = assign_allele_rank(cat).merge(preds, on=["locus", "allele_rank"], how="left")
    n = pd.to_numeric(m["n"], errors="coerce").to_numpy(dtype=float)
    lcf = pd.to_numeric(m["lcf"], errors="coerce").to_numpy(dtype=float)
    pok = pd.to_numeric(m["pok"], errors="coerce").to_numpy(dtype=float)
    non_spanning = (m["non_spanning"] == True).to_numpy(dtype=bool)  # NaN (unmatched join) -> False
    true_r = m["true_repeats"].to_numpy(dtype=float)
    drr_truth = m["drr_truth"].to_numpy(dtype=float)
    corrected = np.round(np.where(lcf > 0, n / np.where(lcf > 0, lcf, np.nan), n))
    ref = np.round(true_r - drr_truth)
    locus = m["locus"].to_numpy()
    raw = m["category"].to_numpy(dtype=object)
    # Alleles with no prediction (pok NaN) -- capped out of ``predict_lcf_pok`` or unmatched by the
    # join -- are marked None so bin_counts drops them from the corrected panels; otherwise they would
    # sit there as raw EH and dilute the LCF-corrected accuracy. The raw "category" column is untouched.
    applied = ~np.isnan(pok)
    for key, _, _, gate in CORRECTION_VARIANTS:
        if gate is None:
            continue
        g = _gate_mask(lcf, pok, non_spanning, gate)
        cat_g = _corrected_category(n, corrected, g, ref, np.round(true_r), drr_truth, locus)
        res = raw.copy()
        res[~applied] = None
        res[g] = cat_g[g]
        m["category__" + key] = res
    return m


def categorize_parquet(parquet_path, model_path, corrected_cap=None):
    """Categorizes the called alleles of a per-allele parquet (raw + each LCF-correction variant)
    WITHOUT the TSV.

    Used for the held-out pool, whose for-comparison TSVs aren't cached and whose catalog/coverage
    vary per sample. Computes every category except "No Call" (the parquet has only called alleles;
    no-call is a tiny fraction for WGS) from the parquet's own ``eh`` / ``true`` /
    ``num_repeats_in_reference`` columns, applies the deployed model inline (row-aligned, so no join)
    -- capped at ``corrected_cap`` per parquet for the corrected variants -- and returns a frame with
    ``category``, one ``category__<key>`` per ``CORRECTION_VARIANTS`` gate, ``xbin``, ``motif``,
    ``locus``.
    """
    import model as M
    import features
    df = pd.read_parquet(parquet_path)
    df = df[pd.to_numeric(df["true"], errors="coerce").notna()].reset_index(drop=True)
    eh = pd.to_numeric(df["eh"], errors="coerce").to_numpy(dtype=float)
    true = pd.to_numeric(df["true"], errors="coerce").to_numpy(dtype=float)
    nref = pd.to_numeric(df["num_repeats_in_reference"], errors="coerce").to_numpy(dtype=float)
    motif = pd.to_numeric(df["motif_size"], errors="coerce").to_numpy(dtype=float)
    purity = pd.to_numeric(df["purity"], errors="coerce").to_numpy(dtype=float)
    locus = df["locus_id"].astype(str).str.replace(r"^chr", "", regex=True).to_numpy()
    # Round each single difference ONCE (matches the str-truth-set convention + classify/xbin, which
    # round the difference, not each operand -- they differ at half-integer reference repeats).
    drr_truth = np.round(true - nref)
    drr_tool = np.round(eh - nref)
    is_ref = drr_tool == 0
    is_hom_ref = (is_ref & pd.DataFrame({"locus": locus, "is_ref": is_ref})
                  .groupby("locus")["is_ref"].transform("all").to_numpy())
    cat = classify(np.round(eh - true), np.round(eh), drr_truth, drr_tool, is_ref, is_hom_ref)

    lcf = np.full(len(df), np.nan)
    pok = np.full(len(df), np.nan)
    mj = M.load(model_path)["genotyping_regimes"]
    comp = {r: (M.compile_genotyping_regime(mj[r]), features.GENOTYPING_REGIME_BRANCH[r])
            for r in features.GENOTYPING_REGIMES}
    sel = np.arange(len(df))
    if corrected_cap and len(df) > corrected_cap:
        sel = np.sort(np.random.default_rng(20260616).choice(sel, corrected_cap, replace=False))
    sub = df.iloc[sel].assign(_row=sel, _regime=features.genotyping_regime_of(
        df.iloc[sel]["genotyping_branch"].to_numpy(),
        pd.to_numeric(df.iloc[sel]["spanning_at_called"], errors="coerce").to_numpy()))
    for r, (cg, branch) in comp.items():
        ss = sub[sub["_regime"] == r]
        if ss.empty:
            continue
        X, _ = features.build_matrix(ss, branch)
        rows = ss["_row"].to_numpy()
        lcf[rows] = M.predict_lcf_json(cg, X)
        pok[rows] = M.predict_proba_json(cg, X)[:, 0]
    non_spanning = np.zeros(len(df), dtype=bool)
    non_spanning[sub["_row"].to_numpy()] = (
        sub["_regime"].to_numpy() == features.GENOTYPING_REGIME_FULL_NONSPANNING)
    corrected = np.round(np.where(lcf > 0, eh / np.where(lcf > 0, lcf, np.nan), eh))
    out = {"category": cat, "xbin": xbin(drr_truth), "motif": motif, "locus": locus, "purity": purity}
    # When corrected_cap < len(df) the model is applied only to the sampled rows; pok stays NaN for the
    # rest. Marking those None (so bin_counts drops them from the corrected panels) keeps the
    # LCF-corrected accuracy on the alleles actually scored, instead of diluting it with raw rows that
    # could never have been corrected. The raw "category" column keeps every allele.
    applied = ~np.isnan(pok)
    for key, _, _, gate in CORRECTION_VARIANTS:
        if gate is None:
            continue
        g = _gate_mask(lcf, pok, non_spanning, gate)
        cat_g = _corrected_category(eh, corrected, g, nref, true, drr_truth, locus)
        c = cat.copy()
        c[~applied] = None
        c[g] = cat_g[g]
        out["category__" + key] = c
    return pd.DataFrame(out)


def bin_counts(cat, category_col, homopolymer, purity_min=None):
    """Tallies a per-category x-bin count matrix for the homopolymer / non-homopolymer subset.

    ``purity_min`` (when not None) additionally keeps only alleles whose truth repeat purity strictly
    exceeds it (NaN purity is dropped). Rows whose ``category_col`` is None are dropped too: a
    corrected variant marks the alleles the model was NOT applied to (capped out / unmatched) as None,
    so they are excluded from the corrected panel's numerator AND denominator rather than silently
    counted as raw (the raw ``category`` column is never None, so the raw panel keeps every allele).
    Returns a dict with ``counts`` (category -> list of ``len(X_LABELS)`` ints), ``alleles_per_bin``
    (total alleles per x-bin), ``same`` / ``total`` scalars and ``loci`` (the set of distinct locus
    ids kept) for the title.
    """
    sel = cat[cat["motif"] == 1] if homopolymer else cat[cat["motif"] > 1]
    if purity_min is not None:
        sel = sel[pd.to_numeric(sel["purity"], errors="coerce") > purity_min]
    sel = sel[sel[category_col].notna()]  # drop alleles the model wasn't applied to (corrected variants)
    nb = len(X_LABELS)
    xb = sel["xbin"].to_numpy()
    cc = sel[category_col].to_numpy()
    counts = {}
    for c in CATEGORIES:
        mask = cc == c
        counts[c] = (np.bincount(xb[mask], minlength=nb)[:nb] if mask.any()
                     else np.zeros(nb, dtype=int)).tolist()
    return {"counts": counts,
            "alleles_per_bin": np.bincount(xb, minlength=nb)[:nb].tolist(),
            "same": int((cc == "Same").sum()), "total": int(len(sel)),
            "loci": set(sel["locus"].to_numpy().tolist())}


# Tool display labels (from the v1 script's TITLE_TOOL_LABELS).
TITLE_TOOL_LABELS = {"EHv5-bw2-optimized": "bw2/EHv5 (optimized-streaming)"}


def plot_accuracy_by_size(data, out_png, tool_label, coverage_label, motif_desc, lcf_note=""):
    """Draws the two-panel stacked-bar accuracy plot from a ``bin_counts`` dict (one motif set, one
    correction state). Left = stacked counts, right = stacked fractions + per-bin allele counts.
    """
    nb = len(X_LABELS)
    x = np.arange(nb)
    counts = {c: np.asarray(data["counts"][c], dtype=float) for c in CATEGORIES}
    col_tot = np.sum([counts[c] for c in CATEGORIES], axis=0)
    stack_order = list(reversed(CATEGORIES))  # bottom -> top

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"wspace": 0.18})
    # Left: stacked counts.
    bottom = np.zeros(nb)
    for c in stack_order:
        ax_l.bar(x, counts[c], 0.9, bottom=bottom, color=CATEGORY_COLORS[c],
                 edgecolor="black", linewidth=0.2)
        bottom += counts[c]
    ax_l.set_ylabel("Number of Alleles", fontsize=12)
    # Right: stacked fractions.
    frac = {c: np.divide(counts[c], col_tot, out=np.zeros(nb), where=col_tot > 0) for c in CATEGORIES}
    bottom = np.zeros(nb)
    for c in stack_order:
        ax_r.bar(x, frac[c], 0.9, bottom=bottom, color=CATEGORY_COLORS[c],
                 edgecolor="black", linewidth=0.2)
        bottom += frac[c]
    ax_r.set_ylabel("Fraction of Alleles", fontsize=12)
    ax_r.set_ylim(0, 1.0)
    ax_r.text(0.0, 1.15, "Alleles Per Bin", transform=ax_r.transAxes, fontsize=10, color="#777777")
    for xi, n in zip(x, data["alleles_per_bin"]):
        if n:
            ax_r.text(xi, 1.018, "{:,}".format(int(n)), ha="center", va="bottom", rotation=45,
                      fontsize=7, color="#777777")
    for ax in (ax_l, ax_r):
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, rotation=90, fontsize=8)
        ax.set_xlabel("True Allele Size Minus Number of Repeats in Reference Genome", fontsize=12)
        ax.set_xlim(-0.6, nb - 0.4)
    # Legend (top -> bottom = CATEGORIES order), placed between the panels' titles.
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CATEGORY_COLORS[c], ec="black", lw=0.2) for c in CATEGORIES]
    ax_r.legend(handles, CATEGORIES, title="%s Call\nvs\nTrue Allele Size" % tool_label,
                fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False)
    pct = 100.0 * data["same"] / data["total"] if data["total"] else float("nan")
    # Reserve the top ~28% of the figure for the title so its lines clear the rotated per-bin allele
    # counts above the right panel; the title is split across 3 lines so the long LCF-correction note
    # fits without widening the figure further.
    fig.subplots_adjust(top=0.72)
    fig.suptitle("%s got %s out of %s alleles (%.1f%%) exactly right\n"
                 "in %s%s\n"
                 "Showing results for %s loci (%s)"
                 % (tool_label, "{:,}".format(data["same"]), "{:,}".format(data["total"]), pct,
                    coverage_label, lcf_note, "{:,}".format(data["total_loci"]), motif_desc),
                 fontsize=13, y=0.995, va="top")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Self-test: reproduce the published HG002 31x 2-6bp numbers (309,621 / 342,406 = 90.4% "Same").
    import sys
    tsv = sys.argv[1] if len(sys.argv) > 1 else (
        "data/real_quick/_downloads/HG002_31x/"
        "HG002.tandem_repeat_genotypes.for_comparison.with_EHv5-bw2-optimized_vs_Truth_columns."
        "alleles.tsv.gz")
    cat = categorize_tsv(tsv, "EHv5-bw2-optimized")
    sub = cat[(cat["motif"] >= 2) & (cat["motif"] <= 6)]
    same = int((sub["category"] == "Same").sum())
    print("2-6bp alleles: %d  (published 342,406)" % len(sub))
    print("Same / exactly-right: %d  (published 309,621)  => %.1f%%  (published 90.4%%)"
          % (same, 100.0 * same / len(sub)))
    print("loci: %d  (published 171,200)" % sub["locus"].nunique())
    print("\ncategory counts:")
    print(sub["category"].value_counts().reindex(CATEGORIES).fillna(0).astype(int).to_string())
