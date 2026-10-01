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


# LCF-correction variants shown on the report's "LCF correction" pill. Each applies
# corrected = round(eh / LCF) only to the alleles passing its gate; "raw" applies no correction.
# (key, pill label, title note, gate). A gate's "pok" is the pOk upper bound; "nonspanning" restricts
# it to full_nonspanning-regime alleles. Gating is on pOk alone (no LCF-magnitude guard), matching the
# eval-side LCF correction.
CORRECTION_VARIANTS = (
    ("raw", "Raw EH", "", None),
    ("p050", "LCF-corrected (pOk < 0.5)", " — LCF-corrected (pOk < 0.5)", {"pok": 0.5}),
    ("p025", "LCF-corrected (pOk < 0.25)", " — LCF-corrected (pOk < 0.25)", {"pok": 0.25}),
    ("p050ns", "LCF-corrected (pOk < 0.5 and non-spanning)",
     " — LCF-corrected (pOk < 0.5, non-spanning only)", {"pok": 0.5, "nonspanning": True}),
)

# Repeat-purity filter for the accuracy-by-size pill. "off" counts every allele; the filtered variant
# keeps only alleles whose truth repeat purity exceeds the threshold (purity is on a 0-1 scale, so 0.95
# = 95% pure). (key, pill label, min purity or None).
PURITY_VARIANTS = (
    ("off", "Off", None),
    ("p95", "> 0.95 pure", 0.95),
)

# pOk-stratum filter for the accuracy-by-size pill: splits alleles by the model's own predicted
# confidence, independent of which LCF-correction variant is selected (so e.g. "Raw EH, pOk < 0.5" shows
# how the uncorrected calls look specifically where the model would consider applying a correction).
# (key, pill label, mode) where mode is None (no filter) / "lt" (pOk < 0.5) / "ge" (pOk >= 0.5).
POK_VARIANTS = (
    ("all", "All", None),
    ("lt050", "pOk < 0.5", "lt"),
    ("ge050", "pOk ≥ 0.5", "ge"),
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


def _share_predictions_within_locus(df, scoreable, called, lcf, pok, non_spanning):
    """Copies each scored allele's prediction onto the duplicate genotype copy of the same call.

    ExpansionHunter makes ONE prediction for a homozygous call, but this report keeps both genotype
    copies (its truth join pairs them with the short and long truth alleles). Only the copy carrying
    its own quality metrics is scored above; this hands the other copy that same prediction, so both
    are categorized under the one correction the deployed binary would actually apply.

    Rows are matched within a ``(locus_id, eh)`` group, which is exactly the set of genotype copies of
    one homozygous call. A parquet predating ``has_own_quality_metrics`` has every row marked
    scoreable, so nothing is shared and the behaviour is unchanged.

    Returns:
        The updated ``(lcf, pok, non_spanning)`` arrays.
    """
    unscored = called & ~scoreable & np.isnan(pok)
    if not unscored.any() or "locus_id" not in df.columns:
        return lcf, pok, non_spanning
    key = pd.MultiIndex.from_arrays([df["locus_id"].to_numpy(),
                                     pd.to_numeric(df["eh"], errors="coerce").to_numpy()])
    scored = ~np.isnan(pok)
    source = pd.DataFrame({"lcf": lcf[scored], "pok": pok[scored],
                           "non_spanning": non_spanning[scored]}, index=key[scored])
    source = source[~source.index.duplicated(keep="first")]
    filled = source.reindex(key[unscored])
    have = filled["pok"].notna().to_numpy()
    target = np.where(unscored)[0][have]
    lcf[target] = filled["lcf"].to_numpy()[have]
    pok[target] = filled["pok"].to_numpy()[have]
    non_spanning[target] = filled["non_spanning"].to_numpy()[have].astype(bool)
    return lcf, pok, non_spanning


def categorize_parquet(parquet_path, model_path, corrected_cap=None):
    """Categorizes the alleles of a per-allele parquet (raw + each LCF-correction variant).

    Rebuilds the str-truth-set "for_comparison" categorization directly from the parquet -- the EH
    calls come from the JSON (already flattened into ``eh`` by ``eh_json``) and the truth from the
    truth-genotypes TSV (already joined into ``true`` / ``num_repeats_in_reference`` / ``purity`` by
    ``dataset.build_combo``) -- so no precomputed comparison TSV is needed. This is the single
    categorization path for every report dataset.

    Rows whose ``eh`` is null are the no-call alleles ``eh_json`` emits for loci EH left uncalled:
    ``classify`` scores them "No Call" (they appear in the raw panel), and because the deployed model
    is applied only to the finite-``eh`` (called) rows below, their ``pok`` stays NaN so they are
    excluded from the LCF-corrected panels (a no-call can't be corrected) -- matching the old TSV path.

    Applies the deployed model inline (row-aligned, so no join) -- capped at ``corrected_cap`` called
    alleles per parquet for the corrected variants -- and returns a frame with ``category``, one
    ``category__<key>`` per ``CORRECTION_VARIANTS`` gate, ``xbin``, ``motif``, ``locus``, ``purity``,
    ``pok``.
    """
    import model as M
    import features
    df = pd.read_parquet(parquet_path)
    # ANALYSIS_OK[imputation]: rows with unparseable truth are dropped by the notna() filter below.
    df = df[pd.to_numeric(df["true"], errors="coerce").notna()].reset_index(drop=True)
    eh = pd.to_numeric(df["eh"], errors="coerce").to_numpy(dtype=float)
    true = pd.to_numeric(df["true"], errors="coerce").to_numpy(dtype=float)
    nref = pd.to_numeric(df["num_repeats_in_reference"], errors="coerce").to_numpy(dtype=float)
    # ANALYSIS_OK[imputation]: remaining eh/nref/motif/purity NaNs propagate into classify()/xbin(),
    # which are NaN-aware by design (see their docstrings). A null eh (no-call row) -> classify "No Call".
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
    model = M.load(model_path)
    # Each regime's matrix is built from the MODEL's own declared feature list: the compiled trees
    # index it positionally, so that list in that order is the only correct matrix for this model
    # (and an older model stays applicable rather than being refused). See model.feature_names_of.
    declared = M.feature_names_of(model, model_path)
    mj = model["genotyping_regimes"]
    comp = {r: (M.compile_genotyping_regime(mj[r]),
                features.GENOTYPING_REGIME_BRANCH[r],
                declared[features.GENOTYPING_REGIME_BRANCH[r]])
            for r in features.GENOTYPING_REGIMES}
    # Only called alleles (finite eh) can be LCF-corrected; no-call rows keep lcf/pok NaN so they drop
    # out of the corrected panels (see docstring). Cap the called-allele apply for speed.
    #
    # ExpansionHunter emits one AlleleQualityMetrics entry -- and therefore ONE prediction -- per
    # allele it scores, and only one for a homozygous call. eh_json still emits both genotype copies
    # (this report's truth join needs them), so scoring the rank-1 copy independently would invent a
    # prediction the deployed binary never makes: on real data the two copies of a hom call get
    # different rounded LCF/pOk for the large majority of pairs, because allele_rank itself is a
    # feature. Score only the rows EH scores, then give each duplicate its twin's prediction, so the
    # corrected panels describe one correction per scored allele exactly as deployment would.
    scoreable = (df["has_own_quality_metrics"].fillna(True).astype(bool).to_numpy()
                 if "has_own_quality_metrics" in df.columns
                 else np.ones(len(df), dtype=bool))
    sel = np.where(np.isfinite(eh) & scoreable)[0]
    if corrected_cap and sel.size > corrected_cap:
        sel = np.sort(np.random.default_rng(20260616).choice(sel, corrected_cap, replace=False))
    # ANALYSIS_OK[imputation]: NaN spanning_at_called -> 0 is genotyping_regime_of's documented
    # default (features.py), routing such alleles to full_nonspanning.
    sub = df.iloc[sel].assign(_row=sel, _regime=features.genotyping_regime_of(
        df.iloc[sel]["genotyping_branch"].to_numpy(),
        pd.to_numeric(df.iloc[sel]["spanning_at_called"], errors="coerce").to_numpy()))
    for r, (cg, branch, names) in comp.items():
        ss = sub[sub["_regime"] == r]
        if ss.empty:
            continue
        X, _ = features.build_matrix(ss, branch, names)
        rows = ss["_row"].to_numpy()
        # Rounded to the 3 decimals ExpansionHunter emits, so the categories describe the deployed
        # binary's output rather than full-precision predictions (model.round_like_emitted).
        lcf[rows] = M.round_like_emitted(M.predict_lcf_json(cg, X))
        pok[rows] = M.round_like_emitted(M.predict_proba_json(cg, X)[:, 0])
    non_spanning = np.zeros(len(df), dtype=bool)
    non_spanning[sub["_row"].to_numpy()] = (
        sub["_regime"].to_numpy() == features.GENOTYPING_REGIME_FULL_NONSPANNING)
    lcf, pok, non_spanning = _share_predictions_within_locus(
        df, scoreable, np.isfinite(eh), lcf, pok, non_spanning)
    corrected = np.round(np.where(lcf > 0, eh / np.where(lcf > 0, lcf, np.nan), eh))
    out = {"category": cat, "xbin": xbin(drr_truth), "motif": motif, "locus": locus, "purity": purity,
           "pok": pok}
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


def bin_counts(cat, category_col, homopolymer, purity_min=None, pok_stratum=None):
    """Tallies a per-category x-bin count matrix for the homopolymer / non-homopolymer subset.

    ``purity_min`` (when not None) additionally keeps only alleles whose truth repeat purity strictly
    exceeds it (NaN purity is dropped). ``pok_stratum`` (when not None) additionally keeps only alleles
    whose predicted ``pok`` is ``< 0.5`` (``"lt"``) or ``>= 0.5`` (``"ge"``); NaN pok (model not applied
    to that allele) is dropped. Rows whose ``category_col`` is None are dropped too: a corrected variant
    marks the alleles the model was NOT applied to (capped out / unmatched) as None, so they are
    excluded from the corrected panel's numerator AND denominator rather than silently counted as raw
    (the raw ``category`` column is never None, so the raw panel keeps every allele).
    Returns a dict with ``counts`` (category -> list of ``len(X_LABELS)`` ints), ``alleles_per_bin``
    (total alleles per x-bin), ``same`` / ``total`` scalars and ``loci`` (the set of distinct locus
    ids kept) for the title.
    """
    sel = cat[cat["motif"] == 1] if homopolymer else cat[cat["motif"] > 1]
    # ANALYSIS_OK[imputation]: NaN purity/pok dropped below, per this function's docstring above.
    if purity_min is not None:
        sel = sel[pd.to_numeric(sel["purity"], errors="coerce") > purity_min]
    if pok_stratum is not None:
        pok = pd.to_numeric(sel["pok"], errors="coerce")
        sel = sel[(pok < 0.5) if pok_stratum == "lt" else (pok >= 0.5)]
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

    ``data`` may carry the distinct-locus count either as a precomputed ``total_loci`` int (what
    ``gen_datasets`` accumulates across samples) or only as the raw ``loci`` set ``bin_counts``
    returns; the title falls back to ``len(data["loci"])`` so a raw ``bin_counts`` dict plots directly.
    """
    total_loci = data.get("total_loci", len(data.get("loci", ())))
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
                    coverage_label, lcf_note, "{:,}".format(total_loci), motif_desc),
                 fontsize=13, y=0.995, va="top")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
