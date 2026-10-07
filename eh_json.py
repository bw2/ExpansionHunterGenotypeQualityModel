"""Extract per-allele feature rows from an ExpansionHunter ``.json`` output.

This is the single source of truth for turning a raw EH JSON (plain or gzipped)
into tidy per-allele rows. Every data source feeds through it so features are
computed identically across coverages, samples, and the two genotyping branches.

A variant's contract is decided per-variant by ``QuickGenotype``:

- ``full`` (``QuickGenotype`` absent/false) -- the full genotyper; carries the
  complete ``AlleleQualityMetrics`` including ``QD`` and the flank-normalized
  depths.
- ``quick`` (``QuickGenotype == true``) -- the heuristic fast path
  (``processLocusFast``); emits the same metrics as approximations but omits
  ``QD`` and the flank-normalized depths.

The allele call ``eh`` always comes from the JSON ``Genotype`` field, never from
any precomputed TSV column. A no-call locus still emits rows (``eh`` None) so the
accuracy-by-size report can count the uncalled truth alleles as "No Call"; those
rows are dropped before training. Only the columns the deployable model and its
report actually consume are emitted (the audit/debug columns from earlier
iterations are intentionally dropped). Pure functions, no global state, no randomness.
"""

import gzip
import re

import ijson

# Genotyping-branch labels (shared with features.genotyping_regime_of). The "quick" branch is
# EH's QuickGenotype fast path (processLocusFast); "full" is the full genotyper.
BRANCH_FULL = "full"
BRANCH_QUICK = "quick"

_COUNTS_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")

# The raw ExpansionHunter output JSON fields this parser reads (for the report): per-locus fields,
# per-variant top-level fields and per-allele ``AlleleQualityMetrics.Alleles[].<field>`` values. Field
# names are leaf names. Source of truth for the field names is ``extract_variant_rows`` below.
EH_OUTPUT_FIELDS = [
    ("Coverage", "Per-locus read depth over the locus's two reference flanks."),
    ("CountsOfInrepeatReads", "Per-variant: reads lying entirely inside the repeat, by size (full genotyper only)."),
    ("ReferenceRepeatPurity", "Per-variant: fraction of the reference repeat region matching a perfect motif tiling."),
    ("Depth", "Per-allele read depth."),
    ("HighQualityUnambiguousReads", "Per-allele high-quality unambiguous read count."),
    ("StrandBiasBinomialPhred", "Strand-bias binomial Phred score."),
    ("MeanInsertedBasesWithinRepeats", "Mean inserted bases within the repeat."),
    ("MeanDeletedBasesWithinRepeats", "Mean deleted bases within the repeat."),
    ("ReadRepeatPurity", "Per-allele: pooled base-weighted fraction of in-repeat read bases matching the motif."),
    ("LeftFlankNormalizedDepth", "Left-flank-normalized depth (full genotyper only)."),
    ("RightFlankNormalizedDepth", "Right-flank-normalized depth (full genotyper only)."),
]


def _open_binary(path):
    """Opens ``path`` for binary reading (what ijson's C backend wants), transparently decompressing ``.gz``."""
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


LOCI_SCANNED_FOR_TYPICAL_READ_LENGTH = 10_000


def typical_read_length_in_file(path):
    """Estimates the read length EH skipped loci by: the largest per-locus ``ReadLength`` among the
    first ``LOCI_SCANNED_FOR_TYPICAL_READ_LENGTH`` loci, or None if none of them reports one.

    EH skips loci too wide for the sample's typical read length, which it takes as the longest of the
    first 1000 reads (probeTypicalReadLength in ExpansionHunter-bw2), but each locus's ``ReadLength`` in
    the JSON is the MEAN length of that locus's reads. A mean can only be at or below the longest read,
    and across thousands of loci some see only full-length reads, so the largest per-locus mean is the
    closest estimate the JSON offers. It can still fall a few bp short on samples with trimmed reads.
    """
    largest = None
    with _open_binary(path) as f:
        for i, (_, locus_result) in enumerate(ijson.kvitems(f, "LocusResults", use_float=True)):
            if i == LOCI_SCANNED_FOR_TYPICAL_READ_LENGTH:
                break
            if locus_result.get("ReadLength"):
                largest = max(largest or 0, int(locus_result["ReadLength"]))
    return largest


def _sample_id_in_file(path):
    """Returns ``SampleParameters.SampleId`` from an EH JSON file, or None.

    EH writes ``SampleParameters`` at the head of the file, so this reads only that far.
    """
    with _open_binary(path) as f:
        return next(ijson.items(f, "SampleParameters.SampleId"), None)


def parse_counts(s):
    """Parses an EH counts string like ``"(20, 8), (22, 1)"`` into ``[(20, 8), (22, 1)]``."""
    if not s:
        return []
    return [(int(a), int(b)) for a, b in _COUNTS_RE.findall(s)]


def parse_genotype(s):
    """Parses an EH ``Genotype`` string (e.g. ``"20/97"``) into a list of ints.

    Returns ``None`` for a no-call (``"./."``, ``"."``, empty) or a non-integer field.
    """
    if not s:
        return None
    parts = str(s).split("/")
    if not all(p.strip().lstrip("-").isdigit() for p in parts):
        return None
    return [int(p) for p in parts]


def parse_ci(s, n_alleles):
    """Parses ``GenotypeConfidenceInterval`` (e.g. ``"20-20/95-153"``) into per-allele (lo, hi).

    Returns ``n_alleles`` ``(lo, hi)`` tuples; missing/malformed entries are ``(None, None)``.
    """
    parts = str(s).split("/") if s else []
    out = []
    for i in range(n_alleles):
        lo = hi = None
        if i < len(parts) and "-" in parts[i]:
            a, b = parts[i].split("-", 1)
            if a.strip().lstrip("-").isdigit() and b.strip().lstrip("-").isdigit():
                lo, hi = int(a), int(b)
        out.append((lo, hi))
    return out


def parse_reference_region(s):
    """Parses ``"chr12:6936716-6936773"`` into ``(chrom, start_0based, end)`` (or all-None)."""
    m = _REGION_RE.match(str(s or ""))
    if not m:
        return (None, None, None)
    return (m.group(1), int(m.group(2)), int(m.group(3)))


def _sum_at(counts, value):
    return sum(c for size, c in counts if size == value)


def _sum_above(counts, value):
    return sum(c for size, c in counts if size > value)


def _aqm_for_allele(aqm_alleles, aqm_by_number, rank, eh):
    """Returns the AlleleQualityMetrics dict for one allele.

    Aligns by 1-based ``AlleleNumber == rank + 1``; if that row's ``AlleleSize``
    disagrees with the called ``eh`` (and another allele matches by size), prefer
    the size match. Falls back to ``{}`` when nothing aligns.
    """
    aqm = aqm_by_number.get(rank + 1)
    if aqm is None or (aqm.get("AlleleSize") not in (None, eh) and len(aqm_alleles) > rank):
        match = [a for a in aqm_alleles if a.get("AlleleSize") == eh]
        aqm = match[0] if match else (aqm or {})
    return aqm or {}


# A no-call locus is emitted as this many rows (ranks 0..N-1). The truth catalog always carries two
# alleles per locus (Short/Long), so two no-call rows join one-to-one with the two truth alleles.
NO_CALL_RANKS = 2


def _no_call_rows(locus_result, sample_id, branch, motif_size, reference_repeat_purity,
                  ref_size_bp, num_ref):
    """Yields ``NO_CALL_RANKS`` no-call rows for one uncalled variant (eh + read fields None)."""
    for rank in range(NO_CALL_RANKS):
        yield {
            "locus_id": locus_result.get("LocusId"),
            "sample_id": sample_id,
            "allele_rank": rank,
            "motif_size": motif_size,
            "reference_repeat_purity": reference_repeat_purity,
            "ref_size_bp": ref_size_bp,
            "num_repeats_in_reference": num_ref,
            "eh": None,
            "eh_minus_ref": None,
            # A no-call has no genotype and so no per-allele quality metrics. These rows exist only
            # for the report's truth join and are dropped from training by `missing_eh_or_true`
            # before the has_own_quality_metrics filter is ever consulted.
            "has_own_quality_metrics": False,
            "n_alleles": None,
            "n_distinct_alleles": None,
            "ci_start": None,
            "ci_end": None,
            "ci_width": None,
            "spanning_total": None,
            "hq_unamb_total": None,
            "flanking_total": None,
            "spanning_at_called": None,
            "spanning_above_called": None,
            "flanking_above_called": None,
            "support_frac": None,
            "flanking_frac": None,
            "coverage": None,
            "depth": None,
            "hq_unambiguous_reads": None,
            "strand_bias_phred": None,
            "mean_inserted_bases": None,
            "mean_deleted_bases": None,
            "read_repeat_purity": None,
            "genotyping_branch": branch,
            "left_flank_norm_depth": None,
            "right_flank_norm_depth": None,
            "inrepeat_total": None,
        }


def extract_variant_rows(variant, locus_result, sample_id):
    """Yields one row dict per allele of a single EH variant.

    A no-call (``Genotype`` == ``"./."`` / empty) still yields ``NO_CALL_RANKS`` rows with ``eh`` (and
    every read/allele-specific field) set to None -- only the catalog fields (motif, reference size,
    repeat purity) are populated. The truth join then attaches ``true`` per ``(locus_id, allele_rank)``
    so the accuracy-by-size report can count these as "No Call" (a truth allele EH left uncalled). They
    carry ``eh=None`` on purpose, so ``dataset.label_and_filter`` drops them (``missing_eh_or_true``)
    before training -- they exist only for the report side, never the model fit.
    """
    branch = BRANCH_QUICK if bool(variant.get("QuickGenotype", False)) else BRANCH_FULL
    repeat_unit = variant.get("RepeatUnit") or ""
    motif_size = len(repeat_unit) or None
    reference_repeat_purity = variant.get("ReferenceRepeatPurity")
    _, start, end = parse_reference_region(variant.get("ReferenceRegion"))
    ref_size_bp = (end - start) if (start is not None and end is not None) else None
    num_ref = (ref_size_bp / motif_size) if (ref_size_bp and motif_size) else None

    genotype = parse_genotype(variant.get("Genotype"))
    if genotype is None:
        yield from _no_call_rows(locus_result, sample_id, branch, motif_size,
                                 reference_repeat_purity, ref_size_bp, num_ref)
        return

    n_alleles = len(genotype)
    cis = parse_ci(variant.get("GenotypeConfidenceInterval"), n_alleles)
    spanning = parse_counts(variant.get("CountsOfSpanningReads"))
    flanking = parse_counts(variant.get("CountsOfFlankingReads"))
    spanning_total = sum(c for _, c in spanning)
    flanking_total = sum(c for _, c in flanking)
    hq_total = sum(c for _, c in parse_counts(
        variant.get("CountsOfHighQualityUnambiguousReads")))
    # Flanking share of the locus's informative reads. 0 when there are no flanking reads at all,
    # which also covers the 0/0 case (no reads of either kind) rather than emitting NaN.
    flanking_frac = (flanking_total / (flanking_total + spanning_total)) if flanking_total else 0.0
    # Per-locus (not per-allele) read depth, shared by every allele of every variant at the locus.
    coverage = locus_result.get("Coverage")

    aqm_alleles = (variant.get("AlleleQualityMetrics") or {}).get("Alleles") or []
    aqm_by_number = {a.get("AlleleNumber"): a for a in aqm_alleles}

    for rank, eh in enumerate(genotype):
        ci_lo, ci_hi = cis[rank]
        ci_width = (ci_hi - ci_lo) if (ci_lo is not None and ci_hi is not None) else None
        span_at = _sum_at(spanning, eh)
        aqm = _aqm_for_allele(aqm_alleles, aqm_by_number, rank, eh)
        depth = aqm.get("Depth")
        row = {
            # Whether ExpansionHunter reported quality metrics for THIS allele specifically.
            # It emits one AlleleQualityMetrics entry per allele for a het call, but only ONE for a
            # homozygous or hemizygous call -- and it scores the model once per entry. So the rank-1
            # row of a hom call is a row inference can never produce: _aqm_for_allele falls back to
            # the size match and hands it the rank-0 allele's read metrics, giving two rows with
            # identical features (bar allele_rank) joined to two DIFFERENT truth alleles.
            # dataset.label_and_filter drops these before training; the accuracy-by-size report reads
            # the parquet directly and still sees both rows, which is what its truth join needs.
            "has_own_quality_metrics": aqm_by_number.get(rank + 1) is not None,
            "locus_id": locus_result.get("LocusId"),
            "sample_id": sample_id,
            "allele_rank": rank,
            "motif_size": motif_size,
            "reference_repeat_purity": reference_repeat_purity,
            "ref_size_bp": ref_size_bp,
            "num_repeats_in_reference": num_ref,
            "eh": eh,
            "eh_minus_ref": (eh - num_ref) if num_ref is not None else None,
            "n_alleles": n_alleles,
            "n_distinct_alleles": len(set(genotype)),
            "ci_start": ci_lo,
            "ci_end": ci_hi,
            "ci_width": ci_width,
            "spanning_total": spanning_total,
            "hq_unamb_total": hq_total,
            "flanking_total": flanking_total,
            "spanning_at_called": span_at,
            "spanning_above_called": _sum_above(spanning, eh),
            "flanking_above_called": _sum_above(flanking, eh),
            "support_frac": (span_at / spanning_total) if spanning_total else None,
            "flanking_frac": flanking_frac,
            "coverage": coverage,
            "depth": depth,
            "hq_unambiguous_reads": aqm.get("HighQualityUnambiguousReads"),
            "strand_bias_phred": aqm.get("StrandBiasBinomialPhred"),
            "mean_inserted_bases": aqm.get("MeanInsertedBasesWithinRepeats"),
            "mean_deleted_bases": aqm.get("MeanDeletedBasesWithinRepeats"),
            "read_repeat_purity": aqm.get("ReadRepeatPurity"),
            "genotyping_branch": branch,
        }
        if branch == BRANCH_FULL:
            row["left_flank_norm_depth"] = aqm.get("LeftFlankNormalizedDepth")
            row["right_flank_norm_depth"] = aqm.get("RightFlankNormalizedDepth")
            # The fast path (processLocusFast) never counts in-repeat reads and always writes an empty
            # table, so only the full genotyper's count carries information.
            row["inrepeat_total"] = sum(c for _, c in parse_counts(variant.get("CountsOfInrepeatReads")))
        else:
            row["left_flank_norm_depth"] = None
            row["right_flank_norm_depth"] = None
            row["inrepeat_total"] = None
        yield row


def extract_rows(eh_json, sample_id=None):
    """Yields per-allele row dicts for every variant in an EH JSON path or parsed dict.

    Args:
        eh_json: Path to a ``.json`` / ``.json.gz`` file, or an already-parsed dict.
        sample_id: Overrides the JSON ``SampleParameters.SampleId`` (used to stamp the
            sample+coverage key from the GCS path).

    Yields:
        Per-allele row dicts (see ``extract_variant_rows``).

    A path is read one ``LocusResults`` record at a time (ijson) rather than with ``json.load``: a
    JSON on the 5.65M-locus TRExplorer v2.1 catalog would need ~50GB as parsed Python objects (1.08GB
    for 108,701 loci), against ~20MB streamed. ``use_float=True`` makes numbers come back as the same
    floats ``json.load`` returns, not Decimals.
    """
    if isinstance(eh_json, str):
        sid = sample_id or _sample_id_in_file(eh_json)
        with _open_binary(eh_json) as f:
            for _, locus_result in ijson.kvitems(f, "LocusResults", use_float=True):
                for variant in (locus_result.get("Variants") or {}).values():
                    yield from extract_variant_rows(variant, locus_result, sid)
        return
    sid = sample_id or (eh_json.get("SampleParameters") or {}).get("SampleId")
    for locus_result in eh_json.get("LocusResults", {}).values():
        for variant in (locus_result.get("Variants") or {}).values():
            yield from extract_variant_rows(variant, locus_result, sid)
