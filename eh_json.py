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
any precomputed TSV column. Only the columns the deployable model and its report
actually consume are emitted (the audit/debug columns from earlier iterations are
intentionally dropped). Pure functions, no global state, no randomness.
"""

import gzip
import json
import re

# Genotyping-branch labels (shared with features.genotyping_regime_of). The "quick" branch is
# EH's QuickGenotype fast path (processLocusFast); "full" is the full genotyper.
BRANCH_FULL = "full"
BRANCH_QUICK = "quick"

_COUNTS_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")


def _open(path):
    """Opens ``path`` for text reading, transparently decompressing ``.gz``."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


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


def extract_variant_rows(variant, locus_result, sample_id):
    """Yields one row dict per allele of a single EH variant (or nothing for a no-call)."""
    genotype = parse_genotype(variant.get("Genotype"))
    if genotype is None:
        return  # no-call: eh undefined, drop

    branch = BRANCH_QUICK if bool(variant.get("QuickGenotype", False)) else BRANCH_FULL
    repeat_unit = variant.get("RepeatUnit") or ""
    motif_size = len(repeat_unit) or None
    _, start, end = parse_reference_region(variant.get("ReferenceRegion"))
    ref_size_bp = (end - start) if (start is not None and end is not None) else None
    num_ref = (ref_size_bp / motif_size) if (ref_size_bp and motif_size) else None

    n_alleles = len(genotype)
    cis = parse_ci(variant.get("GenotypeConfidenceInterval"), n_alleles)
    spanning = parse_counts(variant.get("CountsOfSpanningReads"))
    flanking = parse_counts(variant.get("CountsOfFlankingReads"))
    spanning_total = sum(c for _, c in spanning)
    hq_total = sum(c for _, c in parse_counts(
        variant.get("CountsOfHighQualityUnambiguousReads")))

    aqm_alleles = (variant.get("AlleleQualityMetrics") or {}).get("Alleles") or []
    aqm_by_number = {a.get("AlleleNumber"): a for a in aqm_alleles}

    for rank, eh in enumerate(genotype):
        ci_lo, ci_hi = cis[rank]
        ci_width = (ci_hi - ci_lo) if (ci_lo is not None and ci_hi is not None) else None
        span_at = _sum_at(spanning, eh)
        aqm = _aqm_for_allele(aqm_alleles, aqm_by_number, rank, eh)
        depth = aqm.get("Depth")
        row = {
            "locus_id": locus_result.get("LocusId"),
            "sample_id": sample_id,
            "allele_rank": rank,
            "motif_size": motif_size,
            "ref_size_bp": ref_size_bp,
            "num_repeats_in_reference": num_ref,
            "coverage": locus_result.get("Coverage"),
            "eh": eh,
            "eh_minus_ref": (eh - num_ref) if num_ref is not None else None,
            "ci_start": ci_lo,
            "ci_end": ci_hi,
            "ci_width": ci_width,
            "spanning_total": spanning_total,
            "hq_unamb_total": hq_total,
            "spanning_at_called": span_at,
            "spanning_above_called": _sum_above(spanning, eh),
            "flanking_above_called": _sum_above(flanking, eh),
            "support_frac": (span_at / spanning_total) if spanning_total else None,
            "depth": depth,
            "hq_unambiguous_reads": aqm.get("HighQualityUnambiguousReads"),
            "strand_bias_phred": aqm.get("StrandBiasBinomialPhred"),
            "mean_inserted_bases": aqm.get("MeanInsertedBasesWithinRepeats"),
            "mean_deleted_bases": aqm.get("MeanDeletedBasesWithinRepeats"),
            "genotyping_branch": branch,
        }
        if branch == BRANCH_FULL:
            row["left_flank_norm_depth"] = aqm.get("LeftFlankNormalizedDepth")
            row["right_flank_norm_depth"] = aqm.get("RightFlankNormalizedDepth")
        else:
            row["left_flank_norm_depth"] = None
            row["right_flank_norm_depth"] = None
        yield row


def extract_rows(eh_json, sample_id=None):
    """Yields per-allele row dicts for every variant in an EH JSON path or parsed dict.

    Args:
        eh_json: Path to a ``.json`` / ``.json.gz`` file, or an already-parsed dict.
        sample_id: Overrides the JSON ``SampleParameters.SampleId`` (used to stamp the
            sample+coverage key from the GCS path).

    Yields:
        Per-allele row dicts (see ``extract_variant_rows``).
    """
    if isinstance(eh_json, str):
        with _open(eh_json) as f:
            eh_json = json.load(f)
    sid = sample_id or (eh_json.get("SampleParameters") or {}).get("SampleId")
    for locus_result in eh_json.get("LocusResults", {}).values():
        for variant in (locus_result.get("Variants") or {}).values():
            yield from extract_variant_rows(variant, locus_result, sid)
