"""Shared ExpansionHunter-JSON feature extractor (the single source of truth).

This module turns a raw ExpansionHunter ``.json`` output into tidy **per-allele**
rows. It is used by every data source (real GCS shards, regenerated real
optimized-streaming runs, and simulated runs) so that features are computed
identically across sources -- the real<->sim feature-equivalence guarantee in the
plan (GENOTYPE_QUALITY_METRICS_PLAN.md sec 5).

Two "contracts" are exposed (sec 5.1 / 5.2):

- ``full``  -- the full genotyper's records (seeking / streaming / low-mem-streaming,
  and optimized-streaming's full-genotyper fallback). Carries the complete
  ``AlleleQualityMetrics`` including ``QD`` and the flank-normalized depths.
- ``fast``  -- optimized-streaming's heuristic fast path (``processLocusFast``),
  flagged in the JSON by ``"QuickGenotype": true`` on the variant. Emits the same
  metrics as approximations but omits ``QD`` and the flank-normalized depths.

A variant's contract is decided **per variant** by the presence of
``QuickGenotype == true`` (sec 5.3). ``extract_rows`` auto-routes each variant to
its contract and tags every row with ``genotyping_branch in {full, fast}``.

The label call ``eh`` (sec 3) always comes from the JSON ``Genotype`` field here,
never from any precomputed TSV column.

Determinism: pure functions, no global state, no randomness.
"""

import json
import re


# --- Contracts -------------------------------------------------------------
# Raw columns the extractor guarantees per contract. ``features.py`` selects the
# per-branch model feature subset from these; the parity tests assert the
# extractor populates exactly these (non-None) for each contract on a fixture.

# Columns common to both contracts (locus-level + per-allele, always present).
COMMON_FIELDS = [
    # keys / identifiers (not model features by themselves)
    "locus_id", "variant_id", "sample_id",
    # locus-level
    "repeat_unit", "motif_size", "ref_chrom", "ref_start", "ref_end",
    "ref_size_bp", "num_repeats_in_reference", "coverage", "read_length",
    "fragment_length", "allele_count", "n_alleles_called",
    # per-allele genotype
    "allele_rank", "eh", "eh_minus_ref",
    "ci_start", "ci_end", "ci_width",
    "is_long", "is_hemi", "is_multi",
    # read-class counts (locus-level totals)
    "spanning_total", "flanking_total", "inrepeat_total", "hq_unamb_total",
    "frac_spanning", "frac_flanking", "frac_inrepeat",
    # per-allele read profile / support
    "spanning_at_called", "spanning_above_called", "flanking_above_called",
    "support_frac",
    # per-allele spanning-mode + sister-allele features (deep-dive Mode-3 / Mode-4 fixes)
    "sister_eh", "het_gap",
    "spanning_mode_naive", "spanning_mode_naive_support",
    "spanning_mode_allele", "spanning_mode_allele_support",
    "eh_minus_spanning_mode_naive", "eh_minus_spanning_mode_allele",
    # per-allele AlleleQualityMetrics (present in both contracts)
    "depth", "hq_unambiguous_reads", "strand_bias_phred",
    "mean_inserted_bases", "mean_deleted_bases",
    # branch
    "quick_genotype", "genotyping_branch",
]

# Columns present ONLY on the full contract (omitted/None on the fast path).
FULL_ONLY_FIELDS = ["qd", "eh_q", "left_flank_norm_depth", "right_flank_norm_depth"]

FULL_CONTRACT = COMMON_FIELDS + FULL_ONLY_FIELDS
FAST_CONTRACT = list(COMMON_FIELDS)

_COUNTS_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_counts(s):
    """Parses an EH counts string like ``"(20, 8), (22, 1)"`` into ``[(20, 8), (22, 1)]``.

    An empty tuple string ``"()"`` (or empty/None) returns ``[]``.
    """
    if not s:
        return []
    return [(int(a), int(b)) for a, b in _COUNTS_RE.findall(s)]


def parse_genotype(s):
    """Parses an EH ``Genotype`` string (e.g. ``"20/97"``, ``"20"``) into a list of ints.

    Returns ``None`` for a no-call (``"./."``, ``"."``, empty) or any non-integer field.
    """
    if not s:
        return None
    parts = str(s).split("/")
    if not all(p.strip().lstrip("-").isdigit() for p in parts):
        return None
    return [int(p) for p in parts]


def parse_ci(s, n_alleles):
    """Parses ``GenotypeConfidenceInterval`` (e.g. ``"20-20/95-153"``) into per-allele (lo, hi).

    Returns a list of ``n_alleles`` ``(lo, hi)`` int tuples. Missing / malformed
    entries fall back to ``(None, None)`` for that allele.
    """
    result = []
    parts = (str(s).split("/") if s else [])
    for i in range(n_alleles):
        lo = hi = None
        if i < len(parts) and "-" in parts[i]:
            a, b = parts[i].split("-", 1)
            if a.strip().lstrip("-").isdigit() and b.strip().lstrip("-").isdigit():
                lo, hi = int(a), int(b)
        result.append((lo, hi))
    return result


def parse_reference_region(s):
    """Parses ``"chr12:6936716-6936773"`` into ``(chrom, start_0based, end)``.

    Returns ``(None, None, None)`` if it does not match.
    """
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", str(s or ""))
    if not m:
        return (None, None, None)
    return (m.group(1), int(m.group(2)), int(m.group(3)))


def _sum_counts(counts):
    return sum(c for _, c in counts)


def _counts_above(counts, threshold):
    return sum(c for size, c in counts if size > threshold)


def _counts_at(counts, value):
    return sum(c for size, c in counts if size == value)


def _trim_singletons(counts):
    """Drops singleton clusters (n<2) when any multi-read cluster exists (outlier suppression)."""
    multi = [c for c in counts if c[1] >= 2]
    return multi if multi else counts


def _densest(counts):
    """Returns (repeat_count, n_reads) of the densest cluster, or (None, 0) if empty."""
    if not counts:
        return None, 0
    rep, n = max(counts, key=lambda p: p[1])
    return rep, n


def spanning_mode_features(spanning, eh, sister_eh):
    """Computes the spanning-read-mode features for one allele.

    The spanning-read histogram's dominant cluster directly estimates a spanned allele's size
    (the deep-dive's Mode-4 fix). Two views are returned because a single mode cannot, on its own,
    tell a spurious-allele over-call (needs the locus-global mode) from a genuine het (needs the
    per-allele mode) -- the model arbitrates using these plus ``sister_eh`` / ``het_gap``:

      * NAIVE = densest cluster over ALL spanning reads (singleton-trimmed). Recovers the truth
        when EH invented a spurious second allele (homozygous mis-called het).
      * PER-ALLELE = densest cluster among reads nearer this allele than the sister; ``None`` when
        no multi-read cluster lands on this allele (so it never drags a true-het allele toward the
        dominant sister, and never fabricates a contraction of an EH-exact call).

    Args:
        spanning: Parsed ``CountsOfSpanningReads`` ``[(repeat_count, n_reads), ...]``.
        eh: This allele's EH call.
        sister_eh: The other allele's EH call (``None`` for hemizygous / single-allele loci).

    Returns:
        ``(naive_mode, naive_support, allele_mode, allele_support)`` (modes are ``None`` /
        supports ``0`` when undefined).
    """
    pt = _trim_singletons(spanning)
    naive_mode, naive_sup = _densest(pt)
    if not pt or sister_eh is None or sister_eh == eh:
        return naive_mode, naive_sup, naive_mode, naive_sup  # hom / hemi: per-allele == naive
    mine = [(r, n) for r, n in pt if abs(r - eh) <= abs(r - sister_eh)]
    allele_mode, allele_sup = _densest(mine) if mine else (None, 0)
    return naive_mode, naive_sup, allele_mode, allele_sup


def _sister_eh(genotype, rank):
    """Returns the sister allele's EH size for allele ``rank`` (None if hemizygous).

    Diploid: the other allele. Multi-allelic (>2): the other allele nearest this one in size.
    """
    n = len(genotype)
    if n <= 1:
        return None
    if n == 2:
        return genotype[1 - rank]
    others = [genotype[j] for j in range(n) if j != rank]
    return min(others, key=lambda a: abs(a - genotype[rank]))


def extract_variant_rows(variant, locus_result, sample_id):
    """Yields one row dict per allele of a single EH variant.

    ``variant`` is the dict under ``LocusResults[locus].Variants[variant_id]``.
    The contract (full/fast) is decided by ``QuickGenotype``; the returned rows
    carry ``genotyping_branch`` accordingly and only the columns valid for that
    contract are populated (full-only columns are absent on fast rows).
    """
    genotype = parse_genotype(variant.get("Genotype"))
    if genotype is None:
        return  # no-call: eh undefined, drop (plan sec 4)

    quick = bool(variant.get("QuickGenotype", False))
    branch = "fast" if quick else "full"

    repeat_unit = variant.get("RepeatUnit") or ""
    motif_size = len(repeat_unit) if repeat_unit else None
    chrom, start, end = parse_reference_region(variant.get("ReferenceRegion"))
    ref_size_bp = (end - start) if (start is not None and end is not None) else None
    num_ref = (ref_size_bp / motif_size) if (ref_size_bp is not None and motif_size) else None

    n_alleles = len(genotype)
    cis = parse_ci(variant.get("GenotypeConfidenceInterval"), n_alleles)

    spanning = parse_counts(variant.get("CountsOfSpanningReads"))
    flanking = parse_counts(variant.get("CountsOfFlankingReads"))
    inrepeat = parse_counts(variant.get("CountsOfInrepeatReads"))
    hq = parse_counts(variant.get("CountsOfHighQualityUnambiguousReads"))
    spanning_total = _sum_counts(spanning)
    flanking_total = _sum_counts(flanking)
    inrepeat_total = _sum_counts(inrepeat)
    hq_total = _sum_counts(hq)
    reads_total = spanning_total + flanking_total + inrepeat_total

    aqm_alleles = (variant.get("AlleleQualityMetrics") or {}).get("Alleles") or []
    aqm_by_number = {a.get("AlleleNumber"): a for a in aqm_alleles}

    for rank, eh in enumerate(genotype):
        ci_lo, ci_hi = cis[rank]
        ci_width = (ci_hi - ci_lo) if (ci_lo is not None and ci_hi is not None) else None
        sister_eh = _sister_eh(genotype, rank)
        sp_naive, sp_naive_sup, sp_allele, sp_allele_sup = spanning_mode_features(
            spanning, eh, sister_eh)
        # AQM allele aligned by 1-based AlleleNumber == rank+1; fall back by size.
        aqm = aqm_by_number.get(rank + 1)
        if aqm is None or (aqm.get("AlleleSize") not in (None, eh) and len(aqm_alleles) > rank):
            match = [a for a in aqm_alleles if a.get("AlleleSize") == eh]
            aqm = match[0] if match else (aqm or {})
        aqm = aqm or {}

        row = {
            "locus_id": locus_result.get("LocusId"),
            "variant_id": variant.get("VariantId"),
            "sample_id": sample_id,
            "repeat_unit": repeat_unit,
            "motif_size": motif_size,
            "ref_chrom": chrom,
            "ref_start": start,
            "ref_end": end,
            "ref_size_bp": ref_size_bp,
            "num_repeats_in_reference": num_ref,
            "coverage": locus_result.get("Coverage"),
            "read_length": locus_result.get("ReadLength"),
            "fragment_length": locus_result.get("FragmentLength"),
            "allele_count": locus_result.get("AlleleCount"),
            "n_alleles_called": n_alleles,
            "allele_rank": rank,
            "eh": eh,
            "eh_minus_ref": (eh - num_ref) if num_ref is not None else None,
            "ci_start": ci_lo,
            "ci_end": ci_hi,
            "ci_width": ci_width,
            "is_long": (rank == n_alleles - 1 and n_alleles > 1),
            "is_hemi": (n_alleles == 1),
            "is_multi": (bool(locus_result.get("AlleleCount") and locus_result["AlleleCount"] > 2)),
            "spanning_total": spanning_total,
            "flanking_total": flanking_total,
            "inrepeat_total": inrepeat_total,
            "hq_unamb_total": hq_total,
            "frac_spanning": (spanning_total / reads_total) if reads_total else None,
            "frac_flanking": (flanking_total / reads_total) if reads_total else None,
            "frac_inrepeat": (inrepeat_total / reads_total) if reads_total else None,
            "spanning_at_called": _counts_at(spanning, eh),
            "spanning_above_called": _counts_above(spanning, eh),
            "flanking_above_called": _counts_above(flanking, eh),
            "support_frac": (_counts_at(spanning, eh) / spanning_total) if spanning_total else None,
            "sister_eh": sister_eh,
            "het_gap": (eh - sister_eh) if sister_eh is not None else None,
            "spanning_mode_naive": sp_naive,
            "spanning_mode_naive_support": sp_naive_sup,
            "spanning_mode_allele": sp_allele,
            "spanning_mode_allele_support": sp_allele_sup,
            "eh_minus_spanning_mode_naive": (eh - sp_naive) if sp_naive is not None else None,
            "eh_minus_spanning_mode_allele": (eh - sp_allele) if sp_allele is not None else None,
            "depth": aqm.get("Depth"),
            "hq_unambiguous_reads": aqm.get("HighQualityUnambiguousReads"),
            "strand_bias_phred": aqm.get("StrandBiasBinomialPhred"),
            "mean_inserted_bases": aqm.get("MeanInsertedBasesWithinRepeats"),
            "mean_deleted_bases": aqm.get("MeanDeletedBasesWithinRepeats"),
            "quick_genotype": quick,
            "genotyping_branch": branch,
        }

        if branch == "full":
            qd = aqm.get("QD")
            depth = aqm.get("Depth")
            row["qd"] = qd
            # EH's per-allele quality Q is not emitted in JSON; QD == Q / Depth,
            # so recover the JSON-native Q as qd * depth (plan sec 5.1 / sec 8).
            row["eh_q"] = (qd * depth) if (qd is not None and depth is not None) else None
            row["left_flank_norm_depth"] = aqm.get("LeftFlankNormalizedDepth")
            row["right_flank_norm_depth"] = aqm.get("RightFlankNormalizedDepth")

        yield row


def extract_rows(eh_json, sample_id=None):
    """Yields per-allele row dicts for every variant in an EH JSON object or path.

    ``eh_json`` may be a parsed dict or a path to a ``.json`` file. ``sample_id``
    overrides the JSON ``SampleParameters.SampleId`` when given (useful for the
    GCS/sim row keys, which encode sample+coverage in the path/filename).
    """
    if isinstance(eh_json, str):
        with open(eh_json) as f:
            eh_json = json.load(f)
    sid = sample_id or (eh_json.get("SampleParameters") or {}).get("SampleId")
    for locus_result in eh_json.get("LocusResults", {}).values():
        for variant in (locus_result.get("Variants") or {}).values():
            yield from extract_variant_rows(variant, locus_result, sid)
