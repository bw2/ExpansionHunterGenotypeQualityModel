"""Download the training data from GCS and assemble the per-branch parquets.

For each ``(sample, coverage)`` combo on each genotyping branch this module:

  1. Idempotently downloads the per-shard EH JSON (``*.json.gz``) from
     ``gs://str-truth-set-v2/tool_results/...`` plus two truth sources: the tool-independent
     truth-genotypes TSV (``TRUTH_CATALOG_ROOT``, used for the actual truth join, shared across a
     sample's coverages) and the legacy ``for_comparison`` ``*.alleles.tsv.gz`` (kept only as a
     side effect for ``gen_datasets.py``'s report-side tool-comparison columns). Existing local
     files are not re-fetched.
  2. Extracts per-allele feature rows from the JSON via ``eh_json.extract_rows``
     (``eh`` and every feature come from the JSON, never a TSV).
  3. Joins the truth-genotypes TSV on ``(locus_id, allele_rank)`` -- ascending-by-truth-value
     Short/Long pairing -- attaching only ``true`` / ``purity`` / the negative-control flag (always
     False for this source; see ``_load_truth_from_genotypes_tsv``).
  4. Writes one parquet per combo, then assembles + labels + filters into
     ``data/parquet/{quick,full}.parquet``.

Single EH source -- the optimized-streaming run (``EHv5-bw2-optimized``), the variant
deployed in production. Both branches are carved from the SAME run by ``genotyping_branch``
so each expert trains on exactly the subpopulation it is served at deploy time:
  - ``quick`` -- the ``QuickGenotype`` fast-path rows (``genotyping_branch == "quick"``),
    routed to the ``quick`` genotyping_regime.
  - ``full``  -- the full-genotyper FALLBACK rows (``genotyping_branch == "full"``: the loci
    the fast path punted on), routed to ``full_spanning`` / ``full_nonspanning``.

This replaces the earlier design that sourced ``full`` from a separate low-mem-streaming
``EHv5`` run over ALL loci -- an easier, different locus population than the fast-path fallback
the full experts actually see at deploy, which left the full experts train/serve-skewed. The old
leakage concern (optimized fallback rows colliding with ``EHv5`` full rows on the same loci) no
longer applies: there is no ``EHv5`` pool to collide with, within one optimized run a locus is
either fast-path-called OR fallback (never both), and ``quick`` / ``full`` are separate experts;
cross-coverage repeats of a locus are held out together by the chromosome-clean CV.

The committed model is real-data-only (HG002 10x/20x/31x + CHM1_CHM13 46x); no
simulated rows. Determinism: no randomness anywhere in the ingestion path.
Coding rules: no type hints, Google docstrings, ``print()``, ``gcloud`` (macOS).
"""

import argparse
import base64
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd

import eh_json
import features

GCS_ROOT = "gs://str-truth-set-v2/tool_results"
# Local checkout used only to compare a downloaded EH JSON's stamped build version against the
# current ExpansionHunter-bw2 HEAD (see ``_check_freshness``). Skipped if not present.
EXPANSIONHUNTER_BW2_REPO = os.path.expanduser("~/code/ExpansionHunter-bw2")
# Tool-independent truth catalog: built directly from the dipcall/long-read-assembly pipeline, one
# file per sample (not per sample+coverage -- truth doesn't depend on short-read coverage). Never
# tied to a specific EH run, so unlike the `for_comparison` TSV (GCS_ROOT above) it cannot go stale
# relative to one. See `_load_truth_from_genotypes_tsv`.
TRUTH_CATALOG_ROOT = "gs://str-truth-set-v2/filter_vcf_v2"
HERE = os.path.dirname(os.path.abspath(__file__))

# Single source variant: the optimized-streaming run deployed in production. Its QuickGenotype
# rows feed the `quick` branch and its full-genotyper fallback rows feed the `full` branch (the
# split happens in assemble_branch by genotyping_branch). The variant token is also templated into
# the truth-TSV column headers. SOURCE_SUBDIR keeps its historical name ("real_quick") so existing
# local downloads/parquets are reused; it now feeds BOTH branches.
SOURCE_VARIANT = "EHv5-bw2-optimized"
SOURCE_SUBDIR = "real_quick"

# (sample, coverage-dir label) -- illumina WGS only.
COMBOS = [
    ("HG002", "10x"),
    ("HG002", "20x"),
    ("HG002", "31x"),
    ("CHM1_CHM13", "46x"),
]

# 43-held-out-HPRC samples promoted into the training pool for ancestry/sex diversity at large
# (sparse, single-genome-dominated) allele sizes -- HG002+CHM1_CHM13 alone left the full_nonspanning
# tail overfit to 2 genomes (helps in-sample, hurts on held-out). Picked to cover every population
# present in the 43-sample panel and balance sex (7 male / 7 female including HG002): NA12878 (CEU),
# HG03492 (PJL), HG00621 (CHS), HG02080 (KHV), HG01106 (PUR), HG01258 (CLM), HG01928 (PEL),
# HG02055 (ACB), HG02622 (GWD), HG03453 (MSL), HG03125 (ESN), NA18906 (YRI), NA20129 (ASW). Their
# per-allele parquets are already built by ``heldout.build_sample`` under ``data_eval_43/real_43/``;
# ``main()`` symlinks them into the training subdir rather than re-downloading. Removed from
# ``heldout.SAMPLES`` so they are not double-counted in the external validation set.
PROMOTED_HELDOUT_SAMPLES = (
    "NA12878", "HG03492", "HG00621", "HG02080", "HG01106", "HG01258", "HG01928",
    "HG02055", "HG02622", "HG03453", "HG03125", "NA18906", "NA20129",
)

VALID_CHROMS = set(str(i) for i in range(1, 23)) | {"X", "Y"}


def _combo_dir(sample, variant, cov_label):
    return "%s/%s/illumina/%s/%s_coverage/" % (GCS_ROOT, sample, variant, cov_label)


def _truth_tsv_remote(sample, variant, cov_label):
    return ("%s%s.tandem_repeat_genotypes.for_comparison.with_%s_vs_Truth_columns.alleles.tsv.gz"
            % (_combo_dir(sample, variant, cov_label), sample, variant))


def _truth_genotypes_tsv_remote(sample):
    return "%s/%s/%s.tandem_repeat_genotypes.tsv.gz" % (TRUTH_CATALOG_ROOT, sample, sample)


def _list_json_inputs(sample, variant, cov_label):
    """Lists the combo's JSON shard paths (prefers a single combined file if present)."""
    listing = subprocess.run(["gsutil", "ls", _combo_dir(sample, variant, cov_label) + "json/"],
                             capture_output=True, text=True, check=True).stdout.split()
    jsons = [p for p in listing if p.endswith(".json") or p.endswith(".json.gz")]
    combined = sorted(p for p in jsons if ".shard" not in os.path.basename(p))
    return combined if combined else sorted(p for p in jsons if ".shard" in os.path.basename(p))


def _download(remote_paths, dest_dir):
    """Downloads ``remote_paths`` into ``dest_dir`` (skips files already present, retries).

    Uses a single multi-file ``gcloud storage cp`` (parallelized) and retries the
    whole batch on failure; each retry only re-fetches the still-missing files, so a
    transient bad-hash leaves no partial file and recovers cleanly.
    """
    os.makedirs(dest_dir, exist_ok=True)
    locals_ = [os.path.join(dest_dir, os.path.basename(p)) for p in remote_paths]
    for attempt in (1, 2, 3, 4):
        missing = [r for r, l in zip(remote_paths, locals_) if not os.path.exists(l)]
        if not missing:
            break
        print("    attempt %d: downloading %d/%d file(s) -> %s"
              % (attempt, len(missing), len(locals_), dest_dir))
        subprocess.run(["gcloud", "storage", "cp"] + missing + [dest_dir + os.sep])
    still = [r for r, l in zip(remote_paths, locals_) if not os.path.exists(l)]
    if still:
        raise RuntimeError("failed to download after 4 attempts: %s" % still)
    return locals_


def _gcs_md5(remote_path):
    """Returns the base64 md5 hash GCS reports for ``remote_path`` via ``gsutil stat``, or None."""
    result = subprocess.run(["gsutil", "stat", remote_path], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.strip().startswith("Hash (md5):"):
            return line.split(":", 1)[1].strip()
    return None


def _local_md5(path):
    """Returns the base64 md5 hash of a local file, in the same form ``_gcs_md5`` returns."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def _bw2_head_sha():
    """Returns ExpansionHunter-bw2's current short commit sha, or None if that checkout isn't present
    locally (the EH-build-staleness check in ``_check_freshness`` is then skipped)."""
    if not os.path.isdir(EXPANSIONHUNTER_BW2_REPO):
        return None
    result = subprocess.run(["git", "-C", EXPANSIONHUNTER_BW2_REPO, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _json_eh_version(path):
    """Returns the ``Version`` stamped in a JSON shard's ``SampleParameters`` -- the short commit sha
    of the ExpansionHunter-bw2 build that produced it. Absent entirely on builds that predate that
    stamping (before 2026-07-01); ``"unknown"`` if the build couldn't capture its own commit sha
    (e.g. a Docker build without ``.git`` in the build context)."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f).get("SampleParameters", {}).get("Version")


def _check_freshness(desc, sources):
    """Detects stale local downloads and, if any are found, refuses to proceed silently.

    ``sources`` is a list of ``(label, remote_path, local_path)``. For each entry whose
    ``local_path`` already exists (nothing to check for a first-time download -- ``_download``
    will simply fetch the current bucket version), two independent signals are checked:

    1. Local-cache staleness: does the local file's md5 still match what's currently in the bucket?
       Catches the truth catalog (or the for_comparison TSV) having been regenerated upstream since
       we last downloaded it.
    2. EH-build staleness (labels starting with ``"json"`` only): does the JSON's stamped build
       commit sha match the local ExpansionHunter-bw2 checkout's current HEAD? Catches training on
       EH calls produced by a build older than what's checked out now. Skipped if that repo isn't
       present locally.

    Any issue found prints a summary and prompts ``(e)xit / (u)pdate / (i)gnore`` on a TTY;
    non-interactively (e.g. a backgrounded rebuild) it prints the same summary and exits nonzero --
    a deliberate refusal, not a warning, since silently training on stale inputs is exactly the bug
    class this exists to catch. ``(u)pdate`` only has an automatic fix for local-cache staleness (it
    deletes the stale file so the next ``_download`` call re-fetches it); an EH-build-staleness issue
    has no automatic fix here since that requires re-running ExpansionHunter-bw2 externally, so
    ``(u)pdate`` removes what it can and then re-prompts (rather than silently returning) whenever an
    EH-build issue is still unresolved, forcing an explicit ``(e)xit``/``(i)gnore`` choice for it.
    """
    issues, stale_locals, unfixable = [], [], False
    for label, remote, local in sources:
        if not os.path.exists(local):
            continue
        remote_md5 = _gcs_md5(remote)
        if remote_md5 and remote_md5 != _local_md5(local):
            issues.append("%s: local copy no longer matches the current bucket content (%s)"
                          % (label, remote))
            stale_locals.append(local)
        if label.startswith("json"):
            version, head = _json_eh_version(local), _bw2_head_sha()
            if head and version != head:
                issues.append("%s: produced by EH build %r, current ExpansionHunter-bw2 HEAD is %r"
                              % (label, version, head))
                unfixable = True
    if not issues:
        return
    print("\n=== STALE DATA SOURCE(S): %s ===" % desc)
    for issue in issues:
        print("  - %s" % issue)
    if not sys.stdin.isatty():
        print("  non-interactive run -- refusing to proceed on stale inputs. Re-run interactively to "
             "decide, or delete the affected file(s) under _downloads/ and re-run with --force.")
        sys.exit(1)
    while True:
        resp = input("  (e)xit / (u)pdate / (i)gnore? ").strip().lower()
        if resp in ("e", "exit"):
            sys.exit(1)
        if resp in ("i", "ignore"):
            return
        if resp in ("u", "update"):
            for local in stale_locals:
                os.remove(local)
            stale_locals = []
            print("  removed stale local file(s) -- they will be re-downloaded now.")
            if unfixable:
                print("  Note: a JSON produced by an older EH build can't be 'updated' this way -- that "
                     "requires re-running ExpansionHunter-bw2 externally. Choose (e)xit or (i)gnore to "
                     "proceed despite this.")
                continue
            return
        print("  please answer e/u/i")


def _load_truth_from_genotypes_tsv(tsv_path):
    """Loads the tool-independent truth-genotypes TSV (see ``TRUTH_CATALOG_ROOT``) and reshapes it
    from wide (Short/Long allele columns) to long (one row per allele).

    Built directly from the dipcall/long-read-assembly pipeline, never tied to a specific EH run --
    unlike the ``for_comparison`` TSV (``_truth_tsv_remote``, still used by ``build_combo`` for the
    report-side tool-comparison columns), it cannot go stale relative to one. Contains only variant
    (non-reference) loci, no negative-control rows, so ``is_negative_locus`` is always False. HOM/HEMI
    rows have ``NumRepeatsShortAllele == NumRepeatsLongAllele`` (no NaNs in either column), so every
    locus safely yields exactly two allele rows: ``allele_rank=0`` from the Short columns,
    ``allele_rank=1`` from the Long columns (ascending-by-truth-value pairing). A handful of loci
    (3 in HG002, identical values across their duplicate rows) appear twice in the source file --
    dropped via ``drop_duplicates`` before reshaping. Returns the
    ``{LocusId, allele_rank, true, purity, is_negative_locus}`` contract ``_join_truth`` expects.
    """
    cols = ["LocusId", "NumRepeatsShortAllele", "NumRepeatsLongAllele",
           "RepeatPurityShortAllele", "RepeatPurityLongAllele"]
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=cols,
                     dtype={"LocusId": str}).drop_duplicates("LocusId")
    short = df[["LocusId", "NumRepeatsShortAllele", "RepeatPurityShortAllele"]].rename(
        columns={"NumRepeatsShortAllele": "true", "RepeatPurityShortAllele": "purity"})
    short["allele_rank"] = 0
    long_ = df[["LocusId", "NumRepeatsLongAllele", "RepeatPurityLongAllele"]].rename(
        columns={"NumRepeatsLongAllele": "true", "RepeatPurityLongAllele": "purity"})
    long_["allele_rank"] = 1
    out = pd.concat([short, long_], ignore_index=True)
    # ANALYSIS_OK[imputation]: malformed true/purity become NaN; label_and_filter drops NaN-true rows
    # as "missing_eh_or_true" downstream.
    for c in ("true", "purity"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["is_negative_locus"] = False
    return out[["LocusId", "allele_rank", "true", "purity", "is_negative_locus"]]


# Symmetric locus-catalog agreement check (see ``_assert_catalog_agreement``): a JSON download and its
# truth-comparison TSV are supposed to come from the same EH run/catalog. A global-only tolerance would
# miss a mismatch concentrated in one allele-size range -- exactly the failure mode this exists to
# catch (found live: ~1,560 large-allele truth loci were silently absent from a "real_quick" JSON
# download that the truth TSV still listed real, non-no-call EH calls for; global mismatch was only
# ~0.2% of the whole catalog, but ~45% within the largest-allele bin).
_CATALOG_MISMATCH_GLOBAL_MAX = 0.02   # refuse if >2% of the combined locus catalog disagrees overall
_CATALOG_MISMATCH_PERBIN_MAX = 0.15   # refuse if any truth-allele-size bin disagrees by more than this
_CATALOG_SIZE_BIN_EDGES = (20, 50, 100, 200)  # truth repeat-count bin edges (open-ended below/above)


def _size_bin_label(true_repeats):
    """Buckets a truth allele size (repeats) into one of the coarse ``_CATALOG_SIZE_BIN_EDGES`` bins."""
    edges = (0,) + _CATALOG_SIZE_BIN_EDGES + (float("inf"),)
    for lo, hi in zip(edges, edges[1:]):
        if lo <= true_repeats < hi:
            return ("%g+" % lo) if hi == float("inf") else ("%g-%g" % (lo, hi))
    return "?"


def _assert_catalog_agreement(json_df, tsv_df, source_desc, fatal=True):
    """Raises (or, if ``fatal=False``, prints a WARNING and continues) when the JSON and truth-TSV
    locus catalogs disagree beyond tolerance.

    Compares the two locus_id sets directly (symmetric -- independent of join direction), both overall
    and within each truth-allele-size bucket (bucketed from the TSV's own ``true`` column, the only
    side that carries a truth size). Call this BEFORE joining so a silent catalog mismatch is refused
    at ingestion time instead of quietly dropping alleles downstream with no visible signal.
    ``fatal=False`` is for report-generation call sites that intentionally apply the model to a capped
    subsample of the parquet (e.g. ``accuracy_by_size.predict_lcf_pok``'s ``cap``): a capped ``json_df``
    always looks like a catalog mismatch, so those sites can only ever WARN, not enforce.
    """
    json_loci, tsv_loci = set(json_df["locus_id"]), set(tsv_df["LocusId"])
    union = json_loci | tsv_loci
    if not union:
        return
    only_json, only_tsv = json_loci - tsv_loci, tsv_loci - json_loci
    global_frac = len(only_json | only_tsv) / len(union)

    bins = {}
    for locus, true_val in tsv_df.set_index("LocusId")["true"].items():
        if pd.isna(true_val):
            continue
        label = _size_bin_label(true_val)
        missing, total = bins.setdefault(label, [0, 0])
        bins[label][1] += 1
        bins[label][0] += int(locus in only_tsv)
    worst_bin, worst_frac = None, 0.0
    for label, (missing, total) in bins.items():
        frac = missing / total if total else 0.0
        if frac > worst_frac:
            worst_bin, worst_frac = label, frac

    violated = global_frac > _CATALOG_MISMATCH_GLOBAL_MAX or worst_frac > _CATALOG_MISMATCH_PERBIN_MAX
    if violated:
        msg = (
            "%s: JSON and truth-TSV locus catalogs disagree -- global %.2f%% (%d only-in-JSON, "
            "%d only-in-TSV out of %d total loci; limit %.2f%%), worst truth-size bin '%s' at "
            "%.1f%% missing (limit %.1f%%). They likely come from different EH runs/catalog "
            "versions -- re-download matching JSON/TSV sources before proceeding. Example "
            "mismatched loci: %s" % (
                source_desc, 100 * global_frac, len(only_json), len(only_tsv), len(union),
                100 * _CATALOG_MISMATCH_GLOBAL_MAX, worst_bin, 100 * worst_frac,
                100 * _CATALOG_MISMATCH_PERBIN_MAX, sorted(only_tsv or only_json)[:5]))
        if fatal:
            raise RuntimeError(msg)
        print("  WARNING: %s" % msg, flush=True)
        return
    print("  catalog agreement (%s): %.2f%% global mismatch, worst bin '%s' %.1f%% missing"
          % (source_desc, 100 * global_frac, worst_bin, 100 * worst_frac), flush=True)


def _join_truth(json_df, tsv_df, source_desc):
    """Left-joins truth onto the JSON rows by ``(locus_id, allele_rank)`` (keys asserted unique).

    A leading ``chr`` is stripped from both locus ids before the join: some catalogs emit
    ``chr1-..`` LocusIds in the JSON while the truth TSV uses ``1-..`` (or vice versa), and the
    two must compare equal. ``source_desc`` (e.g. ``"HG002 31x"``) names this combo in the catalog-
    agreement check's error message.
    """
    json_df["locus_id"] = json_df["locus_id"].astype(str).str.replace(r"^chr", "", regex=True)
    tsv_df["LocusId"] = tsv_df["LocusId"].astype(str).str.replace(r"^chr", "", regex=True)
    assert not json_df.duplicated(["locus_id", "allele_rank"]).any(), \
        "JSON (locus_id, allele_rank) key is not unique"
    assert not tsv_df.duplicated(["LocusId", "allele_rank"]).any(), \
        "TSV (LocusId, allele_rank) key is not unique"
    # fatal=False for now: every current combo already fails this at 73-85% missing in the largest
    # (>=200 repeat) truth-size bin -- traced to gs://str-truth-set-v2 itself, where the "for_comparison"
    # TSVs were built from an OLDER version of the json/ shards than what's in the bucket today (the
    # locus is present in that sample's OWN "*.alleles.tsv.gz" json-flattening, absent from the current
    # json/ shards). Flip to fatal once the upstream bucket is regenerated consistently.
    _assert_catalog_agreement(json_df, tsv_df, source_desc, fatal=False)
    merged = json_df.merge(tsv_df, how="left", left_on=["locus_id", "allele_rank"],
                           right_on=["LocusId", "allele_rank"], validate="one_to_one"
                           ).drop(columns=["LocusId"])
    return merged


def build_combo(variant, subdir, sample, cov_label, data_dir, force):
    """Downloads + joins one combo and writes its per-combo parquet (all rows, both branches)."""
    out_path = os.path.join(data_dir, subdir, "%s_%s.parquet" % (sample, cov_label))
    print("=== %s %s -> %s ===" % (sample, cov_label, out_path))

    dl_dir = os.path.join(data_dir, subdir, "_downloads", "%s_%s" % (sample, cov_label))
    # Sample-level (not sample+coverage-level) dir: HG002's 3 coverages share one download of the
    # tool-independent truth catalog instead of re-fetching it 3x (truth doesn't depend on coverage).
    genotypes_dl_dir = os.path.join(data_dir, subdir, "_downloads", sample)
    json_remote = _list_json_inputs(sample, variant, cov_label)
    truth_tsv_remote = _truth_tsv_remote(sample, variant, cov_label)
    genotypes_tsv_remote = _truth_genotypes_tsv_remote(sample)
    # Checked even when the parquet cache below is about to be reused -- an --force-free run must
    # still detect that the inputs it would otherwise silently keep trusting have moved on.
    _check_freshness("%s %s" % (sample, cov_label), [
        ("json shard %d" % i, r, os.path.join(dl_dir, os.path.basename(r)))
        for i, r in enumerate(json_remote)
    ] + [
        ("for_comparison TSV", truth_tsv_remote, os.path.join(dl_dir, os.path.basename(truth_tsv_remote))),
        ("truth-genotypes TSV", genotypes_tsv_remote,
         os.path.join(genotypes_dl_dir, os.path.basename(genotypes_tsv_remote))),
    ])

    if os.path.exists(out_path) and not force:
        n = len(pd.read_parquet(out_path, columns=["eh"]))
        print("    parquet exists; skipping (use --force to rebuild)  [%d rows]" % n)
        return n

    print("    %d JSON file(s) + 1 truth TSV" % len(json_remote))
    json_local = _download(json_remote, dl_dir)
    # Kept for its side effect: gen_datasets.py's report path for hg002_genome/hg002_exome reads this
    # same downloaded for_comparison TSV directly for tool-comparison columns -- not used for _join_truth.
    _download([truth_tsv_remote], dl_dir)
    genotypes_tsv_local = _download([genotypes_tsv_remote], genotypes_dl_dir)[0]

    rows = []
    for path in json_local:
        rows.extend(eh_json.extract_rows(path, sample_id="%s_%s" % (sample, cov_label)))
    json_df = pd.DataFrame(rows)
    tsv_df = _load_truth_from_genotypes_tsv(genotypes_tsv_local)
    merged = _join_truth(json_df, tsv_df, "%s %s" % (sample, cov_label))
    merged = merged.drop(columns=["sample_id"])
    for c in merged.select_dtypes("float64").columns:
        merged[c] = merged[c].astype("float32")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    matched = int(merged["true"].notna().sum())
    print("    %d rows, %d matched truth (%.4f), %d quick / %d full"
          % (len(merged), matched, matched / max(len(merged), 1),
             int((merged["genotyping_branch"] == "quick").sum()),
             int((merged["genotyping_branch"] == "full").sum())))
    return len(merged)


def _chrom_from_locus(locus_id):
    """Returns the normalized chromosome ({1..22,X,Y}) from a ``locus_id`` like ``1-5917-..``."""
    chrom = locus_id.astype(str).str.split("-").str[0].str.replace(r"^chr", "", regex=True)
    return chrom.where(chrom.isin(VALID_CHROMS))


def label_and_filter(df):
    """Adds ``chrom`` + the q/direction labels and drops unusable rows.

    Drops chrM/unknown contigs, negative-control loci, missing/non-positive ``eh``/``true``, and
    missing motif, then derives the labels + genotyping-regime routing via ``features.add_labels``.
    Shared by the training-pool assembly and the held-out-sample benchmark so both apply identical
    filtering. Does NOT filter on truth repeat purity -- impure loci are kept in both training and
    eval (purity is available to the report's opt-in stratification pill via ``accuracy_by_size.py``
    instead of being used as a blanket exclusion).

    Returns:
        A ``(kept_df, drop_counts)`` tuple. ``kept_df`` has ``chrom`` + the label columns added and
        the heavy ``locus_id`` / ``is_negative_locus`` columns dropped.
    """
    df = df.copy()
    df["chrom"] = _chrom_from_locus(df["locus_id"])
    # ANALYSIS_OK[imputation]: NaN eh/true/motif drive the missing_eh_or_true/missing_motif_size drops below.
    eh = pd.to_numeric(df["eh"], errors="coerce")
    true = pd.to_numeric(df["true"], errors="coerce")
    motif = pd.to_numeric(df["motif_size"], errors="coerce")
    negative = df["is_negative_locus"].fillna(False).astype(bool)

    keep = pd.Series(True, index=df.index)
    drops = {}
    for name, bad in (("chrM_or_unknown_contig", df["chrom"].isna()),
                      ("negative_control_locus", negative),
                      ("missing_eh_or_true", eh.isna() | true.isna()),
                      ("nonpositive_eh_or_true", (eh <= 0) | (true <= 0)),
                      ("missing_motif_size", motif.isna() | (motif <= 0))):
        bad = bad & keep
        drops[name] = int(bad.sum())
        keep &= ~bad

    df = df[keep].reset_index(drop=True)
    features.add_labels(df)
    df = df.drop(columns=[c for c in ("locus_id", "is_negative_locus") if c in df.columns])
    return df, drops


def assemble_branch(data_dir, branch, subdir):
    """Assembles one branch's per-combo parquets, labels + filters, writes data/parquet/<branch>."""
    parts = sorted(glob.glob(os.path.join(data_dir, subdir, "*.parquet")))
    if not parts:
        print("\n%s branch: no per-combo parquets in %s/ -- skipping" % (branch, subdir))
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # Both branches are carved from the same optimized-streaming run by genotyping_branch:
    # quick = fast-path QuickGenotype rows; full = full-genotyper fallback rows (the deploy-matched
    # subpopulation the full experts are served).
    df = df[df["genotyping_branch"] == ("quick" if branch == "quick" else "full")].reset_index(drop=True)

    n0 = len(df)
    df, drops = label_and_filter(df)

    out_path = os.path.join(data_dir, "parquet", "%s.parquet" % branch)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    print("\n==================== %s branch ====================" % branch.upper())
    print("  input %d -> kept %d" % (n0, len(df)))
    for k, v in drops.items():
        print("    dropped %-26s %d" % (k, v))
    print("  genotyping_regimes:", df["genotyping_regime"].value_counts().to_dict())
    print("  directions:", df["direction"].value_counts().to_dict())
    print("  wrote %d rows -> %s" % (len(df), out_path))


def _link_promoted_heldout_samples(data_dir, force):
    """Symlinks each ``PROMOTED_HELDOUT_SAMPLES`` parquet into the training subdir.

    Builds the parquet via ``heldout.build_sample`` under ``data_eval_43/real_43/`` first if it
    isn't already there (same schema as a per-combo training parquet), then reuses it instead of
    re-downloading; ``assemble_branch`` picks it up via its ``*.parquet`` glob. Imports ``heldout``
    lazily here (not at module level) since ``heldout`` itself imports ``dataset``. ``force`` is
    threaded through from ``main()`` so ``--force`` also rebuilds promoted-sample parquets instead
    of silently reusing stale ones.
    """
    import heldout
    for sample in PROMOTED_HELDOUT_SAMPLES:
        src = heldout.build_sample(sample, os.path.join(HERE, "data_eval_43"), force=force)
        dst = os.path.join(data_dir, SOURCE_SUBDIR, "%s.parquet" % sample)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    parser.add_argument("--force", action="store_true", help="rebuild even if parquets exist")
    args = parser.parse_args()

    for sample, cov_label in COMBOS:
        build_combo(SOURCE_VARIANT, SOURCE_SUBDIR, sample, cov_label, args.data_dir, args.force)
    _link_promoted_heldout_samples(args.data_dir, args.force)
    for branch in ("quick", "full"):
        assemble_branch(args.data_dir, branch, SOURCE_SUBDIR)


if __name__ == "__main__":
    main()
