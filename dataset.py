"""Download the training data from GCS and assemble the per-branch parquets.

For each ``(sample, coverage)`` combo on each genotyping branch this module:

  1. Idempotently downloads the per-shard EH JSON (``*.json.gz``) from
     ``gs://str-truth-set-v2/tool_results/...`` plus the tool-independent truth-genotypes TSV
     (``TRUTH_CATALOG_ROOT``, used for the actual truth join, shared across a sample's coverages).
     A local file is kept only if it is present AND still current; ``_check_freshness`` re-downloads
     any local copy whose bucket object is newer (by last-modified) or whose content (md5) has changed.
  2. Extracts per-allele feature rows from the JSON via ``eh_json.extract_rows``
     (``eh`` and every feature come from the JSON, never a TSV).
  3. Filters the truth-genotypes TSV to EH's catalog loci (primary contigs + variant, non-hom-ref;
     see ``_load_truth_from_genotypes_tsv``) and joins it on ``(locus_id, allele_rank)`` --
     ascending-by-truth-value Short/Long pairing -- attaching only ``true`` / ``purity`` / the
     negative-control flag (always False for this source).
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

The committed model is real-data-only: the four COMBOS below (HG002 10x/20x/31x + CHM1_CHM13 46x)
plus the 50 samples in PROMOTED_HELDOUT_SAMPLES (54 training sources), which ``main()`` symlinks into
the same subdir so ``assemble_branch``'s glob picks them up. No simulated rows. Determinism: no
randomness anywhere in the ingestion path.
Coding rules: no type hints, Google docstrings, ``print()``, ``gcloud`` (macOS).
"""

import argparse
import base64
import email.utils
import glob
import gzip
import hashlib
import json
import os
import re
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
# tied to a specific EH run, so it cannot go stale relative to one. See
# `_load_truth_from_genotypes_tsv`.
TRUTH_CATALOG_ROOT = "gs://str-truth-set-v2/filter_vcf_v2"
# The truth set exists once per catalog it was genotyped against, under
# {TRUTH_CATALOG_ROOT}/{sample}/{TRUTH_CATALOG_NAME}_genotypes/. combined_321_catalog covers every sample in
# PROMOTED_HELDOUT_SAMPLES and heldout.SAMPLES; combined_43_catalog only covers the 43-sample short-read
# benchmark panel plus HG002 and CHM1_CHM13.
TRUTH_CATALOG_NAME = "combined_321_catalog"
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

# Single-coverage 1kGP/HPRC samples that join HG002+CHM1_CHM13 in the training pool, for ancestry/sex
# diversity at large (sparse, single-genome-dominated) allele sizes -- HG002+CHM1_CHM13 alone left
# the full_nonspanning tail overfit to 2 genomes (helps in-sample, hurts on held-out). Their per-allele
# parquets are built by ``heldout.build_sample`` under ``data_eval_43/real_43/`` (historical dir name)
# and ``main()`` symlinks them into the training subdir rather than re-downloading. Disjoint from
# ``heldout.SAMPLES`` so they are not double-counted in the external validation set.
#
# Panel as of 2026-10-01: 50 samples, so the pool is 54 sources with the 4 COMBOS. Drawn from the 138
# 1kGP samples that have a DipCall high-confidence BED, a truth-genotypes TSV and a Broad short-read
# CRAM (plus NA12878, whose short reads live under tool_results instead of the 1kGP CRAM list):
#   - the first 13 are the original promotion (2026-07), one per population of the 43-sample HPRC
#     panel: NA12878 (CEU), HG03492 (PJL), HG00621 (CHS), HG02080 (KHV), HG01106 (PUR),
#     HG01258 (CLM), HG01928 (PEL), HG02055 (ACB), HG02622 (GWD), HG03453 (MSL), HG03125 (ESN),
#     NA18906 (YRI), NA20129 (ASW);
#   - the next 37 were added deterministically for diversity (31 on 2026-09-25, then 6 more on
#     2026-10-01 by extending the same pick sequence, which left the first 31 unchanged): repeatedly take the 1kGP population
#     with the fewest training samples (ties: the larger candidate pool, then name), within it the
#     sex with fewer training samples (HG002 counted as male), and within that the sample with the
#     most autosomal high-confidence bases (assembly-derived sex, chrY >= 500 kb = male). Excluded
#     from the candidates: the 30 previously held-out samples (kept held out so old and new models
#     stay comparable), HG00512 (father of held-out HG00514), and the ten samples of the pOk
#     fast-path diagnosis cohort (HG00738, HG01940, HG01975, HG01993, HG02004, HG02015, HG02074,
#     HG02293, HG03654, HG03942), so the diagnosed defects can be re-measured on them after
#     retraining. Also excluded, from training and held-out alike (updated 2026-09-28), are the
#     samples in str-truth-set-v2's filter_vcfs_v2/samples_excluded_from_downstream_analyses.tsv:
#     HGSVC2 males whose DipCall truth lost almost all of chrX/chrY because their h1/h2 assemblies
#     are not split into X- and Y-carrying haplotypes (7 of them are among the 138: HG00512, HG01505,
#     HG02011, HG02492, HG03065, HG03371, HG03732, NA19650). Result: 15 populations are represented
#     (1 to 5 samples each; IBS, ITU and MXL had only excluded samples), 25 female / 26 male
#     including HG002.
# The 37 additions have no EHv5-bw2-optimized run under GCS_ROOT yet: each needs
# ``<sample>/illumina/EHv5-bw2-optimized/<cov>_coverage/combined_catalog_43_samples_1.6M_loci/json/``
# (``heldout._catalog_base``) before ``heldout.build_sample`` can ingest it.
PROMOTED_HELDOUT_SAMPLES = (
    "NA12878", "HG03492", "HG00621", "HG02080", "HG01106", "HG01258", "HG01928",
    "HG02055", "HG02622", "HG03453", "HG03125", "NA18906", "NA20129",
    "HG03804", "HG04228", "HG02615", "HG02135", "HG03710", "HG03927", "HG00408",
    "HG01433", "HG02273", "HG01192", "HG02451", "HG03688", "NA19983", "NA12329",
    "HG02965", "HG02647", "HG02129", "HG03239", "HG03816", "HG00658", "HG01346",
    "HG01934", "HG01074", "HG01960", "HG04204", "HG02841", "HG02071", "HG03669",
    "HG03831", "HG00706", "HG01150", "HG01943", "HG01081", "HG02258", "HG04115",
    "HG03041", "HG02083",
)

VALID_CHROMS = set(str(i) for i in range(1, 23)) | {"X", "Y"}

# The EH-build stamp, read straight out of the decompressed JSON bytes (see _json_eh_version).
_VERSION_RE = re.compile(rb'"Version"\s*:\s*"([^"]*)"')
_VERSION_CARRY_BYTES = 256  # comfortably longer than the longest possible key/value spelling


def _combo_dir(sample, variant, cov_label):
    return "%s/%s/illumina/%s/%s_coverage/" % (GCS_ROOT, sample, variant, cov_label)


def _truth_genotypes_tsv_remote(sample):
    return "%s/%s/%s_genotypes/%s.tandem_repeat_genotypes.tsv.gz" % (
        TRUTH_CATALOG_ROOT, sample, TRUTH_CATALOG_NAME, sample)


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


def _gcs_stat(remote_path):
    """Returns ``(md5, mtime)`` for a cloud object from ONE ``gsutil stat`` call.

    Both signals come out of the same output, and ``_check_freshness`` needs both for every cached
    file, so statting twice doubled the network round-trips for no gain (~180 process launches on a
    fully-cached held-out build where ~90 suffice).

    ``md5`` is the base64 hash GCS reports, or None if absent. ``mtime`` is a Unix timestamp,
    preferring the object's ``Update time`` and falling back to ``Creation time`` (an object never
    updated since upload reports only the latter), or None if neither parses.

    A FAILED stat (no network, no credentials, object gone) also yields ``(None, None)``, which
    ``_check_freshness`` cannot distinguish from "unchanged" -- it keeps the local copy. That is the
    right default for a cache check (an unreachable bucket must not delete local data), but it is
    silent, so the failure is printed here.
    """
    result = subprocess.run(["gsutil", "stat", remote_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("    WARNING: `gsutil stat %s` failed (%s); keeping the local copy unchecked"
              % (remote_path, (result.stderr or "").strip().splitlines()[-1:] or "no stderr"),
              flush=True)
        return None, None
    md5, times = None, {}
    for line in result.stdout.splitlines():
        s = line.strip()
        if s.startswith("Hash (md5):"):
            md5 = s.split(":", 1)[1].strip()
            continue
        for key in ("Update time:", "Creation time:"):
            if s.startswith(key):
                try:
                    times[key] = email.utils.parsedate_to_datetime(s.split(":", 1)[1].strip()).timestamp()
                except (TypeError, ValueError, IndexError):
                    pass
    return md5, times.get("Update time:", times.get("Creation time:"))


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
    """Returns the short commit sha of the ExpansionHunter-bw2 build that produced a JSON shard.

    Current builds stamp it in ``RunInfo.Version``; older builds put it in ``SampleParameters.Version``
    (checked as a fallback). Absent entirely on builds that predate the stamping (before 2026-07-01);
    ``"unknown"`` if the build couldn't capture its own commit sha (e.g. a Docker build without
    ``.git`` in the build context).

    Scans the decompressed bytes for the key instead of parsing the JSON: these shards are ~90 MB
    gzipped and ``json.load`` cost 9 seconds and 3.3 GB of RSS to retrieve a 7-character string,
    which the freshness check pays once per shard on every run. The LAST match wins, which reproduces
    the RunInfo-over-SampleParameters preference (ExpansionHunter writes ``SampleParameters`` at the
    head of the file and ``RunInfo`` at the end) while still finding the fallback in an old shard
    that has only ``SampleParameters``. ``Version`` appears nowhere else in EH's output.
    """
    op = gzip.open if path.endswith(".gz") else open
    found, carry = None, b""
    with op(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            block = carry + chunk
            matches = _VERSION_RE.findall(block)
            if matches:
                found = matches[-1].decode()
            # Keep a tail long enough that a key/value split across the chunk boundary still matches.
            carry = block[-_VERSION_CARRY_BYTES:]
    return found


def assert_eh_build_matches(desc, labelled_json_paths):
    """Exits nonzero if any EH output JSON was produced by a build other than ExpansionHunter-bw2 HEAD.

    ``labelled_json_paths`` is a list of ``(label, path)``. A mismatch means the calls in that JSON came
    from a different EH build, which matters beyond provenance: a build can change what an existing
    field MEANS without changing its name (the ``genotype_quality_model_update`` branch redefined
    ``LocusResults.Coverage`` for the optimized-streaming fast path, and ``coverage`` is a model
    feature). No re-download fixes that -- EH has to be re-run -- so this is a hard refusal.

    Skipped entirely when the ExpansionHunter-bw2 checkout is not present locally (``_bw2_head_sha``
    returns None), since there is then nothing to compare against.

    NOTE: the sha is stamped into the binary by CMake at CONFIGURE time, so a rebuild inside an
    existing build directory reports whatever sha that directory was configured with. Re-run ``cmake``
    before generating JSONs meant for training.
    """
    head = _bw2_head_sha()
    if not head:
        return
    stale = [(label, _json_eh_version(path)) for label, path in labelled_json_paths]
    stale = [(label, version) for label, version in stale if version != head]
    if not stale:
        return
    print("\n=== STALE EH BUILD: %s ===" % desc)
    for label, version in stale:
        print("  - %s: produced by EH build %r, current ExpansionHunter-bw2 HEAD is %r"
              % (label, version, head))
    print("  refusing to proceed -- regenerate these JSONs with the current ExpansionHunter-bw2 "
          "build, or remove the local ExpansionHunter-bw2 checkout to skip this check.")
    sys.exit(1)


def _check_freshness(desc, sources):
    """Auto-refreshes stale local downloads and refuses only on an unfixable EH-build mismatch.

    ``sources`` is a list of ``(label, remote_path, local_path)``. For each entry whose ``local_path``
    already exists (a first-time download needs no check -- ``_download`` fetches the current bucket
    version), two signals are evaluated:

    1. Cache staleness: the local copy is out of date if its md5 no longer matches the bucket object's,
       OR the bucket object's last-modified time is newer than the local file's mtime. Such files are
       DELETED here so the subsequent ``_download`` re-fetches (overwrites with) the current bucket
       version -- an automatic update, no prompt. (Re-downloading a file also means its about-to-be-
       overwritten build sha is not judged below.)
    2. EH-build staleness, via ``assert_eh_build_matches`` (labels starting with ``"json"``, and only
       for files being KEPT -- not the ones already scheduled for re-download, whose sha is about to
       be overwritten). Files that do NOT exist locally yet are skipped here and checked by the caller
       AFTER ``_download``, since a first download can just as easily land a JSON from the wrong EH
       build as a cached one can.

    Returns the number of stale local files removed for re-download (0 if none). No caller needs it
    to trigger a rebuild -- deleting the local file already makes ``_parquet_reusable`` fail its
    "every upstream still exists" test -- so it is reported for logging and for the tests that assert
    on the refresh decision.
    """
    redownload, keep_json = [], []
    for label, remote, local in sources:
        if not os.path.exists(local):
            continue
        remote_md5, cloud_mtime = _gcs_stat(remote)
        if (remote_md5 and remote_md5 != _local_md5(local)) or \
                (cloud_mtime is not None and cloud_mtime > os.path.getmtime(local)):
            redownload.append((label, local))
            continue  # being overwritten -- don't judge its stale build sha
        if label.startswith("json"):
            keep_json.append((label, local))
    if redownload:
        print("  refreshing %d stale local file(s) (bucket newer or content changed): %s"
              % (len(redownload), ", ".join(l for _, l in redownload)), flush=True)
        for _, local in redownload:
            os.remove(local)
    assert_eh_build_matches(desc, keep_json)
    return len(redownload)


def rebuild_command_for(path):
    """Returns the command that actually rebuilds ``path``.

    The two builders write to different trees and neither can rebuild the other's output, so a
    message that names one command unconditionally sends the reader to a no-op: ``heldout.py`` writes
    only ``<data_eval_43>/real_43/<sample>.parquet``, while the training-combo parquets under
    ``data/<SOURCE_SUBDIR>/`` come only from ``dataset.py``.
    """
    if os.path.basename(os.path.dirname(os.path.realpath(path))) == "real_43":
        return ("python3 heldout.py --build-only --force --samples %s"
                % os.path.splitext(os.path.basename(path))[0])
    return "python3 dataset.py --force"


def parquet_contract_complaint(path):
    """Returns why a cached parquet no longer matches the current feature contract, or None if it does.

    Three things go stale without any upstream file changing, so an mtime comparison cannot see them:

    - a column the contract has since gained (``features.missing_feature_columns``),
    - ``has_own_quality_metrics``, which is not a feature but decides which rows
      ``label_and_filter`` keeps, so a parquet without it silently trains on alleles
      ExpansionHunter never scores, and
    - float32 feature columns, left by the pipeline that downcast every float64 column before
      writing. ``features.build_matrix`` now refuses those, because the C++ scorer feeds the model
      full doubles and training on quantized values puts tree thresholds ~1 float32 ULP away from
      what inference sees.

    All three are rebuild-able, so reporting them here lets the normal (no ``--force``) run fix itself
    instead of failing several steps later with an assertion that names neither the file nor the fix.
    """
    import pyarrow.parquet as pq  # local: only the parquet-facing callers need this dependency

    missing = features.missing_feature_columns([path]).get(path)
    if missing:
        return "missing feature column(s) %s" % missing
    if "has_own_quality_metrics" not in set(pq.ParquetFile(path).schema.names):
        return ("missing the has_own_quality_metrics column, so the second genotype copy of every "
                "homozygous call (an allele ExpansionHunter never scores) cannot be filtered out")
    contract = set(features.FULL_FEATURES)
    narrowed = sorted(f.name for f in pq.ParquetFile(path).schema_arrow
                      if f.name in contract and str(f.type) == "float")
    if narrowed:
        return ("%d feature column(s) are float32 (e.g. %s); training requires float64 end to end"
                % (len(narrowed), ", ".join(narrowed[:3])))
    return None


def _parquet_reusable(out_path, upstream_locals, force):
    """Returns True iff the cached per-combo/per-sample parquet can be reused as-is.

    Reuse requires: not ``force``, the parquet exists, every upstream local source (EH JSON shards +
    truth TSV) still exists, the parquet is at least as new as all of them, AND it still satisfies the
    current feature contract (``parquet_contract_complaint``). If ``_check_freshness`` just deleted a
    stale upstream (so it is missing / about to be re-downloaded with a newer mtime), or an upstream is
    otherwise newer, the parquet is rebuilt rather than silently reused.
    """
    if force or not os.path.exists(out_path):
        return False
    if not all(os.path.exists(u) for u in upstream_locals):
        return False
    if os.path.getmtime(out_path) < max((os.path.getmtime(u) for u in upstream_locals), default=0):
        return False
    complaint = parquet_contract_complaint(out_path)
    if complaint:
        print("    rebuilding %s: %s" % (os.path.basename(out_path), complaint))
        return False
    return True


def _load_truth_from_genotypes_tsv(tsv_path):
    """Loads the tool-independent truth-genotypes TSV (see ``TRUTH_CATALOG_ROOT``), applies the
    EH-catalog-generation filters, and reshapes it from wide (Short/Long allele columns) to long.

    Built directly from the dipcall/long-read-assembly pipeline, never tied to a specific EH run, so it
    cannot go stale relative to one. It is a superset of the loci ExpansionHunter actually genotypes:
    ``str-truth-set-v2/run_tools/convert_truth_set_to_variant_catalogs.py`` builds the EH catalog from
    this same table by keeping only (a) primary-assembly contigs (chr1-22, X, Y) and (b) *variant*
    (non-reference) loci with parseable repeat counts -- a locus whose Short AND Long alleles both equal
    the reference is hom-ref and is dropped. We apply those same two filters here so the truth we join
    against is the set of loci EH's catalog contains (this is what closes most of the JSON-vs-truth
    catalog gap; see ``_assert_catalog_agreement``). The IlluminaEHv5-only prefilters (>=500 bp
    reference-interval cap, >5 flanking Ns) are deliberately NOT applied: they are specific to the
    official Illumina build, whereas the ``EHv5-bw2-optimized`` source tolerates large loci, and the
    held-out validation depends on evaluating those large alleles.

    ``is_negative_locus`` is always False (no negative-control rows). HOM/HEMI rows have
    ``NumRepeatsShortAllele == NumRepeatsLongAllele``, so every kept locus yields exactly two allele
    rows: ``allele_rank=0`` from the Short columns, ``allele_rank=1`` from the Long columns
    (ascending-by-truth-value pairing). Duplicate LocusId rows are collapsed via ``drop_duplicates``
    before filtering. Returns the ``{LocusId, allele_rank, true, purity, is_negative_locus}`` contract
    ``_join_truth`` expects.
    """
    cols = ["LocusId", "Chrom", "NumRepeatsInReference",
           "NumRepeatsShortAllele", "NumRepeatsLongAllele",
           "RepeatPurityShortAllele", "RepeatPurityLongAllele"]
    df = pd.read_csv(tsv_path, sep="\t", compression="gzip", usecols=cols,
                     dtype={"LocusId": str, "Chrom": str}).drop_duplicates("LocusId")

    # EH-catalog-generation filters (convert_truth_set_to_variant_catalogs.py): primary contigs only,
    # and variant (non-hom-ref) loci with parseable repeat counts. NaN ref/short/long -> not parseable.
    n0 = len(df)
    # ANALYSIS_OK[imputation]: nothing is imputed -- errors="coerce" turns an unparseable repeat
    # count into NaN precisely so the `parseable` mask below drops that locus, mirroring the EH
    # catalog-generation filter. No NaN survives into any downstream value.
    ref = pd.to_numeric(df["NumRepeatsInReference"], errors="coerce")
    short_n = pd.to_numeric(df["NumRepeatsShortAllele"], errors="coerce")
    long_n = pd.to_numeric(df["NumRepeatsLongAllele"], errors="coerce")
    primary = df["Chrom"].astype(str).str.replace(r"^chr", "", regex=True).isin(VALID_CHROMS)
    parseable = ref.notna() & short_n.notna() & long_n.notna()
    variant = ~((short_n == ref) & (long_n == ref))
    df = df[primary & parseable & variant]
    print("    truth catalog: %d loci -> %d after EH-catalog filters (primary contig + variant), "
          "dropped %d" % (n0, len(df), n0 - len(df)))

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


# Symmetric locus-catalog agreement check (see ``_assert_catalog_agreement``): after the truth is
# filtered to EH's catalog rules (_load_truth_from_genotypes_tsv), the JSON download and that filtered
# truth should cover close to the same loci. A global-only tolerance would miss a mismatch concentrated
# in one allele-size range -- exactly the failure mode this exists to catch (found live: large-allele
# truth loci silently absent from a "real_quick" JSON download that the truth still listed EH calls
# for; global mismatch was small but the largest-allele bin was heavily affected).
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
    ``fatal=False`` downgrades a mismatch to a printed warning instead of raising (see ``_join_truth``,
    the only current caller, for why).
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
    # fatal=False: the truth is now filtered to EH's catalog rules (primary contig + variant, see
    # _load_truth_from_genotypes_tsv), which removes the bulk of the JSON-vs-truth catalog gap, but two
    # residual, expected mismatches remain that are NOT staleness to abort on:
    #   1. Training combos (real_quick): those JSON shards predate the current pipeline and were produced
    #      with a tighter (~300bp) reference-interval cap than today's 500bp, so the truth still lists
    #      some 300-500bp variant loci absent from the JSON. Regenerate real_quick to close this.
    #   2. Held-out samples: their JSON genotypes the 1.6M-locus combined_catalog_43_samples catalog,
    #      a SUPERSET of any one sample's variant truth, so the JSON legitimately has many loci the
    #      per-sample truth doesn't (the mismatch is in the only-in-JSON direction).
    # Left-joining on the JSON keeps only JSON-called alleles; unmatched ones drop downstream. Flip to
    # fatal once real_quick is regenerated with the current pipeline and the held-out case is handled.
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
    genotypes_tsv_remote = _truth_genotypes_tsv_remote(sample)
    json_locals = [os.path.join(dl_dir, os.path.basename(r)) for r in json_remote]
    genotypes_tsv_local = os.path.join(genotypes_dl_dir, os.path.basename(genotypes_tsv_remote))
    # Checked even when the parquet cache below is about to be reused -- an --force-free run must
    # still detect that the inputs it would otherwise silently keep trusting have moved on. Deletes any
    # locally-stale (bucket-newer / content-changed) copy so _download re-fetches it below.
    _check_freshness("%s %s" % (sample, cov_label),
                     [("json shard %d" % i, r, l) for i, (r, l) in enumerate(zip(json_remote, json_locals))]
                     + [("truth-genotypes TSV", genotypes_tsv_remote, genotypes_tsv_local)])

    if _parquet_reusable(out_path, json_locals + [genotypes_tsv_local], force):
        n = len(pd.read_parquet(out_path, columns=["eh"]))
        print("    parquet up-to-date; skipping (use --force to rebuild)  [%d rows]" % n)
        return n

    print("    %d JSON file(s) + 1 truth TSV" % len(json_remote))
    json_local = _download(json_remote, dl_dir)
    genotypes_tsv_local = _download([genotypes_tsv_remote], genotypes_dl_dir)[0]
    # _check_freshness could only judge shards that already existed locally. Anything just fetched
    # (a first download, or a refreshed object) is judged here, before it is parsed into features.
    assert_eh_build_matches("%s %s" % (sample, cov_label),
                            [("json shard %d" % i, p) for i, p in enumerate(json_local)])

    rows = []
    for path in json_local:
        rows.extend(eh_json.extract_rows(path, sample_id="%s_%s" % (sample, cov_label)))
    json_df = pd.DataFrame(rows)
    tsv_df = _load_truth_from_genotypes_tsv(genotypes_tsv_local)
    merged = _join_truth(json_df, tsv_df, "%s %s" % (sample, cov_label))
    merged = merged.drop(columns=["sample_id"])
    # Feature columns stay float64 all the way to fit(). Downcasting here used to halve the parquet
    # and frame size, but it permanently quantized every value: sklearn's HistGradientBoosting upcasts
    # back to float64 internally (X_DTYPE) and bins to uint8, so the downcast bought nothing at fit
    # time while forcing the C++ scorer to reproduce it exactly. See features.build_matrix.

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

    Drops chrM/unknown contigs, negative-control loci, missing/non-positive ``eh``/``true``, missing
    motif, and alleles ExpansionHunter never scores (``has_own_quality_metrics`` False -- the second
    genotype copy of a homozygous call), then derives the labels + genotyping-regime routing via
    ``features.add_labels``.
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

    # ExpansionHunter scores the model once per AlleleQualityMetrics entry, and emits only ONE entry
    # for a homozygous or hemizygous call. The extractor still yields one row per genotype copy (the
    # accuracy-by-size report's truth join needs two rows per locus), so a hom call's rank-1 row is a
    # row inference can never produce: same features as its rank-0 twin, but joined to the LONG truth
    # allele. Training and isotonic calibration must not see them. Parquets built before this column
    # existed have no such marker, so they are left alone rather than silently half-filtered -- the
    # freshness guards catch those separately.
    no_own_metrics = (~df["has_own_quality_metrics"].fillna(True).astype(bool)
                      if "has_own_quality_metrics" in df.columns
                      else pd.Series(False, index=df.index))

    keep = pd.Series(True, index=df.index)
    drops = {}
    for name, bad in (("chrM_or_unknown_contig", df["chrom"].isna()),
                      ("negative_control_locus", negative),
                      ("missing_eh_or_true", eh.isna() | true.isna()),
                      ("nonpositive_eh_or_true", (eh <= 0) | (true <= 0)),
                      ("missing_motif_size", motif.isna() | (motif <= 0)),
                      ("no_own_quality_metrics", no_own_metrics)):
        bad = bad & keep
        drops[name] = int(bad.sum())
        keep &= ~bad

    df = df[keep].reset_index(drop=True)
    features.add_labels(df)
    df = df.drop(columns=[c for c in ("locus_id", "is_negative_locus") if c in df.columns])
    return df, drops


def _assert_parts_share_feature_contract(parts):
    """Raises if any per-combo parquet no longer matches the current extractor/feature contract.

    ``assemble_branch`` concatenates these parts, and ``pd.concat`` silently fills a column that
    only some parts carry with NaN -- so a part built before a ``features.py`` feature addition
    would contribute all-NaN values for the new feature instead of failing. Checking up front turns
    that into an actionable error naming the stale parquets.

    Uses the same ``parquet_contract_complaint`` as ``_parquet_reusable`` and the eval path, so all
    three agree on what "off-contract" means: a missing feature column, a missing
    ``has_own_quality_metrics``, or float32 feature storage. The rebuild command is chosen per path,
    since ``dataset.py`` and ``heldout.py`` each write only their own tree.
    """
    stale = {p: c for p, c in ((p, parquet_contract_complaint(p)) for p in parts) if c}
    if stale:
        raise RuntimeError(
            "%d per-combo parquet(s) no longer match the current feature contract and would "
            "contribute all-NaN or quantized columns if concatenated: %s."
            % (len(stale), "; ".join("%s %s -- rebuild with: %s" % (p, c, rebuild_command_for(p))
                                     for p, c in sorted(stale.items()))))


def assemble_branch(data_dir, branch, subdir):
    """Assembles one branch's per-combo parquets, labels + filters, writes data/parquet/<branch>."""
    parts = sorted(glob.glob(os.path.join(data_dir, subdir, "*.parquet")))
    if not parts:
        print("\n%s branch: no per-combo parquets in %s/ -- skipping" % (branch, subdir))
        return
    _assert_parts_share_feature_contract(parts)
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
    """Syncs the training subdir's promoted-sample symlinks to ``PROMOTED_HELDOUT_SAMPLES``.

    Builds each promoted sample's parquet via ``heldout.build_sample`` under ``data_eval_43/real_43/``
    if it isn't already there (same schema as a per-combo training parquet), then symlinks it into the
    training subdir instead of re-downloading; ``assemble_branch`` picks it up via its ``*.parquet``
    glob. Imports ``heldout`` lazily here (not at module level) since ``heldout`` itself imports
    ``dataset``. ``force`` is threaded through from ``main()`` so ``--force`` also rebuilds
    promoted-sample parquets instead of silently reusing stale ones.

    Symlinks for samples no longer in the tuple are REMOVED. ``PROMOTED_HELDOUT_SAMPLES`` is a tuning
    knob, and de-promoting a sample puts it back in ``heldout.SAMPLES``; leaving its symlink behind
    would keep feeding it to training via that glob while the held-out benchmark scores it, which is
    train/eval leakage that no test catches (``dataset_tests`` compares the two Python lists, not the
    filesystem). Only symlinks are removed -- the real per-combo parquets are never touched.
    """
    import heldout
    subdir = os.path.join(data_dir, SOURCE_SUBDIR)
    for sample in PROMOTED_HELDOUT_SAMPLES:
        src = heldout.build_sample(sample, os.path.join(HERE, "data_eval_43"), force=force)
        dst = os.path.join(subdir, "%s.parquet" % sample)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    orphaned = [p for p in sorted(glob.glob(os.path.join(subdir, "*.parquet")))
                if os.path.islink(p)
                and os.path.splitext(os.path.basename(p))[0] not in PROMOTED_HELDOUT_SAMPLES]
    for path in orphaned:
        os.remove(path)
    if orphaned:
        print("  unlinked %d de-promoted sample(s) from the training pool: %s"
              % (len(orphaned), ", ".join(os.path.basename(p) for p in orphaned)))


def _upstream_source_files(data_dir):
    """Returns everything the assembled branch parquets derive from, for the freshness comparison.

    Two layers, because ``assemble_branch`` reads the per-combo parquets and those read the downloads:

    - the per-combo parquets themselves (``data_dir/<SOURCE_SUBDIR>/*.parquet``, promoted-sample
      symlinks included). Rebuilding one of these without re-running ``assemble_branch`` leaves the
      branch parquet older than its own direct input, which no JSON/TSV mtime reflects.
    - the raw EH JSON shards and truth-genotypes TSVs for the training combos (under
      ``data_dir/<SOURCE_SUBDIR>/_downloads``) and for the promoted held-out samples (under
      ``data_eval_43/real_43/_downloads/<sample>``). Only the promoted samples' download dirs are
      scanned there -- the other held-out samples never feed the training pool, so a newer download
      of one of them must not flag the pool as stale.
    """
    files = glob.glob(os.path.join(data_dir, SOURCE_SUBDIR, "*.parquet"))
    combo_dl = os.path.join(data_dir, SOURCE_SUBDIR, "_downloads")
    files += glob.glob(os.path.join(combo_dl, "**", "*.json.gz"), recursive=True)
    files += glob.glob(os.path.join(combo_dl, "**", "*.tandem_repeat_genotypes.tsv.gz"), recursive=True)
    for sample in PROMOTED_HELDOUT_SAMPLES:
        sample_dl = os.path.join(HERE, "data_eval_43", "real_43", "_downloads", sample)
        files += glob.glob(os.path.join(sample_dl, "*.json.gz"))
        files += glob.glob(os.path.join(sample_dl, "*.tandem_repeat_genotypes.tsv.gz"))
    return files


def assert_parquets_up_to_date(data_dir, branches=("quick", "full")):
    """Exits nonzero if a branch parquet is missing, older than an upstream JSON/TSV, or off-contract.

    A guard for the downstream train/report steps, covering two kinds of staleness:

    - **Refreshed inputs**: an upstream file whose mtime is newer than the branch parquet -- an EH
      JSON or truth-genotypes TSV that was re-downloaded, or a per-combo parquet that was rebuilt
      (see ``_upstream_source_files``) -- after the branch parquet was assembled.
    - **Contract drift**: a parquet missing a feature column the contract has since gained, or holding
      float32 feature columns from the old downcasting pipeline (``parquet_contract_complaint``).

    What it does NOT catch is a changed FORMULA for a column that already exists: editing
    ``eh_json.py``'s ``flanking_frac`` expression touches no upstream file's mtime and no column name,
    so a parquet built by the old code still passes. Re-run ``dataset.py --force`` yourself after
    changing how an existing feature is computed.

    Raises ``SystemExit`` with an actionable message rather than silently training on stale data.
    """
    upstream = _upstream_source_files(data_dir)
    for branch in branches:
        parquet = os.path.join(data_dir, "parquet", "%s.parquet" % branch)
        if not os.path.exists(parquet):
            sys.exit("ERROR: %s is missing -- run `python3 dataset.py --data-dir %s` to build the "
                     "parquets before training/reporting." % (parquet, data_dir))
        pq_mtime = os.path.getmtime(parquet)
        newer = sorted(f for f in upstream if os.path.getmtime(f) > pq_mtime)
        if newer:
            sys.exit("ERROR: %s is STALE -- %d upstream file(s) are newer than it (e.g. %s). "
                     "Re-run `python3 dataset.py --data-dir %s --force` to rebuild it."
                     % (parquet, len(newer), newer[0], data_dir))
        complaint = parquet_contract_complaint(parquet)
        if complaint:
            sys.exit("ERROR: %s does not match the current feature contract -- %s. Re-run "
                     "`python3 dataset.py --data-dir %s --force` to rebuild it."
                     % (parquet, complaint, data_dir))
    print("  parquet freshness OK: %s newer than all %d upstream source file(s)"
          % (", ".join(branches), len(upstream)), flush=True)


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
