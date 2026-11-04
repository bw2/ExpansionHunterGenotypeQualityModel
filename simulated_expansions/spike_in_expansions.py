"""Makes training rows for long expansions by adding simulated reads of an expanded allele to real people's reads at
known disease loci, then genotyping them the way the training runs did.

Healthy genomes have almost no alleles longer than ExpansionHunter can measure, so the training tables teach the model
that a long call without spanning reads is almost always an over-call, and it says "too long" for real expansions too.
These rows add long alleles of known size. Everything but the expanded allele is a real person's reads, so the
evidence ExpansionHunter sees around it (coverage, reads mis-mapped from elsewhere, base qualities, mates) is real.

Subcommands, run in this order:

    make-minicrams   For each person, the reads ExpansionHunter needs at the disease loci (``disease_loci.catalog.json``),
                     read from the person's CRAM (``short_read_paths.tsv``) into a small CRAM ("minicram") with
                     str_analysis.make_minicram_for_expansion_hunter.
    check-parity     Genotypes each un-spiked minicram and compares its rows (``eh_json.extract_rows``) with the
                     training run's JSON at the same loci. Any difference means a spiked minicram would not be
                     genotyped the way the training reads were.
    spike            For each person, the loci inside their high-confidence regions get simulated expanded alleles
                     (``expanded_allele_reads``) of the locus's pathogenic motif, or of one of the structures
                     published for patients' alleles there (``pathogenic_allele_structures.tsv``,
                     ``drawn_allele_structure``), in spiked samples of one genotype per locus
                     (``plan_spiked_samples``). Each of the locus's ``simulated_sizes_bp`` (those of
                     ``SIZES_BP`` no longer than the largest pathogenic allele reported there, plus the smallest and
                     largest pathogenic sizes reported there and, where the smallest is under ``SIZES_BP[0]``, one
                     between (``reported_pathogenic_sizes_bp``); all longer than the person's allele) goes once to a
                     carrier, one
                     simulated allele with the person's own, where both truth alleles are equal and the person's
                     short reads don't contradict them, and at autosomal recessive diseases' loci where both of the
                     person's alleles are shorter than ``SIZES_BP[0]`` (and the reads don't contradict them either),
                     once as the longer of a patient's two simulated alleles (``spikeable_loci``,
                     ``simulated_genotypes``). The person's read pairs that lie near the locus are thinned by half for
                     a carrier (all of them for a patient, and at a male's chrX and chrY loci;
                     ``read_pairs_within_windows``), the simulated pairs are aligned to the whole genome with bwa and
                     merged in, and the sample's truth (the person's allele and the simulated one, or the two
                     simulated ones) is written next to its reads.
    downsample       For every spiked sample of another work folder (``--source-work-dir``), a copy downsampled to a
                     clinical genome's coverage and a matched control of the person's own reads downsampled alike,
                     with the person's own truth (``downsample_spiked_samples``), as this work folder's spiked samples.
    genotype         Genotypes the spiked samples.
    tables           Turns each spiked sample's ExpansionHunter JSON and truth into a per-source parquet the way the
                     training tables are built (``dataset.extract_rows_and_join_truth``).

Genotyping runs on Hail Batch in the training runs' ExpansionHunter image (``genotype_on_hail_batch``) with their
settings: optimized-streaming mode, ``--max-depth 150``, motif composition output for all loci, their UCSC hg38
reference, and the person's sex, read from the calls of the person's training table (most chrX and chrY loci get one
allele in males).

``simulated_expansions/make_spiked_samples.sh`` runs every subcommand in order with the settings of the spiked samples
the shipped model was trained on, in two work folders: SPIKED for the spiked samples and LOWER_COVERAGE for their
lower-coverage copies (``work_spiked_samples`` and ``work_lower_coverage_copies_of_spiked_samples`` there). One at a
time:
    python3 simulated_expansions/spike_in_expansions.py make-minicrams --work-dir SPIKED --reference-fasta HG38_FA \
        --billing-project PROJECT
    python3 simulated_expansions/spike_in_expansions.py check-parity --work-dir SPIKED --training-tables-dir TABLES \
        --billing-project PROJECT --gcs-dir gs://...
    python3 simulated_expansions/spike_in_expansions.py spike --work-dir SPIKED --training-tables-dir TABLES \
        --reference-fasta HG38_FA --bwa BWA
    python3 simulated_expansions/spike_in_expansions.py genotype --work-dir SPIKED --billing-project PROJECT \
        --gcs-dir gs://...
    python3 simulated_expansions/spike_in_expansions.py tables --work-dir SPIKED
    python3 simulated_expansions/spike_in_expansions.py downsample --source-work-dir SPIKED --work-dir LOWER_COVERAGE \
        --reference-fasta HG38_FA
    python3 simulated_expansions/spike_in_expansions.py genotype --work-dir LOWER_COVERAGE --billing-project PROJECT \
        --gcs-dir gs://...
    python3 simulated_expansions/spike_in_expansions.py tables --work-dir LOWER_COVERAGE
"""

import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import google.auth
import google.auth.transport.requests
import pandas as pd
import pyfaidx
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import dataset  # noqa: E402
import eh_json  # noqa: E402
from simulated_expansions import expanded_allele_reads as sim  # noqa: E402
from simulated_expansions.build_disease_loci_table import motif_family, rotations  # noqa: E402

# The parent of the calibration's folder (calibrate_simulated_reads.CALIBRATION_DIR), which holds its truth downloads
# and, by default, the bwa index.
WORK_DIR = os.path.join(HERE, "work")
# Where the bwa index of the hg38 the people's reads were aligned to is expected (see the README).
DEFAULT_BWA_INDEX_PREFIX = os.path.join(WORK_DIR, "bwa_index", "Homo_sapiens_assembly38.fasta.64")
DISEASE_LOCI_CATALOG = os.path.join(HERE, "disease_loci.catalog.json")
SHORT_READ_PATHS_TSV = os.path.join(REPO, "short_read_paths.tsv")
# Training people of both sexes whose own long alleles calibrate_simulated_reads.py compares real and simulated reads at,
# none of them a person train.py holds out for calibration.
PILOT_PEOPLE = ("HG01106", "NA12878", "HG03492", "HG01928", "HG02080", "HG01258")
# The training people the spiked samples are made from: every dataset.PROMOTED_HELDOUT_SAMPLES person except the 5 that
# train._calibration_people holds out (HG00621, HG01150, HG03669, HG03688, HG03804), since it refuses a table where a
# held-out person has spiked samples. Sorted, as the spiked samples were made in this order.
SPIKED_PEOPLE = ("HG00408", "HG00658", "HG00706", "HG01074", "HG01081", "HG01106", "HG01192", "HG01258", "HG01346",
                 "HG01433", "HG01928", "HG01934", "HG01943", "HG01960", "HG02055", "HG02071", "HG02080", "HG02083",
                 "HG02129", "HG02135", "HG02258", "HG02273", "HG02451", "HG02615", "HG02622", "HG02647", "HG02841",
                 "HG02965", "HG03041", "HG03125", "HG03239", "HG03453", "HG03492", "HG03710", "HG03816", "HG03831",
                 "HG03927", "HG04115", "HG04204", "HG04228", "NA12329", "NA12878", "NA18906", "NA19983", "NA20129")
TRAINING_IMAGE = ("weisburd/str-analysis-with-expansion-hunter@sha256:"
                  "9e3090a2a69506a513a42ec49d1b9f83f1f0302c828741c3dfde001c629435d8")
TRAINING_REFERENCE_FASTA = "gs://str-truth-set/hg38/ref/hg38.fa"
EXPANSION_HUNTER_ARGS = ("--analysis-mode", "optimized-streaming", "--max-depth", "150",
                         "--output-motif-composition", "all-loci", "--dont-output-consensus-sequences",
                         "--compress-output-files", "--threads", "4")
DISEASE_LOCI_TSV = os.path.join(HERE, "disease_loci.tsv")
# Published structures of patients' expanded alleles at loci where they are not a pure tract of the expanded motif
# (interruptions, blocks of other motifs), with the share of alleles that carry each (see the README).
PATHOGENIC_ALLELE_STRUCTURES_TSV = os.path.join(HERE, "pathogenic_allele_structures.tsv")
# Added to the name of a sample's output JSON that an earlier run downloaded, where the batch just run wrote none for
# it (genotype_on_hail_batch), so the stale output is kept but not used.
OUTPUT_OF_AN_EARLIER_RUN_SUFFIX = ".of_an_earlier_run"
# Expanded allele sizes, from about the read length to well past the fragment length, where ExpansionHunter's calls
# stop growing with the allele, and on to 24 kb, about the largest pathogenic allele reported at C9orf72 (CNBP's reach
# 44 kb), as the patients of loci such as RFC1, C9orf72, CNBP, DMPK, NOP56 and ATXN10 mostly carry alleles of several
# kb. A locus's sizes are these, except those not longer than the person's allele or longer than the largest
# pathogenic allele reported there, plus the smallest and largest pathogenic sizes reported there and, where the
# smallest is shorter than the first one, one between (reported_pathogenic_sizes_bp, simulated_sizes_bp). Each
# goes into one spiked sample per genotype (simulated_genotypes): once as a carrier's simulated allele, and at
# autosomal recessive diseases' loci once more as the longer of a patient's two.
SIZES_BP = (150, 225, 300, 450, 600, 900, 1500, 3000, 4500, 6000, 9000, 12000, 18000, 24000)
# Loci of autosomal recessive diseases, whose patients carry two expanded alleles (inheritance in disease_loci.tsv). An
# X-linked recessive disease's patients are mostly males, with one allele, which the carriers at a male's chrX loci
# already simulate.
AUTOSOMAL_RECESSIVE_INHERITANCE = "AR"
# Reference bases on each side of a simulated allele. The person's own read pairs within this distance of the locus
# are thinned to make room for it, and ExpansionHunter reads 1 kb on each side, so the coverage it sees stays even.
FLANK_BP = 2000
# Each simulated allele loses fragments over its repeat with probability 1 - c^(f * e) (expanded_allele_reads
# .fragment_kept), e drawn from this range. c is fitted to the read pairs at the calibration alleles
# (calibrate_simulated_reads.py), mostly under 300 bp, where e = 1 matches ExpansionHunter's real and simulated rows;
# at the calibration alleles over 300 bp, the real rows had 0.44-0.70 times the simulated rows' in-repeat reads with
# e = 1, which e of about 2-3 matches. Other library preps can lose more or fewer.
FRAGMENT_KEEP_EXPONENT_RANGE = (1.0, 3.0)
# Coverages the downsampled copies of spiked samples are drawn from (downsample_spiked_samples): those of clinical
# genomes, below the training people's 38-50x.
DOWNSAMPLED_COVERAGE_RANGE = (15.0, 30.0)
# Added to a spiked sample's number to name its downsampled copy and its matched control, so they keep the names of
# spiked samples (dataset.is_spiked_sample) without clashing with the spike step's.
DOWNSAMPLED_COPY_NUMBER_OFFSET = 100000
MATCHED_CONTROL_NUMBER_OFFSET = 200000
# Where the person's fragment start rate is measured, in bp from each end of the repeat (inside the minicram's 1 kb
# window, clear of the repeat itself).
COVERAGE_WINDOW_BP = (100, 1000)
TRUTH_COLUMNS = dataset.TRUTH_GENOTYPES_TSV_COLUMNS


def run(command, env=None):
    """Runs a command, printing it, and raises if it fails."""
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def run_gcloud_storage_copies_in_parallel(commands, max_parallel=8):
    """Runs the gcloud copy commands (``run``), up to ``max_parallel`` at a time, as each of hundreds of small copies to
    or from Google Cloud Storage spends most of its time starting up; raises if any fails. Each gcloud copy runs in one
    process (``storage/process_count`` and ``storage/thread_count`` 1, where gcloud copies the files one after the
    other): otherwise each starts extra worker processes, and many copies at once fill the machine's memory to copy a
    few small files."""
    env = dict(os.environ, CLOUDSDK_STORAGE_PROCESS_COUNT="1", CLOUDSDK_STORAGE_THREAD_COUNT="1")
    with concurrent.futures.ThreadPoolExecutor(max_parallel) as pool:
        list(pool.map(lambda command: run(command, env), commands))


def reads_paths(person):
    """``(CRAM path, index path)`` of the person in ``short_read_paths.tsv``."""
    table = pd.read_csv(SHORT_READ_PATHS_TSV, sep="\t").set_index("sample_label")
    return table.at[person, "reads_path"], table.at[person, "reads_index_path"]


def sex_of(person, tables_dir):
    """"male" when most of the person's chrX and chrY loci in the training table got one allele, else "female"."""
    df = pd.read_parquet(os.path.join(tables_dir, "%s.parquet" % person), columns=["locus_id", "n_alleles"])
    sex_chromosome_rows = df[df["locus_id"].str.match(r"^[XY]-")]
    return "male" if len(sex_chromosome_rows) and (sex_chromosome_rows["n_alleles"] == 1).mean() > 0.5 else "female"


def minicram_path(work_dir, person):
    return os.path.join(work_dir, "minicrams", "%s.cram" % person)


def make_minicram(person, work_dir, reference_fasta, catalog, billing_project):
    """Writes the person's minicram of the catalog's loci, unless one newer than the catalog exists (an older one may
    lack reads at loci the catalog has since gained)."""
    out = minicram_path(work_dir, person)
    if os.path.exists(out + ".crai") and os.path.getmtime(out) > os.path.getmtime(catalog):
        print("%s: %s exists" % (person, out))
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cram, crai = reads_paths(person)
    # htslib's reads of gs:// files need an access token, newly issued (valid for an hour) because a cached one could
    # expire while the reads are being fetched. The CRAMs' requester-pays bucket is billed to --gcloud-project.
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    env = dict(os.environ, GCS_OAUTH_TOKEN=credentials.token, GCS_REQUESTER_PAYS_PROJECT=billing_project)
    run([sys.executable, "-u", "-m", "str_analysis.make_minicram_for_expansion_hunter", "--reference-fasta",
         reference_fasta, "--variant-catalog", catalog, "--crai-index-path", crai, "--gcloud-project",
         billing_project, "--output-cram", out, cram], env=env)
    run(["samtools", "index", out])


def sample_names_with_an_output_json_in_gcs(gcs_dir, updated_after=None):
    """The names of the samples whose ``expansion_hunter.json.gz`` is in ``gcs_dir``, those written at or after
    ``updated_after`` (a ``datetime``) where it is given, so one an earlier run left there is left out. Raises if the
    listing fails, since an empty listing would otherwise read as "no sample was genotyped"."""
    listed = subprocess.run(["gcloud", "storage", "ls", "-l", "%s/**expansion_hunter.json.gz" % gcs_dir],
                            capture_output=True, text=True)
    if listed.returncode != 0:
        if "matched no objects" in listed.stderr:
            return set()
        raise RuntimeError("listing %s failed: %s" % (gcs_dir, listed.stderr.strip()))
    names = set()
    for line in listed.stdout.splitlines():
        # "<size> <time the object was written> <url>", and a closing TOTAL line.
        fields = line.split()
        if len(fields) < 3 or not fields[-1].endswith("/expansion_hunter.json.gz"):
            continue
        if updated_after is None or datetime.datetime.fromisoformat(fields[-2]) >= updated_after:
            names.add(fields[-1][len(gcs_dir) + 1:].split("/")[0])
    return names


def genotype_on_hail_batch(samples, catalog, billing_project, gcs_dir, batch_name, only_missing_outputs=False):
    """Genotypes each sample's reads with the training runs' ExpansionHunter image and settings on Hail Batch, and
    downloads each output JSON.

    Every output this batch wrote is downloaded, even when a job failed, and the samples with none are named in the
    error: one job of thousands failing on a transient cloud error would otherwise throw away the whole batch's
    results. Genotyping again with ``only_missing_outputs`` then does just those samples. Outputs are downloaded only
    once the batch ends, so a run stopped while it waits genotypes its samples again.

    ``only_missing_outputs`` leaves out every sample that has an ``out_json``, whatever made it, so it is for
    finishing a batch that failed part way and not for picking up a change: an output stays as it is even if the
    reads, the catalog, the image, the ExpansionHunter settings, the reference or the sample's sex have changed since.
    Delete the outputs to be remade, or leave it off to genotype every sample.

    The training runs' image, not a local build: a macOS build of the same commit gives slightly different read
    depths, strand bias and indel sizes at loci the full genotyper calls, which is where every simulated expansion goes.

    Args:
        samples: Dicts with ``name``, ``reads`` (a local BAM or CRAM with its .bai or .crai next to it), ``sex`` and
            ``out_json`` (the local path to download the output JSON to).
        catalog: The local catalog JSON.
        billing_project: The Hail Batch billing project.
        gcs_dir: A gs:// folder for the uploaded reads and the outputs.
        batch_name: The batch's name.
        only_missing_outputs: Genotype just the samples that have no ``out_json`` (see above).
    """
    import hailtop.batch as hb  # local: only this step needs Hail
    if only_missing_outputs:
        done = [sample for sample in samples if os.path.exists(sample["out_json"])]
        samples = [sample for sample in samples if not os.path.exists(sample["out_json"])]
        print("%d of %d samples have an output JSON already and are left out" % (len(done), len(done) + len(samples)),
              flush=True)
        if not samples:
            return
    run(["gcloud", "storage", "cp", catalog, "%s/catalog.json" % gcs_dir])
    batch = hb.Batch(backend=hb.ServiceBackend(billing_project=billing_project, remote_tmpdir=gcs_dir + "/tmp/"),
                     name=batch_name)
    catalog_input = batch.read_input("%s/catalog.json" % gcs_dir)
    # Keyed "fa.fai" so the index lands at <root>.fa.fai, where htslib looks for it, as the reads' index does below.
    reference = batch.read_input_group(**{"fa": TRAINING_REFERENCE_FASTA, "fa.fai": TRAINING_REFERENCE_FASTA + ".fai"})
    index_extension_of = {sample["name"]: "bai" if sample["reads"].endswith(".bam") else "crai" for sample in samples}
    run_gcloud_storage_copies_in_parallel([["gcloud", "storage", "cp", sample["reads"],
                                            "%s.%s" % (sample["reads"], index_extension_of[sample["name"]]),
                                            "%s/%s/" % (gcs_dir, sample["name"])] for sample in samples])
    for sample in samples:
        extension = "bam" if sample["reads"].endswith(".bam") else "cram"
        index_extension = index_extension_of[sample["name"]]
        remote = "%s/%s/%s" % (gcs_dir, sample["name"], os.path.basename(sample["reads"]))
        reads = batch.read_input_group(**{extension: remote,
                                          "%s.%s" % (extension, index_extension): "%s.%s" % (remote, index_extension)})
        job = batch.new_job(sample["name"])
        job.image(TRAINING_IMAGE)
        job.cpu(1)
        job.storage("10Gi")
        job.command(" ".join(["ExpansionHunter", "--reads", str(reads[extension]), "--reference", str(reference.fa),
                              "--variant-catalog", str(catalog_input), "--sex", sample["sex"], "--output-prefix", "out",
                              *EXPANSION_HUNTER_ARGS]) + " && cp out.json.gz %s" % job.ofile)
        batch.write_output(job.ofile, "%s/%s/expansion_hunter.json.gz" % (gcs_dir, sample["name"]))
    # hailtop reports a failed batch without raising.
    status = batch.run(wait=True).status()
    # Only the outputs this batch wrote count, so one a failed job's sample has from an earlier run is not taken for
    # its result. Hail always reports when a batch was created; without it there is no way to tell the two apart.
    written = sample_names_with_an_output_json_in_gcs(gcs_dir,
                                                      datetime.datetime.fromisoformat(status["time_created"]))
    wrote = [sample for sample in samples if sample["name"] in written]
    missing = [sample for sample in samples if sample["name"] not in written]
    for sample in wrote:
        os.makedirs(os.path.dirname(sample["out_json"]), exist_ok=True)
    run_gcloud_storage_copies_in_parallel([["gcloud", "storage", "cp",
                                            "%s/%s/expansion_hunter.json.gz" % (gcs_dir, sample["name"]),
                                            sample["out_json"]] for sample in wrote])
    # A sample this batch wrote nothing for must not keep an output of an earlier run's, or genotyping again for the
    # missing outputs would leave it out and that older output would be used as its result. Moved, not deleted.
    moved = []
    for sample in missing:
        if os.path.exists(sample["out_json"]):
            os.replace(sample["out_json"], sample["out_json"] + OUTPUT_OF_AN_EARLIER_RUN_SUFFIX)
            moved.append(sample["name"])
    if missing:
        raise RuntimeError("%s ended in state %s; %d of %d samples have no output JSON, the first few: %s. The others "
                           "were downloaded, so only these need genotyping again (the genotype step does just them "
                           "with --only-missing-outputs). %d of them had an output of an earlier run's, moved aside "
                           "with the suffix %s."
                           % (batch_name, status["state"], len(missing), len(samples),
                              ", ".join(sample["name"] for sample in missing[:5]), len(moved),
                              OUTPUT_OF_AN_EARLIER_RUN_SUFFIX))


def rows_at_loci(json_path, locus_ids):
    """``{(locus_id, allele_rank): row}`` of an EH JSON's rows (``eh_json.extract_rows``) at ``locus_ids``, without
    the sample id."""
    rows = {}
    for row in eh_json.extract_rows(json_path):
        if row["locus_id"] in locus_ids:
            row.pop("sample_id", None)
            rows[(row["locus_id"], row["allele_rank"])] = row
    return rows


def differences(training_rows, minicram_rows):
    """Lists ``(locus_id, allele_rank, field, training value, minicram value)`` wherever the two runs' rows differ,
    including rows only one of them has (field "row")."""
    out = []
    for key in sorted(set(training_rows) | set(minicram_rows)):
        a, b = training_rows.get(key), minicram_rows.get(key)
        if a is None or b is None:
            out.append((*key, "row", a is not None, b is not None))
            continue
        for field in sorted(set(a) | set(b)):
            if a.get(field) != b.get(field) and not (pd.isna(a.get(field)) and pd.isna(b.get(field))):
                out.append((*key, field, a.get(field), b.get(field)))
    return out


def check_parity(people, work_dir, catalog, billing_project, gcs_dir, training_tables_dir):
    """Genotypes each person's un-spiked minicram (``genotype_on_hail_batch``) and compares its rows with the training
    run's at the catalog's loci, writing each person's differences to ``parity/<person>.differences.tsv``.

    Returns:
        The people whose rows differ.
    """
    parity_dir = os.path.join(work_dir, "parity")
    genotype_on_hail_batch(
        [{"name": "parity_%s" % person, "reads": minicram_path(work_dir, person),
          "sex": sex_of(person, training_tables_dir),
          "out_json": os.path.join(parity_dir, "%s.json.gz" % person)} for person in people],
        catalog, billing_project, gcs_dir, "simulated_expansions: parity of the un-spiked minicrams")
    with open(catalog) as f:
        locus_ids = {entry["LocusId"] for entry in json.load(f)}
    failed = []
    for person in people:
        # The training run's JSON, from the run the training tables are built from (dataset.EH_RESULTS_ROOT), kept in a
        # folder per run so a new run is downloaded rather than an older one reused.
        training_json = os.path.join(work_dir, "training_eh_json",
                                     os.path.basename(dataset.EH_RESULTS_ROOT.rstrip("/")), "%s.json.gz" % person)
        if not os.path.exists(training_json):
            os.makedirs(os.path.dirname(training_json), exist_ok=True)
            run(["gcloud", "storage", "cp", dataset._list_json_inputs(person)[0], training_json])
        found = differences(rows_at_loci(training_json, locus_ids),
                            rows_at_loci(os.path.join(parity_dir, "%s.json.gz" % person), locus_ids))
        out = os.path.join(parity_dir, "%s.differences.tsv" % person)
        pd.DataFrame(found, columns=["locus_id", "allele_rank", "field", "training_run", "minicram_run"]).to_csv(
            out, sep="\t", index=False)
        print("%s: %d differences at %d loci (%s)" % (person, len(found), len(locus_ids), out))
        if found:
            failed.append(person)
    return failed


def truth_download_dir(work_dir, person):
    """Where the person's truth TSV and BED are downloaded to: a folder per truth version (the last part of
    ``dataset.TRUTH_GENOTYPES_ROOT``), so a new truth is downloaded rather than an older one reused."""
    return os.path.join(work_dir, "truth", os.path.basename(dataset.TRUTH_GENOTYPES_ROOT.rstrip("/")), person)


def person_truth_at_loci(person, locus_ids, work_dir, columns=TRUTH_COLUMNS):
    """The person's truth rows (``columns`` of the truth TSV, which must include ``TRUTH_COLUMNS``) at ``locus_ids``,
    indexed by LocusId (without "chr"), with ``confident``: the locus lies wholly inside one of the person's DipCall
    high-confidence regions. Downloads the truth to ``truth_download_dir`` unless it is there."""
    download_dir = truth_download_dir(work_dir, person)
    sources = dataset._truth_sources(person, download_dir)
    tsv_path, bed_path = dataset._download([remote for _, remote, _ in sources], download_dir)
    chunks = []
    for chunk in pd.read_csv(tsv_path, sep="\t", usecols=columns, dtype={"LocusId": str, "Chrom": str},
                             chunksize=500_000):
        chunk["LocusId"] = chunk["LocusId"].str.replace(r"^chr", "", regex=True)
        chunks.append(chunk[chunk["LocusId"].isin(locus_ids)])
    truth = pd.concat(chunks).drop_duplicates("LocusId").set_index("LocusId")
    truth["confident"] = dataset._inside_high_confidence_regions(truth["Chrom"], truth["Start0Based"], truth["End"],
                                                                 bed_path)
    return truth


def is_hemizygous(chrom, sex):
    return sex == "male" and chrom in ("chrX", "chrY")


def loci_with_truth_contradicted_by_short_reads(person, tables_dir):
    """The loci where the person's training table marks the truth as contradicted by their own short reads
    (``check_truth_genotypes_against_short_reads.py --mark``)."""
    df = pd.read_parquet(os.path.join(tables_dir, "%s.parquet" % person),
                         columns=["locus_id", dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN])
    return set(df.loc[df[dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN].astype(bool), "locus_id"])


def spikeable_loci(loci_table, truth, contradicted_locus_ids):
    """The disease loci inside the person's high-confidence regions where simulated alleles can be spiked, with
    ``person_allele_repeats`` (the person's longer truth allele) and two flags:

    - ``carriers_simulated``: a carrier's simulated allele has a known partner, as both truth alleles are equal (a
      male's chrX and chrY loci have one allele, which the spike replaces), and the person's short reads don't
      contradict that truth (``contradicted_locus_ids``): the allele the person keeps would carry a label the training
      tables leave out, and the person may not be homozygous there. The short reads were checked only at the loci of
      the person's training table where a call disagrees with the truth; a locus missing from the table (mostly
      hom-ref loci it didn't sample) was not checked, and its truth is taken as it is.
    - ``patients_simulated``: an autosomal recessive disease's locus (``AUTOSOMAL_RECESSIVE_INHERITANCE``) where both of
      the person's alleles are shorter than ``SIZES_BP[0]``, so nearly all of their reads lie near the locus and make
      room for a patient's two simulated alleles, whatever the person's genotype (the reads of a longer allele with a
      mate elsewhere would stay), and where the person's short reads don't contradict the truth, which may then miss a
      longer allele.
    """
    keep = []
    for locus in loci_table.to_dict("records"):
        if locus["locus_id"] not in truth.index:
            continue
        row = truth.loc[locus["locus_id"]]
        short, long_ = pd.to_numeric(row["NumRepeatsShortAllele"], errors="coerce"), pd.to_numeric(
            row["NumRepeatsLongAllele"], errors="coerce")
        if not row["confident"] or pd.isna(short) or pd.isna(long_):
            continue
        truth_not_contradicted = locus["locus_id"] not in contradicted_locus_ids
        carriers = short == long_ and truth_not_contradicted
        patients = (locus["inheritance"] == AUTOSOMAL_RECESSIVE_INHERITANCE and truth_not_contradicted
                    and long_ * len(locus["reference_motif"]) < SIZES_BP[0])
        if carriers or patients:
            keep.append({**locus, "person_allele_repeats": int(long_), "carriers_simulated": carriers,
                         "patients_simulated": patients})
    return pd.DataFrame(keep)


def is_dropped(label, read_name, share):
    """Whether the person's read pair ``read_name`` makes room for the simulated allele in sample ``label``: a fixed
    pseudo-random choice with probability ``share`` (from SHA-256, as CRC32 is too uneven over similar read names)."""
    digest = hashlib.sha256(("%s\t%s" % (label, read_name)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < share


def simulated_allele_purity(expanded_motif, reference_motif):
    """The share of the simulated allele's bases that match the reference motif's repeat in its best rotation, scored
    as the truth tables score purity: an N never matches, as the allele holds a real base there (so a GCN repeat is 2/3
    pure, like the person's own allele at the same locus)."""
    return max(sum(a == b != "N" for a, b in zip(rotation, reference_motif.upper()))
               for rotation in rotations(expanded_motif.upper())) / len(reference_motif)


def load_pathogenic_allele_structures(tsv_path=PATHOGENIC_ALLELE_STRUCTURES_TSV):
    """``{locus_id: [row, ...]}`` of the published structures of patients' expanded alleles
    (``pathogenic_allele_structures.tsv``)."""
    structures = {}
    for row in pd.read_csv(tsv_path, sep="\t").to_dict("records"):
        structures.setdefault(row["locus_id"], []).append(row)
    return structures


def drawn_block_count(bounds, rng):
    """A block's copies drawn from ``bounds`` (``LOW-HIGH``, or ``LOW-MEDIAN-HIGH`` for a published median): log-
    uniformly over LOW-HIGH, as such counts cluster at the low end, or half the time log-uniformly over LOW-MEDIAN and
    half the time over MEDIAN-HIGH, so the median is the published one."""
    points = [math.log(float(bound)) for bound in bounds.split("-")]
    if len(points) == 3:
        points = points[:2] if rng.random() < 0.5 else points[1:]
    return math.exp(rng.uniform(*points))


def structure_blocks(blocks_on_plus_strand, copies, rng):
    """The blocks ``[(motif, copies), ...]``, in + strand order, of an allele of ``copies`` motif copies laid out by
    ``blocks_on_plus_strand`` (``;``-separated: ``MOTIF:count=BOUNDS`` copies, ``drawn_block_count``;
    ``MOTIF:share=LOW-HIGH`` of the allele, drawn uniformly; one ``MOTIF:rest``, the copies left; and
    ``MOTIF:added_count=BOUNDS`` copies, drawn like a count, added to the allele's ``copies``: the blocks of the
    reference motif around an inserted motif, at loci whose reported pathogenic sizes count only the inserted motif).
    The counts together take less than half the allele, scaled down to fit a short one, so the rest block stays longer
    than them. Blocks of no copies are left out."""
    drawn = []
    for block in blocks_on_plus_strand.split(";"):
        motif, size = block.split(":")
        if size == "rest":
            drawn.append((motif, "rest", None))
            continue
        kind, bounds = size.split("=")
        if kind in ("count", "added_count"):
            drawn.append((motif, kind, drawn_block_count(bounds, rng)))
        else:
            low, high = (float(bound) for bound in bounds.split("-"))
            drawn.append((motif, kind, rng.uniform(low, high) * copies))
    sized = [(motif, kind, None if n is None else int(round(n))) for motif, kind, n in drawn]
    counted, most_counted = sum(n for _, kind, n in sized if kind == "count"), (copies - 1) // 2
    if counted > most_counted:
        sized = [(motif, kind, n * most_counted // counted if kind == "count" else n) for motif, kind, n in sized]
    rest = copies - sum(n for _, kind, n in sized if kind not in ("rest", "added_count"))
    return [(motif, rest if kind == "rest" else n) for motif, kind, n in sized if (rest if kind == "rest" else n) > 0]


def drawn_allele_structure(structures, expanded_motif, copies, rng):
    """``(structure name, blocks)`` of a simulated allele of ``copies`` copies: one of the locus's published structures
    (``structures``, rows of ``load_pathogenic_allele_structures``) drawn by its ``share_of_alleles``, its blocks from
    ``structure_blocks``, or for the share left, a pure tract of the expanded motif."""
    draw = rng.random()
    for structure in structures:
        if draw < structure["share_of_alleles"]:
            return structure["structure"], structure_blocks(structure["blocks_on_plus_strand"], copies, rng)
        draw -= structure["share_of_alleles"]
    return "pure", [(expanded_motif, copies)]


def repeat_of_blocks(blocks):
    """The ``expanded_allele_reads.Repeat`` of an allele of ``blocks`` (``[(motif, copies), ...]``) between
    ``FLANK_BP`` flanks: its blocks' spans, each of which sets the loss of the fragments over it, and the motif with
    the most copies, which sets the reads' errors inside it (every block motif has one length)."""
    spans, start, copies_by_motif = [], FLANK_BP, collections.Counter()
    for motif, n in blocks:
        spans.append((start, start + n * len(motif), motif))
        start += n * len(motif)
        copies_by_motif[motif] += n
    return sim.Repeat(FLANK_BP, start, copies_by_motif.most_common(1)[0][0], spans)


def blocks_purity(blocks, reference_motif):
    """The share of an allele's bases (``blocks``: ``[(motif, copies), ...]``) that match the reference motif's repeat,
    each block scored by ``simulated_allele_purity``."""
    return (sum(n * len(motif) * simulated_allele_purity(motif, reference_motif) for motif, n in blocks)
            / sum(n * len(motif) for motif, n in blocks))


def loci_closer_than_two_flanks(earlier, later):
    """Whether two loci (dicts with chrom, start_0based, end; ``later`` starting at or after ``earlier``) are closer
    than ``2 * FLANK_BP``, so simulated alleles at both would cover the same reads (e.g. ARX_1 and ARX_2)."""
    return earlier["chrom"] == later["chrom"] and later["start_0based"] - earlier["end"] < 2 * FLANK_BP


def clusters_of_nearby_loci(loci):
    """Groups the loci (dicts with chrom, start_0based, end) into clusters of loci closer than ``2 * FLANK_BP``
    (``loci_closer_than_two_flanks``)."""
    clusters = []
    for locus in sorted(loci, key=lambda l: (l["chrom"], l["start_0based"])):
        if clusters and loci_closer_than_two_flanks(clusters[-1][-1], locus):
            clusters[-1].append(locus)
        else:
            clusters.append([locus])
    return clusters


def groups_of_loci_kept_apart(loci):
    """Groups the loci (dicts with chrom, start_0based, end and expanded_motif) that are never spiked in the same
    sample: loci closer than ``2 * FLANK_BP`` (``loci_closer_than_two_flanks``), and loci whose expanded motifs are one
    repeat (``motif_family``), as bwa puts one expansion's in-repeat reads on the other's repeat (every CAG
    expansion's on TCF4's CTG repeat). A group holds the loci connected through either; the groups and their members
    are in genome order."""
    loci = sorted(loci, key=lambda l: (l["chrom"], l["start_0based"]))
    group_of = list(range(len(loci)))

    def root(i):
        while group_of[i] != i:
            i = group_of[i]
        return i

    for i, a in enumerate(loci):
        for j in range(i + 1, len(loci)):
            b = loci[j]
            if loci_closer_than_two_flanks(a, b) or motif_family(a["expanded_motif"]) == motif_family(
                    b["expanded_motif"]):
                group_of[root(j)] = root(i)
    groups = {}
    for i, locus in enumerate(loci):
        groups.setdefault(root(i), []).append(locus)
    return list(groups.values())


def reported_pathogenic_sizes_bp(locus):
    """The sizes to simulate at the locus besides those of ``SIZES_BP``, from the smallest and largest pathogenic
    alleles reported there (``smallest_reported_pathogenic_bp``, ``largest_reported_pathogenic_bp``), each rounded to
    whole copies of its expanded motif: the smallest; the largest, where it is shorter than the last of ``SIZES_BP``,
    which otherwise reaches past it; and, where the smallest is shorter than the first of ``SIZES_BP``, one halfway
    between it and the shorter of the largest and that first size. So patients' own reported sizes are simulated even
    where they fall between two of ``SIZES_BP`` (TNRC6A's and YEATS2's single reported sizes, 5,500 and 5,000 bp) or
    below all of them (most polyalanine loci). An empty list where either is not reported."""
    smallest, largest = locus["smallest_reported_pathogenic_bp"], locus["largest_reported_pathogenic_bp"]
    if pd.isna(smallest) or pd.isna(largest):
        return []
    sizes = [smallest] + ([largest] if largest < SIZES_BP[-1] else [])
    if smallest < SIZES_BP[0]:
        sizes.append((smallest + min(largest, SIZES_BP[0])) / 2)
    motif_bp = len(locus["expanded_motif"])
    return sorted({int(round(size_bp / motif_bp)) * motif_bp for size_bp in sizes})


def simulated_sizes_bp(locus):
    """The sizes the locus's simulated alleles take, in whole copies of its expanded motif (rounded down) longer than
    the person's allele, each once: those of ``SIZES_BP`` no longer than the largest pathogenic allele reported at the
    locus (``largest_reported_pathogenic_bp``; none where none is reported), as an expansion that long is never seen
    there, and the ``reported_pathogenic_sizes_bp``. Two sizes of the same whole copies are one size (PRNP's 24 bp
    motif makes both 144 and 150 bp six copies). At a locus whose patients carry an inserted motif between blocks of
    the reference one, the size is of the insertion alone, as the reported sizes are, and its blocks are added to it
    (``structure_blocks``)."""
    motif_bp = len(locus["expanded_motif"])
    candidates = [size_bp for size_bp in SIZES_BP if size_bp <= locus["largest_reported_pathogenic_bp"]]
    copies = {size_bp // motif_bp for size_bp in candidates + reported_pathogenic_sizes_bp(locus)}
    return [n * motif_bp for n in sorted(copies) if n > locus["person_allele_repeats"]]


def simulated_genotypes(person, locus):
    """The simulated alleles' sizes at the locus (a row of ``spikeable_loci``), one tuple per spiked sample: each of
    ``simulated_sizes_bp`` once as a carrier's only simulated allele, which pairs with the person's own (a male's chrX
    and chrY loci have no other), where ``carriers_simulated``, and once as the longer of a patient's two simulated
    alleles, which replace both of the person's, where ``patients_simulated``. The patient's shorter allele is drawn
    from the sizes no longer than the longer one, and no shorter than the smallest pathogenic allele reported at the
    locus (``smallest_reported_pathogenic_bp``) where the longer one is that long, as both of a patient's alleles are
    pathogenic (seeded by the person, the locus and the size, so the plan is the same on every run)."""
    sizes = simulated_sizes_bp(locus)
    genotypes = [(size_bp,) for size_bp in sizes] if locus["carriers_simulated"] else []
    if locus["patients_simulated"]:
        smallest_pathogenic_bp = locus["smallest_reported_pathogenic_bp"]
        for i, size_bp in enumerate(sizes):
            shorter_sizes = ([s for s in sizes[:i + 1] if s >= smallest_pathogenic_bp]
                             if size_bp >= smallest_pathogenic_bp else sizes[:i + 1])
            genotypes.append((size_bp, sim.seeded_rng(person, locus["locus_id"], size_bp).choice(shorter_sizes)))
    return genotypes


def plan_spiked_samples(person, loci):
    """Lists ``(label, locus row, simulated allele sizes)`` for the person's spiked samples, each locus's
    ``simulated_genotypes`` once.

    A sample never holds two loci of a group of ``groups_of_loci_kept_apart``: each group lists its members'
    genotypes one member after the other, and sample k gets the group's genotype ``(k + i) % n`` (i: the group's index,
    so a sample mixes sizes; n: the length of its list) for every group with more than k genotypes.
    """
    genotypes_by_group = [[(locus, sizes) for locus in group for sizes in simulated_genotypes(person, locus)]
                          for group in groups_of_loci_kept_apart([locus for locus in loci.to_dict("records")
                                                                  if simulated_sizes_bp(locus)])]
    plan = []
    for k in range(max(map(len, genotypes_by_group), default=0)):
        for i, genotypes in enumerate(genotypes_by_group):
            if k < len(genotypes):
                locus, sizes = genotypes[(k + i) % len(genotypes)]
                plan.append(("%s_spiked_%d" % (person, k), locus, sizes))
    return plan


def windows_around_loci(loci):
    """``[(chrom, start_0based, end), ...]``: each locus (a dict with chrom, start_0based, end) with ``FLANK_BP`` on
    either side, where the person's read pairs make room for simulated ones."""
    return [(locus["chrom"], locus["start_0based"] - FLANK_BP, locus["end"] + FLANK_BP) for locus in loci]


def read_pairs_within_windows(minicram, reference_fasta, windows):
    """``{(chrom, start): names of the read pairs that lie in the window}`` for each window of ``windows_around_loci``:
    pairs whose reads in the window have a mate mapped on the same chromosome that starts in the window too.

    Pairs with an unmapped mate or a mate elsewhere are left out, so simulated reads never replace them: they are mostly
    reads of other parts of the genome. At the pilot people's long alleles, 16% of the reads over the repeat have such
    a mate, against 4% of the reads simulated from those alleles and aligned to the whole genome.
    """
    pairs = {}
    with pysam.AlignmentFile(minicram, reference_filename=reference_fasta) as bam:
        for chrom, start, end in windows:
            within, elsewhere = set(), set()
            for read in bam.fetch(chrom, max(0, start), end):
                if read.is_secondary or read.is_supplementary:
                    continue
                if (read.is_unmapped or read.mate_is_unmapped or read.next_reference_name != chrom
                        or not start <= read.next_reference_start < end):
                    elsewhere.add(read.query_name)
                else:
                    within.add(read.query_name)
            pairs[(chrom, start)] = within - elsewhere
    return pairs


def fragment_start_rate_near_locus(minicram, reference_fasta, chrom, start, end):
    """The rate at which the person's fragments start per bp in ``COVERAGE_WINDOW_BP`` on either side of the repeat."""
    low, high = COVERAGE_WINDOW_BP
    return sim.first_mates_starting_per_bp(minicram, reference_fasta, chrom,
                                           [(start - high, start - low), (end + low, end + high)])


def number_of_simulated_pairs(fragment_start_rate, share, sequence_length, mean_fragment_length):
    """The read pairs that give an allele of ``sequence_length`` bp (with its flanks) ``share`` of the person's
    fragment start rate: one per start position a mean-length fragment fits in, at that rate."""
    return round(fragment_start_rate * share * max(1.0, sequence_length - mean_fragment_length))


def mean_fragment_length(profile):
    return sum(profile.fragment_lengths) / len(profile.fragment_lengths)


def simulation_inputs(minicram, reference_fasta, loci):
    """``(windows, read profile, mean fragment length, reference)`` for simulating alleles at the person's loci (dicts
    with chrom, start_0based, end): ``windows_around_loci``, the person's ``read_profile`` from their reads there, and
    the reference opened as ``expanded_allele_reads`` needs it."""
    windows = windows_around_loci(loci)
    profile = sim.read_profile(minicram, reference_fasta, windows)
    return (windows, profile, mean_fragment_length(profile),
            pyfaidx.Fasta(reference_fasta, one_based_attributes=False, as_raw=True))


def write_simulated_allele_pairs(sequence, fragment_start_rate, share, profile, mean_fragment, name_prefix, fastq1,
                                 fastq2, repeat, repeat_errors, fragment_keep_exponent=1.0):
    """Writes the read pairs of one simulated allele (``sequence``, with its flanks) to the two FASTQs at ``share`` of
    the person's fragment start rate (``number_of_simulated_pairs``), named and seeded by ``name_prefix`` (the sample's
    label, a colon and the allele's id), with the measured errors and fragment loss of real reads over its ``repeat``
    (``expanded_allele_reads.simulate_read_pairs``), and returns the number of fragments drawn before the loss."""
    n_pairs = number_of_simulated_pairs(fragment_start_rate, share, len(sequence), mean_fragment)
    sim.write_fastq_pairs(sim.simulate_read_pairs(sequence, n_pairs, profile, sim.seeded_rng(name_prefix), name_prefix,
                                                  repeat, repeat_errors, fragment_keep_exponent), fastq1, fastq2)
    return n_pairs


def simulate_person(person, sex, loci, plan, work_dir, reference_fasta, repeat_errors, allele_structures, fastq1,
                    fastq2):
    """Writes the simulated read pairs of the person's planned samples to the two FASTQs, with the measured in-repeat
    errors of real reads (``repeat_errors``: ``expanded_allele_reads.RepeatReadErrors``), and returns
    ``(manifest rows, {label: read names of the person's pairs it drops})``.

    Each simulated allele takes the place of one of the person's alleles at the locus (two, or one at a male's chrX and
    chrY loci): it gets that share of the person's fragment start rate, and as many of the person's read pairs near the
    locus as the simulated alleles' share are dropped (half for a carrier's one simulated allele, all for a patient's
    two or a male's one at chrX and chrY). Its repeat takes one of the locus's published structures
    (``allele_structures``: ``load_pathogenic_allele_structures``; ``drawn_allele_structure``, seeded by the allele),
    else a pure tract of the expanded motif. In the manifest, ``copies``, ``size_bp`` and ``simulated_allele_*`` (the
    structure, its blocks as ``MOTIF:copies`` in + strand order, and the purity the truth gets) are the longer
    simulated allele's, and ``shorter_simulated_allele_*`` a patient's other one.
    """
    minicram = minicram_path(work_dir, person)
    windows, profile, mean_fragment, fasta = simulation_inputs(minicram, reference_fasta, loci.to_dict("records"))
    pairs_at_locus, rate = read_pairs_within_windows(minicram, reference_fasta, windows), {}
    manifest, dropped = [], {}
    for label, locus, sizes_bp in plan:
        chrom, start, end, motif = locus["chrom"], locus["start_0based"], locus["end"], locus["expanded_motif"]
        key = (chrom, start)
        if key not in rate:
            rate[key] = fragment_start_rate_near_locus(minicram, reference_fasta, chrom, start, end)
        share_per_allele = 1.0 if is_hemizygous(chrom, sex) else 0.5
        name_prefix = "%s:%s" % (label, locus["locus_id"])
        keep_exponent = sim.seeded_rng(name_prefix, "fragment keep exponent").uniform(*FRAGMENT_KEEP_EXPONENT_RANGE)
        copies = [size_bp // len(motif) for size_bp in sizes_bp]
        n_pairs, alleles = 0, []
        for allele, allele_copies in enumerate(copies):
            allele_prefix = "%s:%d" % (name_prefix, allele)
            structure, blocks = drawn_allele_structure(allele_structures.get(locus["locus_id"], []), motif,
                                                       allele_copies, sim.seeded_rng(allele_prefix, "allele structure"))
            if structure == "pure":
                sequence = sim.expanded_allele_sequence(fasta, chrom, start, end, motif, allele_copies, FLANK_BP)
            else:
                sequence = sim.allele_sequence_with_reference_flanks(
                    fasta, chrom, start, end, "".join(block_motif * n for block_motif, n in blocks), FLANK_BP)
            n_pairs += write_simulated_allele_pairs(
                sequence, rate[key], share_per_allele, profile, mean_fragment, allele_prefix, fastq1, fastq2,
                repeat_of_blocks(blocks), repeat_errors, keep_exponent)
            # The allele's copies are its blocks', which an inserted motif's flanking blocks add to.
            alleles.append({"copies": sum(n for _, n in blocks), "structure": structure,
                            "blocks": ";".join("%s:%d" % block for block in blocks),
                            "purity": blocks_purity(blocks, locus["reference_motif"])})
        names = {name for name in pairs_at_locus[(chrom, start - FLANK_BP)]
                 if is_dropped(label, name, share_per_allele * len(copies))}
        dropped.setdefault(label, set()).update(names)
        shorter = alleles[1] if len(copies) == 2 else {"copies": None, "structure": None, "blocks": None,
                                                        "purity": None}
        manifest.append({"label": label, "person": person, "sex": sex, "locus_id": locus["locus_id"],
                         "gene": locus["gene"], "expanded_motif": motif, "simulated_alleles": len(copies),
                         "copies": alleles[0]["copies"], "size_bp": alleles[0]["copies"] * len(motif),
                         "simulated_allele_structure": alleles[0]["structure"],
                         "simulated_allele_blocks": alleles[0]["blocks"],
                         "simulated_allele_purity": alleles[0]["purity"],
                         "shorter_simulated_allele_copies": shorter["copies"],
                         "shorter_simulated_allele_size_bp": None if shorter["copies"] is None
                         else shorter["copies"] * len(motif),
                         "shorter_simulated_allele_structure": shorter["structure"],
                         "shorter_simulated_allele_blocks": shorter["blocks"],
                         "shorter_simulated_allele_purity": shorter["purity"],
                         "person_allele_repeats": locus["person_allele_repeats"],
                         "hemizygous": is_hemizygous(chrom, sex), "fragment_starts_per_bp": rate[key],
                         "fragment_keep_exponent": keep_exponent, "simulated_fragments_before_loss": n_pairs,
                         "dropped_real_pairs": len(names)})
    return manifest, dropped


def align(fastqs, bwa, bwa_index_prefix, out_bam, threads=10):
    """Aligns the reads of ``fastqs`` (two FASTQs of pairs, or one of single reads) with bwa mem, as the people's reads
    were aligned, to the reference of ``bwa_index_prefix``, into an unsorted BAM."""
    print("%s mem %s %s > %s" % (bwa, bwa_index_prefix, " ".join(fastqs), out_bam), flush=True)
    aligner = subprocess.Popen([bwa, "mem", "-t", str(threads), "-Y", "-K", "100000000", bwa_index_prefix, *fastqs],
                               stdout=subprocess.PIPE)
    subprocess.run(["samtools", "view", "-b", "-o", out_bam, "-"], stdin=aligner.stdout, check=True)
    aligner.stdout.close()
    if aligner.wait() != 0:
        raise RuntimeError("bwa mem failed on %s" % " ".join(fastqs))


def write_samples_with_simulated_reads(fastq1, fastq2, bwa, bwa_index_prefix, simulated_bam, reference_fasta, samples):
    """Aligns the simulated pairs to ``simulated_bam`` (``align``) and writes each sample's reads
    (``write_spiked_reads``): the person's reads without the dropped pairs, plus the simulated reads whose names start
    with the sample's label and a colon.

    Args:
        samples: ``[(label, the person's minicram, names of the person's read pairs to drop, out_bam), ...]``.

    Returns:
        ``{label: number of simulated reads it got}``.
    """
    align([fastq1, fastq2], bwa, bwa_index_prefix, simulated_bam)
    reads_by_label = {}
    with pysam.AlignmentFile(simulated_bam) as bam:
        for read in bam.fetch(until_eof=True):
            reads_by_label.setdefault(read.query_name.split(":")[0], []).append(read)
    for label, minicram, dropped_names, out_bam in samples:
        os.makedirs(os.path.dirname(out_bam), exist_ok=True)
        write_spiked_reads(minicram, reference_fasta, dropped_names, reads_by_label.get(label, []), out_bam)
    return {label: len(reads_by_label.get(label, [])) for label, *_ in samples}


def write_spiked_reads(minicram, reference_fasta, dropped_names, simulated_reads, out_bam):
    """Writes the person's reads without the ``dropped_names`` pairs, plus ``simulated_reads`` (AlignedSegments from
    another header, moved onto the person's by contig name), coordinate-sorted and indexed."""
    unsorted = out_bam + ".unsorted.bam"
    with pysam.AlignmentFile(minicram, reference_filename=reference_fasta) as real:
        with pysam.AlignmentFile(unsorted, "wb", template=real) as out:
            for read in real.fetch(until_eof=True):
                if read.query_name not in dropped_names:
                    out.write(read)
            for read in simulated_reads:
                out.write(pysam.AlignedSegment.from_dict(read.to_dict(), out.header))
    pysam.sort("-o", out_bam, unsorted)
    os.remove(unsorted)
    pysam.index(out_bam)


def write_truth(manifest_rows, truth, out_tsv_gz, out_bed):
    """Writes the spiked sample's truth TSV (``TRUTH_COLUMNS``, the format ``dataset.extract_rows_and_join_truth``
    reads) and a BED covering its spiked loci, from the person's truth and the simulated alleles: at a male's chrX and
    chrY loci the simulated allele twice, as the truth tables write a single allele, a patient's two simulated alleles,
    and a carrier's simulated allele with the person's own. A simulated allele's purity is the manifest's
    (``simulate_person``)."""
    rows = []
    for m in manifest_rows:
        person_row = truth.loc[m["locus_id"]]
        purity = m["simulated_allele_purity"]
        if m["hemizygous"]:
            alleles = [(m["copies"], purity), (m["copies"], purity)]
        else:
            if m["simulated_alleles"] == 2:
                # int: the manifest column is a float one, as carriers leave it empty.
                alleles = [(int(m["shorter_simulated_allele_copies"]), m["shorter_simulated_allele_purity"]),
                           (m["copies"], purity)]
            else:
                alleles = [(m["person_allele_repeats"], person_row["RepeatPurityShortAllele"]), (m["copies"], purity)]
            # Whichever allele came out longer is the long one, as the truth tables write them.
            alleles.sort(key=lambda allele: allele[0])
        rows.append({"LocusId": m["locus_id"], "Chrom": person_row["Chrom"], "Start0Based": person_row["Start0Based"],
                     "End": person_row["End"], "Motif": person_row["Motif"],
                     "NumRepeatsInReference": person_row["NumRepeatsInReference"],
                     "NumRepeatsShortAllele": alleles[0][0], "NumRepeatsLongAllele": alleles[1][0],
                     "RepeatPurityShortAllele": alleles[0][1], "RepeatPurityLongAllele": alleles[1][1]})
    write_truth_table(pd.DataFrame(rows, columns=TRUTH_COLUMNS), out_tsv_gz, out_bed)


def write_truth_table(table, out_tsv_gz, out_bed):
    """Writes a spiked sample's truth rows (``TRUTH_COLUMNS``) as a TSV and a BED covering their loci."""
    table.to_csv(out_tsv_gz, sep="\t", index=False, compression="gzip")
    table.assign(start=table["Start0Based"] - 10, end=table["End"] + 10)[["Chrom", "start", "end"]].to_csv(
        out_bed, sep="\t", index=False, header=False)


def spiked_dir(work_dir, label):
    return os.path.join(work_dir, "spiked", label)


def spike(people, work_dir, reference_fasta, bwa, bwa_index_prefix, training_tables_dir, repeat_read_errors_tsv,
          pathogenic_allele_structures_tsv=PATHOGENIC_ALLELE_STRUCTURES_TSV):
    """Simulates, aligns and merges the reads of every planned spiked sample of ``people`` (see ``SIZES_BP``), with
    the in-repeat read errors of ``repeat_read_errors_tsv`` (``expanded_allele_reads.load_repeat_read_errors``) and
    the published allele structures of ``pathogenic_allele_structures_tsv`` (``load_pathogenic_allele_structures``)."""
    loci_table = pd.read_csv(DISEASE_LOCI_TSV, sep="\t")
    repeat_errors = sim.load_repeat_read_errors(repeat_read_errors_tsv)
    allele_structures = load_pathogenic_allele_structures(pathogenic_allele_structures_tsv)
    out_dir = os.path.join(work_dir, "spiked")
    os.makedirs(out_dir, exist_ok=True)
    fastq1_path, fastq2_path = os.path.join(out_dir, "simulated_1.fq"), os.path.join(out_dir, "simulated_2.fq")
    manifest, dropped, truth_by_person = [], {}, {}
    with open(fastq1_path, "w") as fastq1, open(fastq2_path, "w") as fastq2:
        for person in people:
            sex = sex_of(person, training_tables_dir)
            truth_by_person[person] = person_truth_at_loci(person, set(loci_table["locus_id"]), work_dir)
            loci = spikeable_loci(loci_table, truth_by_person[person],
                                  loci_with_truth_contradicted_by_short_reads(person, training_tables_dir))
            plan = plan_spiked_samples(person, loci)
            print("%s (%s): %d of %d disease loci can take simulated carriers or patients (spikeable_loci), %d of them "
                  "get simulated alleles, in %d carrier genotypes and %d patient genotypes"
                  % (person, sex, len(loci), len(loci_table), len({locus["locus_id"] for _, locus, _ in plan}),
                     sum(len(sizes) == 1 for _, _, sizes in plan), sum(len(sizes) == 2 for _, _, sizes in plan)))
            rows, person_dropped = simulate_person(person, sex, loci, plan, work_dir, reference_fasta, repeat_errors,
                                                   allele_structures, fastq1, fastq2)
            manifest += rows
            dropped.update(person_dropped)
    manifest = pd.DataFrame(manifest)
    manifest_path = os.path.join(out_dir, "manifest.tsv")
    # The spiked samples of people an earlier run spiked and this one doesn't stay listed, so genotype and tables keep
    # them.
    earlier = pd.read_csv(manifest_path, sep="\t") if os.path.exists(manifest_path) else manifest.iloc[:0]
    pd.concat([earlier[~earlier["person"].isin(people)], manifest], ignore_index=True).to_csv(
        manifest_path, sep="\t", index=False)
    write_samples_with_simulated_reads(
        fastq1_path, fastq2_path, bwa, bwa_index_prefix, os.path.join(out_dir, "simulated.bam"), reference_fasta,
        [(label, minicram_path(work_dir, rows["person"].iloc[0]), dropped[label],
          os.path.join(spiked_dir(work_dir, label), "reads.bam"))
         for label, rows in manifest.groupby("label", sort=False)])
    for label, rows in manifest.groupby("label", sort=False):
        person = rows["person"].iloc[0]
        write_truth(rows.to_dict("records"), truth_by_person[person],
                    os.path.join(spiked_dir(work_dir, label), "truth.tsv.gz"),
                    os.path.join(spiked_dir(work_dir, label), "truth.bed"))
        print("%s: %d spiked loci, %d simulated fragments before the loss over the repeats, %d real pairs dropped"
              % (label, len(rows), rows["simulated_fragments_before_loss"].sum(), len(dropped[label])))


def spiked_sample_number(label):
    """The number k of a spiked sample's label (``<person>_spiked_<k>``)."""
    return int(label.rsplit("_", 1)[1])


def first_read_length(minicram, reference_fasta):
    """The length of the first read of the person's minicram (the training people's reads are all one length)."""
    with pysam.AlignmentFile(minicram, reference_filename=reference_fasta) as reads:
        return next(reads.fetch(until_eof=True)).query_length


def downsample_read_pairs(in_reads, fraction, seed, reference_fasta, out_bam):
    """Writes a share ``fraction`` of the read pairs of ``in_reads`` (a BAM, or a CRAM of ``reference_fasta``) to
    ``out_bam``, both mates of a pair or neither (``samtools view -s``, which keeps a pair by its name's hash and
    ``seed``), and indexes it."""
    pysam.view("-b", "-s", "%d%s" % (seed, ("%.6f" % fraction)[1:]), "-T", reference_fasta, "-o", out_bam, in_reads,
               catch_stdout=False)
    pysam.index(out_bam)


def write_persons_own_truth(locus_ids, truth, out_tsv_gz, out_bed):
    """Writes a matched control's truth TSV and BED (as ``write_truth``): the person's own truth at ``locus_ids``."""
    write_truth_table(truth.loc[list(locus_ids)].reset_index()[TRUTH_COLUMNS], out_tsv_gz, out_bed)


def downsample_spiked_samples(source_work_dir, work_dir, reference_fasta, coverage_range=DOWNSAMPLED_COVERAGE_RANGE):
    """Writes, for every spiked sample of ``source_work_dir``, a copy downsampled to a coverage drawn from
    ``coverage_range`` (seeded by the sample) and a matched control, the person's own reads downsampled by the same
    share, with the person's own truth at the same loci, as spiked samples of ``work_dir`` with their manifest.

    Without the controls, a model would learn that low coverage at these loci means an expansion. A person's coverage
    is twice their reads' length times their fragment start rate near the loci (``fragment_starts_per_bp`` in the
    manifest, its median over the sample's loci). Both keep the same read pairs of the person's (one seed, the
    sample's number k), so they differ only by the simulated alleles. The copy of ``<person>_spiked_<k>`` is named
    with k + ``DOWNSAMPLED_COPY_NUMBER_OFFSET`` and its control with k + ``MATCHED_CONTROL_NUMBER_OFFSET``; their
    manifest rows are the source sample's, with ``source_label``, ``downsampled_coverage``, ``downsample_fraction``
    and ``matched_control``, and a control's ``simulated_alleles`` 0 and simulated allele columns empty. The person's
    reads (minicram) and truth are those of ``source_work_dir``, the ones its spiked samples were made from.
    """
    source = pd.read_csv(os.path.join(source_work_dir, "spiked", "manifest.tsv"), sep="\t")
    simulated_allele_columns = [c for c in source.columns if c in ("copies", "size_bp") or c.startswith(
        ("simulated_allele_", "shorter_simulated_allele_"))]
    manifest, read_length, truth_by_person = [], {}, {}
    for label, rows in source.groupby("label", sort=False):
        person = rows["person"].iloc[0]
        if person not in read_length:
            read_length[person] = first_read_length(minicram_path(source_work_dir, person), reference_fasta)
            truth_by_person[person] = person_truth_at_loci(person, set(source.loc[source["person"] == person,
                                                                                  "locus_id"]), source_work_dir)
        coverage = 2 * read_length[person] * rows["fragment_starts_per_bp"].median()
        target = sim.seeded_rng(label, "downsampled coverage").uniform(*coverage_range)
        fraction = min(1.0, target / coverage)
        k = spiked_sample_number(label)
        copy_label = "%s_spiked_%d" % (person, k + DOWNSAMPLED_COPY_NUMBER_OFFSET)
        control_label = "%s_spiked_%d" % (person, k + MATCHED_CONTROL_NUMBER_OFFSET)
        for out_label, in_reads in ((copy_label, os.path.join(spiked_dir(source_work_dir, label), "reads.bam")),
                                    (control_label, minicram_path(source_work_dir, person))):
            os.makedirs(spiked_dir(work_dir, out_label), exist_ok=True)
            downsample_read_pairs(in_reads, fraction, k, reference_fasta,
                                  os.path.join(spiked_dir(work_dir, out_label), "reads.bam"))
        for name in ("truth.tsv.gz", "truth.bed"):
            shutil.copy(os.path.join(spiked_dir(source_work_dir, label), name),
                        os.path.join(spiked_dir(work_dir, copy_label), name))
        write_persons_own_truth(rows["locus_id"], truth_by_person[person],
                                os.path.join(spiked_dir(work_dir, control_label), "truth.tsv.gz"),
                                os.path.join(spiked_dir(work_dir, control_label), "truth.bed"))
        downsampled = rows.assign(source_label=label, downsampled_coverage=target, downsample_fraction=fraction)
        manifest.append(downsampled.assign(label=copy_label, matched_control=False))
        manifest.append(downsampled.assign(label=control_label, matched_control=True, simulated_alleles=0,
                                           **{c: None for c in simulated_allele_columns}))
        print("%s: %.1fx to %.1fx (%.3f of the read pairs), as %s and its control %s"
              % (label, coverage, target, fraction, copy_label, control_label))
    os.makedirs(os.path.join(work_dir, "spiked"), exist_ok=True)
    pd.concat(manifest, ignore_index=True).to_csv(os.path.join(work_dir, "spiked", "manifest.tsv"), sep="\t",
                                                  index=False)


def spiked_labels(work_dir):
    """``(label, sex)`` of the spiked samples ``spike`` wrote, from its manifest."""
    return list(pd.read_csv(os.path.join(work_dir, "spiked", "manifest.tsv"), sep="\t")
                .drop_duplicates("label")[["label", "sex"]].itertuples(index=False))


def genotype_spiked_samples(work_dir, catalog, billing_project, gcs_dir, only_missing_outputs=False):
    """Genotypes every spiked sample with ``genotype_on_hail_batch``, or with ``only_missing_outputs`` just those
    without an output JSON, to finish a batch that failed part way."""
    genotype_on_hail_batch(
        [{"name": label, "reads": os.path.join(spiked_dir(work_dir, label), "reads.bam"), "sex": sex,
          "out_json": os.path.join(spiked_dir(work_dir, label), "expansion_hunter.json.gz")}
         for label, sex in spiked_labels(work_dir)],
        catalog, billing_project, gcs_dir, "simulated_expansions: genotype the spiked samples", only_missing_outputs)


def write_tables(work_dir, tables_dir):
    """Writes each spiked sample's per-source parquet to ``tables_dir``, built from its ExpansionHunter JSON and truth
    by ``dataset.extract_rows_and_join_truth``, as the training tables are. No row is marked as contradicted by the
    short reads: the simulated allele's truth is known, and ``spikeable_loci`` leaves out the loci where the person's
    reads were found to contradict the truth of the allele they keep (those the check judged; the rest were not
    checked).

    A sample with no rows gets no parquet, and its count is printed: that is a matched control whose person carries
    the reference allele at every locus of its sample and whose hom-ref loci all fell outside the hom-ref sample.
    Two kinds of parquet an earlier run left in ``tables_dir`` are removed, so neither is trained on: one the
    manifest no longer lists, and one belonging to a sample that now has no rows."""
    os.makedirs(tables_dir, exist_ok=True)
    labels = {label for label, _ in spiked_labels(work_dir)}
    for name in os.listdir(tables_dir):
        label = name[:-len(".parquet")]
        if name.endswith(".parquet") and dataset.is_spiked_sample(label) and label not in labels:
            os.remove(os.path.join(tables_dir, name))
    without_rows = []
    for label, _ in spiked_labels(work_dir):
        directory = spiked_dir(work_dir, label)
        df = dataset.extract_rows_and_join_truth(
            [os.path.join(directory, "expansion_hunter.json.gz")], os.path.join(directory, "truth.tsv.gz"),
            os.path.join(directory, "truth.bed"), label, label, keep_hom_ref_loci=True,
            allow_sample_with_no_variant_truth_loci=True)
        if df.empty:
            # A matched control of a person who carries the reference allele at every locus of its sample, whose
            # hom-ref loci all fell outside the hom-ref sample as well. It has no rows to train on, and a parquet an
            # earlier run wrote for it must go, or those rows would be trained on as this sample's.
            without_rows.append(label)
            stale = os.path.join(tables_dir, "%s.parquet" % label)
            if os.path.exists(stale):
                os.remove(stale)
            continue
        df = df.drop(columns=["sample_id"])
        df[dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN] = False
        df.to_parquet(os.path.join(tables_dir, "%s.parquet" % label), index=False)
        print("%s: %d rows" % (label, len(df)))
    if without_rows:
        print("%d samples had no rows and no parquet, the first few: %s"
              % (len(without_rows), ", ".join(without_rows[:5])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["make-minicrams", "check-parity", "spike", "downsample", "genotype",
                                            "tables"])
    parser.add_argument("--person", action="append", help="default: the %d SPIKED_PEOPLE" % len(SPIKED_PEOPLE))
    parser.add_argument("--work-dir", required=True,
                        help="the folder the step reads and writes, such as make_spiked_samples.sh's "
                             "work_spiked_samples/ (no default, so no step picks up an older run's folder)")
    parser.add_argument("--source-work-dir", help="the work folder whose spiked samples get downsampled copies and "
                                                  "matched controls in --work-dir (downsample)")
    parser.add_argument("--reference-fasta", help="a local copy of the reference the training runs used "
                                                  "(TRAINING_REFERENCE_FASTA, UCSC hg38), indexed (make-minicrams, spike, "
                                                  "downsample)")
    parser.add_argument("--catalog", default=DISEASE_LOCI_CATALOG)
    parser.add_argument("--training-tables-dir",
                        help="the folder of the training people's per-source parquets (<person>.parquet, with "
                             "n_alleles, from which each person's sex is read, and the truth_contradicted_by_short_reads "
                             "column) of the training runs under dataset.EH_RESULTS_ROOT (check-parity, spike)")
    parser.add_argument("--bwa", default="bwa", help="the bwa executable (spike)")
    parser.add_argument("--bwa-index", default=DEFAULT_BWA_INDEX_PREFIX,
                        help="bwa index prefix of the hg38 the people's reads were aligned to (spike)")
    parser.add_argument("--repeat-read-errors", default=sim.REPEAT_READ_ERRORS_TSV,
                        help="the errors of real reads inside long repeats, which the simulated reads get there "
                             "(calibrate_simulated_reads.py measure-read-errors writes it) (spike)")
    parser.add_argument("--pathogenic-allele-structures", default=PATHOGENIC_ALLELE_STRUCTURES_TSV,
                        help="the published structures of patients' expanded alleles that the simulated alleles take "
                             "(spike)")
    parser.add_argument("--billing-project", help="Hail Batch and requester-pays billing project (make-minicrams, "
                                                  "check-parity, genotype)")
    parser.add_argument("--gcs-dir", help="gs:// folder for the reads genotyped on Hail Batch (check-parity, genotype)")
    parser.add_argument("--only-missing-outputs", action="store_true",
                        help="genotype only the samples without an output JSON, to finish a batch that failed part "
                             "way; an output stays as it is even if the reads or the settings have changed since, so "
                             "delete the ones to be remade (genotype)")
    parser.add_argument("--tables-dir", help="where the spiked samples' per-source parquets go (tables); default: "
                                             "the tables folder of --work-dir")
    args = parser.parse_args()
    people = args.person or SPIKED_PEOPLE
    if args.command in ("make-minicrams", "check-parity", "genotype") and not args.billing_project:
        parser.error("%s needs --billing-project" % args.command)
    if args.command in ("check-parity", "genotype") and not args.gcs_dir:
        parser.error("%s needs --gcs-dir" % args.command)
    if args.command in ("make-minicrams", "spike", "downsample") and not args.reference_fasta:
        parser.error("%s needs --reference-fasta" % args.command)
    if args.command == "downsample" and not args.source_work_dir:
        parser.error("downsample needs --source-work-dir")
    if args.command in ("check-parity", "spike") and not args.training_tables_dir:
        parser.error("%s needs --training-tables-dir" % args.command)
    if args.command == "make-minicrams":
        for person in people:
            make_minicram(person, args.work_dir, args.reference_fasta, args.catalog, args.billing_project)
    elif args.command == "check-parity":
        failed = check_parity(people, args.work_dir, args.catalog, args.billing_project, args.gcs_dir,
                              args.training_tables_dir)
        if failed:
            sys.exit("parity check failed for %s" % ", ".join(failed))
    elif args.command == "spike":
        spike(people, args.work_dir, args.reference_fasta, args.bwa, args.bwa_index, args.training_tables_dir,
              args.repeat_read_errors, args.pathogenic_allele_structures)
    elif args.command == "downsample":
        downsample_spiked_samples(args.source_work_dir, args.work_dir, args.reference_fasta)
    elif args.command == "genotype":
        genotype_spiked_samples(args.work_dir, args.catalog, args.billing_project, args.gcs_dir,
                                args.only_missing_outputs)
    else:
        write_tables(args.work_dir, args.tables_dir or os.path.join(args.work_dir, "tables"))


if __name__ == "__main__":
    main()
