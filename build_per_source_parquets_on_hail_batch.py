"""Builds the per-source parquets of all 135 sources (the 54 training sources and the 81 held-out samples) on Hail
Batch, one job per source next to its ExpansionHunter JSON, then downloads them to where the local builders keep them.

Built locally one source at a time (``dataset.py`` and ``heldout.py --build-only``), the 135 sources take about a day
and ~95 GB of downloads on this RAM-limited machine. Each job here runs the same code on one source
(``dataset.write_per_source_parquet``, shipped into the job with the modules it imports) and writes the source's
parquet and the record of the cloud versions it was built from (``<name>.parquet.sources.json``) to --output-dir.
Every parquet keeps a sample of the loci where the truth equals the reference (``keep_hom_ref_loci``), so it serves
both training populations: today's recipe keeps only its variant-locus rows (``dataset.variant_locus_rows``).

``--download`` then copies each parquet and its record to the path the local builders write: ``data/real_quick/`` for
the 4 training combos (``dataset.build_combo``), ``data_eval_43/real_43/`` for the 50 promoted and 81 held-out samples
(``heldout.build_sample``). There ``dataset.py --keep-hom-ref-loci`` and ``heldout.py --build-only --keep-hom-ref-loci``
reuse them without extracting again, because ``dataset._parquet_reusable`` accepts a parquet whose record matches the
current cloud versions of its sources and whose EH build still counts as current against ExpansionHunter-bw2 HEAD. The
cloud versions are read here, before the jobs read the files, so a source that changes after that is rebuilt locally
rather than trusted.

Outputs already in --output-dir are not built again (step_pipeline skips a job whose outputs exist). After a change to
the extraction code, pass ``--force-build-step`` to rebuild them.

Usage:
    python3 build_per_source_parquets_on_hail_batch.py --dry-run
    python3 build_per_source_parquets_on_hail_batch.py --no-wait -s HG00738   # one source first
    python3 build_per_source_parquets_on_hail_batch.py --no-wait
    python3 build_per_source_parquets_on_hail_batch.py --download

Coding rules: no type hints, Google docstrings, ``print()``.
"""

import concurrent.futures
import glob
import hashlib
import importlib.metadata
import io
import json
import os
import resource
import shlex
import sys
import tarfile

import dataset

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.basename(os.path.abspath(__file__))

# The modules a job imports: this script, dataset.py and what dataset.py imports.
CODE_FILES_SHIPPED_INTO_EACH_JOB = (SCRIPT_NAME, "dataset.py", "eh_json.py", "features.py")
# The packages those modules import. Each job installs the versions installed here, so a parquet built in a job is the
# same as one built on this machine (pandas 3, for example, would store the string columns differently).
PACKAGES_THE_EXTRACTION_IMPORTS = ("numpy", "pandas", "pyarrow", "ijson")
# The same Python as this machine's python3 (3.11.13).
PYTHON_IMAGE = "python:3.11.13-slim@sha256:9bffe4353b925a1656688797ebc68f9c525e79b1d377a764d232182a519eeec4"
# The first argument that makes this script build one source inside a job instead of launching the batch.
BUILD_ONE_SOURCE_IN_JOB_ARG = "--build-one-source-in-job"
# Where each job writes its outputs: the disk Hail Batch mounts for the job's storage request.
JOB_OUTPUT_DIR = "/io"
# How many sources' cloud versions are read at once (each read runs `gsutil ls` or `gsutil stat`).
PARALLEL_CLOUD_VERSION_READS = 8


def default_output_dir():
    """Returns the cloud folder for the parquets built from ``dataset.EH_RESULTS_ROOT``'s JSONs: a sibling of that
    folder, named after it."""
    return "%s/genotype_quality_model_parquets_from_%s" % (os.path.dirname(dataset.EH_RESULTS_ROOT),
                                                            os.path.basename(dataset.EH_RESULTS_ROOT))


def sources_to_build(data_dir, heldout_data_dir):
    """Returns one dict per source, in the order the local builders build them.

    Each dict holds ``name`` (the parquet's base name), ``sample`` (the truth sample id), ``sample_label`` (the folder
    under ``dataset.EH_RESULTS_ROOT``), ``row_sample_id`` and ``source_desc`` (what the local builder passes to
    ``dataset.extract_rows_and_join_truth``) and ``local_parquet_path`` (where the local builder writes it):

    - the training combos as ``dataset.build_combo`` builds them, under ``<data_dir>/<dataset.SOURCE_SUBDIR>/``;
    - the promoted and held-out samples as ``heldout.build_sample`` builds them, under ``<heldout_data_dir>/real_43/``.
    """
    import heldout  # local: only the launcher needs it, and it imports the model code a job does not ship
    sources = []
    for sample, cov_label, sample_label in dataset.COMBOS:
        name = "%s_%s" % (sample, cov_label)
        sources.append({"name": name, "sample": sample, "sample_label": sample_label, "row_sample_id": name,
                        "source_desc": "%s %s" % (sample, cov_label),
                        "local_parquet_path": os.path.join(data_dir, dataset.SOURCE_SUBDIR, name + ".parquet")})
    for sample in list(dataset.PROMOTED_HELDOUT_SAMPLES) + list(heldout.SAMPLES):
        sources.append({"name": sample, "sample": sample, "sample_label": sample, "row_sample_id": sample,
                        "source_desc": sample,
                        "local_parquet_path": os.path.join(heldout_data_dir, "real_43", sample + ".parquet")})
    return sources


def cloud_sources_and_versions(source):
    """Returns ``(sources, cloud_versions)`` for one source, as the local builders compute them: ``sources`` is the
    ``(label, remote)`` list of its JSON shards, truth-genotypes TSV and high-confidence BED, and ``cloud_versions`` is
    ``{remote: [md5, mtime]}`` (``dataset._cloud_versions``)."""
    json_remotes = dataset._list_json_inputs(source["sample_label"])
    sources = ([("json shard %d" % i, remote) for i, remote in enumerate(json_remotes)]
               + [(label, remote) for label, remote, _ in dataset._truth_sources(source["sample"], "")])
    return sources, dataset._cloud_versions([(label, remote, None) for label, remote in sources])


def job_spec(source, job_sources, cloud_versions):
    """Returns what a job needs to build ``source``: ``job_sources`` is its ``(label, remote, path inside the job)``
    list and ``cloud_versions`` the versions read by ``cloud_sources_and_versions``."""
    return {"out_path": os.path.join(JOB_OUTPUT_DIR, source["name"] + ".parquet"),
            "row_sample_id": source["row_sample_id"], "source_desc": source["source_desc"],
            "sources": [list(s) for s in job_sources], "cloud_versions": cloud_versions}


def build_one_source_in_job(spec):
    """Job side: builds one source's parquet and record from the files the job localized (see ``job_spec``).

    Raises:
        RuntimeError: if no record was written, which ``dataset._record_sources_and_remove_downloads`` skips when a
            cloud version is unknown; without it the local builders could not reuse the parquet.
    """
    sources = [tuple(s) for s in spec["sources"]]
    local_by_label = {label: local for label, _, local in sources}
    merged = dataset.write_per_source_parquet(
        spec["out_path"], [local for label, _, local in sources if label.startswith("json")],
        local_by_label["truth-genotypes TSV"], local_by_label["high-confidence BED"], spec["row_sample_id"],
        spec["source_desc"], True, sources, spec["cloud_versions"])
    record_path = dataset._sources_record_path(spec["out_path"])
    if not os.path.exists(record_path):
        raise RuntimeError("%s: no record of the sources was written" % spec["source_desc"])
    with open(record_path) as f:
        eh_builds = sorted(set(json.load(f)["eh_build_by_json"].values()))
    # Linux reports ru_maxrss in KiB.
    print("%s: %d rows (%d at variant loci), EH build %s, peak memory %.1f GiB"
          % (spec["source_desc"], len(merged), int(merged["variant_locus"].sum()), ", ".join(eh_builds),
             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20), flush=True)


def code_bundle_and_its_hash():
    """Returns ``(tar.gz bytes, hash)`` of ``CODE_FILES_SHIPPED_INTO_EACH_JOB``. The hash covers the files' names and
    contents, so a changed file gets a new cloud path."""
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in CODE_FILES_SHIPPED_INTO_EACH_JOB:
            with open(os.path.join(HERE, name), "rb") as f:
                content = f.read()
            digest.update(name.encode() + b"\0" + content + b"\0")
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue(), digest.hexdigest()[:12]


def upload_code_bundle(output_dir, dry_run):
    """Uploads the code bundle to ``<output_dir>/code/`` unless it is already there, and returns its cloud path."""
    import hailtop.fs as hfs  # local: only the launcher needs it, and a job does not install it
    bundle, bundle_hash = code_bundle_and_its_hash()
    cloud_path = "%s/code/per_source_parquet_code.%s.tar.gz" % (output_dir, bundle_hash)
    if hfs.exists(cloud_path):
        print("%s is already up to date" % cloud_path)
    elif dry_run:
        print("would upload the code bundle to %s" % cloud_path)
    else:
        print("uploading the code bundle to %s" % cloud_path)
        with hfs.open(cloud_path, "wb") as f:
            f.write(bundle)
    return cloud_path


def download_built_parquets(sources, output_dir, staging_dir):
    """Copies each source's parquet and record from ``output_dir`` to its ``local_parquet_path``, replacing what is there.

    Every built source is copied, even one whose local record equals the built one: the record names only the inputs
    and their EH build, so a variant-only parquet built locally from the same inputs, or an earlier build of the same
    job, has the same record. The files are downloaded into ``staging_dir`` first and then moved into place, the
    parquet before its record, so an interrupted download never pairs a new record with an old parquet.
    ``staging_dir`` is emptied first: ``dataset._download`` skips a file that is already there, which would keep a file
    left by an interrupted earlier download. A source with no built parquet yet is reported and skipped.

    Returns:
        The number of sources copied.
    """
    for leftover in glob.glob(os.path.join(staging_dir, "*")):
        os.remove(leftover)
    built = _names_in_cloud_folder(output_dir)
    missing = [s["name"] for s in sources
               if s["name"] + ".parquet" not in built or s["name"] + ".parquet.sources.json" not in built]
    if missing:
        print("not built yet, skipped: %s" % ", ".join(missing))
    ready = [s for s in sources if s["name"] not in missing]
    records = dataset._download(["%s/%s.parquet.sources.json" % (output_dir, s["name"]) for s in ready], staging_dir)
    parquets = dataset._download(["%s/%s.parquet" % (output_dir, s["name"]) for s in ready], staging_dir)
    for source, record, parquet in zip(ready, records, parquets):
        os.makedirs(os.path.dirname(source["local_parquet_path"]), exist_ok=True)
        os.replace(parquet, source["local_parquet_path"])
        os.replace(record, dataset._sources_record_path(source["local_parquet_path"]))
    print("downloaded %d built source(s)" % len(ready))
    return len(ready)


def _names_in_cloud_folder(folder):
    """Returns the base names of the files directly in a cloud folder."""
    import hailtop.fs as hfs  # local: only the launcher needs it
    return {os.path.basename(entry.path.rstrip("/")) for entry in hfs.ls(folder)}


def add_build_step(bp, source, remote_sources, cloud_versions, code_bundle_cloud_path, args):
    """Adds the job that builds one source from its ``(label, remote)`` list (``cloud_sources_and_versions``)."""
    from step_pipeline import Localize  # local: only the launcher needs it
    step = bp.new_step("Build per-source parquet: %s" % source["name"], arg_suffix="build-step", step_number=1,
                       image=PYTHON_IMAGE, cpu=args.cpu, memory=args.memory, storage=args.storage,
                       localize_by=Localize.COPY, output_dir=args.output_dir)
    local_code_bundle = step.input(code_bundle_cloud_path)
    job_sources = [(label, remote, step.input(remote).local_path) for label, remote in remote_sources]
    pins = ["%s==%s" % (package, importlib.metadata.version(package)) for package in PACKAGES_THE_EXTRACTION_IMPORTS]
    spec = job_spec(source, job_sources, cloud_versions)
    step.command("set -euxo pipefail")
    step.command("cd %s" % JOB_OUTPUT_DIR)
    step.command("python3 -m pip install --no-cache-dir --quiet %s" % " ".join(pins))
    step.command("mkdir -p code && tar xzf %s -C code" % local_code_bundle)
    step.command("python3 code/%s %s %s" % (SCRIPT_NAME, BUILD_ONE_SOURCE_IN_JOB_ARG, shlex.quote(json.dumps(spec))))
    step.command("ls -lh %s" % JOB_OUTPUT_DIR)
    step.output(spec["out_path"])
    step.output(dataset._sources_record_path(spec["out_path"]))


def launch_or_download():
    """Parses the arguments, then submits the batch, or with ``--download`` copies the built parquets home."""
    from step_pipeline import pipeline, Backend  # local: only the launcher needs it
    bp = pipeline("build_per_source_parquets_on_hail_batch", backend=Backend.HAIL_BATCH_SERVICE,
                  config_file_path="~/.step_pipeline")
    parser = bp.get_config_arg_parser()
    parser.add_argument("--output-dir", default=default_output_dir(),
                        help="cloud folder for the parquets and their records (default: %(default)s)")
    parser.add_argument("--data-dir", default=os.path.join(HERE, "data"),
                        help="dataset.py's --data-dir, which holds the training combos' parquets")
    parser.add_argument("--heldout-data-dir", default=os.path.join(HERE, "data_eval_43"),
                        help="heldout.py's --data-dir, which holds the promoted and held-out samples' parquets")
    parser.add_argument("-s", "--source", action="append",
                        help="build only this source (e.g. HG002_10x or HG00738); repeatable")
    parser.add_argument("--cpu", type=float, default=2)
    parser.add_argument("--memory", default="highmem", help="lowmem, standard, highmem, or an amount like 12Gi")
    parser.add_argument("--storage", default="20Gi", help="job disk; one source's inputs are about 1 GB")
    parser.add_argument("--download", action="store_true",
                        help="copy the built parquets and records to their local paths instead of launching jobs")
    args = bp.parse_known_args()

    sources = sources_to_build(args.data_dir, args.heldout_data_dir)
    if args.source:
        unknown = sorted(set(args.source) - {s["name"] for s in sources})
        if unknown:
            parser.error("unknown source(s): %s" % ", ".join(unknown))
        sources = [s for s in sources if s["name"] in args.source]

    if args.download:
        download_built_parquets(sources, args.output_dir,
                                os.path.join(args.data_dir, "_per_source_parquets_from_hail_batch"))
        print("next: python3 dataset.py --keep-hom-ref-loci --training-population variant_loci "
              "--training-population all_loci; python3 heldout.py --build-only --keep-hom-ref-loci")
        return

    bp.set_name("build_per_source_parquets_on_hail_batch: %d source(s) -> %s" % (len(sources), args.output_dir))
    # Only the outputs: step_pipeline cannot parse a listing that includes the code/ subfolder.
    bp.precache_file_paths(os.path.join(args.output_dir, "*.parquet*"))
    print("reading the cloud versions of %d source(s)" % len(sources), flush=True)
    with concurrent.futures.ThreadPoolExecutor(PARALLEL_CLOUD_VERSION_READS) as pool:
        versions = list(pool.map(cloud_sources_and_versions, sources))
    unknown = [s["name"] for s, (_, cloud_versions) in zip(sources, versions) if [None, None] in cloud_versions.values()]
    if unknown:
        sys.exit("could not read the cloud version of every input of %s; nothing submitted" % ", ".join(unknown))
    code_bundle_cloud_path = upload_code_bundle(args.output_dir, args.dry_run)
    for source, (remote_sources, cloud_versions) in zip(sources, versions):
        add_build_step(bp, source, remote_sources, cloud_versions, code_bundle_cloud_path, args)
    bp.run()


def main():
    if sys.argv[1:2] == [BUILD_ONE_SOURCE_IN_JOB_ARG]:
        build_one_source_in_job(json.loads(sys.argv[2]))
    else:
        launch_or_download()


if __name__ == "__main__":
    main()
