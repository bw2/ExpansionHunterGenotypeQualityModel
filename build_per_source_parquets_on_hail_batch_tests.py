"""Tests for build_per_source_parquets_on_hail_batch.py. Nothing here touches the cloud: the extraction, the cloud
listing and the downloads are mocked."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

import build_per_source_parquets_on_hail_batch as B
import dataset
import heldout


class SourcesToBuildTest(unittest.TestCase):
    def setUp(self):
        self.sources = B.sources_to_build("/d", "/h")

    def test_every_training_and_held_out_source_once(self):
        names = [s["name"] for s in self.sources]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), len(dataset.COMBOS) + len(dataset.PROMOTED_HELDOUT_SAMPLES) + len(heldout.SAMPLES))
        self.assertEqual(len(names), 135)

    def test_a_training_combo_is_named_and_placed_as_build_combo_does(self):
        source = next(s for s in self.sources if s["name"] == "HG002_10x")
        self.assertEqual(source, {"name": "HG002_10x", "sample": "HG002", "sample_label": "HG002_10x",
                                  "row_sample_id": "HG002_10x", "source_desc": "HG002 10x",
                                  "local_parquet_path": "/d/real_quick/HG002_10x.parquet"})

    def test_promoted_and_held_out_samples_are_placed_as_build_sample_does(self):
        for sample in (dataset.PROMOTED_HELDOUT_SAMPLES[0], heldout.SAMPLES[0]):
            source = next(s for s in self.sources if s["name"] == sample)
            self.assertEqual(source, {"name": sample, "sample": sample, "sample_label": sample, "row_sample_id": sample,
                                      "source_desc": sample, "local_parquet_path": "/h/real_43/%s.parquet" % sample})


class BuildOneSourceInJobTest(unittest.TestCase):
    """A job's parquet and record are what the local builders write, and the local builders reuse them once
    downloaded."""

    TRUTH_DF = pd.DataFrame({"LocusId": ["1-1-2-A", "1-1-2-A"], "allele_rank": [0, 1], "true": [9.0, 21.0],
                             "purity": [1.0, 1.0], "is_negative_locus": [False, False]})
    TRUTH_BY_LOCUS = pd.DataFrame({"reference_repeats": [8.0, 4.0], "is_hom_ref": [False, True], "purity": [1.0, 1.0]},
                                  index=pd.Index(["1-1-2-A", "1-5-6-A"], name="LocusId"))
    CLOUD_VERSIONS = {"gs://eh/S/json/a.json.gz": ["md5a", 1.0], "gs://truth/S.tsv.gz": ["md5t", 2.0],
                      "gs://beds/S.dip.bed.gz": ["md5b", 3.0]}

    @staticmethod
    def _row(locus_id, rank, eh):
        return {"locus_id": locus_id, "allele_rank": rank, "eh": eh, "motif_size": 3, "genotyping_branch": "quick",
                "sample_id": "S"}

    def _build_in_job(self, job_dir):
        """Runs ``build_one_source_in_job`` the way a job does, with its spec through JSON, and returns the spec."""
        job_sources = []
        for label, remote in (("json shard 0", "gs://eh/S/json/a.json.gz"), ("truth-genotypes TSV", "gs://truth/S.tsv.gz"),
                              ("high-confidence BED", "gs://beds/S.dip.bed.gz")):
            local = os.path.join(job_dir, "inputs", os.path.basename(remote))
            os.makedirs(os.path.dirname(local), exist_ok=True)
            open(local, "w").close()
            job_sources.append((label, remote, local))
        source = {"name": "S", "row_sample_id": "S", "source_desc": "S"}
        spec = B.job_spec(source, job_sources, self.CLOUD_VERSIONS)
        spec["out_path"] = os.path.join(job_dir, "S.parquet")
        rows = [self._row("chr1-1-2-A", 0, 10.0), self._row("chr1-1-2-A", 1, 20.0),   # variant locus
                self._row("chr1-5-6-A", 0, 4.0), self._row("chr1-5-6-A", 1, 4.0)]     # hom-ref, always kept below
        with mock.patch.object(dataset.eh_json, "extract_rows", return_value=rows) as extract_rows, \
             mock.patch.object(dataset.eh_json, "typical_read_length_in_file", return_value=150), \
             mock.patch.object(dataset, "_load_truth_from_genotypes_tsv",
                               return_value=(self.TRUTH_DF.copy(), {"1-1-2-A", "1-5-6-A"})) as load_truth, \
             mock.patch.object(dataset, "_load_reference_repeats_and_hom_ref_truth", return_value=self.TRUTH_BY_LOCUS), \
             mock.patch.object(dataset, "_locus_hash_fraction", return_value=0.0), \
             mock.patch.object(dataset, "_assert_catalog_agreement"), \
             mock.patch.object(dataset, "_json_eh_version", return_value="e90307c"):
            B.build_one_source_in_job(json.loads(json.dumps(spec)))
        self.assertEqual(extract_rows.call_args.args[0], job_sources[0][2])
        self.assertEqual(load_truth.call_args.args[:2], (job_sources[1][2], job_sources[2][2]))
        return spec, job_sources

    def test_writes_an_all_loci_parquet_and_its_record_and_removes_the_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            spec, job_sources = self._build_in_job(d)
            merged = pd.read_parquet(spec["out_path"])
            self.assertNotIn("sample_id", merged.columns)
            self.assertEqual(list(merged["variant_locus"]), [True, True, False, False])
            with open(dataset._sources_record_path(spec["out_path"])) as f:
                self.assertEqual(json.load(f), {"cloud_versions": self.CLOUD_VERSIONS,
                                                "eh_build_by_json": {"gs://eh/S/json/a.json.gz": "e90307c"}})
            self.assertFalse(any(os.path.exists(local) for _, _, local in job_sources))

    def test_the_local_builders_reuse_the_downloaded_parquet_while_its_sources_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            spec, _ = self._build_in_job(os.path.join(d, "job"))
            local_path = os.path.join(d, "real_43", "S.parquet")
            os.makedirs(os.path.dirname(local_path))
            shutil.copy(spec["out_path"], local_path)
            shutil.copy(dataset._sources_record_path(spec["out_path"]), dataset._sources_record_path(local_path))
            # As heldout.build_sample lists them: the local download paths, which do not exist.
            sources = [(label, remote, os.path.join(d, "real_43", "_downloads", "S", os.path.basename(remote)))
                       for label, remote, _ in spec["sources"]]
            versions = {remote: tuple(version) for remote, version in self.CLOUD_VERSIONS.items()}
            with mock.patch.object(dataset, "_bw2_head_sha", return_value="e90307c"), \
                 mock.patch.object(dataset, "parquet_contract_complaint", return_value=None):
                with mock.patch.object(dataset, "_gcs_stat", side_effect=lambda remote: versions[remote]):
                    self.assertTrue(dataset._parquet_reusable(local_path, sources, force=False))
                versions["gs://eh/S/json/a.json.gz"] = ("changed", 9.0)
                with mock.patch.object(dataset, "_gcs_stat", side_effect=lambda remote: versions[remote]):
                    self.assertFalse(dataset._parquet_reusable(local_path, sources, force=False))


class DownloadBuiltParquetsTest(unittest.TestCase):
    """``download_built_parquets`` with a local folder standing in for the cloud one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cloud = os.path.join(self.tmp.name, "cloud")
        os.makedirs(self.cloud)
        self.sources = [{"name": name, "local_parquet_path": os.path.join(self.tmp.name, "local", name + ".parquet")}
                        for name in ("A", "B", "C")]
        for name in ("A", "B"):   # C is not built yet
            self._put_in_cloud(name, "parquet " + name, "record " + name)

    def tearDown(self):
        self.tmp.cleanup()

    def _put_in_cloud(self, name, parquet, record):
        for suffix, content in ((".parquet", parquet), (".parquet.sources.json", record)):
            with open(os.path.join(self.cloud, name + suffix), "w") as f:
                f.write(content)

    def _download(self):
        def fake_download(remote_paths, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            locals_ = []
            for remote in remote_paths:
                local = os.path.join(dest_dir, os.path.basename(remote))
                if not os.path.exists(local):
                    shutil.copy(remote, local)
                locals_.append(local)
            return locals_
        with mock.patch.object(B, "_names_in_cloud_folder", side_effect=lambda folder: set(os.listdir(folder))), \
             mock.patch.object(dataset, "_download", side_effect=fake_download):
            return B.download_built_parquets(self.sources, self.cloud, os.path.join(self.tmp.name, "staging"))

    def _local(self, name, suffix=".parquet"):
        with open(os.path.join(self.tmp.name, "local", name + suffix)) as f:
            return f.read()

    def test_copies_each_built_parquet_and_record_and_skips_the_unbuilt(self):
        self.assertEqual(self._download(), 2)
        self.assertEqual((self._local("A"), self._local("A", ".parquet.sources.json")), ("parquet A", "record A"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "local", "C.parquet")))

    def test_replaces_a_local_parquet_even_when_its_record_matches(self):
        # A variant-only parquet built locally from the same inputs has the same record as the built one.
        self._download()
        with open(os.path.join(self.tmp.name, "local", "A.parquet"), "w") as f:
            f.write("variant-only parquet A")
        self._put_in_cloud("B", "parquet B2", "record B2")
        self.assertEqual(self._download(), 2)
        self.assertEqual((self._local("A"), self._local("A", ".parquet.sources.json")), ("parquet A", "record A"))
        self.assertEqual((self._local("B"), self._local("B", ".parquet.sources.json")), ("parquet B2", "record B2"))

    def test_a_file_left_in_the_staging_folder_is_not_used(self):
        staging = os.path.join(self.tmp.name, "staging")
        os.makedirs(staging)
        with open(os.path.join(staging, "A.parquet.sources.json"), "w") as f:
            f.write("record from an interrupted download")
        self._download()
        self.assertEqual(self._local("A", ".parquet.sources.json"), "record A")


class CodeBundleTest(unittest.TestCase):
    def test_bundle_holds_every_module_a_job_imports(self):
        import io
        import tarfile
        bundle, bundle_hash = B.code_bundle_and_its_hash()
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
            self.assertEqual(sorted(tar.getnames()), sorted(B.CODE_FILES_SHIPPED_INTO_EACH_JOB))
        self.assertEqual(B.code_bundle_and_its_hash()[1], bundle_hash)


if __name__ == "__main__":
    unittest.main()
