import datetime
import math
import os
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import pandas as pd
import pysam

from simulated_expansions import spike_in_expansions as spike


class DifferencesTest(unittest.TestCase):
    def test_lists_differing_fields_and_rows_only_one_run_has(self):
        training = {("L1", 0): {"eh": 5.0, "coverage": 30.0, "ci_width": math.nan},
                    ("L2", 0): {"eh": 7.0}}
        minicram = {("L1", 0): {"eh": 5.0, "coverage": 31.0, "ci_width": math.nan},
                    ("L3", 1): {"eh": 9.0}}
        self.assertEqual(spike.differences(training, minicram),
                         [("L1", 0, "coverage", 30.0, 31.0), ("L2", 0, "row", True, False),
                          ("L3", 1, "row", False, True)])

    def test_identical_rows_have_no_differences(self):
        rows = {("L1", 0): {"eh": 5.0, "flanking_total": None}}
        self.assertEqual(spike.differences(rows, {("L1", 0): {"eh": 5.0, "flanking_total": None}}), [])


class GenotypeOnHailBatchTest(unittest.TestCase):
    """The parts of the step that run locally: which samples it genotypes and what it downloads."""

    @staticmethod
    def samples(directory, names):
        return [{"name": name, "reads": os.path.join(directory, name, "reads.bam"), "sex": "female",
                 "out_json": os.path.join(directory, name, "expansion_hunter.json.gz")} for name in names]

    BATCH_CREATED = "2026-01-01T00:00:00Z"

    def genotype(self, directory, names, written_in_gcs, batch_state="success", only_missing_outputs=False):
        """Runs the step with Hail Batch, the uploads and the GCS listing mocked; returns
        ``(the output JSONs it downloaded, the samples it sent to the batch)`` and keeps the listing's calls."""
        downloads, hailtop = [], mock.MagicMock()
        hailtop.batch.Batch.return_value.run.return_value.status.return_value = {
            "state": batch_state, "time_created": self.BATCH_CREATED}
        jobs = hailtop.batch.Batch.return_value.new_job
        with mock.patch.dict("sys.modules", {"hailtop": hailtop, "hailtop.batch": hailtop.batch}), \
                mock.patch.object(spike, "run"), \
                mock.patch.object(spike, "sample_names_with_an_output_json_in_gcs",
                                  return_value=set(written_in_gcs)) as listing, \
                mock.patch.object(spike, "run_gcloud_storage_copies_in_parallel",
                                  side_effect=lambda commands: downloads.extend(commands)):
            try:
                spike.genotype_on_hail_batch(self.samples(directory, names), os.path.join(directory, "catalog.json"),
                                             "project", "gs://bucket/run", "a batch", only_missing_outputs)
            finally:
                # The uploads go through the same helper, so the downloads are the copies out of the run's folder.
                self.downloaded = [command[-1] for command in downloads
                                   if command[-2].startswith("gs://bucket/run")]
                self.genotyped = [call.args[0] for call in jobs.call_args_list]
                self.listing_calls = listing.call_args_list
        return self.downloaded, self.genotyped

    def test_every_sample_is_genotyped_unless_only_missing_outputs_is_asked_for(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("A", "B"):
                os.makedirs(os.path.join(d, name))
            open(os.path.join(d, "A", "expansion_hunter.json.gz"), "w").close()
            # By default an existing output is no reason to leave a sample out: the reads or the settings may differ.
            self.assertEqual(self.genotype(d, ["A", "B"], written_in_gcs=["A", "B"])[1], ["A", "B"])
            downloaded, genotyped = self.genotype(d, ["A", "B"], written_in_gcs=["A", "B"],
                                                  only_missing_outputs=True)
            self.assertEqual(genotyped, ["B"])
            self.assertEqual(downloaded, [os.path.join(d, "B", "expansion_hunter.json.gz")])

    def test_no_batch_is_submitted_when_only_the_missing_outputs_are_asked_for_and_none_are(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("A", "B"):
                os.makedirs(os.path.join(d, name))
                open(os.path.join(d, name, "expansion_hunter.json.gz"), "w").close()
            self.assertEqual(self.genotype(d, ["A", "B"], written_in_gcs=[], only_missing_outputs=True), ([], []))

    def test_a_failed_job_still_downloads_the_others_and_names_the_missing_ones(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("A", "B", "C"):
                os.makedirs(os.path.join(d, name))
            with self.assertRaises(RuntimeError) as raised:
                self.genotype(d, ["A", "B", "C"], written_in_gcs=["A", "C"], batch_state="failure")
            self.assertIn("B", str(raised.exception))
            self.assertIn("1 of 3 samples have no output JSON", str(raised.exception))
            # The two that the batch did write are downloaded even though it failed.
            self.assertEqual(self.downloaded, [os.path.join(d, name, "expansion_hunter.json.gz")
                                               for name in ("A", "C")])

    def test_a_missing_samples_output_from_an_earlier_run_is_moved_aside_so_the_rerun_redoes_it(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("A", "B"):
                os.makedirs(os.path.join(d, name))
                open(os.path.join(d, name, "expansion_hunter.json.gz"), "w").close()
            with self.assertRaises(RuntimeError) as raised:
                self.genotype(d, ["A", "B"], written_in_gcs=["A"], batch_state="failure")
            self.assertIn("1 of them had an output of an earlier run's", str(raised.exception))
            self.assertFalse(os.path.exists(os.path.join(d, "B", "expansion_hunter.json.gz")))
            self.assertTrue(os.path.exists(os.path.join(d, "B", "expansion_hunter.json.gz"
                                                        + spike.OUTPUT_OF_AN_EARLIER_RUN_SUFFIX)))
            # So the rerun for the missing outputs genotypes B instead of keeping the output of the earlier run.
            self.assertEqual(self.genotype(d, ["A", "B"], written_in_gcs=["B"], only_missing_outputs=True)[1], ["B"])

    def test_only_the_outputs_written_once_this_batch_started_are_looked_for(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "A"))
            self.genotype(d, ["A"], written_in_gcs=["A"])
        self.assertEqual([call.args[1] for call in self.listing_calls],
                         [datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)])


class SampleNamesWithAnOutputJsonInGcsTest(unittest.TestCase):
    LISTING = ("    1234  2026-03-01T10:00:00Z  gs://bucket/run/A/expansion_hunter.json.gz\n"
               "    2345  2026-01-01T10:00:00Z  gs://bucket/run/B_spiked_2/expansion_hunter.json.gz\n"
               "    3456  2026-03-01T10:00:00Z  gs://bucket/run/A/reads.bam\n"
               "TOTAL: 3 objects, 7035 bytes\n")

    def listed(self, updated_after=None, returncode=0, stdout=None, stderr=""):
        with mock.patch.object(spike.subprocess, "run", return_value=mock.MagicMock(
                returncode=returncode, stdout=self.LISTING if stdout is None else stdout, stderr=stderr)):
            return spike.sample_names_with_an_output_json_in_gcs("gs://bucket/run", updated_after)

    def test_takes_the_sample_name_of_each_output_json_and_leaves_other_files_out(self):
        self.assertEqual(self.listed(), {"A", "B_spiked_2"})

    def test_leaves_out_an_output_written_before_the_given_time(self):
        self.assertEqual(self.listed(datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc)), {"A"})

    def test_a_folder_with_no_output_is_empty_but_a_failed_listing_raises(self):
        self.assertEqual(self.listed(returncode=1, stdout="", stderr="ERROR: ... matched no objects"), set())
        with self.assertRaises(RuntimeError):
            self.listed(returncode=1, stdout="", stderr="ERROR: (gcloud.storage.ls) HTTPError 503: backend error")


class RunGcloudStorageCopiesInParallelTest(unittest.TestCase):
    COMMANDS = [["gcloud", "storage", "cp", "file_%d" % i, "gs://bucket/"] for i in range(20)]

    def test_runs_every_copy_in_one_gcloud_process_at_most_max_parallel_at_once(self):
        ran, running, peak, lock = [], [0], [0], threading.Lock()

        def run(command, env):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.01)
            with lock:
                running[0] -= 1
                ran.append((tuple(command), env["CLOUDSDK_STORAGE_PROCESS_COUNT"],
                            env["CLOUDSDK_STORAGE_THREAD_COUNT"]))

        with mock.patch.object(spike, "run", side_effect=run):
            spike.run_gcloud_storage_copies_in_parallel(self.COMMANDS, max_parallel=4)
        self.assertEqual(sorted(ran), sorted((tuple(command), "1", "1") for command in self.COMMANDS))
        self.assertLessEqual(peak[0], 4)

    def test_raises_if_a_copy_fails(self):
        def run(command, env):
            if command[3] == "file_7":
                raise subprocess.CalledProcessError(1, command)

        with mock.patch.object(spike, "run", side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                spike.run_gcloud_storage_copies_in_parallel(self.COMMANDS, max_parallel=4)


class SexTest(unittest.TestCase):
    def sex_with_one_allele_share(self, share_of_one_allele):
        with tempfile.TemporaryDirectory() as d:
            n = 100
            one = int(n * share_of_one_allele)
            pd.DataFrame({"locus_id": ["X-%d-%d-A" % (i, i + 10) for i in range(n)] + ["1-5-15-A"] * 50,
                          "n_alleles": [1] * one + [2] * (n - one) + [1] * 50}).to_parquet(
                os.path.join(d, "P.parquet"))
            return spike.sex_of("P", tables_dir=d)

    def test_mostly_one_allele_on_chrx_is_male(self):
        self.assertEqual(self.sex_with_one_allele_share(0.95), "male")

    def test_mostly_two_alleles_on_chrx_is_female_whatever_the_autosomes_show(self):
        self.assertEqual(self.sex_with_one_allele_share(0.05), "female")


LOCI = pd.DataFrame([
    {"locus_id": "19-45770204-45770264-CAG", "gene": "DMPK", "chrom": "chr19", "start_0based": 45770204,
     "end": 45770264, "reference_motif": "CAG", "expanded_motif": "CAG", "inheritance": "AD",
     "smallest_reported_pathogenic_bp": 150, "largest_reported_pathogenic_bp": 12000},
    {"locus_id": "4-39348424-39348479-AAAAG", "gene": "RFC1", "chrom": "chr4", "start_0based": 39348424,
     "end": 39348479, "reference_motif": "AAAAG", "expanded_motif": "AAGGG", "inheritance": "AR",
     "smallest_reported_pathogenic_bp": 2000, "largest_reported_pathogenic_bp": 13750},
    {"locus_id": "X-67545316-67545385-GCA", "gene": "AR", "chrom": "chrX", "start_0based": 67545316,
     "end": 67545385, "reference_motif": "GCA", "expanded_motif": "GCA", "inheritance": "XR",
     "smallest_reported_pathogenic_bp": 114, "largest_reported_pathogenic_bp": 204}])
# Loci whose pathogenic alleles stay under 150 bp, and one with no reported pathogenic sizes.
SHORT_LOCI = pd.DataFrame([
    {"locus_id": "X-25013649-25013697-NGC", "gene": "ARX", "chrom": "chrX", "start_0based": 25013649, "end": 25013697,
     "reference_motif": "NGC", "expanded_motif": "NGC", "inheritance": "XR", "smallest_reported_pathogenic_bp": 51,
     "largest_reported_pathogenic_bp": 81},
    {"locus_id": "9-130681606-130681639-GCC", "gene": "PRDM12", "chrom": "chr9", "start_0based": 130681606,
     "end": 130681639, "reference_motif": "GCC", "expanded_motif": "GCC", "inheritance": "AR",
     "smallest_reported_pathogenic_bp": 54, "largest_reported_pathogenic_bp": 57},
    {"locus_id": "1-1435798-1435818-GGCGCGGAGC", "gene": "VWA1", "chrom": "chr1", "start_0based": 1435798,
     "end": 1435818, "reference_motif": "GGCGCGGAGC", "expanded_motif": "GGCGCGGAGC", "inheritance": "AR",
     "smallest_reported_pathogenic_bp": float("nan"), "largest_reported_pathogenic_bp": float("nan")}])


def truth_table(rows):
    return pd.DataFrame(rows, columns=["LocusId"] + spike.TRUTH_COLUMNS[1:] + ["confident"]).set_index("LocusId")


def spikeable(loci, person_allele_repeats):
    """The loci as ``spikeable_loci`` returns them where the person is homozygous with short alleles."""
    return loci.assign(person_allele_repeats=person_allele_repeats, carriers_simulated=True,
                       patients_simulated=loci["inheritance"] == spike.AUTOSOMAL_RECESSIVE_INHERITANCE)


class SpikeableLociTest(unittest.TestCase):
    def test_carriers_where_confident_and_homozygous_patients_at_recessive_loci_where_both_alleles_are_short(self):
        truth = truth_table([
            ["19-45770204-45770264-CAG", "chr19", 45770204, 45770264, "CAG", 20, 12, 12, 1.0, 1.0, True],
            ["4-39348424-39348479-AAAAG", "chr4", 39348424, 39348479, "AAAAG", 11, 11, 13, 1.0, 1.0, True],
            ["X-67545316-67545385-GCA", "chrX", 67545316, 67545385, "GCA", 23, 21, 21, 1.0, 1.0, False]])
        loci = spike.spikeable_loci(LOCI, truth, set())
        self.assertEqual(loci[["locus_id", "person_allele_repeats", "carriers_simulated",
                               "patients_simulated"]].values.tolist(),
                         [["19-45770204-45770264-CAG", 12, True, False], ["4-39348424-39348479-AAAAG", 13, False, True]])
        # Where the person's reads contradict the truth, it is no carrier's known partner allele, and may miss a long
        # allele whose reads would stay in a patient's sample.
        self.assertEqual(spike.spikeable_loci(LOCI, truth, {"4-39348424-39348479-AAAAG"})["locus_id"].tolist(),
                         ["19-45770204-45770264-CAG"])
        self.assertTrue(spike.spikeable_loci(LOCI, truth, {"19-45770204-45770264-CAG",
                                                           "4-39348424-39348479-AAAAG"}).empty)

    def test_no_patients_where_an_allele_of_the_person_is_as_long_as_the_first_size_or_unknown(self):
        rfc1 = LOCI.iloc[[1]]
        for short, long_ in ((11, 30), (30, 30), ("NA", "NA")):
            truth = truth_table([["4-39348424-39348479-AAAAG", "chr4", 39348424, 39348479, "AAAAG", 11, short, long_,
                                  1.0, 1.0, True]])
            loci = spike.spikeable_loci(rfc1, truth, set())
            self.assertEqual(loci["patients_simulated"].tolist() if len(loci) else [],
                             [False] if short == long_ != "NA" else [])

    def test_loci_whose_truth_the_persons_reads_contradict(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"locus_id": ["A", "A", "B"], spike.dataset.TRUTH_CONTRADICTED_BY_SHORT_READS_COLUMN:
                          [False, True, False]}).to_parquet(os.path.join(d, "P.parquet"))
            self.assertEqual(spike.loci_with_truth_contradicted_by_short_reads("P", d), {"A"})

    def test_hemizygous_means_a_males_x_or_y(self):
        self.assertTrue(spike.is_hemizygous("chrX", "male"))
        self.assertFalse(spike.is_hemizygous("chrX", "female"))
        self.assertFalse(spike.is_hemizygous("chr19", "male"))


class SizesTest(unittest.TestCase):
    def test_reported_sizes_are_the_smallest_the_largest_and_one_between_in_whole_motif_copies(self):
        arx, prdm12, vwa1 = SHORT_LOCI.to_dict("records")
        self.assertEqual(spike.reported_pathogenic_sizes_bp(arx), [51, 66, 81])
        # Halfway is 55.5 bp, 18.5 copies, which rounds to the smallest's 18.
        self.assertEqual(spike.reported_pathogenic_sizes_bp(prdm12), [54, 57])
        self.assertEqual(spike.reported_pathogenic_sizes_bp(vwa1), [])

    def test_the_halfway_size_comes_only_where_pathogenic_alleles_start_below_the_first_size(self):
        dmpk, _, ar = LOCI.to_dict("records")
        self.assertEqual(spike.reported_pathogenic_sizes_bp(ar), [114, 132, 204])
        # DMPK's smallest pathogenic allele is as long as the first size, so no size between.
        self.assertEqual(spike.reported_pathogenic_sizes_bp(dmpk), [150, 12000])

    def test_the_largest_reported_size_is_left_out_where_the_sizes_reach_past_it(self):
        # C9orf72's reported maximum, 24,528 bp, is past the last size, which is simulated instead.
        c9 = {"expanded_motif": "GGCCCC", "smallest_reported_pathogenic_bp": 186,
              "largest_reported_pathogenic_bp": 24528, "person_allele_repeats": 10}
        self.assertEqual(spike.reported_pathogenic_sizes_bp(c9), [186])
        self.assertEqual(spike.simulated_sizes_bp(c9)[-1], spike.SIZES_BP[-1])

    def test_sizes_of_the_same_whole_motif_copies_are_one_size(self):
        # PRNP's 24 bp motif: the halfway size (144 bp) and 150 bp are both 6 copies.
        prnp = {"expanded_motif": "CCTCATGGTGGTGGCTGGGGGCAG", "smallest_reported_pathogenic_bp": 120,
                "largest_reported_pathogenic_bp": 384, "person_allele_repeats": 4}
        self.assertEqual(spike.reported_pathogenic_sizes_bp(prnp), [120, 144, 384])
        self.assertEqual(spike.simulated_sizes_bp(prnp), [120, 144, 216, 288, 384])
        # TBP: halfway between 147 bp and 150 bp is 49.5 GCA copies, which rounds to the 50 copies SIZES_BP gives.
        tbp = {"expanded_motif": "GCA", "smallest_reported_pathogenic_bp": 147, "largest_reported_pathogenic_bp": 198,
               "person_allele_repeats": 37}
        self.assertEqual(spike.simulated_sizes_bp(tbp), [147, 150, 198])

    def test_sizes_stop_at_the_largest_pathogenic_allele_and_skip_those_not_longer_than_the_persons_allele(self):
        dmpk, rfc1, ar = LOCI.assign(person_allele_repeats=[12, 40, 21]).to_dict("records")
        self.assertEqual(spike.simulated_sizes_bp(dmpk), [s for s in spike.SIZES_BP if s <= 12000])
        # 40 AAAAG copies are 200 bp: the 150 bp size is no expansion. RFC1's reported sizes are 2,000 and 13,750 bp.
        self.assertEqual(spike.simulated_sizes_bp(rfc1),
                         sorted([s for s in spike.SIZES_BP if s // 5 > 40 and s <= 13750] + [2000, 13750]))
        self.assertEqual(spike.simulated_sizes_bp(ar), [114, 132, 150, 204])  # AR: 204 bp at most
        arx, _, vwa1 = SHORT_LOCI.assign(person_allele_repeats=[20, 0, 2]).to_dict("records")
        self.assertEqual(spike.simulated_sizes_bp(arx), [66, 81])  # 51 bp are 17 copies, no more than the person's 20
        self.assertEqual(spike.simulated_sizes_bp(vwa1), [])


class SimulatedGenotypesTest(unittest.TestCase):
    def test_a_recessive_locus_gets_each_size_once_as_a_carriers_and_once_as_a_patients_longer_allele(self):
        rfc1 = spikeable(LOCI, 11).to_dict("records")[1]
        sizes = spike.simulated_sizes_bp(rfc1)
        genotypes = spike.simulated_genotypes("P", rfc1)
        self.assertEqual([g for g in genotypes if len(g) == 1], [(size_bp,) for size_bp in sizes])
        patients = [g for g in genotypes if len(g) == 2]
        self.assertEqual([longer for longer, _ in patients], sizes)
        self.assertTrue(all(shorter in sizes and shorter <= longer for longer, shorter in patients))
        # Where the longer allele is pathogenic (2 kb at RFC1), so is the shorter one; below, any shorter size.
        self.assertTrue(all(shorter >= 2000 for longer, shorter in patients if longer >= 2000))
        self.assertGreater(len({shorter for longer, shorter in patients if longer < 2000}), 1)
        self.assertEqual(spike.simulated_genotypes("P", rfc1), genotypes)
        # Each person draws the shorter alleles on their own.
        self.assertNotEqual(spike.simulated_genotypes("Q", rfc1), genotypes)

    def test_other_loci_get_carriers_only(self):
        for locus in spikeable(LOCI, [12, 11, 21]).to_dict("records")[::2]:
            self.assertEqual(spike.simulated_genotypes("P", locus),
                             [(size_bp,) for size_bp in spike.simulated_sizes_bp(locus)])

    def test_a_heterozygous_person_gets_patients_only(self):
        rfc1 = spikeable(LOCI, 13).assign(carriers_simulated=False).to_dict("records")[1]
        self.assertEqual([len(g) for g in spike.simulated_genotypes("P", rfc1)],
                         [2] * len(spike.simulated_sizes_bp(rfc1)))


class AlleleStructuresTest(unittest.TestCase):
    def test_blocks_add_up_to_the_allele_with_counts_shares_and_the_rest(self):
        rng = spike.sim.seeded_rng("test")
        blocks = spike.structure_blocks("AAAGG:count=15-25;AAGGG:rest;AAAGG:count=10-10", 600, rng)
        self.assertEqual([motif for motif, _ in blocks], ["AAAGG", "AAGGG", "AAAGG"])
        self.assertTrue(15 <= blocks[0][1] <= 25 and blocks[2][1] == 10)
        self.assertEqual(sum(n for _, n in blocks), 600)
        shares = spike.structure_blocks("GAAAT:share=0.2-0.3;AAAAT:rest", 1000, rng)
        self.assertTrue(200 <= shares[0][1] <= 300 and sum(n for _, n in shares) == 1000)

    def test_counts_take_less_than_half_of_a_short_allele(self):
        # FXN's GGA tract of 22-342 triplets in alleles of 40 and 41 copies.
        for copies in (40, 41):
            blocks = spike.structure_blocks("GGA:count=22-44-342;GAA:rest", copies, spike.sim.seeded_rng("test"))
            self.assertLess(blocks[0][1], blocks[1][1])
            self.assertEqual(sum(n for _, n in blocks), copies)

    def test_counts_with_a_published_median_have_that_median(self):
        counts = [spike.drawn_block_count("22-44-342", spike.sim.seeded_rng("count", i)) for i in range(4000)]
        self.assertAlmostEqual(sum(count < 44 for count in counts) / len(counts), 0.5, delta=0.03)
        self.assertTrue(22 <= min(counts) and max(counts) <= 342)

    def test_repeat_of_blocks_spans_each_block_and_takes_the_motif_with_the_most_copies(self):
        # MARCHF6: two ATTTT blocks outnumber the ATTTC block between them.
        repeat = spike.repeat_of_blocks([("ATTTT", 40), ("ATTTC", 45), ("ATTTT", 15)])
        self.assertEqual((repeat.start, repeat.end, repeat.motif), (spike.FLANK_BP, spike.FLANK_BP + 500, "ATTTT"))
        self.assertEqual(repeat.blocks, [(spike.FLANK_BP, spike.FLANK_BP + 200, "ATTTT"),
                                         (spike.FLANK_BP + 200, spike.FLANK_BP + 425, "ATTTC"),
                                         (spike.FLANK_BP + 425, spike.FLANK_BP + 500, "ATTTT")])

    def test_added_counts_are_added_to_the_alleles_copies_instead_of_taken_from_them(self):
        # DAB1's reference-motif blocks around a pathogenic insertion of 30 copies.
        blocks = spike.structure_blocks("AAAAT:added_count=75-91;GAAAT:rest;AAAAT:added_count=57-76", 30,
                                        spike.sim.seeded_rng("test"))
        self.assertEqual([motif for motif, _ in blocks], ["AAAAT", "GAAAT", "AAAAT"])
        self.assertEqual(blocks[1][1], 30)
        self.assertTrue(75 <= blocks[0][1] <= 91 and 57 <= blocks[2][1] <= 76)
        self.assertGreater(sum(n for _, n in blocks), 30)

    def test_the_committed_insertion_loci_make_alleles_longer_than_the_simulated_insertion(self):
        structures = spike.load_pathogenic_allele_structures()
        loci = pd.read_csv(spike.DISEASE_LOCI_TSV, sep="\t").set_index("gene")
        for gene in ("DAB1", "BEAN1", "STARD7", "TNRC6A", "RAPGEF2", "YEATS2"):
            locus = loci.loc[gene]
            blocks = spike.structure_blocks(structures[locus["locus_id"]][0]["blocks_on_plus_strand"], 100,
                                            spike.sim.seeded_rng(gene))
            # One block is the pathogenic motif, of the simulated size, and the others add reference-motif copies.
            pathogenic = [n for motif, n in blocks if locus["expanded_motif"] in spike.rotations(motif)]
            self.assertEqual(pathogenic, [100], gene)
            self.assertGreater(sum(n for _, n in blocks), 100, gene)
            # The added copies are of the reference motif, so the allele is purer than the pathogenic motif alone.
            self.assertGreater(spike.blocks_purity(blocks, locus["reference_motif"]),
                               spike.simulated_allele_purity(locus["expanded_motif"], locus["reference_motif"]), gene)

    def test_structures_are_drawn_by_their_share_and_the_rest_are_pure(self):
        structures = [{"structure": "composite", "share_of_alleles": 0.25,
                       "blocks_on_plus_strand": "GGA:count=5-5;GAA:rest"}]
        drawn = [spike.drawn_allele_structure(structures, "GAA", 100, spike.sim.seeded_rng("allele", i))
                 for i in range(2000)]
        self.assertAlmostEqual(sum(name == "composite" for name, _ in drawn) / len(drawn), 0.25, delta=0.03)
        self.assertIn(("pure", [("GAA", 100)]), drawn)
        self.assertIn(("composite", [("GGA", 5), ("GAA", 95)]), drawn)
        self.assertEqual(spike.drawn_allele_structure([], "GAA", 100, spike.sim.seeded_rng("allele")),
                         ("pure", [("GAA", 100)]))

    def test_the_committed_structures_fit_their_loci(self):
        loci = pd.read_csv(spike.DISEASE_LOCI_TSV, sep="\t").set_index("locus_id")
        for locus_id, structures in spike.load_pathogenic_allele_structures().items():
            locus = loci.loc[locus_id]
            self.assertLessEqual(sum(s["share_of_alleles"] for s in structures), 1.0, locus_id)
            for structure in structures:
                blocks = [block.split(":") for block in structure["blocks_on_plus_strand"].split(";")]
                self.assertEqual([size for _, size in blocks].count("rest"), 1, locus_id)
                # Blocks of the motif's length keep the allele's copies, the truth's count, whole.
                self.assertTrue(all(len(motif) == len(locus["expanded_motif"]) for motif, _ in blocks), locus_id)
                most_shared = sum(float(size.split("=")[1].split("-")[1]) for _, size in blocks
                                  if size.startswith("share="))
                self.assertLess(most_shared, 1.0, locus_id)


class PlanTest(unittest.TestCase):
    def test_each_locus_gets_each_genotype_once_in_samples_of_one_genotype_per_locus(self):
        loci = spikeable(pd.concat([LOCI, SHORT_LOCI], ignore_index=True), [12, 11, 21, 16, 11, 2])
        plan = spike.plan_spiked_samples("P", loci)
        by_locus = {}
        for label, locus, sizes in plan:
            by_locus.setdefault(locus["locus_id"], []).append(sizes)
        self.assertEqual(sorted(by_locus), sorted(loci["locus_id"][:5]))  # VWA1 has no sizes
        for locus in loci.iloc[:5].to_dict("records"):
            self.assertEqual(sorted(by_locus[locus["locus_id"]]), sorted(spike.simulated_genotypes("P", locus)))
        self.assertEqual(len({(label, locus["locus_id"]) for label, locus, _ in plan}), len(plan))
        self.assertEqual(plan[0][0], "P_spiked_0")
        # DMPK and AR share a repeat, so they take turns: 12 + 4 samples (DMPK's alleles reach 12 kb), fewer than
        # RFC1's 28 (14 carriers, 14 patients), which sets the count.
        self.assertEqual(len({label for label, _, _ in plan}), 28)

    def test_nearby_loci_take_turns_and_are_never_spiked_in_the_same_sample(self):
        loci = pd.DataFrame([
            {"locus_id": "ARX_1", "gene": "ARX", "chrom": "chrX", "start_0based": 25013649, "end": 25013697,
             "expanded_motif": "NGC", "person_allele_repeats": 16},
            # A motif of another repeat, so only the distance keeps the two loci apart.
            {"locus_id": "ARX_2", "gene": "ARX", "chrom": "chrX", "start_0based": 25013529, "end": 25013565,
             "expanded_motif": "GAA", "person_allele_repeats": 12},
            {"locus_id": "DMPK", "gene": "DMPK", "chrom": "chr19", "start_0based": 45770204, "end": 45770264,
             "expanded_motif": "CAG", "person_allele_repeats": 12}]).assign(
            inheritance="AD", smallest_reported_pathogenic_bp=150, largest_reported_pathogenic_bp=spike.SIZES_BP[-1],
            carriers_simulated=True, patients_simulated=False)
        self.assertEqual([[l["locus_id"] for l in c] for c in spike.clusters_of_nearby_loci(loci.to_dict("records"))],
                         [["DMPK"], ["ARX_2", "ARX_1"]])
        loci_by_sample, sizes = self.plan_by_sample_and_locus(loci)
        self.assertTrue(all(not {"ARX_1", "ARX_2"} <= set(ids) for ids in loci_by_sample.values()))
        for locus_id in ("ARX_1", "ARX_2", "DMPK"):
            self.assertEqual(sorted(sizes[locus_id]), [(size_bp,) for size_bp in spike.SIZES_BP])
        self.assertEqual(len(loci_by_sample), 2 * len(spike.SIZES_BP))

    def test_loci_of_one_repeat_take_turns_wherever_they_are(self):
        # CAG and CTG are one repeat read from either strand: bwa puts one expansion's in-repeat reads on the other.
        loci = pd.DataFrame([
            {"locus_id": "DMPK", "gene": "DMPK", "chrom": "chr19", "start_0based": 45770204, "end": 45770264,
             "expanded_motif": "CAG", "person_allele_repeats": 12},
            {"locus_id": "TCF4", "gene": "TCF4", "chrom": "chr18", "start_0based": 55586155, "end": 55586227,
             "expanded_motif": "CTG", "person_allele_repeats": 20},
            {"locus_id": "RFC1", "gene": "RFC1", "chrom": "chr4", "start_0based": 39348424, "end": 39348479,
             "expanded_motif": "AAGGG", "person_allele_repeats": 11}]).assign(
            inheritance="AD", smallest_reported_pathogenic_bp=150, largest_reported_pathogenic_bp=spike.SIZES_BP[-1],
            carriers_simulated=True, patients_simulated=False)
        self.assertEqual(sorted(sorted(l["locus_id"] for l in g)
                                for g in spike.groups_of_loci_kept_apart(loci.to_dict("records"))),
                         [["DMPK", "TCF4"], ["RFC1"]])
        loci_by_sample, sizes = self.plan_by_sample_and_locus(loci)
        self.assertTrue(all(not {"DMPK", "TCF4"} <= set(ids) for ids in loci_by_sample.values()))
        for locus_id in ("DMPK", "TCF4", "RFC1"):
            self.assertEqual(sorted(sizes[locus_id]), [(size_bp,) for size_bp in spike.SIZES_BP])

    @staticmethod
    def plan_by_sample_and_locus(loci):
        loci_by_sample, sizes = {}, {}
        for label, locus, sizes_bp in spike.plan_spiked_samples("P", loci):
            loci_by_sample.setdefault(label, []).append(locus["locus_id"])
            sizes.setdefault(locus["locus_id"], []).append(sizes_bp)
        return loci_by_sample, sizes

    def test_dropping_is_a_fixed_choice_at_about_the_given_share(self):
        names = ["read%d" % i for i in range(4000)]
        dropped = [n for n in names if spike.is_dropped("P_spiked_0", n, 0.5)]
        self.assertAlmostEqual(len(dropped) / len(names), 0.5, delta=0.03)
        self.assertEqual(dropped, [n for n in names if spike.is_dropped("P_spiked_0", n, 0.5)])
        self.assertEqual([n for n in names if spike.is_dropped("P_spiked_0", n, 1.0)], names)

    def test_purity_of_the_simulated_allele_against_the_reference_motif(self):
        self.assertEqual(spike.simulated_allele_purity("CAG", "CAG"), 1.0)
        self.assertEqual(spike.simulated_allele_purity("AGC", "CAG"), 1.0)
        self.assertAlmostEqual(spike.simulated_allele_purity("AAGGG", "AAAAG"), 0.6)
        # The allele holds real bases where the motif says N, and they never match an N, as in the truth tables.
        self.assertAlmostEqual(spike.simulated_allele_purity("NGC", "NGC"), 2 / 3)


class WriteTruthTest(unittest.TestCase):
    def test_carrier_patient_and_hemizygous_truth_alleles(self):
        truth = truth_table([
            ["19-45770204-45770264-CAG", "chr19", 45770204, 45770264, "CAG", 20, 12, 12, 0.9, 0.9, True],
            ["X-67545316-67545385-GCA", "chrX", 67545316, 67545385, "GCA", 23, 21, 21, 1.0, 1.0, True],
            ["4-39348424-39348479-AAAAG", "chr4", 39348424, 39348479, "AAAAG", 11, 11, 11, 1.0, 1.0, True]])
        manifest = pd.DataFrame([
            {"locus_id": "19-45770204-45770264-CAG", "expanded_motif": "CAG", "simulated_alleles": 1, "copies": 100,
             "simulated_allele_purity": 1.0, "shorter_simulated_allele_copies": None,
             "shorter_simulated_allele_purity": None, "person_allele_repeats": 12, "hemizygous": False},
            {"locus_id": "X-67545316-67545385-GCA", "expanded_motif": "GCA", "simulated_alleles": 1, "copies": 50,
             "simulated_allele_purity": 1.0, "shorter_simulated_allele_copies": None,
             "shorter_simulated_allele_purity": None, "person_allele_repeats": 21, "hemizygous": True},
            {"locus_id": "4-39348424-39348479-AAAAG", "expanded_motif": "AAGGG", "simulated_alleles": 2, "copies": 300,
             "simulated_allele_purity": 0.6, "shorter_simulated_allele_copies": 90,
             "shorter_simulated_allele_purity": 0.6, "person_allele_repeats": 11, "hemizygous": False}])
        with tempfile.TemporaryDirectory() as d:
            spike.write_truth(manifest.to_dict("records"), truth, os.path.join(d, "t.tsv.gz"),
                              os.path.join(d, "t.bed"))
            written = pd.read_csv(os.path.join(d, "t.tsv.gz"), sep="\t", dtype=str)
            bed = pd.read_csv(os.path.join(d, "t.bed"), sep="\t", header=None)
        # Whole numbers, though the manifest's column of a patient's shorter allele is a float one.
        self.assertEqual(written[["NumRepeatsShortAllele", "NumRepeatsLongAllele"]].values.tolist(),
                         [["12", "100"], ["50", "50"], ["90", "300"]])
        # A patient's alleles are both simulated: AAGGG is 3/5 AAAAG.
        self.assertEqual(written[["RepeatPurityShortAllele", "RepeatPurityLongAllele"]].astype(float).values.tolist(),
                         [[0.9, 1.0], [1.0, 1.0], [0.6, 0.6]])
        self.assertEqual(bed.values.tolist(), [["chr19", 45770194, 45770274], ["chrX", 67545306, 67545395],
                                               ["chr4", 39348414, 39348489]])

    def test_whichever_allele_came_out_longer_is_the_long_one(self):
        truth = truth_table([["4-39348424-39348479-AAAAG", "chr4", 39348424, 39348479, "AAAAG", 11, 11, 11, 1.0, 1.0,
                              True]])
        # A patient whose allele planned as the shorter one came out longer, as added blocks can make it.
        manifest = [{"locus_id": "4-39348424-39348479-AAAAG", "expanded_motif": "AAGGG", "simulated_alleles": 2,
                     "copies": 90, "simulated_allele_purity": 0.6, "shorter_simulated_allele_copies": 300,
                     "shorter_simulated_allele_purity": 0.8, "person_allele_repeats": 11, "hemizygous": False}]
        with tempfile.TemporaryDirectory() as d:
            spike.write_truth(manifest, truth, os.path.join(d, "t.tsv.gz"), os.path.join(d, "t.bed"))
            written = pd.read_csv(os.path.join(d, "t.tsv.gz"), sep="\t")
        self.assertEqual(written[["NumRepeatsShortAllele", "NumRepeatsLongAllele"]].values.tolist(), [[90, 300]])
        self.assertEqual(written[["RepeatPurityShortAllele", "RepeatPurityLongAllele"]].values.tolist(), [[0.6, 0.8]])


class SimulatePersonTest(unittest.TestCase):
    LOCI = pd.DataFrame([
        {"locus_id": "19-45770204-45770264-CAG", "gene": "DMPK", "chrom": "chr19", "start_0based": 45770204,
         "end": 45770264, "reference_motif": "CAG", "expanded_motif": "CAG", "person_allele_repeats": 12},
        {"locus_id": "X-67545316-67545385-GCA", "gene": "AR", "chrom": "chrX", "start_0based": 67545316,
         "end": 67545385, "reference_motif": "GCA", "expanded_motif": "GCA", "person_allele_repeats": 21},
        {"locus_id": "4-39348424-39348479-AAAAG", "gene": "RFC1", "chrom": "chr4", "start_0based": 39348424,
         "end": 39348479, "reference_motif": "AAAAG", "expanded_motif": "AAGGG", "person_allele_repeats": 11}])
    NAMES = ["pair%d" % i for i in range(2000)]

    def simulate(self, plan, allele_structures=None):
        written, self.repeat_motifs = [], []

        def write_simulated_allele_pairs(sequence, rate, share, profile, mean_fragment, name_prefix, fastq1, fastq2,
                                         repeat, repeat_errors, fragment_keep_exponent):
            written.append((name_prefix, sequence, share, fragment_keep_exponent))
            self.repeat_motifs.append(repeat.motif)
            return 10

        pairs = {(locus["chrom"], locus["start_0based"] - spike.FLANK_BP): set(self.NAMES)
                 for locus in self.LOCI.to_dict("records")}
        with mock.patch.object(spike, "simulation_inputs", return_value=(None, None, 400, None)), \
                mock.patch.object(spike, "read_pairs_within_windows", return_value=pairs), \
                mock.patch.object(spike, "fragment_start_rate_near_locus", return_value=0.1), \
                mock.patch.object(spike.sim, "expanded_allele_sequence",
                                  side_effect=lambda fasta, chrom, start, end, motif, copies, flank: motif * copies), \
                mock.patch.object(spike.sim, "allele_sequence_with_reference_flanks",
                                  side_effect=lambda fasta, chrom, start, end, repeat, flank: repeat), \
                mock.patch.object(spike, "write_simulated_allele_pairs", side_effect=write_simulated_allele_pairs):
            manifest, dropped = spike.simulate_person("P", "male", self.LOCI, plan, "work", None, None,
                                                      allele_structures or {}, None, None)
        return manifest, dropped, written

    def test_an_allele_takes_a_published_structure_by_its_share_and_its_purity_counts_every_block(self):
        rfc1 = self.LOCI.to_dict("records")[2]
        structures = {rfc1["locus_id"]: [{"structure": "AAAGG-AAGGG", "share_of_alleles": 1.0,
                                          "blocks_on_plus_strand": "AAGGG:share=0.3-0.3;AAAGG:rest"}]}
        manifest, _, written = self.simulate([("P_spiked_0", rfc1, (1000, 1000))], structures)
        self.assertEqual([sequence for _, sequence, _, _ in written], ["AAGGG" * 60 + "AAAGG" * 140] * 2)
        # The reads follow the main block's motif, not the expanded motif AAGGG.
        self.assertEqual(self.repeat_motifs, ["AAAGG", "AAAGG"])
        row = manifest[0]
        self.assertEqual([row["simulated_allele_structure"], row["simulated_allele_blocks"],
                          row["shorter_simulated_allele_structure"]], ["AAAGG-AAGGG", "AAGGG:60;AAAGG:140",
                                                                       "AAAGG-AAGGG"])
        # Against AAAAG, AAGGG matches 3 of 5 bases and AAAGG 4 of 5.
        self.assertAlmostEqual(row["simulated_allele_purity"], (60 * 0.6 + 140 * 0.8) / 200)
        pure, _, _ = self.simulate([("P_spiked_0", rfc1, (1000,))])
        self.assertEqual([pure[0]["simulated_allele_structure"], pure[0]["simulated_allele_blocks"]],
                         ["pure", "AAGGG:200"])
        self.assertAlmostEqual(pure[0]["simulated_allele_purity"], 0.6)

    def test_added_blocks_count_towards_the_alleles_copies_in_the_manifest(self):
        rfc1 = self.LOCI.to_dict("records")[2]
        structures = {rfc1["locus_id"]: [{"structure": "insertion", "share_of_alleles": 1.0,
                                          "blocks_on_plus_strand": "AAAAG:added_count=10-10;AAGGG:rest"}]}
        manifest, _, written = self.simulate([("P_spiked_0", rfc1, (1000,))], structures)
        row = manifest[0]
        # The 1,000 bp size is the inserted block's 200 copies; the 10 added copies make the allele 210.
        self.assertEqual([row["copies"], row["size_bp"], row["simulated_allele_blocks"]],
                         [210, 1050, "AAAAG:10;AAGGG:200"])
        self.assertEqual(len(written[0][1]), 1050)

    def test_shares_of_coverage_and_dropped_pairs_for_carriers_patients_and_a_males_x(self):
        dmpk, ar, rfc1 = self.LOCI.to_dict("records")
        manifest, dropped, written = self.simulate(
            [("P_spiked_0", dmpk, (300,)), ("P_spiked_1", ar, (150,)), ("P_spiked_2", rfc1, (1500, 450))])
        self.assertEqual([(prefix, len(sequence), share) for prefix, sequence, share, _ in written],
                         [("P_spiked_0:19-45770204-45770264-CAG:0", 300, 0.5),
                          ("P_spiked_1:X-67545316-67545385-GCA:0", 150, 1.0),
                          ("P_spiked_2:4-39348424-39348479-AAAAG:0", 1500, 0.5),
                          ("P_spiked_2:4-39348424-39348479-AAAAG:1", 450, 0.5)])
        self.assertAlmostEqual(len(dropped["P_spiked_0"]) / len(self.NAMES), 0.5, delta=0.05)
        self.assertEqual(dropped["P_spiked_1"], set(self.NAMES))
        self.assertEqual(dropped["P_spiked_2"], set(self.NAMES))
        columns = ["simulated_alleles", "copies", "size_bp", "shorter_simulated_allele_copies",
                   "shorter_simulated_allele_size_bp", "hemizygous", "simulated_fragments_before_loss"]
        self.assertEqual([[row[c] for c in columns] for row in manifest],
                         [[1, 100, 300, None, None, False, 10], [1, 50, 150, None, None, True, 10],
                          [2, 300, 1500, 90, 450, False, 20]])
        # Both of a patient's alleles come from one library, so they lose fragments over the repeat alike.
        self.assertEqual([exponent for *_, exponent in written[2:]], [manifest[2]["fragment_keep_exponent"]] * 2)


class DownsampleSpikedSamplesTest(unittest.TestCase):
    @staticmethod
    def write_pairs(path, names):
        header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for i, name in enumerate(names):
                for mate, start in ((0x40, 100 + i), (0x80, 300 + i)):
                    read = pysam.AlignedSegment(out.header)
                    read.query_name, read.query_sequence, read.cigarstring = name, "A" * 100, "100M"
                    read.flag, read.reference_name, read.reference_start = 0x1 | mate, "chr1", start
                    read.next_reference_name, read.next_reference_start = "chr1", 400 - start
                    read.mapping_quality, read.query_qualities = 60, pysam.qualitystring_to_array("I" * 100)
                    out.write(read)
        pysam.sort("-o", path + ".sorted.bam", path)
        os.replace(path + ".sorted.bam", path)
        pysam.index(path)

    def test_a_downsampled_copy_and_a_matched_control_keep_the_same_share_of_the_persons_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            source, work = os.path.join(d, "source"), os.path.join(d, "work")
            fasta = os.path.join(d, "ref.fa")
            with open(fasta, "w") as f:
                f.write(">chr1\n" + "ACGT" * 250 + "\n")
            pysam.faidx(fasta)
            real = ["real%d" % i for i in range(400)]
            os.makedirs(os.path.join(work, "minicrams"))
            self.write_pairs(spike.minicram_path(work, "P"), real)
            os.makedirs(spike.spiked_dir(source, "P_spiked_3"))
            self.write_pairs(os.path.join(spike.spiked_dir(source, "P_spiked_3"), "reads.bam"),
                             real[:200] + ["P_spiked_3:L:0:%d" % i for i in range(200)])
            for name in ("truth.tsv.gz", "truth.bed"):
                open(os.path.join(spike.spiked_dir(source, "P_spiked_3"), name), "w").close()
            # 0.1 fragment starts per bp of 100 bp reads: 20x.
            pd.DataFrame([{"label": "P_spiked_3", "person": "P", "sex": "female", "locus_id": "L",
                           "simulated_alleles": 1, "copies": 100, "size_bp": 300, "simulated_allele_structure": "pure",
                           "fragment_starts_per_bp": 0.1}]).to_csv(
                os.path.join(source, "spiked", "manifest.tsv"), sep="\t", index=False)
            truth = truth_table([["L", "chr1", 100, 130, "CAG", 10, 10, 12, 1.0, 1.0, True]])
            with mock.patch.object(spike, "person_truth_at_loci", return_value=truth):
                spike.downsample_spiked_samples(source, work, fasta, coverage_range=(10.0, 10.0))
            manifest = pd.read_csv(os.path.join(work, "spiked", "manifest.tsv"), sep="\t")
            copy, control = (spike.spiked_dir(work, "P_spiked_%d" % (3 + offset)) for offset in (
                spike.DOWNSAMPLED_COPY_NUMBER_OFFSET, spike.MATCHED_CONTROL_NUMBER_OFFSET))
            names = {}
            for name, folder in (("copy", copy), ("control", control)):
                with pysam.AlignmentFile(os.path.join(folder, "reads.bam")) as reads:
                    names[name] = {read.query_name for read in reads.fetch(until_eof=True)}
            control_truth = pd.read_csv(os.path.join(control, "truth.tsv.gz"), sep="\t")
            self.assertTrue(os.path.exists(os.path.join(copy, "truth.bed")))
        self.assertEqual(manifest[["label", "matched_control", "simulated_alleles", "source_label"]].values.tolist(),
                         [["P_spiked_100003", False, 1, "P_spiked_3"], ["P_spiked_200003", True, 0, "P_spiked_3"]])
        self.assertEqual(manifest["downsample_fraction"].tolist(), [0.5, 0.5])
        self.assertTrue(manifest.loc[1, ["copies", "size_bp", "simulated_allele_structure"]].isna().all())
        # About half of each kind of pair is kept, and the same person's pairs in both.
        self.assertAlmostEqual(len(names["control"]) / len(real), 0.5, delta=0.1)
        self.assertEqual({n for n in names["copy"] if n.startswith("real")}, {n for n in names["control"]
                                                                               if n in real[:200]})
        self.assertEqual(control_truth[["NumRepeatsShortAllele", "NumRepeatsLongAllele"]].values.tolist(), [[10, 12]])


class WriteSpikedReadsTest(unittest.TestCase):
    @staticmethod
    def write_bam(path, contigs, reads):
        header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": name, "LN": 1000} for name in contigs]}
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for name, contig, start in reads:
                read = pysam.AlignedSegment(out.header)
                read.query_name, read.query_sequence, read.flag = name, "A" * 10, 0
                read.reference_name, read.reference_start, read.cigarstring = contig, start, "10M"
                read.mapping_quality, read.query_qualities = 60, pysam.qualitystring_to_array("I" * 10)
                out.write(read)
        pysam.index(path)

    def test_drops_the_named_pairs_and_moves_simulated_reads_onto_the_persons_header(self):
        with tempfile.TemporaryDirectory() as d:
            real, simulated, out = (os.path.join(d, name) for name in ("real.bam", "sim.bam", "out.bam"))
            self.write_bam(real, ["chr1", "chr2"], [("keep", "chr1", 100), ("drop", "chr1", 200)])
            # The simulated reads' header lists the contigs in another order.
            self.write_bam(simulated, ["chr2", "chr1"], [("P_spiked_0:L:0", "chr1", 150)])
            with pysam.AlignmentFile(simulated) as bam:
                simulated_reads = list(bam.fetch(until_eof=True))
            spike.write_spiked_reads(real, None, {"drop"}, simulated_reads, out)
            with pysam.AlignmentFile(out) as bam:
                written = [(r.query_name, r.reference_name, r.reference_start) for r in bam.fetch(until_eof=True)]
            self.assertTrue(os.path.exists(out + ".bai"))
        self.assertEqual(written, [("keep", "chr1", 100), ("P_spiked_0:L:0", "chr1", 150)])

    def test_read_pairs_that_lie_in_each_window_leaving_out_those_with_a_mate_elsewhere(self):
        header = {"HD": {"VN": "1.6", "SO": "coordinate"},
                  "SQ": [{"SN": "chr1", "LN": 5000}, {"SN": "chr2", "LN": 5000}]}
        # (name, start, mate contig, mate start or None for an unmapped mate)
        reads = [("both_in_first", 100, "chr1", 300), ("both_in_first", 300, "chr1", 100),
                 ("mate_far_away", 150, "chr1", 4000), ("mate_on_chr2", 160, "chr2", 100),
                 ("mate_unmapped", 170, "chr1", None), ("both_in_second", 2100, "chr1", 2400),
                 ("both_in_second", 2400, "chr1", 2100)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "real.bam")
            with pysam.AlignmentFile(path, "wb", header=header) as out:
                for name, start, mate_contig, mate_start in sorted(reads, key=lambda r: r[1]):
                    read = pysam.AlignedSegment(out.header)
                    read.query_name, read.query_sequence, read.cigarstring = name, "A" * 10, "10M"
                    read.flag = 0x1 | (0x8 if mate_start is None else 0)
                    read.reference_name, read.reference_start, read.mapping_quality = "chr1", start, 60
                    read.next_reference_name = "chr1" if mate_start is None else mate_contig
                    read.next_reference_start = start if mate_start is None else mate_start
                    read.query_qualities = pysam.qualitystring_to_array("I" * 10)
                    out.write(read)
            pysam.index(path)
            pairs = spike.read_pairs_within_windows(path, None, [("chr1", 0, 1000), ("chr1", 2000, 3000)])
        self.assertEqual(pairs, {("chr1", 0): {"both_in_first"}, ("chr1", 2000): {"both_in_second"}})


class SpikeManifestTest(unittest.TestCase):
    def test_spiking_some_people_keeps_the_other_peoples_samples_listed(self):
        def simulate_person(person, *_):
            return ([{"label": "%s_spiked_0" % person, "person": person, "simulated_fragments_before_loss": 1}],
                    {"%s_spiked_0" % person: set()})

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(spike, "sex_of", return_value="female"), \
                mock.patch.object(spike, "person_truth_at_loci"), \
                mock.patch.object(spike, "loci_with_truth_contradicted_by_short_reads", return_value=set()), \
                mock.patch.object(spike, "spikeable_loci",
                                  return_value=spikeable(LOCI.iloc[:1], 12)), \
                mock.patch.object(spike, "simulate_person", side_effect=simulate_person), \
                mock.patch.object(spike, "write_samples_with_simulated_reads"), \
                mock.patch.object(spike, "write_truth"), mock.patch.object(spike.sim, "load_repeat_read_errors"):
            os.makedirs(os.path.join(d, "spiked"))
            pd.DataFrame([{"label": "A_spiked_0", "person": "A"}, {"label": "B_spiked_0", "person": "B"}]).to_csv(
                os.path.join(d, "spiked", "manifest.tsv"), sep="\t", index=False)
            spike.spike(["B"], d, None, "bwa", None, None, None)
            manifest = pd.read_csv(os.path.join(d, "spiked", "manifest.tsv"), sep="\t")
        self.assertEqual(sorted(manifest["label"]), ["A_spiked_0", "B_spiked_0"])


class WriteTablesTest(unittest.TestCase):
    """``write_tables``: which spiked samples get a per-source parquet."""

    @staticmethod
    def rows_of(label):
        """One row for every sample but the control that has none, which stands for a person who carries the
        reference allele at every locus of its sample."""
        if label == "B_spiked_200000":
            return pd.DataFrame()
        return pd.DataFrame([{"sample_id": label, "locus_id": "1-1-2-A", "eh": 10.0}])

    def write(self, directory, labels):
        pd.DataFrame([{"label": label, "sex": "female"} for label in labels]).to_csv(
            os.path.join(directory, "spiked", "manifest.tsv"), sep="\t", index=False)
        with mock.patch.object(spike.dataset, "extract_rows_and_join_truth",
                               side_effect=lambda *args, **kwargs: self.rows_of(args[3])) as extract:
            spike.write_tables(directory, os.path.join(directory, "tables"))
        # Without this the extractor raises on a control whose person is hom-ref at every locus of its sample.
        for call in extract.call_args_list:
            self.assertIs(call.kwargs["allow_sample_with_no_variant_truth_loci"], True)
            self.assertIs(call.kwargs["keep_hom_ref_loci"], True)
        return sorted(name for name in os.listdir(os.path.join(directory, "tables")))

    def test_a_sample_without_rows_gets_no_parquet_and_loses_the_one_an_earlier_run_wrote(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "spiked"))
            os.makedirs(os.path.join(d, "tables"))
            # An earlier run wrote a parquet for the sample that now has no rows; training on it would use its rows
            # as this sample's.
            stale = os.path.join(d, "tables", "B_spiked_200000.parquet")
            pd.DataFrame([{"eh": 1.0}]).to_parquet(stale, index=False)
            self.assertEqual(self.write(d, ["A_spiked_100000", "B_spiked_200000"]),
                             ["A_spiked_100000.parquet"])
            self.assertFalse(os.path.exists(stale))

    def test_a_parquet_of_a_sample_the_manifest_no_longer_lists_is_removed(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "spiked"))
            os.makedirs(os.path.join(d, "tables"))
            pd.DataFrame([{"eh": 1.0}]).to_parquet(os.path.join(d, "tables", "C_spiked_0.parquet"), index=False)
            self.assertEqual(self.write(d, ["A_spiked_100000"]), ["A_spiked_100000.parquet"])


class NumberOfSimulatedPairsTest(unittest.TestCase):
    def test_one_pair_per_mean_fragment_start_position_at_the_share_of_the_rate(self):
        self.assertEqual(spike.number_of_simulated_pairs(0.1, 0.5, 4400, 400), 200)
        # An allele shorter than a mean fragment still gets the pairs of one start position's worth of fragments.
        self.assertEqual(spike.number_of_simulated_pairs(2.0, 0.5, 300, 400), 1)


if __name__ == "__main__":
    unittest.main()
