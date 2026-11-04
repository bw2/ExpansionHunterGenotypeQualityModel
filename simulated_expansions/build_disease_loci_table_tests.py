import io
import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from simulated_expansions import build_disease_loci_table as table


class MotifTest(unittest.TestCase):
    def test_reverse_complement(self):
        self.assertEqual(table.reverse_complement("AAGGN"), "NCCTT")

    def test_the_motifs_of_one_repeat_share_a_family_and_n_is_a_letter(self):
        self.assertEqual({table.motif_family(m) for m in ("CAG", "AGC", "GCA", "CTG", "TGC", "gct")}, {"AGC"})
        self.assertNotEqual(table.motif_family("CGG"), table.motif_family("CAG"))
        self.assertNotEqual(table.motif_family("NGC"), table.motif_family("CAG"))

    def test_motifs_match_any_rotation_on_either_strand_with_n_as_a_wildcard(self):
        self.assertTrue(table.motifs_match("CAG", "AGC"))
        self.assertTrue(table.motifs_match("CAG", "CTG"))
        self.assertTrue(table.motifs_match("GCN", "NGC"))
        self.assertTrue(table.motifs_match("NGC", "GCC"))
        self.assertFalse(table.motifs_match("CGC", "CGA"))
        self.assertFalse(table.motifs_match("CAG", "CAGG"))

    def test_pathogenic_motif_written_in_the_genes_orientation_is_turned_to_the_plus_strand(self):
        self.assertEqual(table.plus_strand_pathogenic_motif("ATTTC", "AAAAT"), "GAAAT")

    def test_pathogenic_motif_already_on_the_plus_strand_is_kept(self):
        self.assertEqual(table.plus_strand_pathogenic_motif("AAGGG", "AAAAG"), "AAGGG")
        self.assertEqual(table.plus_strand_pathogenic_motif("TGGAA", "TAAAA"), "TGGAA")
        self.assertEqual(table.plus_strand_pathogenic_motif("ATTTC", "TTTTA"), "ATTTC")


class MatchTrainingLocusTest(unittest.TestCase):
    DISEASE_LOCUS = {"chrom": "chr8", "start_0based": 104588969, "end": 104588997, "metadata_motif": "CGC"}

    def test_a_locus_with_the_same_motif_is_preferred_over_one_overlapping_more(self):
        loci = [{"locus_id": "a", "chrom": "chr8", "start_0based": 104588961, "end": 104589000, "motif": "CGA"},
                {"locus_id": "b", "chrom": "chr8", "start_0based": 104588970, "end": 104588997, "motif": "CGC"}]
        self.assertEqual(table.match_training_locus(self.DISEASE_LOCUS, loci)["locus_id"], "b")

    def test_without_a_motif_match_the_locus_overlapping_most_is_taken(self):
        loci = [{"locus_id": "a", "chrom": "chr8", "start_0based": 104588961, "end": 104589000, "motif": "CGA"},
                {"locus_id": "b", "chrom": "chr8", "start_0based": 104588990, "end": 104588999, "motif": "GAT"}]
        self.assertEqual(table.match_training_locus(self.DISEASE_LOCUS, loci)["locus_id"], "a")

    def test_no_locus_on_the_chromosome_or_of_another_motif_length_is_none(self):
        loci = [{"locus_id": "a", "chrom": "chr9", "start_0based": 104588961, "end": 104589000, "motif": "CGC"},
                {"locus_id": "b", "chrom": "chr8", "start_0based": 104588961, "end": 104589000, "motif": "CG"}]
        self.assertIsNone(table.match_training_locus(self.DISEASE_LOCUS, loci))


class BuildTableTest(unittest.TestCase):
    def test_expanded_motif_is_the_pathogenic_one_only_where_it_differs_and_each_locus_gets_its_strchive_range(self):
        disease_loci = pd.DataFrame([
            {"disease_locus_id": "DAB1", "gene": "DAB1", "chrom": "chr1", "start_0based": 57367043, "end": 57367118,
             "metadata_motif": "AAAAT", "metadata_pathogenic_motif": "ATTTC", "normal_max": None,
             "pathogenic_min": 31, "inheritance": "AD"},
            {"disease_locus_id": "DMPK", "gene": "DMPK", "chrom": "chr19", "start_0based": 45770204, "end": 45770264,
             "metadata_motif": "CAG", "metadata_pathogenic_motif": "CAG", "normal_max": 34, "pathogenic_min": 50,
             "inheritance": "AD"},
            {"disease_locus_id": "MISSING", "gene": "X", "chrom": "chr2", "start_0based": 1, "end": 10,
             "metadata_motif": "CAG", "metadata_pathogenic_motif": "CAG", "normal_max": 1, "pathogenic_min": 2,
             "inheritance": "AD"}])
        training_loci = [
            {"locus_id": "1-57367043-57367118-AAAAT", "chrom": "chr1", "start_0based": 57367043, "end": 57367118,
             "motif": "AAAAT"},
            {"locus_id": "19-45770204-45770264-CAG", "chrom": "chr19", "start_0based": 45770204, "end": 45770264,
             "motif": "CAG"}]
        strchive_loci = [
            {"id": "DM1_DMPK", "chrom": "chr19", "start_hg38": 45770204, "stop_hg38": 45770266, "motif_len": 3,
             "pathogenic_min": 50, "pathogenic_max": 4000},
            {"id": "SCA37_DAB1", "chrom": "chr1", "start_hg38": 57367043, "stop_hg38": 57367119, "motif_len": 5,
             "pathogenic_max": None},
            {"id": "ELSEWHERE", "chrom": "chr19", "start_hg38": 1, "stop_hg38": 30, "motif_len": 3,
             "pathogenic_max": 9}]
        built, unmatched = table.build_table(disease_loci, training_loci, strchive_loci)
        self.assertEqual(unmatched, ["MISSING"])
        self.assertEqual(built.set_index("disease_locus_id")["expanded_motif"].to_dict(),
                         {"DAB1": "GAAAT", "DMPK": "CAG"})
        by_locus = built.set_index("disease_locus_id")
        self.assertEqual(by_locus.at["DMPK", "largest_reported_pathogenic_bp"], 12000)
        self.assertEqual(by_locus.at["DMPK", "smallest_reported_pathogenic_bp"], 150)
        self.assertTrue(pd.isna(by_locus.at["DAB1", "smallest_reported_pathogenic_bp"]))
        self.assertEqual(by_locus.at["DMPK", "strchive_ids"], "DM1_DMPK")
        self.assertEqual(table.catalog_entries(built)[1],
                         {"LocusId": "19-45770204-45770264-CAG", "LocusStructure": "(CAG)*",
                          "ReferenceRegion": "chr19:45770204-45770264", "VariantType": "Repeat"})


class LoadStrchiveLociTest(unittest.TestCase):
    RECORDS = [{"id": "DM1_DMPK", "chrom": "chr19", "pathogenic_max": 4000}]

    def test_reads_a_local_copy(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "STRchive-loci.json")
            with open(path, "w") as f:
                json.dump(self.RECORDS, f)
            self.assertEqual(table.load_strchive_loci(path), self.RECORDS)

    def test_downloads_an_https_url(self):
        with mock.patch.object(table.urllib.request, "urlopen",
                               return_value=io.BytesIO(json.dumps(self.RECORDS).encode())) as urlopen:
            self.assertEqual(table.load_strchive_loci(table.STRCHIVE_LOCI_JSON), self.RECORDS)
        urlopen.assert_called_once_with(table.STRCHIVE_LOCI_JSON)


class OverlappingStrchiveIdsAndPathogenicAlleleRangeBpTest(unittest.TestCase):
    LOCUS = {"chrom": "chr19", "start_0based": 1000, "end": 1060}

    def test_takes_the_smallest_and_largest_over_the_overlapping_strchive_loci_that_report_them(self):
        strchive_loci = [
            {"id": "B", "chrom": "chr19", "start_hg38": 1050, "stop_hg38": 1100, "motif_len": 3, "pathogenic_min": 30,
             "pathogenic_max": 50},
            {"id": "A", "chrom": "chr19", "start_hg38": 990, "stop_hg38": 1030, "motif_len": 5, "pathogenic_min": 12,
             "pathogenic_max": 40},
            {"id": "C", "chrom": "chr19", "start_hg38": 1000, "stop_hg38": 1060, "motif_len": 3,
             "pathogenic_min": None, "pathogenic_max": None},
            {"id": "BEYOND_THE_END", "chrom": "chr19", "start_hg38": 1060, "stop_hg38": 1090, "motif_len": 3,
             "pathogenic_min": 1, "pathogenic_max": 900},
            {"id": "OTHER_CHROMOSOME", "chrom": "chr1", "start_hg38": 1000, "stop_hg38": 1060, "motif_len": 3,
             "pathogenic_min": 1, "pathogenic_max": 900}]
        self.assertEqual(table.overlapping_strchive_ids_and_pathogenic_allele_range_bp(self.LOCUS, strchive_loci),
                         ("A,B,C", 60, 200))

    def test_nan_where_no_overlapping_strchive_locus_reports_a_minimum_or_maximum(self):
        strchive_ids, smallest_bp, largest_bp = table.overlapping_strchive_ids_and_pathogenic_allele_range_bp(
            self.LOCUS, [{"id": "C", "chrom": "chr19", "start_hg38": 1000, "stop_hg38": 1060, "motif_len": 3,
                          "pathogenic_min": None, "pathogenic_max": None}])
        self.assertEqual(strchive_ids, "C")
        self.assertTrue(pd.isna(smallest_bp))
        self.assertTrue(pd.isna(largest_bp))

    def test_empty_ids_and_nan_where_no_strchive_locus_overlaps(self):
        strchive_ids, smallest_bp, largest_bp = table.overlapping_strchive_ids_and_pathogenic_allele_range_bp(
            self.LOCUS, [])
        self.assertEqual(strchive_ids, "")
        self.assertTrue(pd.isna(smallest_bp))
        self.assertTrue(pd.isna(largest_bp))


class KnownDiseaseLociTest(unittest.TestCase):
    def test_reads_location_motifs_and_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "loci.tsv")
            pd.DataFrame({"LocusId": ["RFC1", "DMPK"], table.LOCATION_COLUMN: ["chr4:39348425-39348479",
                                                                                 "chr19:45770205-45770264"],
                          "RU": ["AAAAG", "CAG"], "Pathogenic Motif": ["AAGGG  (+ACAGG)", None],
                          "Gene": ["RFC1", "DMPK"], "NormalMax": [None, 34], "PathogenicMin": [400, 50],
                          "InheritanceMode": ["AR", "AD"]}).to_csv(path, sep="\t", index=False)
            loci = table.known_disease_loci(path)
        self.assertEqual(loci["start_0based"].tolist(), [39348424, 45770204])
        self.assertEqual(loci["metadata_pathogenic_motif"].tolist(), ["AAGGG", "CAG"])


if __name__ == "__main__":
    unittest.main()
