"""Tests for the EH-JSON per-allele extractor (parsing, contract routing, gzip)."""

import gzip
import json
import os
import tempfile
import unittest

import eh_json


def _locus(genotype="20/97", quick=False, with_aqm=True):
    """Builds a minimal EH locus_result dict with one variant."""
    variant = {
        "VariantId": "L", "RepeatUnit": "CAG", "ReferenceRegion": "chr1:1000-1030",
        "Genotype": genotype, "GenotypeConfidenceInterval": "20-20/90-105",
        "CountsOfSpanningReads": "(20, 8), (97, 1)",
        "CountsOfFlankingReads": "(98, 2)",
        "CountsOfHighQualityUnambiguousReads": "(20, 6)",
    }
    if quick:
        variant["QuickGenotype"] = True
    if with_aqm:
        variant["AlleleQualityMetrics"] = {"Alleles": [
            {"AlleleNumber": 1, "AlleleSize": 20, "Depth": 30, "QD": 0.4,
             "HighQualityUnambiguousReads": 6, "StrandBiasBinomialPhred": 1.0,
             "MeanInsertedBasesWithinRepeats": 0.1, "MeanDeletedBasesWithinRepeats": 0.0,
             "LeftFlankNormalizedDepth": 1.1, "RightFlankNormalizedDepth": 0.9},
            {"AlleleNumber": 2, "AlleleSize": 97, "Depth": 28, "QD": 0.3,
             "HighQualityUnambiguousReads": 1, "StrandBiasBinomialPhred": 2.0,
             "MeanInsertedBasesWithinRepeats": 0.2, "MeanDeletedBasesWithinRepeats": 0.1,
             "LeftFlankNormalizedDepth": 1.0, "RightFlankNormalizedDepth": 1.0}]}
    return {"LocusId": "1-1000-1030-CAG", "Coverage": 30.0, "Variants": {"L": variant}}


class ParseTest(unittest.TestCase):
    def test_genotype(self):
        self.assertEqual(eh_json.parse_genotype("20/97"), [20, 97])
        self.assertEqual(eh_json.parse_genotype("20"), [20])
        self.assertIsNone(eh_json.parse_genotype("./."))
        self.assertIsNone(eh_json.parse_genotype(""))

    def test_counts(self):
        self.assertEqual(eh_json.parse_counts("(20, 8), (97, 1)"), [(20, 8), (97, 1)])
        self.assertEqual(eh_json.parse_counts("()"), [])

    def test_ci(self):
        self.assertEqual(eh_json.parse_ci("20-20/90-105", 2), [(20, 20), (90, 105)])
        self.assertEqual(eh_json.parse_ci("", 1), [(None, None)])


class ExtractTest(unittest.TestCase):
    def test_full_contract(self):
        rows = list(eh_json.extract_rows({"LocusResults": {"x": _locus()}}, sample_id="S"))
        self.assertEqual(len(rows), 2)
        a, b = rows
        self.assertEqual(a["eh"], 20)
        self.assertEqual(a["genotyping_branch"], "full")
        self.assertEqual(a["motif_size"], 3)
        self.assertEqual(a["spanning_at_called"], 8)   # (20, 8)
        self.assertEqual(b["spanning_at_called"], 1)   # (97, 1)
        self.assertEqual(a["flanking_above_called"], 2)  # (98,2) above eh=20
        self.assertEqual(a["left_flank_norm_depth"], 1.1)

    def test_quick_contract_omits_full_only(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(quick=True, with_aqm=False)}}, sample_id="S"))
        self.assertEqual(rows[0]["genotyping_branch"], "quick")
        self.assertIsNone(rows[0]["left_flank_norm_depth"])

    def test_no_call_dropped(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(genotype="./.")}}, sample_id="S"))
        self.assertEqual(rows, [])

    def test_gzip_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json.gz")
            with gzip.open(path, "wt") as f:
                json.dump({"LocusResults": {"x": _locus()}}, f)
            rows = list(eh_json.extract_rows(path, sample_id="S"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample_id"], "S")


if __name__ == "__main__":
    unittest.main()
