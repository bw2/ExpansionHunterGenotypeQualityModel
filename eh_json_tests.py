"""Tests for the EH-JSON per-allele extractor (parsing, contract routing, gzip)."""

import gzip
import json
import os
import tempfile
import unittest

import eh_json


def _locus(genotype="20/97", quick=False, with_aqm=True):
    """Builds a minimal EH locus_result dict with one variant.

    The AlleleQualityMetrics list mirrors what ExpansionHunter really emits: one entry per DISTINCT
    called allele, so a homozygous or hemizygous genotype gets a single entry (see
    ``HtsLowMemStreamingHelpers.cpp``'s ``if (isHet) {two} else {one}``). Handing a hom fixture two
    entries would hide the very asymmetry ``has_own_quality_metrics`` exists to record.
    """
    variant = {
        "VariantId": "L", "RepeatUnit": "CAG", "ReferenceRegion": "chr1:1000-1030",
        "Genotype": genotype, "GenotypeConfidenceInterval": "20-20/90-105",
        "CountsOfSpanningReads": "(20, 8), (97, 1)",
        "CountsOfFlankingReads": "(98, 2)",
        "CountsOfHighQualityUnambiguousReads": "(20, 6)",
        "CountsOfInrepeatReads": "()" if quick else "(33, 4), (34, 1)",
    }
    if quick:
        variant["QuickGenotype"] = True
    if with_aqm:
        alleles = [
            {"AlleleNumber": 1, "AlleleSize": 20, "Depth": 30, "QD": 0.4,
             "HighQualityUnambiguousReads": 6, "StrandBiasBinomialPhred": 1.0,
             "MeanInsertedBasesWithinRepeats": 0.1, "MeanDeletedBasesWithinRepeats": 0.0,
             "LeftFlankNormalizedDepth": 1.1, "RightFlankNormalizedDepth": 0.9},
            {"AlleleNumber": 2, "AlleleSize": 97, "Depth": 28, "QD": 0.3,
             "HighQualityUnambiguousReads": 1, "StrandBiasBinomialPhred": 2.0,
             "MeanInsertedBasesWithinRepeats": 0.2, "MeanDeletedBasesWithinRepeats": 0.1,
             "LeftFlankNormalizedDepth": 1.0, "RightFlankNormalizedDepth": 1.0}]
        called = eh_json.parse_genotype(genotype) or []
        variant["AlleleQualityMetrics"] = {"Alleles": alleles[:max(1, len(set(called)))]}
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
        self.assertEqual(a["coverage"], 30.0)             # per-locus, shared by both alleles
        self.assertEqual(b["coverage"], 30.0)
        self.assertEqual(a["flanking_total"], 2)          # (98, 2)
        self.assertAlmostEqual(a["flanking_frac"], 2 / 11)  # 2 / (2 flanking + 9 spanning)
        self.assertEqual(a["n_alleles"], 2)
        self.assertEqual(a["n_distinct_alleles"], 2)      # 20/97 is het
        self.assertEqual(a["inrepeat_total"], 5)          # (33, 4), (34, 1); per variant, so shared
        self.assertEqual(b["inrepeat_total"], 5)

    def test_homozygous_genotype_has_one_distinct_allele(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(genotype="20/20")}}, sample_id="S"))
        self.assertEqual([r["n_alleles"] for r in rows], [2, 2])
        self.assertEqual([r["n_distinct_alleles"] for r in rows], [1, 1])

    def test_only_the_first_copy_of_a_homozygous_call_has_its_own_quality_metrics(self):
        # EH emits one AlleleQualityMetrics entry for a hom call and scores the model once per entry,
        # so the rank-1 row is an allele inference never produces -- it must be marked, not silently
        # handed the rank-0 allele's read metrics and the LONG truth allele.
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(genotype="20/20")}}, sample_id="S"))
        self.assertEqual([r["has_own_quality_metrics"] for r in rows], [True, False])

    def test_both_copies_of_a_het_call_have_their_own_quality_metrics(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(genotype="20/97")}}, sample_id="S"))
        self.assertEqual([r["has_own_quality_metrics"] for r in rows], [True, True])
        self.assertEqual([r["depth"] for r in rows], [30, 28])  # each read its OWN entry

    def test_hemizygous_genotype_has_one_allele(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(genotype="20")}}, sample_id="S"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_alleles"], 1)
        self.assertEqual(rows[0]["n_distinct_alleles"], 1)
        self.assertTrue(rows[0]["has_own_quality_metrics"])

    def test_flanking_frac_is_zero_without_flanking_reads(self):
        locus = _locus()
        locus["Variants"]["L"]["CountsOfFlankingReads"] = ""
        rows = list(eh_json.extract_rows({"LocusResults": {"x": locus}}, sample_id="S"))
        self.assertEqual(rows[0]["flanking_total"], 0)
        self.assertEqual(rows[0]["flanking_frac"], 0.0)   # 0, never NaN -- also covers 0 spanning

    def test_quick_contract_omits_full_only(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(quick=True, with_aqm=False)}}, sample_id="S"))
        self.assertEqual(rows[0]["genotyping_branch"], "quick")
        self.assertIsNone(rows[0]["left_flank_norm_depth"])
        self.assertIsNone(rows[0]["inrepeat_total"])  # the fast path never counts in-repeat reads

    def test_no_call_emits_rows_with_null_eh(self):
        rows = list(eh_json.extract_rows(
            {"LocusResults": {"x": _locus(genotype="./.")}}, sample_id="S"))
        self.assertEqual(len(rows), eh_json.NO_CALL_RANKS)
        self.assertEqual([r["allele_rank"] for r in rows], list(range(eh_json.NO_CALL_RANKS)))
        for r in rows:
            self.assertIsNone(r["eh"])              # eh undefined -> dropped from training later
            self.assertIsNone(r["spanning_at_called"])
            for col in ("n_alleles", "n_distinct_alleles", "flanking_total", "flanking_frac",
                        "coverage"):
                self.assertIsNone(r[col])           # no genotype / no reads -> nothing to report
            self.assertEqual(r["motif_size"], 3)    # catalog fields still populated for the report
            self.assertEqual(r["num_repeats_in_reference"], 10)  # (1030-1000)/3
            self.assertEqual(r["genotyping_branch"], "full")

    def test_gzip_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json.gz")
            with gzip.open(path, "wt") as f:
                json.dump({"LocusResults": {"x": _locus()}}, f)
            rows = list(eh_json.extract_rows(path, sample_id="S"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample_id"], "S")

    def test_typical_read_length_is_the_largest_per_locus_mean(self):
        # Per-locus ReadLength is a mean, so the largest one is the best estimate of EH's typical length.
        eh = {"LocusResults": {"x": {"Variants": {}}, "y": dict(_locus(), ReadLength=148),
                               "z": dict(_locus(), ReadLength=151)}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json.gz")
            with gzip.open(path, "wt") as f:
                json.dump(eh, f)
            self.assertEqual(eh_json.typical_read_length_in_file(path), 151)

    def test_path_rows_match_parsed_dict_rows(self):
        # A path is streamed with ijson; its rows (floats included) must equal those from the parsed dict.
        eh = {"SampleParameters": {"SampleId": "FromFile", "Sex": "Female"},
              "LocusResults": {"x": _locus(), "y": _locus()}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            with open(path, "w") as f:
                json.dump(eh, f)
            from_path = list(eh_json.extract_rows(path))
        self.assertEqual(from_path, list(eh_json.extract_rows(eh)))
        self.assertEqual(from_path[0]["sample_id"], "FromFile")


if __name__ == "__main__":
    unittest.main()
