"""Unit tests for the full-branch real join logic in ``data_gcs.py``.

Exercises the rank-pairing / truth-join on a tiny synthetic JSON + TSV pair
written to temp files -- nothing here touches GCS. Covers: size-sorted
``allele_rank`` assignment on the TSV side, het / homozygous-EH pairing, the
unique-key assertion, unmatched (JSON-only) rows kept with NaN truth, and that
``eh`` comes from the JSON while ``true`` / ``purity`` come from the TSV.

Run with:  python3 -m unittest data_gcs_tests -v
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import data_gcs as D


def _variant(variant_id, region, unit, genotype, ci, spanning, aqm_sizes):
    """Builds a minimal full-branch EH variant dict (no QuickGenotype)."""
    return {
        "VariantId": variant_id,
        "VariantType": "Repeat",
        "ReferenceRegion": region,
        "RepeatUnit": unit,
        "Genotype": genotype,
        "GenotypeConfidenceInterval": ci,
        "CountsOfSpanningReads": spanning,
        "CountsOfFlankingReads": "()",
        "CountsOfInrepeatReads": "()",
        "CountsOfHighQualityUnambiguousReads": spanning,
        "AlleleQualityMetrics": {"Alleles": [
            {"AlleleNumber": i + 1, "AlleleSize": sz, "Depth": 30.0, "QD": 1.5,
             "StrandBiasBinomialPhred": 0.0, "ConfidenceIntervalDividedByAlleleSize": 0.1,
             "MeanInsertedBasesWithinRepeats": 0.0, "MeanDeletedBasesWithinRepeats": 0.0,
             "HighQualityUnambiguousReads": 10, "LeftFlankNormalizedDepth": 1.0,
             "RightFlankNormalizedDepth": 1.0}
            for i, sz in enumerate(aqm_sizes)]},
    }


def _write_json(path):
    """Writes a synthetic EH JSON with 4 loci (het, hom-EH, JSON-only, negative)."""
    locus = lambda lid, var: {"LocusId": lid, "Coverage": 30.0, "ReadLength": 150,
                              "FragmentLength": 350, "AlleleCount": 2,
                              "Variants": {lid: dict(var, VariantId=lid)}}
    eh_json = {"SampleParameters": {"SampleId": "x", "Sex": "Male"}, "LocusResults": {
        "A": locus("A", _variant("A", "chrA:1000-1018", "AT", "8/14", "8-8/14-16",
                                  "(8, 10), (14, 5)", [8, 14])),
        "B": locus("B", _variant("B", "chrB:2000-2032", "AT", "16/16", "16-16/16-16",
                                  "(16, 20)", [16, 16])),
        "C": locus("C", _variant("C", "chrC:3000-3018", "AT", "5/9", "5-5/9-11",
                                  "(5, 8), (9, 4)", [5, 9])),
        "D": locus("D", _variant("D", "chrD:4000-4040", "AT", "20/20", "20-20/20-20",
                                  "(20, 25)", [20, 20])),
    }}
    with open(path, "w") as f:
        json.dump(eh_json, f)


# Truth TSV: one row per (LocusId, allele); EHv5 / Truth rows deliberately out of
# size order to exercise the sort. Locus C is absent (JSON-only -> unmatched).
_TSV_COLS = ["LocusId", "NumRepeats: Allele: Truth", "RepeatPurity: Allele: Truth",
             "NumRepeats: Allele: EHv5", "TruthSetOrNegativeLocus",
             "Allele: Concordance: EHv5 vs Truth"]
_TSV_ROWS = [
    # Locus A het, rows out of order: long allele first.
    ["A", "15", "0.95", "14", "TruthSet", "Discordant"],
    ["A", "9", "1.0", "8", "TruthSet", "ExactlyTheSame"],
    # Locus B: EH homozygous (16/16) but truth het (16, 18); rows out of order.
    ["B", "18", "0.90", "16", "TruthSet", "Discordant"],
    ["B", "16", "1.0", "16", "TruthSet", "ExactlyTheSame"],
    # Locus D: negative-control locus (no truth size).
    ["D", "", "1.0", "20", "NegativeLocus", "Discordant"],
    ["D", "", "1.0", "20", "NegativeLocus", "Discordant"],
]


def _write_tsv(path):
    """Writes the synthetic gzipped truth TSV."""
    with gzip.open(path, "wt") as f:
        f.write("\t".join(_TSV_COLS) + "\n")
        for row in _TSV_ROWS:
            f.write("\t".join(row) + "\n")


class JoinLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.json_path = os.path.join(cls.tmp, "syn.json")
        cls.tsv_path = os.path.join(cls.tmp, "syn.alleles.tsv.gz")
        _write_json(cls.json_path)
        _write_tsv(cls.tsv_path)
        cls.json_df = D.load_json_rows([cls.json_path], sample_id="HG002_10x")
        cls.tsv_df = D.load_truth_tsv(cls.tsv_path)
        cls.merged, cls.n_matched = D.join_truth(cls.json_df, cls.tsv_df)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _row(self, df, locus, rank):
        return df[(df["locus_id"] == locus) & (df["allele_rank"] == rank)].iloc[0]

    def test_json_all_full_branch(self):
        self.assertEqual(len(self.json_df), 8)  # 4 loci x 2 alleles
        self.assertTrue((self.json_df["genotyping_branch"] == "full").all())

    def test_truth_rank_assignment(self):
        # Within each LocusId, allele_rank 0/1 follow ascending (EHv5, Truth) size.
        self.assertTrue(self.tsv_df.set_index(["LocusId", "allele_rank"]).index.is_unique)
        a = self.tsv_df[self.tsv_df["LocusId"] == "A"].sort_values("allele_rank")
        self.assertEqual(list(a["tsv_eh"]), [8, 14])      # short allele -> rank 0
        self.assertEqual(list(a["true"]), [9.0, 15.0])
        b = self.tsv_df[self.tsv_df["LocusId"] == "B"].sort_values("allele_rank")
        self.assertEqual(list(b["tsv_eh"]), [16, 16])      # hom-EH tie
        self.assertEqual(list(b["true"]), [16.0, 18.0])   # secondary sort by truth

    def test_match_rate(self):
        # A, B, D matched (6 rows); C is JSON-only (2 unmatched).
        self.assertEqual(self.n_matched, 6)
        self.assertEqual(len(self.merged), 8)
        self.assertAlmostEqual(self.n_matched / len(self.merged), 0.75)

    def test_eh_from_json_truth_from_tsv(self):
        # eh is the JSON call; tsv_eh is the audit copy; true/purity come from TSV.
        r0 = self._row(self.merged, "A", 0)
        self.assertEqual(r0["eh"], 8)            # JSON Genotype
        self.assertEqual(r0["tsv_eh"], 8)        # TSV audit
        self.assertEqual(r0["true"], 9.0)
        self.assertEqual(r0["purity"], 1.0)
        r1 = self._row(self.merged, "A", 1)
        self.assertEqual(r1["eh"], 14)
        self.assertEqual(r1["true"], 15.0)
        self.assertEqual(r1["purity"], 0.95)

    def test_homozygous_pairs_harmlessly(self):
        # Both B truth sizes attach to the (identical-eh) JSON rows.
        b = self.merged[self.merged["locus_id"] == "B"].sort_values("allele_rank")
        self.assertEqual(set(b["true"]), {16.0, 18.0})
        self.assertTrue((b["eh"] == 16).all())

    def test_unmatched_kept_with_nan(self):
        c = self.merged[self.merged["locus_id"] == "C"]
        self.assertEqual(len(c), 2)
        self.assertTrue(c["true"].isna().all())
        self.assertTrue(c["is_negative_locus"].isna().all())

    def test_negative_control_flag(self):
        d = self.merged[self.merged["locus_id"] == "D"]
        self.assertTrue((d["is_negative_locus"] == True).all())
        # Matched negative loci have no truth size.
        self.assertTrue(d["true"].isna().all())

    def test_duplicate_key_raises(self):
        dup = pd.concat([self.json_df, self.json_df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            D.join_truth(dup, self.tsv_df)


class CoverageParseTest(unittest.TestCase):
    def test_nominal_coverage_from_label(self):
        # build_combo derives the nominal coverage column from the dir label.
        self.assertEqual(D.parse_nominal_coverage("10x"), 10.0)
        self.assertEqual(D.parse_nominal_coverage("46x"), 46.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
