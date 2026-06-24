"""Unit tests for build_dataset label/filter/chrom logic (no filesystem / GCS)."""

import unittest

import numpy as np
import pandas as pd

import build_dataset as B


class NormalizeChromTest(unittest.TestCase):
    def test_real_locus_id(self):
        df = pd.DataFrame({"locus_id": ["1-590-600-A", "22-1-2-AC", "X-5-9-G", "chrM-1-2-A"]})
        out = list(B.normalize_chrom(df))
        self.assertEqual(out[:3], ["1", "22", "X"])
        self.assertTrue(pd.isna(out[3]))

    def test_sim_chrom_raw(self):
        df = pd.DataFrame({"locus_id": ["chr12-1-2-CAG", "chrX-1-2-G", "chrM-1-2-A"],
                           "chrom_raw": ["chr12", "chrX", "chrM"]})
        out = list(B.normalize_chrom(df))
        self.assertEqual(out[:2], ["12", "X"])
        self.assertTrue(pd.isna(out[2]))


class LabelsAndFilterTest(unittest.TestCase):
    def make(self):
        return pd.DataFrame({
            "locus_id": ["1-1-2-A", "2-1-2-A", "3-1-2-A", "4-1-2-A", "5-1-2-A",
                         "chrM-1-2-A", "6-1-2-A", "7-1-2-A"],
            "chrom_raw": [None] * 8,
            "eh":      [12, 10, 8, 0, 12, 10, 11, 30],
            "true":    [10.0, 10.0, 10.0, 10.0, np.nan, 10.0, 10.0, 10.0],
            "purity":  [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.5, 1.0],
            "is_negative_locus": [False, False, False, False, False, False, False, True],
            "motif_size": [1, 1, 1, 1, 1, 1, 1, 1],
            "genotyping_branch": ["full"] * 8,
            "spanning_at_called": [5] * 8,
            "source": ["real"] * 8,
        })

    def test_filtering_and_labels(self):
        kept, drops = B.add_labels_and_filter(self.make())
        # Kept: row0 (TOO_LONG), row1 (OK exact), row2 (TOO_SHORT). Dropped: eh=0, true NaN, chrM,
        # impure(0.5), negative-control.
        self.assertEqual(set(kept["locus_id"]), {"1-1-2-A", "2-1-2-A", "3-1-2-A"})
        self.assertEqual(drops["nonpositive_eh_or_true"], 1)
        self.assertEqual(drops["missing_eh_or_true"], 1)
        self.assertEqual(drops["chrM_or_unknown_contig"], 1)
        self.assertEqual(drops["impure_below_0.9"], 1)
        self.assertEqual(drops["negative_control_locus"], 1)
        by = kept.set_index("locus_id")
        self.assertEqual(by.loc["1-1-2-A", "direction"], "TOO_LONG")   # dr=+2
        self.assertEqual(by.loc["2-1-2-A", "direction"], "OK")     # dr=0
        self.assertEqual(by.loc["3-1-2-A", "direction"], "TOO_SHORT")  # dr=-2
        self.assertAlmostEqual(by.loc["1-1-2-A", "q"], 1.2)
        self.assertAlmostEqual(by.loc["1-1-2-A", "t"], np.log(12) - np.log(10), places=6)
        self.assertEqual(list(kept["dir_code"]), [{"OK": 0, "TOO_LONG": 1, "TOO_SHORT": 2}[d]
                                                  for d in kept["direction"]])
        # size-tolerance + regime columns are added.
        self.assertEqual(by.loc["1-1-2-A", "regime"], "full_spanning")
        self.assertEqual(int(by.loc["1-1-2-A", "tol_repeats"]), 0)  # bp=10 (<50) -> exact band
        self.assertAlmostEqual(by.loc["1-1-2-A", "allele_bp"], 10.0)

    def _row(self, eh, true, motif, span=5, branch="full"):
        return pd.DataFrame({"locus_id": ["1-1-2-A"], "chrom_raw": [None], "eh": [eh],
                             "true": [float(true)], "purity": [1.0], "is_negative_locus": [False],
                             "motif_size": [motif], "genotyping_branch": [branch],
                             "spanning_at_called": [span], "source": ["real"]})

    def test_size_tolerance_band(self):
        # Small allele (bp=10 < 50 => tol 0): off-by-one is a miscall, not OK.
        self.assertEqual(B.add_labels_and_filter(self._row(11, 10, 1))[0].iloc[0]["direction"],
                         "TOO_LONG")
        # Large allele (motif 3, true 100 => 300bp => tol 4): off-by-3 is OK.
        self.assertEqual(B.add_labels_and_filter(self._row(103, 100, 3))[0].iloc[0]["direction"],
                         "OK")
        # ...but off-by-5 at 300bp exceeds tol 4 => TOO_LONG.
        self.assertEqual(B.add_labels_and_filter(self._row(105, 100, 3))[0].iloc[0]["direction"],
                         "TOO_LONG")

    def test_regime_routing(self):
        self.assertEqual(B.add_labels_and_filter(self._row(10, 10, 1, span=0))[0].iloc[0]["regime"],
                         "full_nonspanning")
        self.assertEqual(
            B.add_labels_and_filter(self._row(10, 10, 1, branch="fast"))[0].iloc[0]["regime"],
            "fast")


if __name__ == "__main__":
    unittest.main(verbosity=2)
