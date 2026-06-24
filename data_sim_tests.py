"""Smoke test for the simulated-data pipeline (``data_sim.py``).

This actually runs ExpansionHunter (both analysis modes) on ONE locus with a
couple of diploid pairs, so it stays fast while exercising the real merge ->
catalog -> EH -> extract path end to end. It asserts that rows are produced for
both modes, that optimized-streaming triggers the fast path at least once, that
the rank-paired ``true`` values come from the merged pair, and that ``eh`` is
present.

Run with:  python3 -m unittest data_sim_tests -v
"""

import os
import shutil
import tempfile
import unittest

import data_sim as DS
import mode_consistency_and_accuracy_tests as M


LOCUS_ID = "chr12-6936716-6936773-CAG"


class DataSimOneLocusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skip_reason = None
        if M.find_reference() is None:
            cls.skip_reason = "no hg38 reference FASTA (+.fai) found"
        elif not (os.path.exists(M.EH_BINARY) and os.access(M.EH_BINARY, os.X_OK)):
            cls.skip_reason = f"EH binary not found/executable: {M.EH_BINARY}"
        elif not os.path.isdir(os.path.join(M.SIM_DATA_DIR, LOCUS_ID)):
            cls.skip_reason = f"sim locus dir not found: {LOCUS_ID}"
        if cls.skip_reason:
            return

        reference = M.find_reference()
        cls.tmp = tempfile.mkdtemp(prefix="data_sim_test_")
        locus_dir = os.path.join(M.SIM_DATA_DIR, LOCUS_ID)
        _, chrom, start, end, motif = M.parse_locus_dir(LOCUS_ID)
        # Two pairs: a heterozygous and a homozygous genotype.
        sizes = M.allele_sizes_in_dir(locus_dir)
        cls.pairs = [(sizes[0], sizes[-1]), (sizes[0], sizes[0])]
        catalog = os.path.join(cls.tmp, "catalog.json")
        M.write_catalog(LOCUS_ID, chrom, start, end, motif, catalog)

        cls.rows = []
        for n1, n2 in cls.pairs:
            merged_bam = os.path.join(cls.tmp, f"a1_{n1}__a2_{n2}.bam")
            M.build_merged_bam(
                os.path.join(locus_dir, f"sim_{n1}x__10_150_450_50.bam"),
                os.path.join(locus_dir, f"sim_{n2}x__10_150_450_50.bam"),
                merged_bam)
            for mode in DS.ANALYSIS_MODES:
                json_path = DS.run_eh_json(
                    merged_bam, reference, catalog,
                    os.path.join(cls.tmp, f"{n1}_{n2}.{mode}"), mode)
                cls.rows.extend(DS.rows_for_run(
                    json_path, f"sim_{LOCUS_ID}_{n1}_{n2}",
                    tuple(sorted((n1, n2))), chrom, mode))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmp", None):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def test_rows_for_both_modes(self):
        modes = {r["analysis_mode"] for r in self.rows}
        self.assertIn("low-mem-streaming", modes)
        self.assertIn("optimized-streaming", modes)

    def test_low_mem_streaming_is_full_branch(self):
        full = [r for r in self.rows if r["analysis_mode"] == "low-mem-streaming"]
        self.assertGreater(len(full), 0)
        self.assertTrue(all(r["genotyping_branch"] == "full" for r in full))

    def test_optimized_streaming_has_quick_genotype(self):
        opt = [r for r in self.rows if r["analysis_mode"] == "optimized-streaming"]
        self.assertGreater(len(opt), 0)
        self.assertTrue(any(r["quick_genotype"] is True for r in opt))

    def test_true_values_in_merged_pair(self):
        valid_truths = {float(n) for pair in self.pairs for n in pair}
        for r in self.rows:
            self.assertIn(r["true"], valid_truths)

    def test_eh_present_and_sim_labels(self):
        for r in self.rows:
            self.assertIsNotNone(r["eh"])
            self.assertEqual(r["source"], "sim")
            self.assertEqual(r["purity"], 1.0)
            self.assertFalse(r["is_negative_locus"])
            self.assertEqual(r["chrom_raw"], "chr12")


if __name__ == "__main__":
    unittest.main(verbosity=2)
