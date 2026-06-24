"""Parity unit tests for the shared EH-JSON feature extractor.

Asserts that ``eh_json_features`` parses real fixtures correctly and that each
contract (full / fast) populates exactly the columns it guarantees -- the
JSON<->extractor parity check the plan requires (sec 5.4 / sec 9).

Run with:  python3 -m unittest eh_json_features_tests -v
"""

import json
import os
import unittest

import eh_json_features as F


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fixtures")
FULL_FIXTURE = os.path.join(FIXTURE_DIR, "full_branch_fixture.json")
FAST_FIXTURE = os.path.join(FIXTURE_DIR, "sim_optimized-streaming.json")


class ParserTest(unittest.TestCase):
    def test_parse_counts(self):
        self.assertEqual(F.parse_counts("(20, 8), (22, 1)"), [(20, 8), (22, 1)])
        self.assertEqual(F.parse_counts("()"), [])
        self.assertEqual(F.parse_counts(""), [])
        self.assertEqual(F.parse_counts(None), [])

    def test_parse_genotype(self):
        self.assertEqual(F.parse_genotype("20/97"), [20, 97])
        self.assertEqual(F.parse_genotype("20"), [20])
        self.assertIsNone(F.parse_genotype("./."))
        self.assertIsNone(F.parse_genotype(""))

    def test_parse_ci(self):
        self.assertEqual(F.parse_ci("20-20/95-153", 2), [(20, 20), (95, 153)])
        self.assertEqual(F.parse_ci("17-17", 1), [(17, 17)])
        self.assertEqual(F.parse_ci("", 2), [(None, None), (None, None)])

    def test_parse_reference_region(self):
        self.assertEqual(F.parse_reference_region("chr12:6936716-6936773"),
                         ("chr12", 6936716, 6936773))
        self.assertEqual(F.parse_reference_region("bad"), (None, None, None))


class FullContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(FULL_FIXTURE):
            raise unittest.SkipTest(f"missing fixture {FULL_FIXTURE}")
        cls.rows = list(F.extract_rows(FULL_FIXTURE))

    def test_produced_rows(self):
        self.assertGreater(len(self.rows), 100)

    def test_all_full_branch(self):
        # low-mem-streaming shard => every record is full-branch (no QuickGenotype).
        self.assertTrue(all(r["genotyping_branch"] == "full" for r in self.rows))
        self.assertTrue(all(r["quick_genotype"] is False for r in self.rows))

    def test_full_contract_columns_present(self):
        # Every full row must carry every FULL_CONTRACT key; the AQM-derived
        # numeric features must be non-None for at least most rows.
        for r in self.rows[:500]:
            for col in F.FULL_CONTRACT:
                self.assertIn(col, r, f"missing column {col}")
        nonnull = lambda c: sum(1 for r in self.rows if r.get(c) is not None)
        for col in ["eh", "depth", "qd", "eh_q", "strand_bias_phred",
                    "left_flank_norm_depth", "ci_width", "motif_size"]:
            self.assertGreater(nonnull(col), 0.5 * len(self.rows), f"too many null {col}")

    def test_eh_matches_genotype_and_rank(self):
        # Per-locus the short allele (rank 0) eh <= long allele (rank 1) eh.
        by_variant = {}
        for r in self.rows:
            by_variant.setdefault((r["locus_id"], r["variant_id"]), []).append(r)
        for rows in by_variant.values():
            rows.sort(key=lambda r: r["allele_rank"])
            ehs = [r["eh"] for r in rows]
            self.assertEqual(ehs, sorted(ehs))

    def test_eh_q_is_qd_times_depth(self):
        for r in self.rows:
            if r.get("qd") is not None and r.get("depth") is not None:
                self.assertAlmostEqual(r["eh_q"], r["qd"] * r["depth"], places=3)


class FastContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(FAST_FIXTURE):
            raise unittest.SkipTest(f"missing fixture {FAST_FIXTURE}")
        cls.rows = list(F.extract_rows(FAST_FIXTURE))

    def test_fast_rows_tagged(self):
        fast = [r for r in self.rows if r["genotyping_branch"] == "fast"]
        self.assertGreater(len(fast), 0)
        self.assertTrue(all(r["quick_genotype"] is True for r in fast))

    def test_fast_contract_columns_present(self):
        fast = [r for r in self.rows if r["genotyping_branch"] == "fast"]
        for r in fast:
            for col in F.FAST_CONTRACT:
                self.assertIn(col, r, f"missing column {col}")
            # Full-only columns must NOT be present on fast rows.
            for col in F.FULL_ONLY_FIELDS:
                self.assertNotIn(col, r, f"fast row should not carry {col}")

    def test_fast_aqm_features_nonnull(self):
        fast = [r for r in self.rows if r["genotyping_branch"] == "fast"]
        for col in ["eh", "depth", "strand_bias_phred",
                    "flanking_total", "ci_width"]:
            self.assertTrue(any(r.get(col) is not None for r in fast), f"all-null {col}")


class SpanningModeFeatureTest(unittest.TestCase):
    """Locks in the spanning-mode + sister-allele feature logic (deep-dive Mode-3/4 fixes)."""

    def test_het_overcall_naive_recovers_truth(self):
        # EH over-called one allele (46) of a really-32/32 locus: naive mode = 32 (the truth),
        # per-allele mode for the 46 allele = its own (weak) 46 cluster.
        span = [(32, 8), (33, 2), (46, 2)]
        nm, ns, am, asup = F.spanning_mode_features(span, eh=46, sister_eh=32)
        self.assertEqual((nm, ns), (32, 8))
        self.assertEqual((am, asup), (46, 2))

    def test_het_exact_call_not_dragged(self):
        # EH exactly right at 34; sister 20 owns the only multi-read cluster. Per-allele mode must
        # be None (NOT 20) so the model is never told to contract the correct call.
        span = [(20, 4), (34, 1), (37, 1)]
        nm, ns, am, asup = F.spanning_mode_features(span, eh=34, sister_eh=20)
        self.assertEqual((nm, ns), (20, 4))          # singletons trimmed -> only (20,4) survives
        self.assertEqual((am, asup), (None, 0))      # no own-allele cluster -> NaN, no drag

    def test_homozygous_naive_equals_allele(self):
        nm, ns, am, asup = F.spanning_mode_features([(10, 30), (11, 3)], eh=10, sister_eh=10)
        self.assertEqual((nm, ns, am, asup), (10, 30, 10, 30))

    def test_hemizygous_no_sister(self):
        nm, ns, am, asup = F.spanning_mode_features([(8, 12)], eh=8, sister_eh=None)
        self.assertEqual((nm, ns, am, asup), (8, 12, 8, 12))

    def test_no_spanning_reads(self):
        self.assertEqual(F.spanning_mode_features([], eh=50, sister_eh=20), (None, 0, None, 0))

    def test_singleton_trim_keeps_all_when_no_multi(self):
        # All singletons -> nothing to trim; densest is the first max.
        nm, ns, _, _ = F.spanning_mode_features([(5, 1), (9, 1)], eh=5, sister_eh=9)
        self.assertEqual((nm, ns), (5, 1))

    def test_sister_eh_helper(self):
        self.assertEqual(F._sister_eh([20, 34], 0), 34)
        self.assertEqual(F._sister_eh([20, 34], 1), 20)
        self.assertIsNone(F._sister_eh([20], 0))
        self.assertEqual(F._sister_eh([10, 20, 30], 1), 10)   # nearest other allele to 20


if __name__ == "__main__":
    unittest.main(verbosity=2)
