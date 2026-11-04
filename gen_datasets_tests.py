"""Tests for gen_datasets.py's dataset specs, in particular which input tree each one is read from."""

import os
import shutil
import tempfile
import unittest

import gen_datasets
import heldout


class HeldoutEvalDataDirTest(unittest.TestCase):
    """The held-out HPRC parquets come from the named held-out evaluation tree, not a fixed one.

    Several evaluation trees sit side by side, one per ExpansionHunter extraction and truth version, so
    the caller has to be able to name the one that matches the model being scored.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.held_out_sample = sorted(heldout.SAMPLES)[0]

    def _make_eval_tree(self, name, sample_names):
        """Creates ``<tmp>/<name>/real_43/<sample>.parquet`` for each name and returns the tree's path."""
        tree_dir = os.path.join(self.tmp_dir, name)
        os.makedirs(os.path.join(tree_dir, "real_43"))
        for sample_name in sample_names:
            with open(os.path.join(tree_dir, "real_43", "%s.parquet" % sample_name), "w") as handle:
                handle.write("")
        return tree_dir

    def test_the_parquets_come_from_the_named_tree(self):
        wanted = self._make_eval_tree("with_the_fix", [self.held_out_sample])
        self._make_eval_tree("without_the_fix", [self.held_out_sample])
        self.assertEqual(gen_datasets._heldout_parquets(wanted),
                         [os.path.join(wanted, "real_43", "%s.parquet" % self.held_out_sample)])

    def test_a_sample_that_is_not_held_out_is_left_out_of_the_named_tree(self):
        tree_dir = self._make_eval_tree("with_the_fix", [self.held_out_sample, "HG002"])
        self.assertNotIn("HG002", heldout.SAMPLES)  # a training sample, so never scored as held out
        self.assertEqual(gen_datasets._heldout_parquets(tree_dir),
                         [os.path.join(tree_dir, "real_43", "%s.parquet" % self.held_out_sample)])

    def test_the_dataset_specs_read_the_two_trees_separately(self):
        held_out_tree = self._make_eval_tree("held_out_tree", [self.held_out_sample])
        specs = gen_datasets.datasets(data_dir=os.path.join(self.tmp_dir, "training_tree"),
                                      heldout_eval_data_dir=held_out_tree)
        self.assertEqual(specs["heldout_hprc"]["parquets"],
                         [os.path.join(held_out_tree, "real_43", "%s.parquet" % self.held_out_sample)])
        self.assertEqual(specs["hg002_genome"]["parquets"],
                         [os.path.join(self.tmp_dir, "training_tree", "real_quick", "HG002_31x.parquet")])

    def test_the_held_out_label_counts_only_the_named_trees_parquets(self):
        held_out_tree = self._make_eval_tree("held_out_tree", sorted(heldout.SAMPLES)[:3])
        self._make_eval_tree("another_tree", sorted(heldout.SAMPLES))
        specs = gen_datasets.datasets(heldout_eval_data_dir=held_out_tree)
        self.assertEqual(specs["heldout_hprc"]["label"], "3 held-out HPRC samples")
        self.assertEqual(specs["heldout_hprc"]["coverage_label"],
                         "3 held-out HPRC samples (short-read WGS)")

    def test_the_default_tree_is_the_repos_own(self):
        self.assertEqual(gen_datasets.DEFAULT_HELDOUT_EVAL_DATA_DIR,
                         os.path.join(gen_datasets.HERE, "data_eval_43"))


if __name__ == "__main__":
    unittest.main()
